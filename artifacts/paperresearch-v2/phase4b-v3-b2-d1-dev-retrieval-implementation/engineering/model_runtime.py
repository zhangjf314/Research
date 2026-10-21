"""Pinned local batch1 CPU/float32 encoders; no network/model/device fallback."""
import time

import common as c


class Models:
    def __init__(self, budget, manifest):
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
        self.torch, self.budget, self.manifest = torch, budget, manifest
        self.inference_seconds = {}
        self.truncations = {}
        torch.set_num_threads(manifest['execution']['torch_num_threads'])
        torch.set_num_interop_threads(manifest['execution']['torch_num_interop_threads'])
        torch.manual_seed(manifest['execution']['seed'])
        torch.use_deterministic_algorithms(True)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(c.OUT / 'models/embedding', local_files_only=True, trust_remote_code=False, padding_side='left', truncation_side='right')
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(c.OUT / 'models/reranker', local_files_only=True, trust_remote_code=False, padding_side='left', truncation_side='right')
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(c.OUT / 'models/visual', local_files_only=True, truncation_side='right')
        self.clip_processor = CLIPImageProcessor.from_pretrained(c.OUT / 'models/visual', local_files_only=True)
        kwargs = {'local_files_only': True, 'trust_remote_code': False, 'dtype': torch.float32, 'attn_implementation': 'eager', 'device_map': None}
        self.embedding_model = AutoModel.from_pretrained(c.OUT / 'models/embedding', **kwargs).to('cpu').eval()
        self.embedding_model.config.use_cache = False
        budget.check()
        print('MODEL LOAD: embedding CPU float32', flush=True)
        self.reranker_model = AutoModelForCausalLM.from_pretrained(c.OUT / 'models/reranker', **kwargs).to('cpu').eval()
        self.reranker_model.config.use_cache = False
        budget.check()
        print('MODEL LOAD: reranker CPU float32', flush=True)
        self.visual_model = CLIPModel.from_pretrained(c.OUT / 'models/visual', **kwargs).to('cpu').eval()
        budget.check()
        print('MODEL LOAD: visual CPU float32', flush=True)
        self.prefix = self.reranker_tokenizer.encode(manifest['models']['reranker']['input_contract']['prefix'], add_special_tokens=False)
        self.suffix = self.reranker_tokenizer.encode(manifest['models']['reranker']['input_contract']['suffix'], add_special_tokens=False)
        no = self.reranker_tokenizer.encode('no', add_special_tokens=False)
        yes = self.reranker_tokenizer.encode('yes', add_special_tokens=False)
        assert len(no) == len(yes) == 1 and no != yes, 'YES_NO_TOKEN_COMPATIBILITY_FAILURE'
        self.no, self.yes = no[0], yes[0]
        for model in [self.embedding_model, self.reranker_model, self.visual_model]:
            assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'MODEL_DEVICE_DTYPE_MISMATCH'
            assert not model.training

    def _charge(self, purpose, tokens):
        self.budget.check()
        self.budget.charge('max_model_forward_input_records')
        self.budget.charge('max_total_encoder_reranker_padded_tokens', tokens)
        field = {'synthetic': 'max_synthetic_model_input_records', 'index_dense': 'max_document_embeddings_all_passes',
                 'query_dense': 'max_Qwen_query_embeddings', 'index_image': 'max_image_embeddings_all_passes',
                 'query_image_text': 'max_CLIP_query_embeddings', 'rerank': 'max_reranker_pairs'}[purpose]
        self.budget.charge(field)
        self.budget.checkpoint('FORWARD_CHARGED_BEFORE_EXECUTION')

    def _vector(self, value, dimension):
        torch = self.torch
        assert value.shape == (1, dimension) and value.dtype == torch.float32 and value.device.type == 'cpu'
        assert torch.isfinite(value).all() and torch.linalg.vector_norm(value).item() > 0
        value = torch.nn.functional.normalize(value, p=2, dim=1, eps=1e-12)
        assert torch.isfinite(value).all() and abs(torch.linalg.vector_norm(value).item() - 1) < 1e-5
        self.budget.check()
        return value[0].numpy().copy()

    def embed_tokens(self, text):
        original = self.embedding_tokenizer(text, add_special_tokens=True)['input_ids']
        if len(original) > 4096:
            self.truncations['embedding'] = self.truncations.get('embedding', 0) + 1
        inputs = self.embedding_tokenizer(text, padding=True, truncation=True, max_length=4096, add_special_tokens=True, return_tensors='pt')
        assert inputs['input_ids'].shape[0] == 1 and inputs['input_ids'].shape[1] <= 4096
        return inputs

    def dense(self, text, purpose='synthetic'):
        start = time.perf_counter()
        inputs = self.embed_tokens(text)
        self._charge(purpose, inputs['input_ids'].shape[1])
        with self.torch.inference_mode():
            output = self.embedding_model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
            vector = self._vector(output, 1024)
        self.inference_seconds['embedding'] = self.inference_seconds.get('embedding', 0) + time.perf_counter() - start
        return vector

    def rerank_tokens(self, query, document):
        template = self.manifest['models']['reranker']['input_contract']['body_template']
        text = template.format(query=query, document=document)
        raw = self.reranker_tokenizer.encode(text, add_special_tokens=False)
        bound = 4096 - len(self.prefix) - len(self.suffix)
        assert bound > 0
        if len(raw) > bound:
            self.truncations['reranker'] = self.truncations.get('reranker', 0) + 1
        tokens = self.prefix + raw[:bound] + self.suffix
        assert len(tokens) <= 4096
        return {'input_ids': self.torch.tensor([tokens], dtype=self.torch.long),
                'attention_mask': self.torch.ones((1, len(tokens)), dtype=self.torch.long)}

    def rerank(self, query, document, purpose='synthetic'):
        start = time.perf_counter()
        inputs = self.rerank_tokens(query, document)
        self._charge(purpose, inputs['input_ids'].shape[1])
        with self.torch.inference_mode():
            # Retain only required final-position logits; same frozen yes/no score.
            logits = self.reranker_model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[:, -1, :]
            pair = logits[:, [self.no, self.yes]].float()
            score = self.torch.softmax(pair, dim=1)[0, 1].item()
        assert 0 <= score <= 1 and self.torch.isfinite(pair).all(), 'NONFINITE_RERANK_SCORE'
        self.budget.check()
        self.inference_seconds['reranker'] = self.inference_seconds.get('reranker', 0) + time.perf_counter() - start
        return score

    def clip_text_tokens(self, query):
        length = len(self.clip_tokenizer.encode(query, add_special_tokens=True))
        if length > 77:
            self.truncations['clip_text'] = self.truncations.get('clip_text', 0) + 1
        inputs = self.clip_tokenizer(query, padding=True, truncation=True, max_length=77, return_tensors='pt')
        assert inputs['input_ids'].shape[0] == 1 and inputs['input_ids'].shape[1] <= 77
        return inputs

    def clip_text(self, query, purpose='synthetic'):
        start = time.perf_counter()
        inputs = self.clip_text_tokens(query)
        self._charge(purpose, inputs['input_ids'].shape[1])
        with self.torch.inference_mode():
            output = self.visual_model.get_text_features(**inputs, return_dict=True).pooler_output
            vector = self._vector(output, 512)
        self.inference_seconds['clip_text'] = self.inference_seconds.get('clip_text', 0) + time.perf_counter() - start
        return vector

    def clip_image_pixels(self, image):
        from PIL import Image
        assert image.mode == 'RGB' and image.width > 0 and image.height > 0
        side = max(image.width, image.height)
        square = Image.new('RGB', (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        pixels = self.clip_processor(images=square, return_tensors='pt')['pixel_values']
        assert pixels.shape == (1, 3, 224, 224) and pixels.dtype == self.torch.float32
        assert self.torch.isfinite(pixels).all()
        return pixels

    def clip_image(self, image, purpose='synthetic'):
        start = time.perf_counter()
        pixels = self.clip_image_pixels(image)
        self._charge(purpose, 0)
        with self.torch.inference_mode():
            output = self.visual_model.get_image_features(pixel_values=pixels, return_dict=True).pooler_output
            vector = self._vector(output, 512)
        self.inference_seconds['clip_image'] = self.inference_seconds.get('clip_image', 0) + time.perf_counter() - start
        return vector

    def synthetic_preflight(self):
        import numpy as np
        from PIL import Image
        text = 'A synthetic blue rectangle is above a red circle.'
        first, second = self.dense(text), self.dense(text)
        assert np.array_equal(first, second), 'NONDETERMINISTIC_EMBEDDING'
        relevance = self.rerank('Where is the rectangle?', text)
        assert relevance == self.rerank('Where is the rectangle?', text), 'NONDETERMINISTIC_RERANK'
        query_vector = self.clip_text(text)
        image = Image.new('RGB', (80, 40), (64, 128, 192))
        pixels = self.clip_image_pixels(image)
        white = self.torch.tensor([(1 - m) / s for m, s in zip(self.clip_processor.image_mean, self.clip_processor.image_std)])
        assert self.torch.allclose(pixels[0, :, 0, 0], white, atol=1e-5), 'PREPROCESSOR_WHITE_LETTERBOX_MISMATCH'
        image_vector = self.clip_image(image)
        assert np.array_equal(image_vector, self.clip_image(image)), 'NONDETERMINISTIC_CLIP_IMAGE'
        long = 'synthetic ' * 5000
        assert self.embed_tokens(long)['input_ids'].shape == (1, 4096)
        rerank = self.rerank_tokens(text, long)
        assert rerank['input_ids'].shape == (1, 4096)
        assert rerank['input_ids'][0, -len(self.suffix):].tolist() == self.suffix
        assert self.clip_text_tokens(long)['input_ids'].shape == (1, 77)
        assert self.clip_text_tokens(long)['input_ids'][0, -1].item() == self.clip_tokenizer.eos_token_id
        self.budget.check()
        return {'Qwen_embedding_dimension': len(first), 'CLIP_text_dimension': len(query_vector), 'CLIP_image_dimension': len(image_vector),
                'reranker_finite_probability': relevance, 'repeatability_bitwise': True, 'query_document_tokens_bound4096': True,
                'CLIP_tokens_bound77': True, 'image_preprocess_shape': list(pixels.shape), 'white_letterbox_normalization': 'PASS',
                'CPU_float32_all_parameters': 'PASS', 'synthetic_forward_input_records': self.budget.counts.get('max_synthetic_model_input_records', 0),
                'peak_RSS_bytes': self.budget.peak_RSS, 'inference_seconds': self.inference_seconds}
