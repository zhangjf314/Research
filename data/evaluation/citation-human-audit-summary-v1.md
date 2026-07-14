# Citation Human Audit Summary v1

- Reviewer: `OpenAI GPT-5.6 Thinking (AI-assisted manual citation audit)`
- Reviewed at: `2026-07-14`
- Samples: `30/30`
- Pending: `0`

## Label distribution

| Label | Count |
|---|---:|
| `fully_supported` | 5 |
| `partially_supported` | 2 |
| `related_but_insufficient` | 7 |
| `unsupported` | 16 |
| `gold_annotation_too_narrow` | 0 |

## Item-level decisions

| # | Question | Automated | Human label | Decision summary |
|---:|---|---|---|---|
| 1 | `q001` | `semantic_support_non_gold` | `partially_supported` | The cited author-contribution note supports that the work developed Transformer models and replaced RNNs with self-attention, but it does not fully support the broader claim that the architecture is based solely on attention and replaces both recurrent and convolutional layers. |
| 2 | `q001` | `semantic_support_non_gold` | `related_but_insufficient` | The cited author-contribution note concerns development of Transformer models and accelerated research, but it does not establish the stated research problem of improving sequence-transduction quality and efficiency. |
| 3 | `q001` | `semantic_support_non_gold` | `related_but_insufficient` | The citation mentions improved results and accelerated research in an author-contribution note, but it does not directly substantiate superior model quality, parallelizability, or reduced training time relative to prior models. |
| 4 | `q002` | `semantic_support_non_gold` | `fully_supported` | The cited conclusion directly states that Transformer is the first sequence-transduction model based entirely on attention and replaces recurrent layers with multi-headed self-attention. |
| 5 | `q002` | `semantic_support_non_gold` | `related_but_insufficient` | The citation describes the attention-only architecture but provides no evidence that it trains significantly faster than recurrent or convolutional alternatives. |
| 6 | `q002` | `semantic_support_non_gold` | `related_but_insufficient` | The citation identifies the architecture but does not report the WMT English-German or English-French state-of-the-art results claimed. |
| 7 | `q003` | `semantic_support_non_gold` | `fully_supported` | The citation directly states that Transformer replaces recurrent layers with multi-headed self-attention. |
| 8 | `q004` | `semantic_support_non_gold` | `fully_supported` | The abstract directly reports evaluation on WMT 2014 English-German and English-French and gives state-of-the-art translation results. |
| 9 | `q004` | `semantic_support_non_gold` | `unsupported` | The cited passage concerns an attention-head ablation on translation BLEU and contains no evidence about English constituency parsing or small/large parsing datasets. |
| 10 | `q008` | `semantic_support_non_gold` | `unsupported` | The citation is only a bibliographic author entry and provides no support for the next-sentence-prediction claim. |
| 11 | `q007` | `same_gold_page` | `fully_supported` | The cited block explicitly explains that masked language modeling fuses left and right context to pre-train a deep bidirectional Transformer. |
| 12 | `q007` | `same_gold_page` | `unsupported` | The citation is only the heading 'Related Work' and does not support the claim about reducing task-specific architecture engineering or achieving state of the art across many tasks. |
| 13 | `q007` | `same_gold_page` | `unsupported` | The citation is only the heading 'Related Work' and provides no evidence that BERT advances the state of the art on eleven NLP tasks. |
| 14 | `q010` | `same_gold_page` | `related_but_insufficient` | The citation establishes that pre-training benefits sentence- and token-level tasks, but it does not state the unidirectionality limitation or the need for bidirectional context in question answering. |
| 15 | `q015` | `same_gold_page` | `fully_supported` | The cited passage directly states that the paper did not achieve state-of-the-art WMT results and discusses English-only pre-training, backtranslation, and other cross-lingual training as likely reasons. |
| 16 | `q015` | `same_gold_page` | `unsupported` | The citation is about T5 scaling and baseline-1T ablations, not ROUGE/coherence mismatch or extractive summarization on CNN/Daily Mail. |
| 17 | `q015` | `same_gold_page` | `unsupported` | The citation contains only a page number and provides no support for the claim about repetitive maximum-likelihood summaries or summary coherence. |
| 18 | `q017` | `same_gold_page` | `unsupported` | The citation only says 'Equal contribution' and does not identify who optimized the Transformer implementation. |
| 19 | `q017` | `same_gold_page` | `unsupported` | The citation only says 'Equal contribution' and does not identify who created the text datasets. |
| 20 | `q025` | `same_gold_page` | `related_but_insufficient` | The cited text begins a general limitation discussion about scaling language models, but it does not substantiate the specific claimed weaknesses on WIC, ANLI, or reading-comprehension tasks. |
| 21 | `q002` | `weakly_related` | `unsupported` | The citation contains only the fragment 'my' and provides no support for the multi-head-attention motivation. |
| 22 | `q008` | `weakly_related` | `unsupported` | The citation is only a bibliographic author entry and provides no evidence about BERT, bidirectional pre-training, or masked language modeling. |
| 23 | `q014` | `weakly_related` | `unsupported` | The cited Romanian text fragment is unrelated to the claimed benchmark suite and validation-set reporting. |
| 24 | `q016` | `weakly_related` | `unsupported` | The citation is a bibliography entry and provides no support for compute-optimal allocation between model size and training data. |
| 25 | `q019` | `unsupported` | `unsupported` | The citation is an acknowledgements passage and provides no support for the fitted scaling-law parameters or Table 4. |
| 26 | `q026` | `weakly_related` | `unsupported` | The citation is a partial author list and provides no support for parameter-efficient transfer learning. |
| 27 | `q026` | `weakly_related` | `unsupported` | The citation is a partial author list and provides no support for evaluating prompt strategies. |
| 28 | `q032` | `weakly_related` | `partially_supported` | The cited results table provides empirical evidence for LoRA on at least one benchmark setting, but it does not by itself support the full claim that experiments cover WebNLG, GLUE, and WikiSQL. |
| 29 | `q039` | `weakly_related` | `related_but_insufficient` | The citation describes human-preference evaluation on API prompts, but it does not specify the claimed harmful-behavior proxy criteria such as protected-class denigration, sexual content, or violence. |
| 30 | `q046` | `unsupported` | `unsupported` | The cited passage is about an occupation/participant/pronoun coreference example and does not support a claim about mathematical problem-solving benchmarks. |
