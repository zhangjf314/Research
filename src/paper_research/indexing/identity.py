"""Content-backed manifests for isolated qualification indexes."""

import hashlib
import json
from datetime import UTC, datetime

from paper_research.chunking.types import Chunk
from paper_research.indexing.embedding import EmbeddingProvider

SCHEMA_VERSION = "qdrant-index-identity-v1"
BUILDER_VERSION = "phase1q-r2-index-builder-v1"


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def corpus_identity(chunks: list[Chunk], paper_metadata: dict) -> dict:
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    if not ordered or len({c.chunk_id for c in ordered}) != len(ordered):
        raise ValueError("corpus must contain unique, nonempty chunk identities")
    raw = [Chunk.model_validate(c.model_dump()).model_dump() for c in ordered]
    return {
        "paper_count": len({c.paper_id for c in ordered}),
        "chunk_count": len(ordered),
        "sorted_chunk_id_digest": canonical_digest([c.chunk_id for c in ordered]),
        "canonical_metadata_digest": canonical_digest(
            {
                "chunks": [{k: v for k, v in c.items() if k != "chunk_text"} for c in raw],
                "paper_metadata": paper_metadata,
            }
        ),
        "raw_text_digest": canonical_digest([[c.chunk_id, c.chunk_text] for c in ordered]),
    }


def index_manifest(
    chunks: list[Chunk],
    paper_metadata: dict,
    embedding: EmbeddingProvider,
    *,
    collection: str,
    embedding_input: str,
    source_corpus_identity: dict,
    git_commit: str,
) -> dict:
    if embedding_input not in {"raw", "contextual"}:
        raise ValueError("embedding_input must be raw or contextual")
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_name": collection,
        "embedding_provider": embedding.provider_name,
        "embedding_model": embedding.model_name,
        "embedding_revision": embedding.revision,
        "embedding_dimension": embedding.dimensions,
        "distance_metric": "Cosine",
        "embedding_input": embedding_input,
        "query_instruction": "",
        "document_instruction": "",
        "instruction_version": "siliconflow-verbatim-input-v1",
        "instruction_note": (
            "Existing SiliconFlow adapter sends input verbatim, float format; no added instruction."
        ),
        **corpus_identity(chunks, paper_metadata),
        "source_corpus_identity": source_corpus_identity,
        "build_git_commit": git_commit,
        "build_timestamp": datetime.now(UTC).isoformat(),
        "index_builder_version": BUILDER_VERSION,
    }


def assert_runtime_compatible(
    manifest: dict,
    embedding: EmbeddingProvider,
    *,
    collection: str,
    distance: str = "Cosine",
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "collection_name": collection,
        "embedding_provider": embedding.provider_name,
        "embedding_model": embedding.model_name,
        "embedding_revision": embedding.revision,
        "embedding_dimension": embedding.dimensions,
        "distance_metric": distance,
        "query_instruction": "",
        "document_instruction": "",
        "instruction_version": "siliconflow-verbatim-input-v1",
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError("INDEX_EMBEDDING_IDENTITY_MISMATCH: " + ", ".join(mismatched))


def equality_certificate(raw: dict, contextual: dict) -> dict:
    fields = [
        "paper_count",
        "chunk_count",
        "sorted_chunk_id_digest",
        "canonical_metadata_digest",
        "raw_text_digest",
        "embedding_provider",
        "embedding_model",
        "embedding_revision",
        "embedding_dimension",
        "distance_metric",
        "query_instruction",
        "document_instruction",
        "instruction_version",
        "source_corpus_identity",
    ]
    checks = {key: raw.get(key) == contextual.get(key) for key in fields}
    checks["representation_roles"] = (
        raw.get("embedding_input") == "raw" and contextual.get("embedding_input") == "contextual"
    )
    checks["separate_collections"] = raw.get("collection_name") != contextual.get("collection_name")
    valid = all(checks.values())
    return {
        "schema_version": "phase1q-r2-corpus-equality-v1",
        "status": "VALID" if valid else "C0_C1_COMPARISON_INVALID",
        "checks": checks,
        "raw_collection": raw["collection_name"],
        "contextual_collection": contextual["collection_name"],
        "raw_manifest_digest": canonical_digest(raw),
        "contextual_manifest_digest": canonical_digest(contextual),
    }


# S5 strict local schemas. No runtime dependency on research artifact files.
CORPUS_SCHEMAS = {'index_identity': {'type': 'object',
                    'additionalProperties': False,
                    'properties': {'embedding_provider': {'type': 'string', 'minLength': 1},
                                   'embedding_model': {'type': 'string', 'minLength': 1},
                                   'embedding_revision': {'type': 'string', 'minLength': 1},
                                   'embedding_dimension': {'type': 'integer', 'minimum': 1},
                                   'index_version': {'type': 'string', 'minLength': 1},
                                   'collection_name': {'type': 'string', 'minLength': 1},
                                   'distance_metric': {'const': 'Cosine'}},
                    'required': ['embedding_provider',
                                 'embedding_model',
                                 'embedding_revision',
                                 'embedding_dimension',
                                 'index_version',
                                 'collection_name',
                                 'distance_metric']},
 'source_input': {'type': 'object',
                  'additionalProperties': False,
                  'properties': {'source_ref': {'type': 'string', 'minLength': 1},
                                 'pdf_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                                 'byte_count': {'type': 'integer', 'minimum': 1},
                                 'input_relative_path': {'type': 'string', 'minLength': 1}},
                  'required': ['source_ref', 'pdf_sha256', 'byte_count', 'input_relative_path']},
 'admission_intent': {'type': 'object',
                      'additionalProperties': False,
                      'properties': {'schema_version': {'const': 'corpus-admission-intent-v1'},
                                     'admission_id': {'type': 'string',
                                                      'format': 'uuid',
                                                      'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                     'authorization_ref': {'type': 'string', 'minLength': 1},
                                     'authority_id': {'type': 'string', 'minLength': 1},
                                     'closed_input_scope': {'enum': ['EXPLICIT_USER_AUTHORIZED_SYNTHETIC_PDFS',
                                                                      'EXPLICIT_OPERATOR_AUTHORIZED_PDFS']},
                                     'sources': {'type': 'array',
                                                 'items': {'$ref': '#/$defs/source_input'},
                                                 'minItems': 1},
                                     'created_at': {'type': 'string', 'format': 'date-time'},
                                     'intent_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}},
                      'required': ['schema_version',
                                   'admission_id',
                                   'authorization_ref',
                                   'authority_id',
                                   'closed_input_scope',
                                   'sources',
                                   'created_at',
                                   'intent_digest']},
 'document_binding': {'type': 'object',
                      'additionalProperties': False,
                      'properties': {'source_ref': {'type': 'string', 'minLength': 1},
                                     'paper_id': {'type': 'string',
                                                  'format': 'uuid',
                                                  'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                     'pdf_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                                     'parse_blocks_sha256': {'type': 'string',
                                                             'pattern': '^[a-f0-9]{64}$'},
                                     'parse_manifest_sha256': {'type': 'string',
                                                               'pattern': '^[a-f0-9]{64}$'}},
                      'required': ['source_ref',
                                   'paper_id',
                                   'pdf_sha256',
                                   'parse_blocks_sha256',
                                   'parse_manifest_sha256']},
 'admission_commitment': {'type': 'object',
                          'additionalProperties': False,
                          'properties': {'schema_version': {'const': 'corpus-admission-commitment-v1'},
                                         'admission_id': {'type': 'string',
                                                          'format': 'uuid',
                                                          'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                         'intent_digest': {'type': 'string',
                                                           'pattern': '^[a-f0-9]{64}$'},
                                         'documents': {'type': 'array',
                                                       'items': {'$ref': '#/$defs/document_binding'},
                                                       'minItems': 1},
                                         'closed': {'const': True},
                                         'committed_at': {'type': 'string', 'format': 'date-time'},
                                         'admission_digest': {'type': 'string',
                                                              'pattern': '^[a-f0-9]{64}$'}},
                          'required': ['schema_version',
                                       'admission_id',
                                       'intent_digest',
                                       'documents',
                                       'closed',
                                       'committed_at',
                                       'admission_digest']},
 'chunk_binding': {'type': 'object',
                   'additionalProperties': False,
                   'properties': {'paper_id': {'type': 'string',
                                               'format': 'uuid',
                                               'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                  'chunk_id': {'type': 'string', 'minLength': 1},
                                  'canonical_payload_sha256': {'type': 'string',
                                                               'pattern': '^[a-f0-9]{64}$'},
                                  'point_id': {'type': 'string',
                                               'format': 'uuid',
                                               'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'}},
                   'required': ['paper_id', 'chunk_id', 'canonical_payload_sha256', 'point_id']},
 'document_membership': {'type': 'object',
                         'additionalProperties': False,
                         'properties': {'paper_id': {'type': 'string',
                                                     'format': 'uuid',
                                                     'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                        'chunk_ids': {'type': 'array',
                                                      'items': {'type': 'string', 'minLength': 1},
                                                      'minItems': 1,
                                                      'uniqueItems': True},
                                        'chunks_relative_path': {'type': 'string', 'minLength': 1},
                                        'chunks_file_sha256': {'type': 'string',
                                                               'pattern': '^[a-f0-9]{64}$'}},
                         'required': ['paper_id',
                                      'chunk_ids',
                                      'chunks_relative_path',
                                      'chunks_file_sha256']},
 'inventory': {'type': 'object',
               'additionalProperties': False,
               'properties': {'schema_version': {'const': 'authoritative-corpus-inventory-v1'},
                              'inventory_id': {'type': 'string',
                                               'format': 'uuid',
                                               'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                              'generation_id': {'type': 'string',
                                                'format': 'uuid',
                                                'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                              'index_identity': {'$ref': '#/$defs/index_identity'},
                              'source_admission_id': {'type': 'string',
                                                      'format': 'uuid',
                                                      'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                              'source_admission_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                              'expected_document_ids': {'type': 'array',
                                                        'items': {'type': 'string',
                                                                  'format': 'uuid',
                                                                  'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                                        'minItems': 1,
                                                        'uniqueItems': True},
                              'expected_document_count': {'type': 'integer', 'minimum': 1},
                              'expected_chunks': {'type': 'array',
                                                  'items': {'$ref': '#/$defs/chunk_binding'},
                                                  'minItems': 1},
                              'expected_chunk_count': {'type': 'integer', 'minimum': 1},
                              'document_to_chunk_membership': {'type': 'array',
                                                               'items': {'$ref': '#/$defs/document_membership'},
                                                               'minItems': 1},
                              'raw_chunk_id_collision_policy': {'const': 'REJECT_ANY_DUPLICATE_BARE_CHUNK_ID_BEFORE_EMBED_OR_UPSERT'},
                              'prepared_records_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                              'parser_receipt_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                              'chunker_config': {'type': 'object',
                                                 'additionalProperties': False,
                                                 'properties': {'max_tokens': {'type': 'integer',
                                                                               'minimum': 1},
                                                                'overlap_tokens': {'type': 'integer',
                                                                                   'minimum': 0}},
                                                 'required': ['max_tokens', 'overlap_tokens']},
                              'created_at': {'type': 'string', 'format': 'date-time'},
                              'inventory_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}},
               'required': ['schema_version',
                            'inventory_id',
                            'generation_id',
                            'index_identity',
                            'source_admission_id',
                            'source_admission_digest',
                            'expected_document_ids',
                            'expected_document_count',
                            'expected_chunks',
                            'expected_chunk_count',
                            'document_to_chunk_membership',
                            'raw_chunk_id_collision_policy',
                            'prepared_records_digest',
                            'parser_receipt_digest',
                            'chunker_config',
                            'created_at',
                            'inventory_digest']},
 'create_receipt': {'type': 'object',
                    'additionalProperties': False,
                    'properties': {'generation_id': {'type': 'string',
                                                     'format': 'uuid',
                                                     'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                   'collection_name': {'type': 'string', 'minLength': 1},
                                   'existed_before': {'const': False},
                                   'creation_acknowledged': {'const': True},
                                   'observed_initial_point_count': {'const': 0},
                                   'initial_full_scroll_empty': {'const': True},
                                   'exclusive_owner_id': {'type': 'string', 'minLength': 1},
                                   'created_at': {'type': 'string', 'format': 'date-time'},
                                   'receipt_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}},
                    'required': ['generation_id',
                                 'collection_name',
                                 'existed_before',
                                 'creation_acknowledged',
                                 'observed_initial_point_count',
                                 'initial_full_scroll_empty',
                                 'exclusive_owner_id',
                                 'created_at',
                                 'receipt_digest']},
 'physical_point_receipt': {'type': 'object',
                            'additionalProperties': False,
                            'properties': {'point_id': {'type': 'string',
                                                        'format': 'uuid',
                                                        'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                           'paper_id': {'type': 'string',
                                                        'format': 'uuid',
                                                        'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                           'chunk_id': {'type': 'string', 'minLength': 1},
                                           'payload_sha256': {'type': 'string',
                                                              'pattern': '^[a-f0-9]{64}$'},
                                           'vector_f32_sha256': {'type': 'string',
                                                                 'pattern': '^[a-f0-9]{64}$'}},
                            'required': ['point_id',
                                         'paper_id',
                                         'chunk_id',
                                         'payload_sha256',
                                         'vector_f32_sha256']},
 'seal': {'type': 'object',
          'additionalProperties': False,
          'properties': {'schema_version': {'const': 'verified-index-generation-seal-v1'},
                         'generation_id': {'type': 'string',
                                           'format': 'uuid',
                                           'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                         'inventory_id': {'type': 'string',
                                          'format': 'uuid',
                                          'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                         'inventory_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'index_identity': {'$ref': '#/$defs/index_identity'},
                         'create_receipt_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'build_vector_receipt_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'observed_canonical_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'observed_physical_points': {'type': 'array',
                                                      'items': {'$ref': '#/$defs/physical_point_receipt'},
                                                      'minItems': 1},
                         'observed_physical_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'verification_result': {'const': 'EXACT_MEMBERSHIP_PAYLOAD_AND_VECTOR_BINDING_VERIFIED'},
                         'verified_at': {'type': 'string', 'format': 'date-time'},
                         'sealed_at': {'type': 'string', 'format': 'date-time'},
                         'seal_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}},
          'required': ['schema_version',
                       'generation_id',
                       'inventory_id',
                       'inventory_digest',
                       'index_identity',
                       'create_receipt_digest',
                       'build_vector_receipt_digest',
                       'observed_canonical_digest',
                       'observed_physical_points',
                       'observed_physical_digest',
                       'verification_result',
                       'verified_at',
                       'sealed_at',
                       'seal_digest']},
 'active_binding': {'type': 'object',
                    'additionalProperties': False,
                    'properties': {'schema_version': {'const': 'active-corpus-generation-v1'},
                                   'logical_collection': {'type': 'string', 'minLength': 1},
                                   'collection_name': {'type': 'string', 'minLength': 1},
                                   'generation_id': {'type': 'string',
                                                     'format': 'uuid',
                                                     'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                   'inventory_id': {'type': 'string',
                                                    'format': 'uuid',
                                                    'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                                   'inventory_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                                   'seal_digest': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                                   'source_admission_digest': {'type': 'string',
                                                               'pattern': '^[a-f0-9]{64}$'},
                                   'index_identity': {'$ref': '#/$defs/index_identity'},
                                   'publication_state': {'const': 'ACTIVE'},
                                   'published_at': {'type': 'string', 'format': 'date-time'}},
                    'required': ['schema_version',
                                 'logical_collection',
                                 'collection_name',
                                 'generation_id',
                                 'inventory_id',
                                 'inventory_digest',
                                 'seal_digest',
                                 'source_admission_digest',
                                 'index_identity',
                                 'publication_state',
                                 'published_at']},
 'build_vector_receipt': {'type': 'object',
                          'additionalProperties': False,
                          'required': ['schema_version',
                                       'generation_id',
                                       'inventory_digest',
                                       'vectors_relative_path',
                                       'vectors_file_sha256',
                                       'point_count',
                                       'receipt_digest'],
                          'properties': {'schema_version': {'const': 'corpus-build-vector-receipt-v1'},
                                         'generation_id': {'$ref': '#/$defs/inventory/properties/generation_id'},
                                         'inventory_digest': {'$ref': '#/$defs/inventory/properties/inventory_digest'},
                                         'vectors_relative_path': {'type': 'string', 'minLength': 1},
                                         'vectors_file_sha256': {'type': 'string',
                                                                 'pattern': '^[a-f0-9]{64}$'},
                                         'point_count': {'type': 'integer', 'minimum': 1},
                                         'receipt_digest': {'type': 'string',
                                                            'pattern': '^[a-f0-9]{64}$'}}},
 'journal_transition': {'type': 'object',
                        'additionalProperties': False,
                        'properties': {'from_state': {'enum': ['NEW',
                                                               'ADMITTING',
                                                               'PREPARED',
                                                               'BUILDING',
                                                               'VERIFYING',
                                                               'SEALED',
                                                               'FAILED',
                                                               'ABORTED']},
                                       'to_state': {'enum': ['ADMITTING',
                                                             'PREPARED',
                                                             'BUILDING',
                                                             'VERIFYING',
                                                             'SEALED',
                                                             'FAILED',
                                                             'ABORTED']},
                                       'at': {'type': 'string', 'format': 'date-time'},
                                       'error_code': {'type': ['string', 'null']}},
                        'required': ['from_state', 'to_state', 'at', 'error_code']},
 'build_journal': {'type': 'object',
                   'additionalProperties': False,
                   'properties': {'schema_version': {'const': 'corpus-generation-journal-v1'},
                                  'generation_id': {'type': 'string',
                                                    'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
                                                    'format': 'uuid'},
                                  'admission_id': {'type': 'string',
                                                   'pattern': '^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
                                                   'format': 'uuid'},
                                  'state': {'enum': ['NEW',
                                                     'ADMITTING',
                                                     'PREPARED',
                                                     'BUILDING',
                                                     'VERIFYING',
                                                     'SEALED',
                                                     'FAILED',
                                                     'ABORTED']},
                                  'inventory_digest': {'anyOf': [{'type': 'string',
                                                                  'pattern': '^[a-f0-9]{64}$'},
                                                                 {'type': 'null'}]},
                                  'create_receipt_digest': {'anyOf': [{'type': 'string',
                                                                       'pattern': '^[a-f0-9]{64}$'},
                                                                      {'type': 'null'}]},
                                  'build_vector_receipt_digest': {'anyOf': [{'type': 'string',
                                                                             'pattern': '^[a-f0-9]{64}$'},
                                                                            {'type': 'null'}]},
                                  'seal_digest': {'anyOf': [{'type': 'string',
                                                             'pattern': '^[a-f0-9]{64}$'},
                                                            {'type': 'null'}]},
                                  'transition_history': {'type': 'array',
                                                         'items': {'$ref': '#/$defs/journal_transition'}},
                                  'failure': {'anyOf': [{'type': 'object',
                                                         'additionalProperties': False,
                                                         'properties': {'code': {'type': 'string',
                                                                                 'minLength': 1},
                                                                        'stage': {'type': 'string',
                                                                                  'minLength': 1}},
                                                         'required': ['code', 'stage']},
                                                        {'type': 'null'}]},
                                  'updated_at': {'type': 'string', 'format': 'date-time'}},
                   'required': ['schema_version',
                                'generation_id',
                                'admission_id',
                                'state',
                                'inventory_digest',
                                'create_receipt_digest',
                                'build_vector_receipt_digest',
                                'seal_digest',
                                'transition_history',
                                'failure',
                                'updated_at']}}


def trust_digest(value: object, own_field: str | None = None) -> str:
    if own_field is not None:
        value = {k: v for k, v in value.items() if k != own_field}
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def trust_json_loads(raw: str | bytes):
    def nonfinite(value):
        raise ValueError("nonfinite JSON number")
    return json.loads(raw, object_pairs_hook=_strict_pairs, parse_constant=nonfinite)


def validate_trust_record(name: str, value: object) -> None:
    """Validate the frozen schema subset without installing a new dependency."""
    import re
    import uuid

    def equal(a, b):
        return type(a) is type(b) and a == b

    def check(schema, item):
        if "$ref" in schema:
            target = {"$defs": CORPUS_SCHEMAS}
            for key in schema["$ref"][2:].split("/"):
                target = target[key]
            return check(target, item)
        if "anyOf" in schema:
            for choice in schema["anyOf"]:
                try:
                    check(choice, item)
                    return
                except ValueError:
                    pass
            raise ValueError("invalid union type")
        if "const" in schema and not equal(item, schema["const"]):
            raise ValueError("invalid constant")
        if "enum" in schema and not any(equal(item, v) for v in schema["enum"]):
            raise ValueError("invalid enum")
        types = schema.get("type")
        if types:
            types = types if isinstance(types, list) else [types]
            matched = any({"object": type(item) is dict, "array": type(item) is list,
                "integer": type(item) is int, "string": type(item) is str,
                "null": item is None, "boolean": type(item) is bool}.get(t, False)
                for t in types)
            if not matched:
                raise ValueError("invalid strict type")
        if type(item) is dict and schema.get("type") == "object":
            if set(item) != set(schema["required"]):
                raise ValueError("missing or extra fields")
            for key, child in schema["properties"].items():
                check(child, item[key])
        if type(item) is list and schema.get("type") == "array":
            if len(item) < schema.get("minItems", 0):
                raise ValueError("empty members")
            if schema.get("uniqueItems") and len({trust_digest(v) for v in item}) != len(item):
                raise ValueError("duplicate members")
            for child in item:
                check(schema["items"], child)
        if type(item) is int and item < schema.get("minimum", item):
            raise ValueError("invalid minimum")
        if type(item) is str:
            if len(item) < schema.get("minLength", 0):
                raise ValueError("empty string")
            if "pattern" in schema and re.fullmatch(schema["pattern"], item) is None:
                raise ValueError("invalid pattern")
            if schema.get("format") == "uuid" and str(uuid.UUID(item)) != item:
                raise ValueError("invalid UUID")
            if schema.get("format") == "date-time":
                if datetime.fromisoformat(item).tzinfo is None:
                    raise ValueError("timezone required")
    check(CORPUS_SCHEMAS[name], value)


def payload_fields() -> set[str]:
    return set(Chunk.model_fields)


def chunk_file_bytes(chunks: list[Chunk]) -> bytes:
    return "".join(json.dumps(c.model_dump(), ensure_ascii=False) + "\n"
                   for c in chunks).encode("utf-8")


def point_identity(chunk_id: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def vector_f32(vector: list[float], dimension: int) -> list[float]:
    import math
    import struct
    if type(vector) is not list or len(vector) != dimension:
        raise ValueError("invalid vector dimension")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
        raise ValueError("nonfinite or invalid vector")
    return list(struct.unpack("<" + "f" * dimension,
                             struct.pack("<" + "f" * dimension, *vector)))


def vector_f32_digest(vector: list[float], dimension: int) -> str:
    import struct
    values = vector_f32(vector, dimension)
    return hashlib.sha256(struct.pack("<" + "f" * dimension, *values)).hexdigest()


def normalized_f32(vector: list[float], dimension: int) -> list[float]:
    import math
    values = vector_f32(vector, dimension)
    norm = math.sqrt(sum(v * v for v in values))
    return vector_f32([v / norm for v in values] if norm else values, dimension)
