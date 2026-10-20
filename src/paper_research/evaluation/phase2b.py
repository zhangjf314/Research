"""Phase 2B evaluation harness; does not mutate qualified Agent implementation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

from paper_research.agents.research_agent.evidence_contract import AgenticRetrievalConfig
from paper_research.agents.research_agent.state import AgentBudget
from paper_research.config import Settings
from paper_research.providers.factory import build_llm_provider
from paper_research.retrieval.frozen_backend import BACKEND_ID as BACKEND_ID
from paper_research.retrieval.frozen_backend import read_manifest

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/paperresearch-v2/phase2b"
PHASE2A = ROOT / "artifacts/paperresearch-v2/phase2a"
ARMS = ("a0", "a1c", "a1")
SEED = 20261002
LABEL = "REAL_CORPUS_EFFECTIVENESS_BENCHMARK"
ERRORS = [
    "CONTRACT_OMISSION",
    "FALSE_GAP",
    "MISSED_GAP",
    "BAD_QUERY_REFORMULATION",
    "TARGETED_RETRIEVAL_NO_GAIN",
    "RETRIEVER_MISS",
    "VERIFIER_FALSE_SUPPORT",
    "VERIFIER_FALSE_REJECT",
    "PREMATURE_STOP",
    "OVER_SEARCH",
    "FINAL_SYNTHESIS_OMISSION",
    "UNSUPPORTED_GENERATION",
    "BUDGET_EXHAUSTED",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def code_identity():
    expected = read(PHASE2A / "protocol.json")["code_sha256"]
    mismatches = [p for p, h in expected.items() if sha(ROOT / p) != h]
    if mismatches:
        raise ValueError("PHASE2B_CODE_IDENTITY_MISMATCH: " + ",".join(mismatches))
    return expected


def verify_frozen():
    code_identity()
    read_manifest()
    manifest = read(OUT / "benchmark/manifest.json")
    for p, h in manifest["file_sha256"].items():
        if sha(OUT / p) != h:
            raise ValueError("PHASE2B_BENCHMARK_HASH_MISMATCH: " + p)
    protocol = read(OUT / "protocol.json")
    amendment = OUT / "protocol_cost_guard_amendment.json"
    if amendment.exists():
        from paper_research.evaluation.cost_guard import verify_amendment

        verify_amendment(protocol)
        return protocol
    for p, h in protocol["harness_sha256"].items():
        if sha(ROOT / p) != h:
            raise ValueError("PHASE2B_FORMAL_RUN_MUTATION: " + p)
    return protocol


def arm_config(arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return AgenticRetrievalConfig(
        agentic_retrieval_enabled=arm != "a0", max_replans=0 if arm == "a1c" else 4
    )


def budget():
    return AgentBudget(
        max_steps=12, max_tool_calls=6, max_provider_requests=12, max_tokens=60000, max_cost_usd=1
    )


def balanced_order(count):
    return [list(ARMS[i % 3 :] + ARMS[: i % 3]) for i in range(count)]


def agent_input(item):
    # Hard allowlist: no gold, topology, task-family or authoring metadata.
    return {"research_question": item["question"], "paper_ids": item["paper_scope"]}


def settings():
    original = Settings()
    qualified = read(PHASE2A / "protocol.json")["live_smoke"]
    if (original.llm_provider_name or original.llm_provider) != qualified[
        "llm_provider"
    ] or original.llm_model != qualified["llm_model"]:
        raise ValueError("PHASE2B_VERIFIER_CONFIGURATION_MISMATCH")
    return original.model_copy(
        update={
            "llm_max_retries": 0,
            "qa_response_audit_enabled": False,
            "qa_response_audit_store_full_payload": False,
        }
    )


def provider():
    return build_llm_provider(settings())


def provider_identity():
    s = settings()
    return {
        k: getattr(s, k)
        for k in [
            "llm_provider",
            "llm_provider_name",
            "llm_model",
            "llm_base_url",
            "llm_temperature",
            "llm_timeout_seconds",
            "llm_max_retries",
            "llm_response_format",
            "llm_thinking_enabled",
            "llm_stream",
        ]
    }


def structured_call(llm, system, payload, identity, folder, validate, max_output=5000):
    """Two bounded attempts; immutable attempt files, no arm-specific transport retries."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    error = ""
    for attempt in range(1, 3):
        path = folder / f"{identity}-attempt-{attempt}.json"
        if path.exists():
            saved = read(path)
            if saved.get("valid"):
                return saved["payload"]
            error = saved.get("error", "")
            continue
        started = now()
        record = {
            "identity": identity,
            "attempt": attempt,
            "started": started,
            "prompt_hash": digest({"system": system, "input": payload}),
            "provider": provider_identity(),
            "valid": False,
        }
        try:
            result = llm.generate_structured_json(
                system_prompt=system,
                user_prompt=json.dumps(
                    {"input": payload, "validation_error": error}, ensure_ascii=False
                ),
                schema_name="phase2b_evaluation_v1",
                request_context={"task_id": identity, "attempt_number": attempt},
                max_output_tokens=max_output,
            )
            record.update(
                payload=result.payload,
                usage=result.usage.model_dump(),
                requests=result.request_attempt_count,
            )
            validate(result.payload)
            record["valid"] = True
        except Exception as exc:
            import traceback

            error = (str(exc) or traceback.format_exc(limit=2))[:1200]
            record.update(
                error=error,
                error_type=type(exc).__name__,
                requests=record.get("requests", getattr(exc, "api_request_count", 1)),
            )
        record["finished"] = now()
        write(path, record)
        if record["valid"]:
            return record["payload"]
    raise ValueError("PHASE2B_BOUNDED_STRUCTURED_FAILURE: " + identity + ": " + error)


def shuffled_blind_map(items):
    rng = random.Random(SEED)
    mapping = [
        {
            "sample_id": item["sample_id"],
            "arm": arm,
            "response_id": f"response-{rng.getrandbits(96):024x}",
        }
        for item in items
        for arm in ARMS
    ]
    rng.shuffle(mapping)
    return mapping


def blind_payload(question, response_id, contract, evidence, answer, gold):
    # No state, stop reasons, tool/action names or arm identity enters judge input.
    return {
        "response_id": response_id,
        "question": question,
        "proposed_requirements": contract,
        "evidence": evidence,
        "answer": answer,
        "gold_requirements": gold["requirements"],
        "gold_claims": gold["claims"],
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def paired_bootstrap(left, right):
    import numpy as np

    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(SEED)
    sampled = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {
        "point_delta": float(delta.mean()),
        "ci95": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
        "resamples": 10000,
        "seed": SEED,
        "unit": "query",
        "n": len(delta),
    }
