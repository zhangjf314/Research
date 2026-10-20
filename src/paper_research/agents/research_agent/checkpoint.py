from __future__ import annotations

import json
import math
import hashlib
import re
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from paper_research.agents.research_agent.state import AgentState
from paper_research.agents.research_agent.models import CanonicalEvidenceItem, NativeEvidenceItem
from paper_research.agents.research_agent.verifier import (
    REPORT_SAFETY_NAMESPACE, REPORT_SAFETY_RECEIPT, SEMANTIC_SUPPORT_NOT_ASSESSED,
    ResearchSafetyError, validate_research_task_id,
)


def _validate_active_receipt(raw: dict, marker: dict) -> None:
    """Validate the frozen v2 receipt before models can coerce or drop fields."""
    fixed = {
        "mode": "CANONICAL_CANDIDATE_ONLY_REPORT_SUPPRESSED",
        "schema_version": 2,
        "source_identity_schema": "canonical-agent-provenance-v2",
        "policy_id": REPORT_SAFETY_RECEIPT["policy_id"],
        "semantic_status": SEMANTIC_SUPPORT_NOT_ASSESSED,
        "semantic_qualification_id": None,
        "report_eligibility": REPORT_SAFETY_RECEIPT["report_eligibility"],
    }
    required = set(fixed) | {"retrieval_identity", "paper_scope",
                             "generation_binding", "generation_binding_digest"}
    if (not isinstance(marker, dict) or set(marker) != required
            or any(type(marker[key]) is not type(value) or marker[key] != value
                   for key, value in fixed.items())):
        raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")

    def reject(condition: bool) -> None:
        if not condition:
            raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")

    def strings(value: dict, fields: set[str]) -> bool:
        return all(type(value.get(key)) is str and bool(value[key]) for key in fields)

    def identity(value: dict, collection_key: str) -> bool:
        text = {"embedding_provider", "embedding_model", "embedding_revision",
                "index_version", collection_key}
        return (isinstance(value, dict) and set(value) == text | {"embedding_dimension"}
                and strings(value, text) and type(value["embedding_dimension"]) is int
                and value["embedding_dimension"] > 0)

    binding = marker["generation_binding"]
    reject(isinstance(binding, dict) and set(binding) == {
        "schema_version", "generation_id", "inventory_digest", "seal_digest",
        "index_identity", "collection_identity", "source_scope"})
    reject(binding["schema_version"] == "d6-c2-generation-binding-v1")
    reject(type(binding["generation_id"]) is str and re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        binding["generation_id"]) is not None)
    for key in ("inventory_digest", "seal_digest"):
        reject(type(binding[key]) is str and re.fullmatch(r"[0-9a-f]{64}", binding[key]) is not None)
    index = binding["index_identity"]
    reject(isinstance(index, dict) and index.get("distance_metric") == "Cosine")
    reject(identity({key: value for key, value in index.items() if key != "distance_metric"},
                    "collection_name"))
    collection = binding["collection_identity"]
    reject(isinstance(collection, dict) and set(collection) == {"logical_collection", "collection_name"}
           and strings(collection, set(collection))
           and collection["collection_name"] == index["collection_name"])
    retrieval = marker["retrieval_identity"]
    reject(identity(retrieval, "collection"))
    expected = {key: index[key] for key in retrieval if key != "collection"}
    expected["collection"] = collection["collection_name"]
    reject(retrieval == expected)
    scope = binding["source_scope"]
    reject(isinstance(scope, dict) and set(scope) == {"kind", "paper_ids"})
    papers = scope["paper_ids"]
    reject(type(papers) is list and all(type(p) is str and bool(p) for p in papers))
    reject(len(papers) == len(set(papers)))
    reject(scope["kind"] == ("paper" if papers else "global"))
    for paper in papers:
        reject(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", paper) is not None)
    requested = raw.get("paper_ids")
    reject(requested is None or (type(requested) is list
           and all(type(p) is str and bool(p) for p in requested)))
    reject(type(marker["paper_scope"]) is list and marker["paper_scope"] == papers
           and papers == list(dict.fromkeys(requested or [])))
    digest = marker["generation_binding_digest"]
    reject(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
    encoded = json.dumps(binding, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("utf8")
    reject(hashlib.sha256(encoded).hexdigest() == digest)


def checkpoint_kind(raw: dict) -> str:
    """Classify raw data before Pydantic can discard unknown trust fields."""
    if not isinstance(raw, dict) or set(raw) - set(AgentState.model_fields):
        raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
    provenance = raw.get("evidence_provenance", {})
    if not isinstance(provenance, dict):
        raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
    readonly_keys = [key for key in provenance if key.startswith("p1_readonly_retrieval_bridge_")]
    safety_keys = [key for key in provenance if key.startswith("p1_agent_report_safety_")]
    active_keys = [key for key in provenance if key.startswith("p1_active_canonical_provenance_")]
    if sum(bool(keys) for keys in (readonly_keys, safety_keys, active_keys)) > 1:
        raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
    active = bool(active_keys)
    if active:
        if active_keys != ["p1_active_canonical_provenance_v2"] or set(provenance) != set(active_keys):
            raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
        marker = provenance[active_keys[0]]
        _validate_active_receipt(raw, marker)
        items = raw.get("evidence_state", {}).get("items", {})
        if not isinstance(items, dict):
            raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
        for key, value in items.items():
            if isinstance(value, dict) and (
                    value.get("verification_status", "CANDIDATE") != "CANDIDATE"
                    or value.get("supports") or value.get("contradicts")):
                raise ResearchSafetyError("RESEARCH_TRUST_STATE_INVALID")
            try:
                native = isinstance(value, dict) and "native_unit" in value
                if native and raw.get("multimodal_enabled") is not True:
                    raise ValueError("multimodal option required")
                item = (NativeEvidenceItem if native else CanonicalEvidenceItem).model_validate(value)
            except Exception as exc:
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID") from exc
            if (key != item.stable_key
                    or item.generation_binding_digest != marker["generation_binding_digest"]
                    or value["retrieval_identity"] != marker["retrieval_identity"]
                    or any(type(value["retrieval_identity"][field]) is not type(expected)
                           for field, expected in marker["retrieval_identity"].items())
                    or (marker["paper_scope"] and item.paper_id not in marker["paper_scope"])):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
            occurrences = [o if isinstance(o, dict) else o.model_dump() for o in item.occurrences]
            if any(o.get("task_id") != raw.get("task_id")
                   or o.get("paper_scope") != marker["paper_scope"]
                   or type(o.get("step")) is not int or not 0 <= o["step"] <= raw.get("step_count", 0)
                   or o.get("generation_binding_digest") != marker["generation_binding_digest"]
                   for o in occurrences):
                raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
        known = {sq.get("id") for sq in raw.get("subquestions", [])}
        links = raw.get("evidence_state", {}).get("subquestion_evidence", {})
        if (not isinstance(links, dict) or any(sq not in known or not isinstance(keys, list)
                or len(keys) != len(set(keys)) or any(key not in items for key in keys)
                for sq, keys in links.items())):
            raise ResearchSafetyError("CANONICAL_PROVENANCE_INVALID")
    if readonly_keys:
        if readonly_keys != ["p1_readonly_retrieval_bridge_v1"]:
            raise ResearchSafetyError("READONLY_OBSERVATION_INVALID")
        if (raw.get("status") != "PARTIAL" or raw.get("verification_state")
                or raw.get("verified_claims") or raw.get("retrieval_backend_id")
                or raw.get("agentic_config") or raw.get("evidence_contract")
                or any(raw.get("evidence_state", {}).get(key)
                       for key in ("items", "subquestion_evidence", "contradictions", "evidence_gaps"))):
            raise ResearchSafetyError("READONLY_OBSERVATION_INVALID")
        return "READONLY"
    if safety_keys:
        if safety_keys != [REPORT_SAFETY_NAMESPACE] or set(provenance) != {REPORT_SAFETY_NAMESPACE}:
            raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
        marker = provenance[REPORT_SAFETY_NAMESPACE]
        if not isinstance(marker, dict) or set(marker) != set(REPORT_SAFETY_RECEIPT):
            raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
        if any(type(marker[key]) is not type(value) or marker[key] != value
               for key, value in REPORT_SAFETY_RECEIPT.items()):
            raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
    config = raw.get("agentic_config") or {}
    if raw.get("retrieval_backend_id") or config.get("agentic_retrieval_enabled"):
        return "SCIENCE"
    if not safety_keys and not active:
        return "LEGACY"
    invalid = (raw.get("status") not in {"RUNNING", "PAUSED", "PARTIAL", "FAILED"}
               or raw.get("stop_reason") in {"SUCCESS", "CONTRACT_SATISFIED", "EVIDENCE_SUFFICIENT"}
               or any(raw.get(key) for key in ("verified_claims", "unsupported_claims",
                   "contradictions", "evidence_contract", "evidence_assessments", "requirement_coverage")))
    for item in raw.get("evidence_state", {}).get("items", {}).values():
        invalid = invalid or (item.get("verification_status", "CANDIDATE") != "CANDIDATE"
                             or bool(item.get("supports") or item.get("contradicts")))
    counts = [raw.get(key, 0) for key in (
        "step_count", "tool_call_count", "provider_call_count", "plan_version",
        "no_progress_count", "resume_count", "remaining_step_budget", "remaining_tool_budget",
        "remaining_token_budget", "targeted_no_gain_count")]
    counts += [raw.get("token_usage", {}).get(key, 0)
               for key in ("input_tokens", "output_tokens", "total_tokens")]
    counts += [raw.get("retry_state", {}).get(key, 0)
               for key in ("provider_retry_count", "tool_retry_count")]
    invalid = invalid or any(type(value) is not int or value < 0 for value in counts)
    cost = raw.get("estimated_cost", 0)
    invalid = invalid or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
    completed = raw.get("completed_tool_call_ids", [])
    invalid = invalid or len(completed) != len(set(completed))
    result = raw.get("verification_state")
    if result:
        expected = {
            "status": "PARTIAL", "verified_claims": [], "unsupported_claims": [],
            "unresolved_subquestions": [sq["id"] for sq in raw.get("subquestions", [])],
            "evidence_gaps": [SEMANTIC_SUPPORT_NOT_ASSESSED], "contradictions": [],
            "recommended_next_action": "FINISH",
        }
        invalid = invalid or result != expected or bool(raw.get("resolved_subquestions"))
        invalid = invalid or any(sq.get("status") != "OPEN" for sq in raw.get("subquestions", []))
    if invalid:
        raise ResearchSafetyError("RESEARCH_TRUST_STATE_INVALID")
    return "CURRENT_C2" if active else "CURRENT_C1"


def require_current_resume(state: AgentState) -> None:
    kind = checkpoint_kind(state.model_dump(mode="json"))
    code = {"READONLY": "READONLY_OBSERVATION_TERMINAL",
            "CURRENT_C1": "C1_PROVENANCE_CHECKPOINT_RESUME_FORBIDDEN",
            "SCIENCE": "RESEARCH_BACKEND_OUT_OF_SCOPE",
            "LEGACY": "LEGACY_CHECKPOINT_RESUME_FORBIDDEN"}.get(kind)
    if code:
        raise ResearchSafetyError(code, 409)
    if state.status not in {"PAUSED", "RUNNING"}:
        raise ResearchSafetyError("RESEARCH_TRUST_TERMINAL", 409)


class JsonResearchAgentCheckpointStore:
    def __init__(self, root: Path = Path(".runtime/research-agent/checkpoints")) -> None:
        self.root = root

    @contextmanager
    def task_write_lease(self, task_id: str):
        """Serialize one task's POST/resume writers across OS processes."""
        if os.name == "nt":
            import msvcrt
        elif os.name == "posix":
            import fcntl
        else:
            raise ResearchSafetyError("RESEARCH_TASK_LOCK_UNSUPPORTED", 503)

        path = self.path_for(task_id)
        self.root.mkdir(parents=True, exist_ok=True)
        lease_path = path.with_suffix(".lock")
        with lease_path.open("a+b") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            try:
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ResearchSafetyError("RESEARCH_TASK_BUSY", 409) from exc
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def save(self, state: AgentState, phase: str) -> str:
        path = self.path_for(state.task_id)
        kind = checkpoint_kind(state.model_dump(mode="json"))
        if kind not in {"CURRENT_C1", "CURRENT_C2", "READONLY"}:
            raise ResearchSafetyError("LEGACY_CHECKPOINT_WRITE_FORBIDDEN", 409)
        if path.exists():
            previous = self.load(state.task_id)
            if kind not in {"CURRENT_C1", "CURRENT_C2"} or checkpoint_kind(previous.model_dump(mode="json")) != kind:
                raise ResearchSafetyError("LEGACY_CHECKPOINT_WRITE_FORBIDDEN", 409)
        checkpoint_id = f"{state.task_id}-{len(state.checkpoint_chain) + 1:04d}-{phase}"
        if kind == "CURRENT_C2":
            staged = state.model_copy(deep=True)
            staged.checkpoint_id = checkpoint_id
            staged.checkpoint_chain.append(checkpoint_id)
            payload = staged.model_dump_json(indent=2)
            checkpoint_kind(json.loads(payload))
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            with temporary.open("x", encoding="utf8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            state.checkpoint_id = checkpoint_id
            state.checkpoint_chain.append(checkpoint_id)
            return checkpoint_id
        state.checkpoint_id = checkpoint_id
        state.checkpoint_chain.append(checkpoint_id)
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        return checkpoint_id

    def load(self, task_id: str) -> AgentState:
        path = self.path_for(task_id)
        if not path.exists():
            raise KeyError(f"checkpoint not found: {task_id}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"),
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            kind = checkpoint_kind(raw)
            if raw.get("task_id") != task_id:
                raise ResearchSafetyError("RESEARCH_TRUST_SCHEMA_INVALID")
            state = AgentState.model_validate(raw)
            if kind == "READONLY":
                # Reuse D2's complete strict snapshot validation, including ignored nested fields.
                from paper_research.api.routes.research import _readonly_snapshot
                _readonly_snapshot(state)
            return state
        except ResearchSafetyError:
            raise
        except Exception as exc:
            code = "READONLY_OBSERVATION_INVALID" if locals().get("kind") == "READONLY" else "RESEARCH_TRUST_SCHEMA_INVALID"
            raise ResearchSafetyError(code) from exc

    def path_for(self, task_id: str) -> Path:
        validate_research_task_id(task_id)
        path = self.root / f"{task_id}.json"
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ResearchSafetyError("INVALID_RESEARCH_TASK_ID", 422)
        return path
