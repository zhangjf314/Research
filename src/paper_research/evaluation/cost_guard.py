"""Execution-only accounting, transactional HTTP continuation and immutable registry."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import threading
import time
import uuid
from collections import defaultdict

import httpx

from paper_research.agents.research_agent.checkpoint import JsonResearchAgentCheckpointStore
from paper_research.evaluation.phase2b import (
    OUT,
    ROOT,
    digest,
    now,
    read,
    rows,
    settings,
    sha,
    write,
)

NOT_OBSERVED = "NOT_OBSERVED"
LOCK = threading.RLock()
AMENDMENT = OUT / "protocol_cost_guard_amendment.json"


class InfrastructureInterruption(BaseException):
    """Bypass scientific failure handlers; the harness restores its transaction."""


def append(path, value):
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            handle.flush()


def atomic_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary, value)
    temporary.replace(path)


def verify_amendment(protocol):
    amendment = read(AMENDMENT)
    if sha(OUT / "protocol.json") != amendment["base_protocol_hash"]:
        raise ValueError("PHASE2B_CG_SCIENTIFIC_PROTOCOL_MISMATCH: base protocol")
    if sha(OUT / "benchmark/manifest.json") != protocol["benchmark_manifest_sha256"]:
        raise ValueError("PHASE2B_CG_SCIENTIFIC_PROTOCOL_MISMATCH: benchmark")
    for path, expected in amendment["harness_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("FORMAL_RESUME_IDENTITY_MISMATCH: " + path)
    for path, names in amendment["scientific_ast_sha256"].items():
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        nodes = {getattr(n, "name", ""): n for n in tree.body}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        for name, expected in names.items():
            observed = hashlib.sha256(
                ast.dump(nodes[name], include_attributes=False).encode()
            ).hexdigest()
            if observed != expected:
                raise ValueError("PHASE2B_CG_SCIENTIFIC_PROTOCOL_MISMATCH: " + name)
    return amendment


def identity():
    return {
        "protocol_hash": sha(OUT / "protocol.json"),
        "amendment_hash": sha(AMENDMENT),
        "benchmark_hash": sha(OUT / "benchmark/manifest.json"),
        "backend_hash": read(OUT / "protocol.json")["backend_manifest_sha256"],
    }


class FormalRunRegistry:
    def __init__(self, root=OUT):
        self.root = root
        self.path = root / "execution/formal_run_registry.json"
        self.identity = identity()
        self.data = (
            read(self.path) if self.path.exists() else {"identity": self.identity, "runs": {}}
        )
        if self.data["identity"] != self.identity:
            raise ValueError("FORMAL_RESUME_IDENTITY_MISMATCH")

    def entry(self, sample, arm):
        key = sample + "/" + arm
        return self.data["runs"].setdefault(
            key,
            {
                "sample_id": sample,
                "arm": arm,
                "run_identity": digest({**self.identity, "sample": sample, "arm": arm}),
                "status": "NOT_STARTED",
                **self.identity,
            },
        )

    def completed(self, sample, arm, path):
        entry = self.entry(sample, arm)
        if entry["status"] == "COMPLETED":
            if not path.exists() or sha(path) != entry["result_sha256"]:
                raise ValueError("FORMAL_RESUME_IDENTITY_MISMATCH: completed result")
            return True
        if path.exists():
            record = read(path)
            if record.get("execution_identity") != entry["run_identity"]:
                raise ValueError("FORMAL_RESUME_IDENTITY_MISMATCH: orphan result")
            self.update(
                sample, arm, "COMPLETED", result_sha256=sha(path), completion="FORMAL_RUN_COMPLETE"
            )
            return True
        return False

    def update(self, sample, arm, status, **extra):
        entry = self.entry(sample, arm)
        if entry["status"] == "COMPLETED" and status != "COMPLETED":
            raise ValueError("FORMAL_RUN_COMPLETE is immutable")
        entry.update(status=status, updated_at=now(), **extra)
        atomic_write(self.path, self.data)


def balance_snapshot(reason, root=OUT):
    s = settings()
    record = {
        "timestamp": now(),
        "reason": reason,
        "provider": "deepseek",
        "balance_delta_may_include_external_usage": True,
    }
    try:
        response = httpx.get(
            s.llm_base_url.rstrip("/") + "/user/balance",
            headers={"Authorization": "Bearer " + s.llm_api_key},
            timeout=30,
        )
        record["http_status"] = response.status_code
        response.raise_for_status()
        info = next(x for x in response.json()["balance_infos"] if x["currency"] == "CNY")
        record.update(
            balance=float(info["total_balance"]), currency="CNY", request_status="SUCCESS"
        )
    except Exception as exc:
        record.update(
            balance=NOT_OBSERVED, currency="CNY", request_status="FAILED", error=type(exc).__name__
        )
    append(root / "cost/balance_snapshots.jsonl", record)
    return record


def usage_summary(root=OUT):
    path = root / "cost/formal_usage.jsonl"
    events = rows(path) if path.exists() else []
    groups = defaultdict(
        lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_cny": 0.0,
            "missing_usage_requests": 0,
        }
    )
    for event in events:
        key = event["provider"] + "/" + event["phase"]
        group = groups[key]
        group["requests"] += 1
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            if isinstance(event[field], (int, float)):
                group[field] += event[field]
        if event["total_tokens"] == NOT_OBSERVED:
            group["missing_usage_requests"] += 1
        if isinstance(event["estimated_cost_cny"], (int, float)):
            group["estimated_cost_cny"] += event["estimated_cost_cny"]
    formal = [e for e in events if e["phase"] in ("formal_arm_execution", "posthoc_evaluation")]
    known = sum(
        e["reported_cost"]
        if isinstance(e["reported_cost"], (int, float)) and e.get("reported_cost_currency") == "CNY"
        else e["estimated_cost_cny"]
        if isinstance(e["estimated_cost_cny"], (int, float))
        else 0
        for e in formal
    )
    snapshots = root / "cost/balance_snapshots.jsonl"
    observed = (
        [s["balance"] for s in rows(snapshots) if isinstance(s.get("balance"), (int, float))]
        if snapshots.exists()
        else []
    )
    gross = sum(max(0, a - b) for a, b in zip(observed, observed[1:], strict=False))
    summary = {
        "timestamp": now(),
        "providers_by_phase": dict(groups),
        "formal_requests": len(formal),
        "combined_observable_estimated_cny": known,
        "conservative_account_gross_decrease_cny": gross,
        "guard_cost_cny": max(known, gross),
        "actual_cost": "NOT_ASSESSED",
        "balance_delta_may_include_external_usage": True,
        "siliconflow_unreported_cost": "NOT_ASSESSED",
        "preparation_excluded": True,
    }
    arm_totals = defaultdict(lambda: {"requests": 0, "observed_total_tokens": 0})
    operations = defaultdict(
        lambda: {
            "requests": 0,
            "input_characters": 0,
            "observed_input_tokens": 0,
            "observed_output_tokens": 0,
            "missing_input_tokens": 0,
            "estimated_cny": 0.0,
        }
    )
    for event in formal:
        arm = arm_totals[event["phase"] + "/" + event.get("arm", "NONE")]
        arm["requests"] += 1
        if isinstance(event["total_tokens"], int):
            arm["observed_total_tokens"] += event["total_tokens"]
        operation = operations[event["provider"] + "/" + event.get("operation_type", "other")]
        operation["requests"] += 1
        operation["input_characters"] += event.get("input_characters", 0)
        for field in ("input_tokens", "output_tokens"):
            if isinstance(event[field], int):
                operation["observed_" + field] += event[field]
        if not isinstance(event["input_tokens"], int):
            operation["missing_input_tokens"] += 1
        if isinstance(event["estimated_cost_cny"], (int, float)):
            operation["estimated_cny"] += event["estimated_cost_cny"]
    summary["by_arm_and_phase"] = dict(arm_totals)
    summary["by_provider_operation"] = dict(operations)
    registry = root / "execution/formal_run_registry.json"
    if registry.exists():
        entries = list(read(registry)["runs"].values())
        completed = [e for e in entries if e["status"] == "COMPLETED"]
        counts = defaultdict(int)
        for entry in completed:
            counts[entry["sample_id"]] += 1
        summary.update(
            completed_runs=len(completed),
            completed_samples=sum(n == 3 for n in counts.values()),
            remaining_runs=180 - len(completed),
            remaining_formal_request_upper_bound=(180 - len(completed)) * 12,
        )
    write(root / "cost/provider_input_size_audit.json", dict(operations))
    write(root / "cost/formal_usage_summary.json", summary)
    return summary


class CostGuard:
    def __init__(self, root=OUT):
        self.root = root

    def before_sample(self, sample, already_started=False):
        if already_started:
            return
        if usage_summary(self.root)["guard_cost_cny"] >= 60:
            raise InfrastructureInterruption("COST_HARD_PAUSE")
        record = balance_snapshot("before_sample:" + sample, self.root)
        if not isinstance(record["balance"], (int, float)) or record["balance"] < 20:
            raise InfrastructureInterruption("INSUFFICIENT_START_BALANCE")

    def boundary(self, sample, completed_arms):
        if set(completed_arms) != {"a0", "a1c", "a1"}:
            raise ValueError("cost threshold requires complete sample")
        balance_snapshot("complete_sample:" + sample, self.root)
        summary = usage_summary(self.root)
        cost = summary["guard_cost_cny"]
        status = (
            "COST_HARD_PAUSE"
            if cost >= 60
            else "COST_SOFT_CHECKPOINT"
            if cost >= 30
            else "CONTINUE"
        )
        write(
            self.root / "execution/progress.json",
            {"status": status, "completed_sample": sample, **summary},
        )
        if status == "COST_HARD_PAUSE":
            raise InfrastructureInterruption(status)
        return status


class TransactionCheckpointStore(JsonResearchAgentCheckpointStore):
    """Retain the last safe pre-operation state alongside the qualified checkpoint."""

    def __init__(self, root):
        super().__init__(root)
        self.safe = None
        self.interrupted = False
        self.segment = "initial"
        self.ordinal = 0

    def load(self, task_id):
        state = super().load(task_id)
        recovery = self.root / "safe_transaction.json"
        if recovery.exists():
            saved = read(recovery)
            self.safe = saved["state"]
            self.segment = saved["segment"]
            from paper_research.agents.research_agent.state import AgentState

            state = AgentState.model_validate(self.safe)
        return state

    def save(self, state, phase):
        if self.interrupted:
            write(self.root / "infrastructure_checkpoint.json", state.model_dump(mode="json"))
            return state.checkpoint_id or "INFRASTRUCTURE_CHECKPOINT"
        checkpoint = super().save(state, phase)
        if phase != "POST_USAGE_ACCOUNTING":
            self.safe = state.model_dump(mode="json")
            # No wall-clock, checkpoint-chain or resume-count fields in replay identity.
            self.segment = digest(
                {
                    "phase": phase,
                    "steps": state.step_count,
                    "tools": state.tool_call_count,
                    "provider": state.provider_call_count,
                    "replans": len(state.replan_history),
                }
            )
            self.ordinal = 0
            atomic_write(
                self.root / "safe_transaction.json", {"state": self.safe, "segment": self.segment}
            )
        return checkpoint


class RequestObserver:
    def __init__(
        self, folder, sample, arm, phase="formal_arm_execution", checkpoints=None, root=OUT
    ):
        self.folder, self.sample, self.arm, self.phase = folder, sample, arm, phase
        self.checkpoints, self.root = checkpoints, root
        self.operation = "verifier"
        self.retry = 0
        self.bindings = []
        self.ordinal = 0
        self.segment = "generation"

    def wrap_backend(self, backend):
        original = backend.search
        journal = self.folder / "backend_continuation.json"

        def search(*args, **kwargs):
            segment = self.checkpoints.segment
            before = len(backend.calls)
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                if isinstance(exc, httpx.TransportError) or type(exc).__name__ in {
                    "ResponseHandlingException",
                    "ReadTimeout",
                    "ConnectTimeout",
                }:
                    self.checkpoints.interrupted = True
                    raise InfrastructureInterruption(type(exc).__name__) from exc
                raise
            saved = read(journal) if journal.exists() else {}
            calls = list(backend.calls[before:])
            if segment in saved and saved[segment] != calls:
                self.checkpoints.interrupted = True
                raise InfrastructureInterruption("FORMAL_RESUME_IDENTITY_MISMATCH: retrieval")
            saved[segment] = calls
            atomic_write(journal, saved)
            return result

        backend.search = search
        self.bindings.append((backend, original, "search"))

    def attach(self, client, provider_name):
        original = client.send

        def send(request, **kwargs):
            store = self.checkpoints
            segment = (
                store.segment if store and self.operation != "answer_generation" else self.segment
            )
            ordinal = (
                store.ordinal if store and self.operation != "answer_generation" else self.ordinal
            )
            if store and self.operation != "answer_generation":
                store.ordinal += 1
            else:
                self.ordinal += 1
            request.read()
            request_hash = hashlib.sha256(
                request.method.encode() + str(request.url).encode() + request.content
            ).hexdigest()
            path = self.folder / "http_continuation" / f"{segment}-{ordinal:04d}.json"
            if path.exists():
                saved = read(path)
                if saved["request_hash"] != request_hash:
                    if store:
                        store.interrupted = True
                    raise InfrastructureInterruption(
                        "FORMAL_RESUME_IDENTITY_MISMATCH: HTTP continuation"
                    )
                append(
                    self.root / "execution/http_replays.jsonl",
                    {
                        "timestamp": now(),
                        "sample_id": self.sample,
                        "arm": self.arm,
                        "request_hash": request_hash,
                    },
                )
                return httpx.Response(
                    saved["status"],
                    headers=saved["headers"],
                    content=base64.b64decode(saved["body"]),
                    request=request,
                )
            started = time.perf_counter()
            status, body, error = NOT_OBSERVED, {}, None
            try:
                response = original(request, **kwargs)
                response.read()
                status = response.status_code
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if status in (402, 408, 429) or status >= 500:
                    raise InfrastructureInterruption("HTTP " + str(status))
                # Preserve every returned non-infrastructure response, including schema failures.
                atomic_write(
                    path,
                    {
                        "request_hash": request_hash,
                        "status": status,
                        "headers": dict(response.headers),
                        "body": base64.b64encode(response.content).decode(),
                    },
                )
                return response
            except (httpx.TransportError, InfrastructureInterruption) as exc:
                error = type(exc).__name__ + ": " + str(exc)[:180]
                if store:
                    store.interrupted = True
                raise InfrastructureInterruption(error) from exc
            finally:
                try:
                    payload = json.loads(request.content)
                except ValueError:
                    payload = {}
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                inputs = usage.get("prompt_tokens", usage.get("input_tokens", NOT_OBSERVED))
                outputs = usage.get("completion_tokens", usage.get("output_tokens", NOT_OBSERVED))
                cached = usage.get(
                    "prompt_cache_hit_tokens",
                    usage.get("prompt_tokens_details", {}).get("cached_tokens", NOT_OBSERVED),
                )
                estimate = NOT_OBSERVED
                if (
                    provider_name == "deepseek"
                    and isinstance(inputs, int)
                    and isinstance(outputs, int)
                ):
                    cache = cached if isinstance(cached, int) else 0
                    estimate = ((inputs - cache) * 2 + cache * 0.04 + outputs * 8) / 1_000_000
                operation = "other" if provider_name == "siliconflow" else self.operation
                record = {
                    "timestamp": now(),
                    "sample_id": self.sample,
                    "arm": self.arm,
                    "phase": self.phase,
                    "provider": provider_name,
                    "model": payload.get("model", NOT_OBSERVED),
                    "reported_model": body.get("model", NOT_OBSERVED),
                    "operation_type": operation,
                    "input_tokens": inputs,
                    "output_tokens": outputs,
                    "cached_input_tokens": cached,
                    "total_tokens": usage.get("total_tokens", NOT_OBSERVED),
                    "input_characters": len(request.content.decode("utf-8", errors="replace")),
                    "request_success": isinstance(status, int) and 200 <= status < 300,
                    "http_status": status,
                    "retry_count": self.retry,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "reported_cost": usage.get("cost", NOT_OBSERVED),
                    "reported_cost_currency": usage.get("currency", NOT_OBSERVED),
                    "estimated_cost_cny": estimate,
                    "estimate_basis": (
                        "DeepSeek official peak upper tariff observed 2026-10-02; "
                        "CNY/M miss2 hit0.04 out8"
                    )
                    if estimate != NOT_OBSERVED
                    else "NOT_ASSESSED",
                    "endpoint": request.url.path,
                    "request_hash": request_hash,
                    "error": error,
                }
                if self.phase == "posthoc_evaluation":
                    record["combined_semantic_operations"] = [
                        "posthoc_contract_judge",
                        "posthoc_evidence_judge",
                        "posthoc_claim_judge",
                    ]
                append(self.root / "cost/formal_usage.jsonl", record)

        client.send = send
        self.bindings.append((client, original))

    def wrap_llm(self, llm):
        original = llm.generate_structured_json

        def generate(**kwargs):
            schema = kwargs.get("schema_name", "")
            context = kwargs.get("request_context", {})
            self.retry = max(context.get("attempt_number", 1) - 1, 0)
            if self.phase == "posthoc_evaluation":
                self.operation = "posthoc_evidence_judge"
            elif schema == "agent_final_report_draft_v1":
                self.operation = "answer_generation"
                self.segment = "generation-" + str(context.get("attempt_number", 1))
                self.ordinal = 0
            else:
                self.operation = (
                    "replan_verifier"
                    if self.checkpoints
                    and self.checkpoints.safe
                    and self.checkpoints.safe.get("replan_history")
                    else "verifier"
                )
            return original(**kwargs)

        llm.generate_structured_json = generate
        self.attach(llm.client, "deepseek")

    def close(self):
        for binding in self.bindings:
            obj, original = binding[:2]
            setattr(obj, binding[2] if len(binding) == 3 else "send", original)


def canonical_judge_input(payload, root=OUT):
    """Exact content only: preserve all semantic strings and array order."""
    value = copy.deepcopy(payload)
    value.pop("response_id", None)
    content_hash = digest(value)
    path = root / "evaluation/judge_cache/response_ids.json"
    with LOCK:
        ids = read(path) if path.exists() else {}
        if content_hash not in ids:
            ids[content_hash] = "response-" + uuid.uuid4().hex[:24]
            atomic_write(path, ids)
    value["response_id"] = ids[content_hash]
    # JSON field-order/serialization whitespace only; never normalize source prose.
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True)), content_hash


def judge_cache_key(judge_type, prompt, model, payload):
    return digest(
        {
            "judge_type": judge_type,
            "prompt_hash": digest(prompt),
            "model": model,
            "canonical_input_hash": digest(payload),
        }
    )
