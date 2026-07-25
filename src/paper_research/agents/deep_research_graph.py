from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from paper_research.agents.providers import (
    ExternalResearchProvider,
    LocalResearchProvider,
)
from paper_research.agents.report_quality import evaluate_report_quality
from paper_research.agents.research_models import ResearchEvidence
from paper_research.agents.research_synthesis_provider import (
    ResearchSynthesisLLMProvider,
)
from paper_research.agents.state import ResearchBudget, ResearchState, initial_state
from paper_research.providers.llm import LLMProviderError

ImportProvider = Callable[[dict], str | None]


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    title: str
    intent: str


SECTION_SPECS = [
    SectionSpec(
        "background",
        "领域背景",
        "该领域解决什么问题、发展背景、核心定义和研究动机是什么？",
    ),
    SectionSpec(
        "methods",
        "主要研究路线",
        "论文提出了哪些 RAG 架构、检索策略、训练方法和技术路线？",
    ),
    SectionSpec(
        "results",
        "实验结果对比",
        "论文报告了哪些数据集、指标、基线、定量结果和消融实验？",
    ),
    SectionSpec(
        "limitations",
        "当前局限",
        "论文明确指出哪些限制、失败情形、成本、延迟和未来工作？",
    ),
]

SECTION_TITLES = {spec.section_id: spec.title for spec in SECTION_SPECS}


class DeepResearchGraph:
    def __init__(
        self,
        local_provider: LocalResearchProvider,
        external_provider: ExternalResearchProvider | None = None,
        import_provider: ImportProvider | None = None,
        synthesis_provider: ResearchSynthesisLLMProvider | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_after: list[str] | None = None,
    ) -> None:
        self.local_provider = local_provider
        self.synthesis_provider = synthesis_provider
        self.external_provider = external_provider
        self.import_provider = import_provider
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build().compile(
            checkpointer=self.checkpointer,
            interrupt_after=interrupt_after,
        )

    def run(
        self,
        query: str,
        *,
        budget: ResearchBudget | None = None,
        paper_ids: list[str] | None = None,
        task_id: str | None = None,
    ) -> ResearchState:
        state = initial_state(query, budget or ResearchBudget(), paper_ids)
        if task_id:
            state["task_id"] = task_id
        config = {"configurable": {"thread_id": state["task_id"]}}
        result = self.graph.invoke(state, config=config)
        if self.graph.get_state(config).next:
            result["status"] = "PAUSED"
        return result

    def resume(self, task_id: str) -> ResearchState:
        config = {"configurable": {"thread_id": task_id}}
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"checkpoint not found: {task_id}")
        if not snapshot.next:
            return snapshot.values
        return self.graph.invoke(None, config=config)

    def _build(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        workflow.add_node("understand", self._understand)
        workflow.add_node("plan", self._plan)
        workflow.add_node("local_search", self._local_search)
        workflow.add_node("assess", self._assess)
        workflow.add_node("external_search", self._external_search)
        workflow.add_node("select_import", self._select_import)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("report", self._report)
        workflow.add_node("validate", self._validate)
        workflow.add_edge(START, "understand")
        workflow.add_edge("understand", "plan")
        workflow.add_edge("plan", "local_search")
        workflow.add_edge("local_search", "assess")
        workflow.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {"external": "external_search", "synthesize": "synthesize"},
        )
        workflow.add_edge("external_search", "select_import")
        workflow.add_conditional_edges(
            "select_import",
            self._route_after_import,
            {"retry_local": "local_search", "synthesize": "synthesize"},
        )
        workflow.add_edge("synthesize", "report")
        workflow.add_edge("report", "validate")
        workflow.add_edge("validate", END)
        return workflow

    def _understand(self, state: ResearchState) -> dict:
        normalized = " ".join(state["original_query"].split())
        return {
            "normalized_query": normalized,
            "research_goal": f"系统梳理并基于论文证据回答：{normalized}",
            "node_history": [*state["node_history"], "understand"],
        }

    def _plan(self, state: ResearchState) -> dict:
        query = state["normalized_query"]
        section_queries = {
            spec.section_id: f"{query}。{spec.intent}"
            for spec in SECTION_SPECS
        }
        return {
            "sub_questions": list(section_queries.values()),
            "section_queries": section_queries,
            "search_queries": [query, *section_queries.values()],
            "research_plan": [
                "使用生产混合检索为各报告章节检索独立证据",
                "构建全局唯一 Evidence Catalog，并记录章节到证据的关系",
                "评估章节证据缺口，必要时检索外部候选论文",
                "基于结构化 synthesis 生成确定性 Markdown 报告",
                "执行 Citation 与报告重复质量门禁",
            ],
            "node_history": [*state["node_history"], "plan"],
        }

    def _local_search(self, state: ResearchState) -> dict:
        budget = ResearchBudget.model_validate(state["budget"])
        catalog = dict(state.get("evidence_catalog", {}))
        section_evidence_ids = {
            spec.section_id: list(state.get("section_evidence_ids", {}).get(spec.section_id, []))
            for spec in SECTION_SPECS
        }
        previous = len(catalog)
        local_evidence = list(state.get("local_evidence", []))

        for section_id, query in state["section_queries"].items():
            for raw in self.local_provider.search(
                query, state.get("requested_paper_ids") or None, limit=8
            ):
                evidence = ResearchEvidence.model_validate(raw)
                if not self._is_usable_evidence(evidence):
                    continue
                citation_id = self._citation_id(catalog, evidence)
                existing = catalog.get(citation_id)
                if existing is None:
                    evidence = evidence.model_copy(update={"target_sections": [section_id]})
                    item = evidence.model_dump()
                    item["citation_id"] = citation_id
                    catalog[citation_id] = item
                    local_evidence.append(item)
                else:
                    targets = list(
                        dict.fromkeys([*existing.get("target_sections", []), section_id])
                    )
                    existing["target_sections"] = targets
                if citation_id not in section_evidence_ids[section_id]:
                    section_evidence_ids[section_id].append(citation_id)
                if len(catalog) >= budget.max_evidence_items:
                    break

        no_new = state["no_new_evidence_rounds"] + 1 if len(catalog) == previous else 0
        section_evidence_ids = self._normalize_section_evidence_links(
            catalog,
            section_evidence_ids,
        )
        estimated_tokens = sum(len(item["text"].split()) for item in catalog.values())
        return {
            "local_evidence": local_evidence,
            "evidence_catalog": catalog,
            "section_evidence_ids": section_evidence_ids,
            "previous_evidence_count": previous,
            "no_new_evidence_rounds": no_new,
            "estimated_tokens": estimated_tokens,
            "iteration_count": state["iteration_count"] + 1,
            "node_history": [*state["node_history"], "local_search"],
        }

    def _assess(self, state: ResearchState) -> dict:
        gaps = []
        for spec in SECTION_SPECS:
            ids = state.get("section_evidence_ids", {}).get(spec.section_id, [])
            if not ids:
                gaps.append(f"{SECTION_TITLES[spec.section_id]}：缺少可用证据")
            elif spec.section_id == "results" and not self._has_result_evidence(
                [state["evidence_catalog"][citation_id] for citation_id in ids]
            ):
                gaps.append("实验结果对比：当前证据不足以完成可靠的实验结果比较")
        stop_reason = self._budget_stop_reason(state)
        return {
            "evidence_gaps": gaps,
            "stop_reason": stop_reason,
            "node_history": [*state["node_history"], "assess"],
        }

    def _route_after_assessment(self, state: ResearchState) -> str:
        if not state["evidence_gaps"] or state.get("stop_reason"):
            return "synthesize"
        if self.external_provider is None:
            return "synthesize"
        return "external"

    def _external_search(self, state: ResearchState) -> dict:
        assert self.external_provider is not None
        query = " ".join(state["evidence_gaps"][:2]) or state["normalized_query"]
        candidates = self.external_provider.search(query, limit=10)
        return {
            "candidate_papers": candidates,
            "external_search_count": state["external_search_count"] + 1,
            "node_history": [*state["node_history"], "external_search"],
        }

    def _select_import(self, state: ResearchState) -> dict:
        budget = ResearchBudget.model_validate(state["budget"])
        selected = [
            candidate
            for candidate in state["candidate_papers"]
            if candidate.get("pdf_url")
        ][: budget.max_papers]
        imported: list[dict] = []
        if self.import_provider:
            for candidate in selected:
                paper_id = self.import_provider(candidate)
                if paper_id:
                    imported.append({**candidate, "paper_id": paper_id})
        external_evidence = [
            {
                "paper_id": item.get("paper_id") or item.get("source_id"),
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "source_url": item.get("source_url"),
            }
            for item in selected
        ]
        return {
            "selected_papers": imported or selected,
            "external_evidence": external_evidence,
            "node_history": [*state["node_history"], "select_import"],
        }

    def _route_after_import(self, state: ResearchState) -> str:
        can_retry = (
            self.import_provider
            and state["selected_papers"]
            and not self._budget_stop_reason(state)
        )
        if can_retry:
            return "retry_local"
        return "synthesize"

    def _synthesize(self, state: ResearchState) -> dict:
        if self.synthesis_provider is not None:
            try:
                result = self.synthesis_provider.synthesize(
                    question=state["normalized_query"],
                    section_queries=state["section_queries"],
                    evidence_catalog=state["evidence_catalog"],
                    section_evidence_ids=state.get("section_evidence_ids", {}),
                    contradictions=self._find_contradictions(state["evidence_catalog"]),
                    request_context={
                        "run_id": state["task_id"],
                        "request_id": state["task_id"],
                    },
                )
            except LLMProviderError as exc:
                status = (
                    "FAILED_PROVIDER_SCHEMA"
                    if exc.error_code == "FAILED_PROVIDER_SCHEMA"
                    else "FAILED_PROVIDER"
                )
                usage = (
                    exc.error_details.get("usage")
                    if isinstance(exc.error_details, dict)
                    else {}
                )
                usage_record_count = (
                    int(exc.error_details.get("usage_record_count", 0))
                    if isinstance(exc.error_details, dict)
                    else 0
                )
                if not usage and exc.usage_records:
                    input_tokens = sum(record.usage.input_tokens for record in exc.usage_records)
                    output_tokens = sum(record.usage.output_tokens for record in exc.usage_records)
                    total_tokens = sum(record.usage.total_tokens for record in exc.usage_records)
                    costs = [
                        record.usage.estimated_cost_usd
                        for record in exc.usage_records
                        if record.usage.estimated_cost_usd is not None
                    ]
                    usage = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": sum(costs) if costs else None,
                        "usage_source": "provider_reported"
                        if all(
                            record.usage.usage_source == "provider_reported"
                            for record in exc.usage_records
                        )
                        else "estimated",
                    }
                    usage_record_count = len(exc.usage_records)
                return {
                    "status": status,
                    "stop_reason": str(exc),
                    "synthesis": {},
                    "provider_completed_request_count": exc.api_request_count,
                    "request_attempt_count": exc.api_request_count,
                    "usage_record_count": usage_record_count,
                    "active_reserved_tokens": 0,
                    "model_usage": usage,
                    "node_history": [*state["node_history"], "synthesize_llm"],
                }
            return {
                "synthesis": result.synthesis.model_dump(),
                "model_usage": result.usage.model_dump(),
                "llm_provider": result.provider,
                "llm_model": result.model,
                "request_attempt_count": result.request_attempt_count,
                "provider_completed_request_count": result.provider_completed_request_count,
                "usage_record_count": 1 if result.usage.total_tokens else 0,
                "active_reserved_tokens": 0,
                "node_history": [*state["node_history"], "synthesize_llm"],
            }

        sections = {}
        for spec in SECTION_SPECS:
            evidence_items = [
                state["evidence_catalog"][citation_id]
                for citation_id in state.get("section_evidence_ids", {}).get(spec.section_id, [])
                if citation_id in state["evidence_catalog"]
            ]
            sections[spec.section_id] = self._synthesize_section(spec, evidence_items)
        synthesis = {
            "title": "深度研究报告",
            "executive_summary": self._executive_summary(state, sections),
            "sections": sections,
            "consensus": self._section_claims(sections, ["background", "methods"]),
            "disagreements": self._find_contradictions(state["evidence_catalog"]),
            "research_gaps": state.get("evidence_gaps", []),
        }
        return {
            "synthesis": synthesis,
            "contradictions": synthesis["disagreements"],
            "node_history": [*state["node_history"], "synthesize"],
        }

    def _report(self, state: ResearchState) -> dict:
        if state.get("status") in {"FAILED_PROVIDER_SCHEMA", "FAILED_PROVIDER"}:
            return {
                "draft_report": "",
                "report_quality": None,
                "node_history": [*state["node_history"], "render_report"],
            }
        synthesis = state["synthesis"]
        synthesis_sections = synthesis.get("sections", {})
        if isinstance(synthesis_sections, list):
            sections_by_id = {
                section["section_id"]: section
                for section in synthesis_sections
            }
        else:
            sections_by_id = synthesis_sections
        catalog = state["evidence_catalog"]
        lines = [
            "# 深度研究报告",
            "",
            "## 摘要",
            synthesis["executive_summary"],
            "",
            "## 1. 研究问题与范围",
            state["research_goal"],
            "",
            "## 2. 检索与证据范围",
            f"- 检索子问题：{len(state['section_queries'])} 个章节意图。",
            f"- 全局唯一证据：{len(catalog)} 条。",
            f"- 检索后证据缺口：{len(state.get('evidence_gaps', []))} 条。",
            "",
        ]
        section_texts: dict[str, str] = {}
        for offset, spec in enumerate(SECTION_SPECS, start=3):
            section = sections_by_id[spec.section_id]
            section_lines = [
                f"## {offset}. {spec.title}",
                section["summary"],
            ]
            for claim in section["claims"]:
                citations = " ".join(claim["citation_ids"])
                section_lines.append(f"- {claim['text']} {citations}".rstrip())
            if section.get("insufficient_evidence"):
                section_lines.append(f"- 证据不足：{section['evidence_gap']}")
            lines.extend([*section_lines, ""])
            section_texts[spec.section_id] = "\n".join(section_lines[1:])

        lines.extend(
            [
                "## 7. 共识与争议",
                self._consensus_to_markdown(synthesis["consensus"]),
                self._claims_to_markdown(
                    synthesis["disagreements"],
                    fallback="当前检索证据未形成明确的方向性冲突。",
                ),
                "",
                "## 8. 研究空白与后续建议",
            ]
        )
        if synthesis["research_gaps"]:
            lines.extend(f"- {gap}" for gap in synthesis["research_gaps"])
        else:
            lines.append("- 当前检索证据覆盖了四个核心章节，但这不等同于完整语义充分性证明。")
        lines.extend(["", "## 9. 参考证据"])
        used_citations = set(re.findall(r"\bE\d{2,3}\b|\[E\d+\]", "\n".join(lines)))
        for citation_id in sorted(used_citations, key=self._citation_sort_key):
            if citation_id not in catalog:
                continue
            item = catalog[citation_id]
            section = " > ".join(item.get("section_path", [])) or "未命名章节"
            lines.append(
                f"- {citation_id} {item['paper_id']}，第 {item['page_start']}"
                f"-{item['page_end']} 页，{section}"
            )
        report = "\n".join(lines)
        quality = evaluate_report_quality(
            report,
            sections=section_texts,
            evidence_catalog=catalog,
            section_evidence_ids=state.get("section_evidence_ids", {}),
        )
        return {
            "draft_report": report,
            "report_quality": quality.model_dump(),
            "node_history": [*state["node_history"], "render_report"],
        }

    def _validate(self, state: ResearchState) -> dict:
        if state.get("status") in {"FAILED_PROVIDER_SCHEMA", "FAILED_PROVIDER"}:
            return {
                "citation_results": [],
                "status": state["status"],
                "stop_reason": state.get("stop_reason"),
                "node_history": [*state["node_history"], "validate_report_quality"],
            }
        catalog = state["evidence_catalog"]
        used = set(re.findall(r"\bE\d{2,3}\b|\[E\d+\]", state["draft_report"]))
        citations = [
            {
                "citation_id": citation_id,
                "evidence_id": item["evidence_id"],
                "paper_id": item["paper_id"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "valid": citation_id in catalog,
                "used": citation_id in used,
            }
            for citation_id, item in sorted(
                catalog.items(),
                key=lambda pair: self._citation_sort_key(pair[0]),
            )
        ]
        quality = state.get("report_quality") or {}
        quality_passed = bool(quality.get("passed", False))
        status = "COMPLETED" if quality_passed else "FAILED_REPORT_QUALITY"
        if not catalog:
            status = "FAILED_RETRIEVAL"
        return {
            "citation_results": citations,
            "status": status,
            "stop_reason": state.get("stop_reason") or (
                "research_complete" if status == "COMPLETED" else status.lower()
            ),
            "node_history": [*state["node_history"], "validate_report_quality"],
        }

    @staticmethod
    def _citation_id(catalog: dict[str, dict[str, Any]], evidence: ResearchEvidence) -> str:
        for citation_id, item in catalog.items():
            if (item["paper_id"], item["evidence_id"]) == evidence.global_key:
                return citation_id
        return f"E{len(catalog) + 1:02d}"

    @staticmethod
    def _normalize_section_evidence_links(
        catalog: dict[str, dict[str, Any]],
        section_evidence_ids: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        """Keep model-visible section allowlists and evidence metadata in sync."""

        section_ids = {spec.section_id for spec in SECTION_SPECS}
        normalized = {
            section_id: [
                citation_id
                for citation_id in dict.fromkeys(section_evidence_ids.get(section_id, []))
                if citation_id in catalog
            ]
            for section_id in section_ids
        }
        for citation_id, item in catalog.items():
            targets = [
                section_id
                for section_id in dict.fromkeys(item.get("target_sections", []))
                if section_id in section_ids
            ]
            for section_id, ids in normalized.items():
                if citation_id in ids and section_id not in targets:
                    targets.append(section_id)
            item["target_sections"] = targets
            for section_id in targets:
                if citation_id not in normalized[section_id]:
                    normalized[section_id].append(citation_id)
        return {spec.section_id: normalized[spec.section_id] for spec in SECTION_SPECS}

    @staticmethod
    def _citation_sort_key(citation_id: str) -> int:
        match = re.search(r"\d+", citation_id)
        return int(match.group()) if match else 0

    @staticmethod
    def _is_usable_evidence(evidence: ResearchEvidence) -> bool:
        text = evidence.text.strip()
        if len(text) < 20:
            return False
        lowered = text.lower()
        low_value_patterns = (
            "references\n",
            "system prompt",
            "{system}",
            "{user}",
            "ignore previous instructions",
        )
        if any(pattern in lowered for pattern in low_value_patterns):
            return False
        section = " > ".join(evidence.section_path).lower()
        if "reference" in section:
            return False
        if lowered.count("\n") > 20 and len(text) < 500:
            return False
        return True

    @staticmethod
    def _has_result_evidence(items: list[dict[str, Any]]) -> bool:
        pattern = re.compile(
            r"\b(\d+(?:\.\d+)?%?|accuracy|f1|bleu|rouge|dataset|baseline|"
            r"ablation|outperform|improve|result|score|benchmark)\b",
            re.I,
        )
        return any(pattern.search(item.get("text", "")) for item in items)

    def _synthesize_section(self, spec: SectionSpec, evidence_items: list[dict[str, Any]]) -> dict:
        if not evidence_items:
            return {
                "section_id": spec.section_id,
                "summary": f"当前检索证据不足以可靠回答“{spec.title}”。",
                "claims": [],
                "insufficient_evidence": True,
                "evidence_gap": f"{spec.title}缺少可用证据。",
            }
        if spec.section_id == "results" and not self._has_result_evidence(evidence_items):
            return {
                "section_id": spec.section_id,
                "summary": "当前证据不足以完成可靠的实验结果比较。",
                "claims": [],
                "insufficient_evidence": True,
                "evidence_gap": "实验结果章节没有检索到足够的指标、数据集、基线或定量结果证据。",
            }
        claims = []
        for item in evidence_items[:3]:
            claims.append(
                {
                    "text": self._claim_from_evidence(spec.section_id, item),
                    "citation_ids": [item["citation_id"]],
                }
            )
        return {
            "section_id": spec.section_id,
            "summary": self._section_summary(spec.section_id, evidence_items),
            "claims": claims,
            "insufficient_evidence": False,
            "evidence_gap": None,
        }

    @staticmethod
    def _section_summary(section_id: str, evidence_items: list[dict[str, Any]]) -> str:
        paper_count = len({item["paper_id"] for item in evidence_items})
        if section_id == "background":
            return f"检索证据覆盖 {paper_count} 篇论文的研究动机、任务背景或问题定义。"
        if section_id == "methods":
            return f"检索证据覆盖 {paper_count} 篇论文的方法设计、系统组件或技术路线。"
        if section_id == "results":
            return f"检索证据覆盖 {paper_count} 篇论文的实验设置、指标或结果描述。"
        return f"检索证据覆盖 {paper_count} 篇论文的局限、成本、失败情形或未来工作。"

    @staticmethod
    def _claim_from_evidence(section_id: str, item: dict[str, Any]) -> str:
        section = " > ".join(item.get("section_path", [])) or "相关章节"
        prefix = {
            "background": "背景证据显示",
            "methods": "方法证据显示",
            "results": "实验结果证据显示",
            "limitations": "局限性证据显示",
        }[section_id]
        snippet = _first_sentence(item.get("text", ""))
        return f"{prefix}，{item['paper_id']} 的 {section} 提供了与该章节问题相关的依据：{snippet}"

    @staticmethod
    def _executive_summary(state: ResearchState, sections: dict[str, dict[str, Any]]) -> str:
        answered = sum(1 for section in sections.values() if not section["insufficient_evidence"])
        return (
            f"本报告围绕“{state['normalized_query']}”进行章节化检索与综合。"
            f"四个核心章节中有 {answered}/4 个章节具备可用证据；"
            "所有引用均来自全局唯一 Evidence Catalog，未使用重复 evidence dump。"
        )

    @staticmethod
    def _section_claims(sections: dict[str, dict[str, Any]], section_ids: list[str]) -> list[dict]:
        claims: list[dict] = []
        for section_id in section_ids:
            claims.extend(sections.get(section_id, {}).get("claims", [])[:1])
        return claims

    @staticmethod
    def _claims_to_markdown(claims: list[dict], *, fallback: str) -> str:
        if not claims:
            return f"- {fallback}"
        return "\n".join(
            f"- {claim['text']} {' '.join(claim.get('citation_ids', []))}".rstrip()
            for claim in claims
        )

    @staticmethod
    def _consensus_to_markdown(claims: list[dict]) -> str:
        if not claims:
            return "- 当前证据主要支持上述章节化结论。"
        citations = " ".join(
            dict.fromkeys(
                citation
                for claim in claims
                for citation in claim.get("citation_ids", [])
            )
        )
        return (
            "- 共识主要来自背景与方法章节的交叉证据，"
            f"而不是重复粘贴原始段落。{citations}"
        ).rstrip()

    @staticmethod
    def _find_contradictions(evidence_catalog: dict[str, dict[str, Any]]) -> list[dict]:
        positive_pattern = re.compile(r"\b(outperform|improve|increase)\b", re.I)
        negative_pattern = re.compile(r"\b(underperform|decrease|worse)\b", re.I)
        positive = [
            item for item in evidence_catalog.values()
            if positive_pattern.search(item.get("text", ""))
        ]
        negative = [
            item for item in evidence_catalog.values()
            if negative_pattern.search(item.get("text", ""))
        ]
        if positive and negative:
            return [
                {
                    "text": "不同证据中存在结果方向不一致，需要人工进一步核验。",
                    "citation_ids": [positive[0]["citation_id"], negative[0]["citation_id"]],
                }
            ]
        return []

    @staticmethod
    def _budget_stop_reason(state: ResearchState) -> str | None:
        budget = ResearchBudget.model_validate(state["budget"])
        if state["iteration_count"] >= budget.max_iterations:
            return "max_iterations"
        if state["external_search_count"] >= budget.max_external_searches:
            return "max_external_searches"
        if state["no_new_evidence_rounds"] >= budget.max_no_new_evidence_rounds:
            return "no_new_evidence"
        if state["estimated_tokens"] >= budget.max_estimated_tokens:
            return "max_estimated_tokens"
        if len(state.get("evidence_catalog", {})) >= budget.max_evidence_items:
            return "max_evidence_items"
        return None


def _first_sentence(text: str, *, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "证据文本为空。"
    match = re.search(r"(.+?[。.!?])\s", cleaned)
    sentence = match.group(1) if match else cleaned
    if len(sentence) > max_chars:
        return sentence[: max_chars - 1].rstrip() + "…"
    return sentence
