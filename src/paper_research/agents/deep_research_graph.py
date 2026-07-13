import re
from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from paper_research.agents.providers import ExternalResearchProvider, LocalResearchProvider
from paper_research.agents.state import ResearchBudget, ResearchState, initial_state

ImportProvider = Callable[[dict], str | None]


class DeepResearchGraph:
    def __init__(
        self,
        local_provider: LocalResearchProvider,
        external_provider: ExternalResearchProvider | None = None,
        import_provider: ImportProvider | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        interrupt_after: list[str] | None = None,
    ) -> None:
        self.local_provider = local_provider
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
        aspects = ["研究背景与问题", "主要方法与技术路线", "实验结果与比较", "局限与研究空白"]
        sub_questions = [f"{query}：{aspect}是什么？" for aspect in aspects]
        return {
            "sub_questions": sub_questions,
            "search_queries": [query, *[f"{query} {aspect}" for aspect in aspects[1:]]],
            "research_plan": [
                "检索本地论文证据",
                "判断子问题覆盖与证据缺口",
                "必要时检索并导入外部论文",
                "综合多论文共识、差异和局限",
                "生成并校验带引用报告",
            ],
            "node_history": [*state["node_history"], "plan"],
        }

    def _local_search(self, state: ResearchState) -> dict:
        budget = ResearchBudget.model_validate(state["budget"])
        evidence = list(state["local_evidence"])
        existing_ids = {
            (item["evidence_id"], item.get("sub_question")) for item in evidence
        }
        for sub_question in state["sub_questions"]:
            for item in self.local_provider.search(
                sub_question, state.get("requested_paper_ids") or None, limit=5
            ):
                evidence_key = (item["evidence_id"], sub_question)
                if evidence_key not in existing_ids:
                    item["sub_question"] = sub_question
                    evidence.append(item)
                    existing_ids.add(evidence_key)
                if len(evidence) >= budget.max_evidence_items:
                    break
        previous = len(state["local_evidence"])
        no_new = state["no_new_evidence_rounds"] + 1 if len(evidence) == previous else 0
        estimated_tokens = sum(len(item["quote"].split()) for item in evidence)
        return {
            "local_evidence": evidence,
            "previous_evidence_count": previous,
            "no_new_evidence_rounds": no_new,
            "estimated_tokens": estimated_tokens,
            "iteration_count": state["iteration_count"] + 1,
            "node_history": [*state["node_history"], "local_search"],
        }

    def _assess(self, state: ResearchState) -> dict:
        covered = {item.get("sub_question") for item in state["local_evidence"]}
        gaps = [question for question in state["sub_questions"] if question not in covered]
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
        contradictions = self._find_contradictions(state["local_evidence"])
        return {
            "contradictions": contradictions,
            "node_history": [*state["node_history"], "synthesize"],
        }

    def _report(self, state: ResearchState) -> dict:
        evidence = state["local_evidence"]
        sections = [
            "# 深度研究报告",
            "",
            "## 1. 研究问题与范围",
            state["research_goal"],
            "",
            "## 2. 检索策略",
            "；".join(state["research_plan"]),
            "",
        ]
        headings = ["领域背景", "主要研究路线", "实验结果对比", "当前局限"]
        for index, question in enumerate(state["sub_questions"]):
            sections.append(f"## {index + 3}. {headings[index]}")
            matches = [item for item in evidence if item.get("sub_question") == question][:5]
            if not matches:
                sections.append("当前证据不足，未形成确定性结论。")
            for item in matches:
                citation = f"[{item['paper_id']}, p.{item['page_start']}]"
                sections.append(f"- {item['quote'][:500]} {citation}")
            sections.append("")
        sections.extend(
            [
                "## 7. 共识与争议",
                "检测到的潜在冲突见结构化 contradictions 字段。",
                "",
                "## 8. 研究空白与后续建议",
                "；".join(state["evidence_gaps"]) or "当前子问题均有至少一条本地证据覆盖。",
                "",
                "## 9. 参考证据",
            ]
        )
        for item in evidence:
            sections.append(
                f"- {item['paper_id']}，第 {item['page_start']} 页，"
                f"{' > '.join(item['section_path']) or '未命名章节'}"
            )
        return {
            "draft_report": "\n".join(sections),
            "node_history": [*state["node_history"], "report"],
        }

    def _validate(self, state: ResearchState) -> dict:
        citations = []
        for item in state["local_evidence"]:
            marker = f"[{item['paper_id']}, p.{item['page_start']}]"
            citations.append(
                {
                    "marker": marker,
                    "evidence_id": item["evidence_id"],
                    "valid": marker in state["draft_report"],
                }
            )
        all_valid = all(item["valid"] for item in citations)
        return {
            "citation_results": citations,
            "status": "COMPLETED" if all_valid else "FAILED_VALIDATION",
            "stop_reason": state.get("stop_reason") or "research_complete",
            "node_history": [*state["node_history"], "validate"],
        }

    @staticmethod
    def _find_contradictions(evidence: list[dict]) -> list[dict]:
        positive_pattern = re.compile(r"\b(outperform|improve|increase)\b", re.I)
        negative_pattern = re.compile(r"\b(underperform|decrease|worse)\b", re.I)
        positive = [item for item in evidence if positive_pattern.search(item["quote"])]
        negative = [item for item in evidence if negative_pattern.search(item["quote"])]
        if positive and negative:
            return [
                {
                    "type": "directional_result_conflict",
                    "evidence_ids": [
                        positive[0]["evidence_id"],
                        negative[0]["evidence_id"],
                    ],
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
        if len(state["local_evidence"]) >= budget.max_evidence_items:
            return "max_evidence_items"
        return None
