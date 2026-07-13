def evaluate_agent_run(state: dict) -> dict[str, float | int | bool]:
    history = state.get("node_history", [])
    sub_questions = state.get("sub_questions", [])
    evidence = state.get("local_evidence", [])
    covered = {item.get("sub_question") for item in evidence}
    tool_nodes = {"local_search", "external_search", "select_import"}
    return {
        "sub_question_coverage": len(covered) / len(sub_questions) if sub_questions else 0.0,
        "search_rounds": history.count("local_search") + history.count("external_search"),
        "external_searches": state.get("external_search_count", 0),
        "selected_papers": len(state.get("selected_papers", [])),
        "tool_calls": sum(node in tool_nodes for node in history),
        "duplicate_node_calls": len(history) - len(set(history)),
        "stopped_with_reason": bool(state.get("stop_reason")),
        "citation_validity": (
            sum(item.get("valid", False) for item in state.get("citation_results", []))
            / len(state.get("citation_results", []))
            if state.get("citation_results")
            else 0.0
        ),
        "estimated_tokens": state.get("estimated_tokens", 0),
    }
