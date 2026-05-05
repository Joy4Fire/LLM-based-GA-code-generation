from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Minimal shared state for the refactored LangGraph baseline."""

    user_input: str
    task_blocks_result: dict[str, Any]
    code_to_optimize_result: dict[str, Any]
    variable_assignments_result: dict[str, Any]
    multivectors_to_be_visualized_result: dict[str, Any]
