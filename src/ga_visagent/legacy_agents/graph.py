from langgraph.graph import END, START, StateGraph

from ga_visagent.legacy_agents.nodes import (
    code_to_optimize_agent_node,
    multivectors_to_be_visualized_agent_node,
    task_block_generator_node,
    variable_assignments_agent_node,
)
from ga_visagent.legacy_agents.state import AgentState


def create_graph():
    """Create a minimal LangGraph with task blocks, code-to-optimize, variable assignments, and visualization generation."""
    workflow = StateGraph(AgentState)
    workflow.add_node("task_block_generator_node", task_block_generator_node)
    workflow.add_node("code_to_optimize_agent_node", code_to_optimize_agent_node)
    workflow.add_node("variable_assignments_agent_node", variable_assignments_agent_node)
    workflow.add_node(
        "multivectors_to_be_visualized_agent_node",
        multivectors_to_be_visualized_agent_node,
    )
    workflow.add_edge(START, "task_block_generator_node")
    workflow.add_edge("task_block_generator_node", "code_to_optimize_agent_node")
    workflow.add_edge("code_to_optimize_agent_node", "variable_assignments_agent_node")
    workflow.add_edge(
        "variable_assignments_agent_node",
        "multivectors_to_be_visualized_agent_node",
    )
    workflow.add_edge("multivectors_to_be_visualized_agent_node", END)
    return workflow.compile()
