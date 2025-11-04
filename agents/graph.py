# graph.py
from langgraph.graph import StateGraph, END

from agents.nodes import extract_information_node, generate_code_node, extract_assignments_node, \
    generate_assignment_code_node, generate_visualization_code_node, integrate_code_node, call_gaalop_api_node, \
    analyze_task_node, router_node
from agents.state import AgentState


def create_graph():
    """创建并返回LangGraph工作流"""

    workflow = StateGraph(AgentState)

    # 1. 添加所有节点，包括新的信息提取器
    workflow.add_node("information_extractor", extract_information_node)
    workflow.add_node("analyzer", analyze_task_node)
    workflow.add_node("code_generator", generate_code_node)
    workflow.add_node("assignment_extractor", extract_assignments_node)
    workflow.add_node("assignment_generator", generate_assignment_code_node)
    workflow.add_node("visualization_generator", generate_visualization_code_node)
    workflow.add_node("integrator", integrate_code_node)
    workflow.add_node("api_caller", call_gaalop_api_node)

    # 2. 设置新的入口点
    workflow.set_entry_point("information_extractor")

    # 3. 定义新的流程顺序
    # 信息提取 -> 任务分解
    workflow.add_edge("information_extractor", "analyzer")

    # 后续流程与之前相同
    workflow.add_edge("analyzer", "code_generator")

    workflow.add_conditional_edges(
        "code_generator",
        router_node,
        {
            "with_assignment": "assignment_extractor",
            "without_assignment": "integrator"
        }
    )

    workflow.add_edge("assignment_extractor", "assignment_generator")
    workflow.add_edge("assignment_generator", "visualization_generator")
    workflow.add_edge("visualization_generator", "integrator")

    workflow.add_edge("integrator", "api_caller")
    workflow.add_edge("api_caller", END)

    # 4. 编译图
    app = workflow.compile()

    return app