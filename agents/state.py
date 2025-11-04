# state.py
from typing import TypedDict, Optional, Dict, Any


class AgentState(TypedDict):
    """
    定义智能体图的状态。
    所有节点都会共享和修改这个状态对象。
    Optional表示某些字段在工作流的某些阶段可能不存在。
    """
    # 基础输出
    user_input: str

    # 由“信息提取智能体”填充的字段
    function_name: str
    target_language: str
    target_space: str

    # 中间状态
    analysis_result: Optional[str]
    generated_code: Optional[str]
    assignment_result: Optional[str]
    assignment_code: Optional[str]
    visualization_code: Optional[str]

    # GAALOPScript脚本
    final_code: Optional[str]

    # 最终的API响应
    api_response_code: Optional[Dict[str, Any]]