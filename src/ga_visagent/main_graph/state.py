from typing import Optional, TypedDict


class MainGraphState(TypedDict, total=False):
    user_input: str
    function_name: str
    target_language: str
    target_space: str
    task_blocks_result: dict
    validated_task_blocks_result: dict
    task_ir_validation_result: dict
    operation_task_blocks_result: dict
    subtask_execution_order: list[int]
    subtask_results: dict
    final_code: str
    gaalop_request_result: dict
    gaalop_compile_result: dict
    gaalop_compile_attempts: list
    gaalop_script_repair_count: int
    gaalop_script_repairs: list
    information_extraction_raw: str
    information_extraction_error: Optional[str]
