from ga_visagent.main_graph.nodes import (
    final_code_assembler_node,
    gaalop_compile_node,
    gaalop_request_builder_node,
    information_extraction_node,
    operation_to_task_block_node,
    subtask_dispatcher_node,
    task_ir_validator_node,
    task_decomposition_node,
)


def run_main_information_extraction(user_input: str) -> dict:
    state = {"user_input": user_input}
    state.update(information_extraction_node(state))
    return state


def run_main_graph(user_input: str) -> dict:
    state = {"user_input": user_input}
    state.update(information_extraction_node(state))
    state.update(task_decomposition_node(state))
    state.update(task_ir_validator_node(state))
    state.update(operation_to_task_block_node(state))
    state.update(subtask_dispatcher_node(state))
    state.update(final_code_assembler_node(state))
    state.update(gaalop_request_builder_node(state))
    state.update(gaalop_compile_node(state))
    return state
