import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_visagent.main_graph.graph import run_main_graph


DEFAULT_INPUT = (
    "In conformal space, create points P1(0,0,0) and P2(1,0,0), "
    "then construct line L from P1 and P2. Visualize L in red. I need Python code."
)


def print_json(title: str, data) -> None:
    print(f"================ {title} ================")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print()


def print_text(title: str, text: str) -> None:
    print(f"================ {title} ================")
    print(text)
    print()


def run_case(user_input: str) -> dict:
    print_text("INPUT", user_input)
    state = run_main_graph(user_input)

    print_json(
        "INFORMATION EXTRACTION OUTPUT",
        {
            "function_name": state.get("function_name"),
            "target_language": state.get("target_language"),
            "target_space": state.get("target_space"),
        },
    )
    print_json("TASK DECOMPOSITION OUTPUT", state.get("task_blocks_result"))
    print_json("TASK IR VALIDATION OUTPUT", state.get("task_ir_validation_result"))
    print_json("OPERATION TO TASK BLOCK OUTPUT", state.get("operation_task_blocks_result"))
    print_json(
        "SUBTASK DISPATCHER OUTPUT",
        {
            "subtask_execution_order": state.get("subtask_execution_order"),
            "subtask_results": state.get("subtask_results"),
        },
    )
    print_text("FINAL CODE OUTPUT", state.get("final_code", ""))
    print_json("GAALOP REQUEST OUTPUT", state.get("gaalop_request_result"))
    return state


def main() -> None:
    user_input = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_INPUT
    try:
        run_case(user_input)
    except Exception as exc:
        print_text("ERROR", str(exc))
        print_text("TRACEBACK", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
