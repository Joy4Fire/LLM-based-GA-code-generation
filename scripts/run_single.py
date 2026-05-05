import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_visagent.legacy_agents.nodes import (
    code_to_optimize_agent_node,
    multivectors_to_be_visualized_agent_node,
    variable_assignments_agent_node,
)


def print_json(title: str, data) -> None:
    print(f"================ {title} ================")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print()


def build_single_task_state() -> dict:
    return {
        "task_blocks_result": {
            "tasks": [
                {
                    "task_id": 1,
                    "task_type": "construct_cga_point",
                    "operation": "construct_point",
                    "inputs": [],
                    "outputs": ["P1"],
                    "depends_on": [],
                    "object_specs": {
                        "name": "P1",
                        "type": "point",
                        "coordinates": [0, 0, 0],
                    },
                    "visualization": {
                        "required": True,
                        "objects": [
                            {
                                "name": "P1",
                                "type": "point",
                                "color": "Red",
                            }
                        ],
                    },
                    "code_to_optimize": {
                        "goal": "Construct CGA point P1.",
                        "formula": "P1 = a1*e1 + b1*e2 + c1*e3 + 0.5*(a1*a1 + b1*b1 + c1*c1)*einf + e0",
                        "parameters": ["a1", "b1", "c1"],
                        "output": "P1",
                    },
                    "variable_assignments": {
                        "a1": 0,
                        "b1": 0,
                        "c1": 0,
                    },
                    "multivectors_to_be_visualized": {
                        "required": True,
                        "objects": [
                            {
                                "name": "P1",
                                "type": "point",
                                "color": "Red",
                            }
                        ],
                    },
                }
            ]
        }
    }


def main() -> None:
    state = build_single_task_state()
    print_json("INPUT TASK BLOCK", state.get("task_blocks_result"))

    state.update(code_to_optimize_agent_node(state))
    state.update(variable_assignments_agent_node(state))
    state.update(multivectors_to_be_visualized_agent_node(state))

    print_json("CODE TO OPTIMIZE OUTPUT", state.get("code_to_optimize_result"))
    print_json("VARIABLE ASSIGNMENTS OUTPUT", state.get("variable_assignments_result"))
    print_json("MULTIVECTORS TO BE VISUALIZED OUTPUT", state.get("multivectors_to_be_visualized_result"))

    final_tasks = state.get("multivectors_to_be_visualized_result", {}).get("tasks", [])
    final_single_task = final_tasks[0] if isinstance(final_tasks, list) and final_tasks else {}
    print_json("FINAL SINGLE TASK RESULT", final_single_task)


if __name__ == "__main__":
    main()
