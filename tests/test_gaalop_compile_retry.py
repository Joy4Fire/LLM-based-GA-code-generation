from copy import deepcopy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ga_visagent.main_graph.nodes as nodes


def build_state() -> dict:
    return {
        "user_input": "Create point P and visualize it.",
        "final_code": (
            "Code to optimize:\n"
            "?P=createPoint(a1,b1,c1)\n\n"
            "Variable assignments:\n"
            "a1=0; b1=0; c1=0;\n\n"
            "Multivectors to be visualized:\n"
            ":Red;\n"
            ":P;"
        ),
        "gaalop_request_result": {
            "visualizationEnabled": True,
            "outputMode": "CODE_AND_VISUALIZATION",
            "codegenPlugins": "JAVA",
            "algebraPlugins": "ALGEBRA_CGA",
            "optimization": {"maxima": False, "cse": False},
            "script": {
                "multivectorsVisualized": ":Red;\n:P;",
                "variableAssignments": "a1=0; b1=0; c1=0;",
                "functionName": "denis",
                "optimizeCode": "?P=createPoint(a1,b1,c1)",
            },
        },
    }


def test_compile_success_without_repair():
    original_call = nodes.call_gaalop_compile_api
    try:
        calls = []

        def fake_call(payload, *, api_url=nodes.GAALOP_COMPILE_API_URL, timeout=120):
            calls.append(deepcopy(payload))
            return {
                "success": True,
                "http_status": 200,
                "response_headers": {},
                "response_text": '{"statusCode":"200"}',
                "response_json": {"statusCode": "200"},
            }

        nodes.call_gaalop_compile_api = fake_call
        result = nodes.gaalop_compile_node(build_state())

        assert len(calls) == 1
        assert result["gaalop_compile_result"]["success"] is True
        assert result["gaalop_script_repair_count"] == 0
    finally:
        nodes.call_gaalop_compile_api = original_call


def test_compile_failure_repairs_script_and_retries():
    original_call = nodes.call_gaalop_compile_api
    original_repair = nodes.repair_gaalop_script_with_llm
    try:
        calls = []

        def fake_call(payload, *, api_url=nodes.GAALOP_COMPILE_API_URL, timeout=120):
            calls.append(deepcopy(payload))
            if len(calls) == 1:
                return {
                    "success": False,
                    "http_status": 400,
                    "response_headers": {},
                    "response_text": "missing semicolon",
                    "response_json": {"statusCode": "400", "message": "missing semicolon"},
                }
            return {
                "success": True,
                "http_status": 200,
                "response_headers": {},
                "response_text": '{"statusCode":"200"}',
                "response_json": {"statusCode": "200"},
            }

        def fake_repair(state, request_payload, compile_result):
            assert "missing semicolon" in compile_result["response_text"]
            return {
                "optimizeCode": "?P=createPoint(a1,b1,c1);",
                "variableAssignments": "a1=0; b1=0; c1=0;",
                "multivectorsVisualized": ":Red;\n:P;",
            }

        nodes.call_gaalop_compile_api = fake_call
        nodes.repair_gaalop_script_with_llm = fake_repair

        result = nodes.gaalop_compile_node(build_state())

        assert len(calls) == 2
        assert calls[1]["script"]["optimizeCode"] == "?P=createPoint(a1,b1,c1);"
        assert result["gaalop_compile_result"]["success"] is True
        assert result["gaalop_script_repair_count"] == 1
        assert result["final_code"].startswith("Code to optimize:\n?P=createPoint")
    finally:
        nodes.call_gaalop_compile_api = original_call
        nodes.repair_gaalop_script_with_llm = original_repair


def main():
    test_compile_success_without_repair()
    test_compile_failure_repairs_script_and_retries()
    print("main_graph_gaalop_compile_retry_test.py passed.")


if __name__ == "__main__":
    main()
