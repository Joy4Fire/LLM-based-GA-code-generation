import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_visagent.main_graph.graph import run_main_graph


API_URL = "http://gacrac.gagis.cn:8080/api/v1/compile"
DEBUG_PAYLOAD_PATH = os.path.join(".", "debug", "gaalop_request_payload.json")
DEFAULT_USER_INPUT = (
    "In conformal space, create a point P with coordinates (0,0,0) and visualize it as red. "
    "I need Java code. Function name is denis. And you should use maxima and cse."
)


def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_debug_payload(payload: dict, path: str = DEBUG_PAYLOAD_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, ensure_ascii=False, indent=2)


def strip_output_prefix_from_optimize_code(optimize_code: str) -> str:
    normalized_lines = []
    for line in optimize_code.splitlines():
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        if stripped.startswith("?"):
            normalized_lines.append(f"{leading}{stripped[1:]}")
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def call_gaalop_compile_api(payload: dict, api_url: str = API_URL, timeout: int = 120):
    if requests is None:
        raise RuntimeError("requests is required to call the Gaalop API. Install it with: pip install requests")

    safe_payload = make_json_safe(payload)
    response = requests.post(
        api_url,
        json=safe_payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )

    print("================ REQUEST URL ================")
    print(api_url)

    print("================ REQUEST PAYLOAD ================")
    print(json.dumps(safe_payload, ensure_ascii=False, indent=2))

    print("================ HTTP STATUS ================")
    print(response.status_code)

    print("================ RESPONSE HEADERS ================")
    print(dict(response.headers))

    print("================ RESPONSE TEXT ================")
    print(response.text)

    if response.status_code >= 400:
        raise RuntimeError(f"Gaalop compile API failed with status {response.status_code}: {response.text}")

    return response


def run_to_code(
    user_input: str,
    *,
    disable_visualization: bool = False,
    strip_output_prefix: bool = False,
):
    print("================ USER INPUT ================")
    print(user_input)

    print("================ RUNNING MAIN GRAPH ================")
    state = run_main_graph(user_input)

    gaalop_request_result = state.get("gaalop_request_result")
    if not gaalop_request_result:
        raise ValueError("run_main_graph did not produce gaalop_request_result")

    payload = make_json_safe(gaalop_request_result)
    if disable_visualization:
        payload["visualizationEnabled"] = False
        payload["outputMode"] = "CODE_ONLY"
        script = payload.get("script")
        if isinstance(script, dict):
            script["multivectorsVisualized"] = ""

    if strip_output_prefix:
        script = payload.get("script")
        optimize_code = script.get("optimizeCode") if isinstance(script, dict) else None
        if isinstance(optimize_code, str):
            script["optimizeCode"] = strip_output_prefix_from_optimize_code(optimize_code)

    save_debug_payload(payload)

    print("================ GAALOP REQUEST PAYLOAD ================")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("================ DEBUG PAYLOAD FILE ================")
    print(DEBUG_PAYLOAD_PATH)

    if state.get("gaalop_compile_result") and not disable_visualization and not strip_output_prefix:
        print("================ GAALOP COMPILE RESULT ================")
        print(json.dumps(make_json_safe(state.get("gaalop_compile_result")), ensure_ascii=False, indent=2))
        return {
            "state": state,
            "response": None,
        }

    print("================ CALLING GAALOP COMPILE API ================")
    response = call_gaalop_compile_api(payload)

    print("================ API RESPONSE ================")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)

    return {
        "state": state,
        "response": response,
    }


def get_user_input_from_args():
    args = sys.argv[1:]
    disable_visualization = False
    strip_output_prefix = False
    user_input_parts = []

    for arg in args:
        if arg == "--no-visualization":
            disable_visualization = True
            continue
        if arg == "--strip-output-prefix":
            strip_output_prefix = True
            continue
        user_input_parts.append(arg)

    user_input = " ".join(user_input_parts).strip() or DEFAULT_USER_INPUT
    return user_input, disable_visualization, strip_output_prefix


if __name__ == "__main__":
    try:
        user_input, disable_visualization, strip_output_prefix = get_user_input_from_args()
        run_to_code(
            user_input,
            disable_visualization=disable_visualization,
            strip_output_prefix=strip_output_prefix,
        )
    except Exception as exc:
        print("================ ERROR ================")
        print(str(exc))
        print("================ TRACEBACK ================")
        traceback.print_exc()
        sys.exit(1)
