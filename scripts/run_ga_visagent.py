import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ga_visagent.main_graph.graph import run_main_graph


DATA_PATH = str(PROJECT_ROOT / "data" / "question.json")
RESULT_DIR = str(PROJECT_ROOT / "results" / "ga_visagent")


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    questions: list[Any] = []

    if isinstance(payload, dict) and "conformal_space_tasks" in payload:
        conformal_tasks = payload.get("conformal_space_tasks")
        if isinstance(conformal_tasks, list):
            questions = conformal_tasks
    elif isinstance(payload, list):
        questions = payload
    elif isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                questions.extend(value)

    normalized_questions = [
        item.strip()
        for item in questions
        if isinstance(item, str) and item.strip()
    ]

    if not normalized_questions:
        raise ValueError(f"No valid string questions found in {path}")

    return normalized_questions


def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_success_result(index: int, user_input: str, state: dict) -> dict:
    return {
        "index": index,
        "success": True,
        "user_input": user_input,
        "function_name": state.get("function_name"),
        "target_language": state.get("target_language"),
        "target_space": state.get("target_space"),
        "task_blocks_result": state.get("task_blocks_result"),
        "validated_task_blocks_result": state.get("validated_task_blocks_result"),
        "task_ir_validation_result": state.get("task_ir_validation_result"),
        "operation_task_blocks_result": state.get("operation_task_blocks_result"),
        "subtask_execution_order": state.get("subtask_execution_order"),
        "subtask_results": state.get("subtask_results"),
        "final_code": state.get("final_code"),
        "gaalop_request_result": state.get("gaalop_request_result"),
    }


def build_error_result(index: int, user_input: str, exc: Exception) -> dict:
    return {
        "index": index,
        "success": False,
        "user_input": user_input,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def save_result(index: int, result: dict) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, f"{index}.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(make_json_safe(result), file, ensure_ascii=False, indent=2)
    return output_path.replace("\\", "/")


def is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text or "readtimeout" in text


def run_main_graph_with_retry(question: str, *, index: int, max_retries: int = 3):
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return run_main_graph(question)
        except Exception as exc:
            if not is_timeout_error(exc):
                raise
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            sleep_seconds = 2 * (2 ** attempt)
            print(f"[{index}] Timeout, retry {attempt + 1}/{max_retries} after {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def run_batch() -> None:
    questions = load_questions(DATA_PATH)
    total = len(questions)
    success_count = 0
    failed_count = 0

    os.makedirs(RESULT_DIR, exist_ok=True)

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{total}] Running...")
        try:
            state = run_main_graph_with_retry(question, index=index)
            result = build_success_result(index, question, state)
            success_count += 1
            output_path = save_result(index, result)
            print(f"[{index}/{total}] Success -> {output_path}")
        except Exception as exc:
            result = build_error_result(index, question, exc)
            failed_count += 1
            output_path = save_result(index, result)
            print(f"[{index}/{total}] Failed -> {output_path}")
            print(f"Error: {exc}")

    print("")
    print("Batch finished.")
    print(f"Total: {total}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    run_batch()
