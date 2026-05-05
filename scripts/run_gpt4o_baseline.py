import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = str(PROJECT_ROOT / "data" / "question.json")
RESULT_DIR = str(PROJECT_ROOT / "results" / "gpt4o_baseline")
DEFAULT_MODEL = "gpt-4o"


GPT4O_PROMPT_TEMPLATE = """
You are an expert in GAALOPScript within the geometric algebra domain. Your task is to understand user input and generate GAALOPScript code consisting of three components: optimized computation code, variable assignment statements, and multivectors to be visualized. Below are the task description and GA formula provided
by the user:
Task Description: {task_description}
GA Formula: {ga_formula}
Target Programming Language: {target_language}
Target Space: {target_space}

Return the GAALOP Web service request payload as one pure JSON object using
this structure:
{{
  "visualizationEnabled": true,
  "outputMode": "CODE_AND_VISUALIZATION",
  "codegenPlugins": "{target_language}",
  "algebraPlugins": "{target_space}",
  "optimization": {{
    "maxima": false,
    "cse": false
  }},
  "script": {{
    "optimizeCode": "",
    "variableAssignments": "",
    "multivectorsVisualized": "",
    "functionName": ""
  }}
}}
""".strip()


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


def split_question(question: str) -> tuple[str, str]:
    markers = [
        "Calculation process:",
        "Calculation Process:",
        "GA Formula:",
        "Formula:",
    ]
    for marker in markers:
        if marker in question:
            task_description, ga_formula = question.split(marker, 1)
            return task_description.strip(), ga_formula.strip()
    return question.strip(), ""


def infer_target_language(question: str) -> str:
    language_patterns = [
        ("PYTHON", r"\bpython\b"),
        ("CLUCALC", r"\bclucalc\b"),
        ("JULIA", r"\bjulia\b"),
        ("VERILOG", r"\bverilog\b"),
        ("CSHARP", r"\bc#\b|\bcsharp\b"),
        ("RUST", r"\brust\b"),
        ("JAVA", r"\bjava\b"),
        ("MATLAB", r"\bmatlab\b"),
        ("CPP", r"\bc\+\+\b|\bcpp\b"),
        ("LATEX", r"\blatex\b"),
    ]
    for plugin_name, pattern in language_patterns:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return plugin_name
    return "PYTHON"


def infer_target_space(question: str) -> str:
    space_patterns = [
        ("ALGEBRA_CGA", r"\bconformal\b|\bcga\b"),
        ("ALGEBRA_3D_PGA", r"\b3d\s+pga\b"),
        ("ALGEBRA_2D_PGA", r"\b2d\s+pga\b"),
        ("ALGEBRA_3D", r"\b3d\b"),
        ("ALGEBRA_2D", r"\b2d\b"),
        ("ALGEBRA_STA", r"\bsta\b|space\s*time"),
        ("ALGEBRA_DCGA", r"\bdcga\b"),
        ("ALGEBRA_CCGA", r"\bccga\b"),
        ("ALGEBRA_QGA", r"\bqga\b"),
    ]
    for plugin_name, pattern in space_patterns:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return plugin_name
    return "ALGEBRA_CGA"


def build_prompt(question: str) -> str:
    task_description, ga_formula = split_question(question)
    target_language = infer_target_language(question)
    target_space = infer_target_space(question)
    return GPT4O_PROMPT_TEMPLATE.format(
        task_description=task_description,
        ga_formula=ga_formula or "Not explicitly provided.",
        target_language=target_language,
        target_space=target_space,
    )


def get_gpt4o_llm(model: str, temperature: float, timeout: float, max_retries: int):
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain_openai is required to run GPT-4o tests. "
            "Install it with: pip install langchain-openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to run GPT-4o tests.")

    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def invoke_llm(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def parse_json_output(output: str) -> dict | None:
    text = str(output or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_success_result(index: int, question: str, prompt: str, output: str) -> dict:
    task_description, ga_formula = split_question(question)
    parsed_output = parse_json_output(output)
    return {
        "index": index,
        "success": True,
        "user_input": question,
        "task_description": task_description,
        "ga_formula": ga_formula,
        "prompt": prompt,
        "output": output,
        "parsed_output": parsed_output,
        "json_parse_success": parsed_output is not None,
    }


def build_error_result(index: int, question: str, prompt: str, exc: Exception) -> dict:
    return {
        "index": index,
        "success": False,
        "user_input": question,
        "prompt": prompt,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def save_result(result_dir: Path, index: int, result: dict) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"{index}.json"
    output_path.write_text(
        json.dumps(make_json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def save_summary(
    result_dir: Path,
    total: int,
    success_count: int,
    failed_count: int,
    json_success_count: int,
) -> Path:
    payload = {
        "model": os.getenv("GPT4O_MODEL", DEFAULT_MODEL),
        "total": total,
        "api_success": success_count,
        "api_failed": failed_count,
        "json_parse_success": json_success_count,
        "json_parse_failed": success_count - json_success_count,
        "note": (
            "This file records API call success only. The prompt asks GPT-4o "
            "to return a GAALOP Web request JSON payload. JSON validity and "
            "GAALOPScript correctness must be judged separately."
        ),
    }
    output_path = result_dir / "summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def run_batch(args: argparse.Namespace) -> None:
    questions = load_questions(args.data)
    result_dir = Path(args.result_dir)
    llm = get_gpt4o_llm(
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    success_count = 0
    failed_count = 0
    json_success_count = 0
    total = len(questions)

    for index, question in enumerate(questions, start=1):
        prompt = build_prompt(question)
        print(f"[{index}/{total}] Running GPT-4o...")
        try:
            output = invoke_llm(llm, prompt)
            result = build_success_result(index, question, prompt, output)
            success_count += 1
            if result.get("json_parse_success"):
                json_success_count += 1
            output_path = save_result(result_dir, index, result)
            print(f"[{index}/{total}] Saved -> {output_path}")
        except Exception as exc:
            result = build_error_result(index, question, prompt, exc)
            failed_count += 1
            output_path = save_result(result_dir, index, result)
            print(f"[{index}/{total}] Failed -> {output_path}")
            print(f"Error: {exc}")

        if args.sleep > 0 and index < total:
            time.sleep(args.sleep)

    summary_path = save_summary(result_dir, total, success_count, failed_count, json_success_count)
    print("")
    print("GPT-4o batch finished.")
    print(f"Total: {total}")
    print(f"API success: {success_count}")
    print(f"API failed: {failed_count}")
    print(f"JSON parse success: {json_success_count}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-style GPT-4o baseline prompt over the CGA question set."
    )
    parser.add_argument("--data", default=DATA_PATH, help="Path to question JSON.")
    parser.add_argument("--result-dir", default=RESULT_DIR, help="Directory for GPT-4o outputs.")
    parser.add_argument("--model", default=os.getenv("GPT4O_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())
