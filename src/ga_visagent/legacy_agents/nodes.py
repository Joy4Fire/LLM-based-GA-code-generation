import json
import os
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any

from ga_visagent.legacy_agents.state import AgentState
from ga_visagent.models.llm_setup import get_llm
from ga_visagent.prompts.code_to_optimize_agent_template import (
    GAALOPSCRIPT_RULES_TEXT,
    build_code_to_optimize_agent_prompt,
)
from ga_visagent.prompts.information_extraction_template import build_information_extraction_prompt
from ga_visagent.prompts.multivectors_to_be_visualized_agent_template import (
    MULTIVECTORS_TO_BE_VISUALIZED_RULES_TEXT,
    build_multivectors_to_be_visualized_agent_prompt,
)
from ga_visagent.prompts.task_block_generation_template import build_task_block_generation_prompt
from ga_visagent.prompts.variable_assignments_agent_template import (
    VARIABLE_ASSIGNMENTS_RULES_TEXT,
    build_variable_assignments_agent_prompt,
)

DEBUG_VERBOSE = False


def _debug_print(*args, **kwargs) -> None:
    if DEBUG_VERBOSE:
        print(*args, **kwargs)


@lru_cache(maxsize=1)
def _get_default_llm():
    return get_llm(
        llm_type=os.getenv("GA_VISAGENT_LLM_TYPE", "lm_studio"),
        model=os.getenv("GA_VISAGENT_LLM_MODEL", "Qwen/Qwen3.6-27B"),
        api_key=os.getenv("GA_VISAGENT_LLM_API_KEY", "local"),
        base_url=os.getenv("GA_VISAGENT_LLM_BASE_URL", "http://localhost:1234/v1"),
        timeout=float(os.getenv("GA_VISAGENT_LLM_TIMEOUT", "120")),
        max_retries=int(os.getenv("GA_VISAGENT_LLM_MAX_RETRIES", "2")),
    )


def _should_use_llm() -> bool:
    return True


def _parse_json_object(content: str) -> dict[str, Any] | None:
    cleaned = (content or "").strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _infer_function_name(user_input: str) -> str:
    explicit_patterns = [
        r"function(?:\s+name)?\s*(?:is|=)?\s*[:：]?\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"函数名称\s*(?:为|是|=)?\s*[:：]?\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"函数名\s*(?:为|是|=)?\s*[:：]?\s*([A-Za-z_][A-Za-z0-9_]*)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    point_match = re.search(r"\b(P\d+)\b", user_input)
    if point_match and re.search(r"\bCGA\b|共形|conformal", user_input, flags=re.IGNORECASE):
        return f"create{point_match.group(1)}InCGA"
    if point_match:
        return f"create{point_match.group(1)}"
    return "extractInformation"


def _infer_target_language(user_input: str) -> str:
    upper_text = user_input.upper()
    if "PYTHON" in upper_text:
        return "PYTHON"
    if "JAVA" in upper_text:
        return "JAVA"
    if "CPP" in upper_text or "C++" in user_input:
        return "CPP"
    if "MATLAB" in upper_text:
        return "MATLAB"
    return "PYTHON"


def _infer_target_space(user_input: str) -> str:
    lower_text = user_input.lower()
    if "cga" in lower_text or "conformal" in lower_text or "共形" in user_input:
        return "ALGEBRA_CGA"
    if "pga" in lower_text or "projective" in lower_text:
        return "ALGEBRA_3D_PGA"
    if "ega" in lower_text or "euclidean" in lower_text:
        return "ALGEBRA_3D"
    return "unknown"


def _rule_based_extract_information(user_input: str) -> dict[str, Any]:
    return {
        "function_name": _infer_function_name(user_input),
        "target_language": _infer_target_language(user_input),
        "target_space": _infer_target_space(user_input),
    }


def extract_information_node(state: AgentState) -> dict[str, Any]:
    """
    Extract function name, target language, and target space from user input.
    This is the only node kept for the LangGraph refactor baseline.
    """
    print("--- Node: Extract Information ---")

    user_input = state.get("user_input", "")
    if not user_input:
        print("Error: user_input is empty.")
        return {}

    fallback_info = _rule_based_extract_information(user_input)
    print(f"Rule-based fallback result: {fallback_info}")

    if not _should_use_llm():
        print("LLM call skipped. Set GA_USE_LLM=1 to enable remote model invocation.")
        print("Node result:")
        print(json.dumps(fallback_info, ensure_ascii=False, indent=2))
        return fallback_info

    try:
        print(
            "LLM config:",
            {
                "llm_type": os.getenv("GA_VISAGENT_LLM_TYPE", "lm_studio"),
                "model": os.getenv("GA_VISAGENT_LLM_MODEL", "Qwen/Qwen3.6-27B"),
                "base_url": os.getenv("GA_VISAGENT_LLM_BASE_URL", "http://localhost:1234/v1"),
                "timeout": float(os.getenv("GA_VISAGENT_LLM_TIMEOUT", "120")),
            },
        )
        llm = _get_default_llm()
        prompt = build_information_extraction_prompt(user_input)
        print("Invoking LLM...")
        result = llm.invoke(prompt)
    except Exception as exc:
        print(f"Error: information extraction invocation failed - {exc}")
        print("Falling back to rule-based extraction.")
        print("Node result:")
        print(json.dumps(fallback_info, ensure_ascii=False, indent=2))
        return fallback_info

    raw_content = getattr(result, "content", str(result))
    print(f"Raw LLM output: {raw_content}")
    parsed = _parse_json_object(raw_content)
    if not parsed:
        print("Error: information extraction returned invalid JSON.")
        print("Falling back to rule-based extraction.")
        print("Node result:")
        print(json.dumps(fallback_info, ensure_ascii=False, indent=2))
        return fallback_info

    extracted_info = {
        "function_name": parsed.get("function_name") or fallback_info["function_name"],
        "target_language": parsed.get("target_language") or fallback_info["target_language"],
        "target_space": parsed.get("target_space") or fallback_info["target_space"],
    }
    print(f"Extracted information: {extracted_info}")
    print("Node result:")
    print(json.dumps(extracted_info, ensure_ascii=False, indent=2))
    return extracted_info


BUILTIN_BASIS_VECTORS = {"e1", "e2", "e3", "e0", "einf"}
KNOWN_COLORS = {
    "red": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "purple": "Purple",
    "cyan": "Cyan",
    "magenta": "Magenta",
    "black": "Black",
    "white": "White",
    "orange": "Orange",
    "红色": "Red",
    "蓝色": "Blue",
    "绿色": "Green",
    "黄色": "Yellow",
    "紫色": "Purple",
    "青色": "Cyan",
    "黑色": "Black",
    "白色": "White",
    "橙色": "Orange",
}
VISUALIZATION_KEYWORDS = r"可视化|显示|绘制|渲染|visualize|plot|render|display"


def _map_target_space_to_space(target_space: str) -> str:
    mapping = {
        "ALGEBRA_CGA": "CGA",
        "ALGEBRA_3D_PGA": "PGA",
        "ALGEBRA_2D_PGA": "PGA",
    }
    return mapping.get(target_space, "unknown")


def _coerce_number(text: str) -> int | float | None:
    value = text.strip()
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d+\.\d+", value):
        return float(value)
    return None


def _clean_formula_text(formula: str | None) -> str | None:
    if formula is None:
        return None
    cleaned = formula.strip().strip("；;。")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _extract_formula_map(user_input: str) -> dict[str, str]:
    formulas: dict[str, str] = {}
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9_]*)\s*=\s*([^\n；;。]+)", user_input, flags=re.MULTILINE):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs.strip("；;。 ")):
            continue
        if not any(token in rhs for token in ("e0", "e1", "e2", "e3", "einf", "*", "+", "-", "/", "^", "∧", ".", "sqrt")):
            continue
        formulas.setdefault(lhs, _clean_formula_text(f"{lhs} = {rhs}") or f"{lhs} = {rhs}")
    return formulas


def _ordered_unique(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_parameters_from_formula(formula: str | None, output: str | None) -> list[str]:
    if not formula:
        return []
    rhs = formula.split("=", 1)[1] if "=" in formula else formula
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs)
    ignore = set(BUILTIN_BASIS_VECTORS)
    if output:
        ignore.add(output)
    ignore.update({"sqrt"})
    return _ordered_unique([token for token in tokens if token not in ignore])


def _infer_object_type(name: str) -> str:
    if not name:
        return "unknown"
    upper_name = name.upper()
    if upper_name.startswith(("P", "X")):
        return "point"
    if upper_name.startswith("L"):
        return "line"
    if upper_name.startswith("C"):
        return "circle"
    if upper_name.startswith("S"):
        return "sphere"
    return "multivector"


def _normalize_color(color: str | None) -> str | None:
    if color is None:
        return None
    normalized = color.strip()
    if not normalized:
        return None
    return KNOWN_COLORS.get(normalized.lower(), KNOWN_COLORS.get(normalized, normalized))


def _extract_explicit_assignments(user_input: str) -> dict[str, int | float]:
    assignments: dict[str, int | float] = {}
    for name, value in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)(?![A-Za-z0-9_])", user_input):
        parsed = _coerce_number(value)
        if parsed is not None:
            assignments[name] = parsed
    return assignments


def _is_placeholder_parameter(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z][A-Za-z0-9_]*", name))


def _extract_point_specs(user_input: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for match in re.finditer(r"\b(P\d+)\b\s*的坐标是\s*\(([^)]+)\)", user_input, flags=re.IGNORECASE):
        coords = [item.strip() for item in match.group(2).split(",") if item.strip()]
        specs.append(
            {
                "name": match.group(1),
                "coordinate_parameters": coords,
            }
        )
    return specs


def _specialize_point_formula(formula: str | None, output: str, coordinate_parameters: list[str]) -> str | None:
    if not formula:
        return None
    specialized = formula
    if "=" in specialized:
        _, rhs = specialized.split("=", 1)
        specialized = f"{output} = {rhs.strip()}"
    if len(coordinate_parameters) == 3:
        replacements = {
            "x": coordinate_parameters[0],
            "y": coordinate_parameters[1],
            "z": coordinate_parameters[2],
        }
        for source, target in replacements.items():
            specialized = re.sub(rf"\b{source}\b", target, specialized)
    return _clean_formula_text(specialized)


def _extract_exact_or_generic_formula(
    output_name: str,
    formulas: dict[str, str],
    generic_candidates: list[str],
    coordinate_parameters: list[str] | None = None,
) -> str | None:
    if output_name in formulas:
        return formulas[output_name]
    for candidate in generic_candidates:
        if candidate in formulas:
            formula = formulas[candidate]
            if coordinate_parameters is not None:
                return _specialize_point_formula(formula, output_name, coordinate_parameters)
            return formula
    return None


def _extract_object_visualization(user_input: str, object_name: str) -> tuple[bool, str | None]:
    color_pattern = r"(红色|蓝色|绿色|黄色|紫色|青色|黑色|白色|橙色|red|blue|green|yellow|purple|cyan|black|white|orange)"
    pattern_after = rf"{re.escape(object_name)}[^。；;\n]{{0,120}}(?:{VISUALIZATION_KEYWORDS})(?:为|成|in)?\s*{color_pattern}?"
    match_after = re.search(pattern_after, user_input, flags=re.IGNORECASE)
    if match_after:
        color = match_after.group(1) if match_after.lastindex else None
        return True, _normalize_color(color)

    pattern_before = rf"(?:{VISUALIZATION_KEYWORDS})[^。；;\n]{{0,80}}{re.escape(object_name)}(?:[^。；;\n]{{0,40}}(?:为|成|in)\s*{color_pattern})?"
    match_before = re.search(pattern_before, user_input, flags=re.IGNORECASE)
    if match_before:
        color = match_before.group(1) if match_before.lastindex else None
        return True, _normalize_color(color)

    return False, None


def _build_visualization_payload(required: bool, objects: list[dict[str, Any]]) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "objects": [],
        }
    return {
        "required": True,
        "objects": objects,
    }


def _build_point_tasks(
    user_input: str,
    formulas: dict[str, str],
    explicit_assignments: dict[str, int | float],
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    tasks: list[dict[str, Any]] = []
    missing_assignments: list[str] = []
    consumed_formulas: set[str] = set()

    for index, spec in enumerate(_extract_point_specs(user_input), start=1):
        output = spec["name"]
        coordinate_parameters = spec["coordinate_parameters"]
        formula = _extract_exact_or_generic_formula(output, formulas, ["P"], coordinate_parameters)
        if output in formulas:
            consumed_formulas.add(output)
        elif "P" in formulas:
            consumed_formulas.add("P")

        parameters = _extract_parameters_from_formula(formula, output) or coordinate_parameters
        variable_assignments: dict[str, Any] = {}
        for parameter in parameters:
            if _is_placeholder_parameter(parameter):
                if parameter in explicit_assignments:
                    variable_assignments[parameter] = explicit_assignments[parameter]
                else:
                    variable_assignments[parameter] = None
                    missing_assignments.append(parameter)

        visualization_required, color = _extract_object_visualization(user_input, output)
        visualization_objects = []
        if visualization_required:
            visualization_objects.append(
                {
                    "name": output,
                    "type": "point",
                    "color": color,
                }
            )

        tasks.append(
            {
                "task_id": index,
                "task_type": "construct_cga_point",
                "code_to_optimize": {
                    "goal": f"在 CGA 空间中，根据坐标 ({', '.join(coordinate_parameters)}) 创建共形点 {output}。",
                    "formula": formula,
                    "parameters": parameters,
                    "output": output,
                },
                "variable_assignments": variable_assignments,
                "multivectors_to_be_visualized": _build_visualization_payload(
                    visualization_required,
                    visualization_objects,
                ),
            }
        )

    return tasks, _ordered_unique(missing_assignments), consumed_formulas


def _build_line_tasks(
    user_input: str,
    formulas: dict[str, str],
    consumed_formulas: set[str],
    next_task_id: int,
) -> tuple[list[dict[str, Any]], set[str]]:
    tasks: list[dict[str, Any]] = []

    for match in re.finditer(r"(?:创建|基于|根据|create|construct)[^。；;\n]{0,40}(?:一条)?直线\s*([A-Za-z][A-Za-z0-9_]*)", user_input, flags=re.IGNORECASE):
        output = match.group(1)
        formula = _extract_exact_or_generic_formula(output, formulas, ["L"])
        if output in formulas:
            consumed_formulas.add(output)
        elif "L" in formulas:
            consumed_formulas.add("L")

        parameters = _extract_parameters_from_formula(formula, output)
        point_inputs = [item for item in parameters if re.fullmatch(r"[PX]\d+", item)]
        goal = f"根据 {', '.join(point_inputs)} 创建直线 {output}。" if point_inputs else f"创建直线 {output}。"

        tasks.append(
            {
                "task_id": next_task_id + len(tasks),
                "task_type": "construct_cga_line_from_two_points",
                "code_to_optimize": {
                    "goal": goal,
                    "formula": formula,
                    "parameters": parameters,
                    "output": output,
                },
                "variable_assignments": {},
                "multivectors_to_be_visualized": {
                    "required": False,
                    "objects": [],
                },
            }
        )

    return tasks, consumed_formulas


def _extract_visualize_clause(user_input: str) -> str | None:
    clauses = re.findall(
        rf"([^。；;\n]*?(?:{VISUALIZATION_KEYWORDS})[^。；;\n]*)",
        user_input,
        flags=re.IGNORECASE,
    )
    return clauses[-1] if clauses else None


def _build_visualize_task(
    user_input: str,
    known_outputs: list[str],
    task_id: int,
) -> dict[str, Any] | None:
    clause = _extract_visualize_clause(user_input)
    if not clause:
        return None

    object_names = [name for name in known_outputs if re.search(rf"\b{re.escape(name)}\b", clause)]
    object_names = _ordered_unique(object_names)
    if len(object_names) < 2:
        return None

    objects = [
        {
            "name": name,
            "type": _infer_object_type(name),
            "color": None,
        }
        for name in object_names
    ]

    return {
        "task_id": task_id,
        "task_type": "visualize_objects",
        "code_to_optimize": {
            "goal": f"可视化 {'、'.join(object_names)}。",
            "formula": None,
            "parameters": object_names,
            "output": None,
        },
        "variable_assignments": {},
        "multivectors_to_be_visualized": {
            "required": True,
            "objects": objects,
        },
    }


def _build_task_block_fallback(user_input: str) -> dict[str, Any]:
    formulas = _extract_formula_map(user_input)
    explicit_assignments = _extract_explicit_assignments(user_input)
    warnings: list[str] = []

    point_tasks, _, consumed_formulas = _build_point_tasks(
        user_input,
        formulas,
        explicit_assignments,
    )

    tasks: list[dict[str, Any]] = list(point_tasks)
    line_tasks, consumed_formulas = _build_line_tasks(
        user_input,
        formulas,
        consumed_formulas,
        next_task_id=len(tasks) + 1,
    )
    tasks.extend(line_tasks)

    visualize_task = _build_visualize_task(
        user_input,
        [task["code_to_optimize"]["output"] for task in tasks if task["code_to_optimize"]["output"]],
        task_id=len(tasks) + 1,
    )
    if visualize_task:
        tasks.append(visualize_task)

    for formula_output, formula in formulas.items():
        if formula_output in consumed_formulas:
            continue
        if "^" in formula or "∧" in formula:
            task_type = "compute_outer_product"
            goal = f"根据公式计算外积结果 {formula_output}。"
        elif "I^{-1}" in formula or "^*" in formula or "对偶" in user_input:
            task_type = "compute_dual"
            goal = f"根据公式计算对偶结果 {formula_output}。"
        elif "sqrt" in formula and re.search(r"\bP\b", formula):
            task_type = "decompose_point_pair"
            goal = f"根据公式分解点对并得到 {formula_output}。"
        else:
            task_type = "unknown"
            goal = f"根据公式计算变量 {formula_output}。"
            warnings.append(f"unrecognized task type for formula output {formula_output}")

        parameters = _extract_parameters_from_formula(formula, formula_output)
        tasks.append(
            {
                "task_id": len(tasks) + 1,
                "task_type": task_type,
                "code_to_optimize": {
                    "goal": goal,
                    "formula": formula,
                    "parameters": parameters,
                    "output": formula_output,
                },
                "variable_assignments": {},
                "multivectors_to_be_visualized": {
                    "required": False,
                    "objects": [],
                },
            }
        )

    return _normalize_task_blocks_result({"tasks": tasks})


def _normalize_task_id(task_id: Any, fallback: int) -> int:
    if isinstance(task_id, int):
        return task_id
    if isinstance(task_id, str):
        digits = re.findall(r"\d+", task_id)
        if digits:
            return int(digits[0])
    return fallback


def _normalize_visualization_objects(raw_objects: Any, output_name: str | None = None) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if isinstance(raw_objects, dict):
        raw_objects = [raw_objects]
    if not isinstance(raw_objects, list):
        raw_objects = []

    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or output_name or "").strip()
        if not name:
            continue
        objects.append(
            {
                "name": name,
                "type": str(item.get("type") or _infer_object_type(name)).strip(),
                "color": _normalize_color(item.get("color")) if item.get("color") is not None else None,
            }
        )
    return objects


def _normalize_visualization_payload(raw_payload: Any, output_name: str | None = None) -> dict[str, Any]:
    if isinstance(raw_payload, list):
        objects = _normalize_visualization_objects(raw_payload, output_name=output_name)
        return _build_visualization_payload(bool(objects), objects)

    if isinstance(raw_payload, dict):
        objects = _normalize_visualization_objects(raw_payload.get("objects"), output_name=output_name)
        required = bool(raw_payload.get("required")) or bool(objects)
        return _build_visualization_payload(required, objects)

    return _build_visualization_payload(False, [])


def _apply_default_visualization_color_payload(payload: Any) -> dict[str, Any]:
    normalized_payload = _normalize_visualization_payload(payload)
    required = bool(normalized_payload.get("required"))
    objects = normalized_payload.get("objects") if isinstance(normalized_payload.get("objects"), list) else []

    updated_objects: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        updated_item = deepcopy(item)
        color = updated_item.get("color")
        color_text = str(color).strip() if color is not None else ""
        if color is None or not color_text or color_text.lower() in {"null", "none"}:
            updated_item["color"] = "Red"
        else:
            updated_item["color"] = _normalize_color(color_text) or "Red"
        updated_objects.append(updated_item)

    return {
        "required": required,
        "objects": updated_objects,
    }


def _merge_visualization_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged_objects: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    required = False

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        required = required or bool(payload.get("required"))
        for item in payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("name") or "").strip(),
                str(item.get("type") or "").strip(),
                item.get("color"),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            merged_objects.append(item)

    return _build_visualization_payload(required or bool(merged_objects), merged_objects)


def _normalize_single_task(task: dict[str, Any], fallback_id: int = 1) -> dict[str, Any]:
    code = task.get("code_to_optimize", {})
    if not isinstance(code, dict):
        code = {}

    output_name = code.get("output")
    output_name = str(output_name).strip() if output_name is not None else None

    normalized_task = {
        "task_id": _normalize_task_id(task.get("task_id"), fallback_id),
        "task_type": str(task.get("task_type") or "unknown").strip() or "unknown",
        "code_to_optimize": {
            "goal": code.get("goal"),
            "formula": _clean_formula_text(code.get("formula")) if code.get("formula") is not None else None,
            "parameters": [str(item).strip() for item in code.get("parameters", []) if str(item).strip()]
            if isinstance(code.get("parameters"), list)
            else [],
            "output": output_name,
        },
        "variable_assignments": task.get("variable_assignments", {}) if isinstance(task.get("variable_assignments"), dict) else {},
        "multivectors_to_be_visualized": _normalize_visualization_payload(
            task.get("multivectors_to_be_visualized"),
            output_name=output_name,
        ),
    }
    return normalized_task


def _normalize_task_blocks_result(parsed: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    parsed_warnings = parsed.get("warnings", [])
    if isinstance(parsed_warnings, list):
        warnings.extend(str(item) for item in parsed_warnings if str(item).strip())

    tasks = parsed.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []

    compute_task: dict[str, Any] | None = None
    compute_index = 1
    visualization_payloads: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        task_type = str(task.get("task_type") or "").strip()

        raw_vis = _normalize_visualization_payload(
            task.get("multivectors_to_be_visualized"),
            output_name=task.get("code_to_optimize", {}).get("output") if isinstance(task.get("code_to_optimize"), dict) else None,
        )
        extra_objects = _normalize_visualization_objects(
            task.get("objects"),
            output_name=task.get("code_to_optimize", {}).get("output") if isinstance(task.get("code_to_optimize"), dict) else None,
        )
        if extra_objects:
            raw_vis = _merge_visualization_payloads(raw_vis, _build_visualization_payload(True, extra_objects))

        if task_type == "visualize_objects":
            visualization_payloads.append(raw_vis)
            warnings.append("merged standalone visualize_objects task into the main task")
            continue

        if compute_task is None:
            compute_task = _normalize_single_task(task, fallback_id=index)
            compute_index = index
            visualization_payloads.append(raw_vis)
        else:
            warnings.append("multiple tasks returned by model; kept the first computational task only")
            visualization_payloads.append(raw_vis)

    if compute_task is None:
        return {
            "tasks": [],
        }

    compute_task["task_id"] = 1
    compute_task["multivectors_to_be_visualized"] = _merge_visualization_payloads(
        compute_task.get("multivectors_to_be_visualized", _build_visualization_payload(False, [])),
        *visualization_payloads,
    )

    return {
        "tasks": [compute_task],
    }


def task_block_generator_node(state: dict) -> dict:
    print("--- Node: Task Block Generator ---")

    user_input = state.get("user_input", "")

    hard_fallback = {
        "tasks": [],
    }

    try:
        llm = _get_default_llm()
        prompt = build_task_block_generation_prompt(user_input=user_input)
        print("Invoking LLM for task block generation...")
        result = llm.invoke(prompt)
        raw_content = getattr(result, "content", str(result))
        print(f"Raw task block output: {raw_content}")
        parsed = _parse_json_object(raw_content)
        if parsed:
            normalized_result = _normalize_task_blocks_result(parsed=parsed)
            print("Node result:")
            print(json.dumps({"task_blocks_result": normalized_result}, ensure_ascii=False, indent=2))
            return {
                "task_blocks_result": normalized_result,
            }
    except Exception as exc:
        print(f"Error: task block generation invocation failed - {exc}")

    try:
        heuristic_result = _build_task_block_fallback(user_input=user_input)
        print("Node result:")
        print(json.dumps({"task_blocks_result": heuristic_result}, ensure_ascii=False, indent=2))
        return {
            "task_blocks_result": heuristic_result,
        }
    except Exception as exc:
        print(f"Error: task block fallback generation failed - {exc}")
        print("Node result:")
        print(json.dumps({"task_blocks_result": hard_fallback}, ensure_ascii=False, indent=2))
        return {
            "task_blocks_result": hard_fallback,
        }


def _looks_like_gaalopscript_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return False
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return False
    if stripped.startswith(("Role", "Task", "Explanation", "Output", "Here")):
        return False
    if stripped.startswith(("#", "//", "import ", "def ", "print(")):
        return False
    if stripped.startswith(":") or "_BGColor" in stripped:
        return True
    if "=" in stripped:
        return True
    if re.search(r"\b(createPoint|sqrt|sin|cos|normal)\b", stripped):
        return True
    if re.search(r"[?^.*+\-/]", stripped):
        return True
    return False


def clean_code_to_optimize_code(raw_text: str) -> str:
    cleaned = (raw_text or "").replace("```gaalop", "").replace("```text", "").replace("```", "")
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _looks_like_gaalopscript_line(line):
            continue
        if not line.endswith(";"):
            line = f"{line};"
        lines.append(line)
    return "\n".join(lines)


def compact_code_to_optimize(code: str) -> str:
    if not isinstance(code, str):
        return ""

    compacted_lines: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.endswith(";"):
            line = f"{line};"
        line = re.sub(r"\s*,\s*", ",", line)
        line = re.sub(r"\(\s+", "(", line)
        line = re.sub(r"\s+\)", ")", line)
        line = re.sub(r"([A-Za-z0-9_])\s+\(", r"\1(", line)
        line = re.sub(r"\s*([=+\-*/^\.])\s*", r"\1", line)
        compacted_lines.append(line)

    return "\n".join(compacted_lines)


def validate_code_to_optimize_code(code: str) -> list[str]:
    errors: list[str] = []
    normalized = (code or "").strip()
    if not normalized:
        return ["code is empty"]

    if "```" in normalized:
        errors.append("markdown code fence is not allowed")
    if "{" in normalized or "}" in normalized:
        errors.append("json-like braces are not allowed")
    if "#pragma" in normalized:
        errors.append("pragma is not allowed")
    if re.search(r"\bdef\b|\bimport\b|\bprint\b", normalized):
        errors.append("python syntax is not allowed")
    if "#" in normalized:
        errors.append("comments are not allowed")
    if not re.search(r"\?[A-Za-z_][A-Za-z0-9_]*", normalized):
        errors.append("at least one ? output variable is required")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line in lines:
        if not line.endswith(";"):
            errors.append(f"missing semicolon: {line}")
        if line.startswith(":") or "_BGColor" in line or re.search(r":(?:Red|Blue|Black|Yellow)\b", line):
            errors.append(f"visualization code is not allowed: {line}")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*[-+]?\d+(?:\.\d+)?\s*;", line):
            errors.append(f"variable assignment is not allowed: {line}")
        if re.fullmatch(r"\?(?:a\d+|x\d+)\s*=\s*createPoint\s*\([^)]*\)\s*;", line):
            errors.append(f"invalid output variable naming: {line}")
        if re.search(r"[\u4e00-\u9fff]", line):
            errors.append(f"natural language line is not allowed: {line}")

    return _ordered_unique(errors)


def fallback_code_from_formula(code_to_optimize: dict) -> str:
    if not isinstance(code_to_optimize, dict):
        return ""

    formula = _clean_formula_text(code_to_optimize.get("formula"))
    output = code_to_optimize.get("output")
    output = str(output).strip() if output is not None else ""

    if formula:
        if "=" in formula:
            lhs, rhs = formula.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs:
                return f"?{lhs} = {rhs};"
        if output:
            return f"?{output} = {formula};"

    return ""


def _build_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    error_lines = "\n".join(f"- {item}" for item in errors)
    return (
        f"{base_prompt}\n\n"
        "Previous output validation errors:\n"
        f"{error_lines}\n\n"
        "Fix only the GAALOPScript Code to optimize code. "
        "Output only pure GAALOPScript code."
    )


def _generate_code_to_optimize_code(
    llm: Any,
    task_id: int | str,
    task_type: str,
    code_to_optimize: dict,
) -> tuple[str, list[str]]:
    base_prompt = build_code_to_optimize_agent_prompt(
        rules_text=GAALOPSCRIPT_RULES_TEXT,
        task_id=int(task_id) if isinstance(task_id, int) else task_id,
        task_type=task_type,
        code_to_optimize=code_to_optimize,
    )

    prompt = base_prompt
    last_errors: list[str] = []
    for _ in range(3):
        raw_content = getattr(llm.invoke(prompt), "content", "")
        cleaned_code = clean_code_to_optimize_code(str(raw_content))
        errors = validate_code_to_optimize_code(cleaned_code)
        if not errors:
            return cleaned_code, []
        last_errors = errors
        prompt = _build_retry_prompt(base_prompt, errors)

    raise ValueError(
        "code_to_optimize generation failed: "
        + "; ".join(_ordered_unique(last_errors))
    )


def code_to_optimize_agent_node(state: dict) -> dict:
    task_blocks_result = state.get("task_blocks_result")
    if not isinstance(task_blocks_result, dict):
        raise ValueError("code_to_optimize_agent_node requires task_blocks_result")

    source_tasks = task_blocks_result.get("tasks", [])
    tasks = deepcopy(source_tasks) if isinstance(source_tasks, list) else []
    updated_task_blocks_result = {
        "tasks": tasks,
    }

    llm = _get_default_llm()

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue

        task_id = task.get("task_id", index + 1)
        task_type = str(task.get("task_type") or "unknown")
        code_block = task.get("code_to_optimize", {})
        if not isinstance(code_block, dict):
            code_block = {}

        generated_code, _ = _generate_code_to_optimize_code(
            llm=llm,
            task_id=task_id,
            task_type=task_type,
            code_to_optimize=code_block,
        )

        generated_code = compact_code_to_optimize(generated_code)
        task["code_to_optimize"] = {
            "code": generated_code,
        }

    result = {
        "code_to_optimize_result": updated_task_blocks_result,
    }
    return result


def _looks_like_assignment_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("```", "Role", "Task", "Explanation", "Output", "Here")):
        return False
    if stripped.startswith(("#", "//", "import ", "def ", "print(")):
        return False
    if stripped.startswith("?") or stripped.startswith(":"):
        return True
    if "=" in stripped:
        return True
    return False


def clean_variable_assignments_code(raw_text: str) -> str:
    cleaned = (raw_text or "").replace("```gaalop", "").replace("```text", "").replace("```", "")
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _looks_like_assignment_line(line):
            continue
        if line.startswith("?") or line.startswith(":"):
            continue
        if re.search(r"[\u4e00-\u9fff]", line):
            continue
        if not line.endswith(";"):
            line = f"{line};"
        lines.append(line)
    return "\n".join(lines)


def compact_variable_assignments(code: str) -> str:
    if not isinstance(code, str):
        return ""

    stripped = code.strip()
    if not stripped:
        return ""
    if stripped == "No variable assignments.":
        return stripped

    statements: list[str] = []
    for part in re.split(r";", stripped.replace("\r", "\n")):
        statement = part.strip()
        if not statement:
            continue
        statement = re.sub(r"\s*=\s*", "=", statement)
        statements.append(f"{statement};")

    return " ".join(statements)


def validate_variable_assignments_code(code: str) -> list[str]:
    errors: list[str] = []
    normalized = (code or "").strip()
    if not normalized:
        return []

    if "```" in normalized:
        errors.append("markdown code fence is not allowed")
    if "{" in normalized or "}" in normalized:
        errors.append("json-like braces are not allowed")
    if "#pragma" in normalized:
        errors.append("pragma is not allowed")
    if re.search(r"\bdef\b|\bimport\b|\bprint\b", normalized):
        errors.append("python syntax is not allowed")
    if "#" in normalized:
        errors.append("comments are not allowed")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    assignment_pattern = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?\s*;",
        re.IGNORECASE,
    )
    for line in lines:
        if not line.endswith(";"):
            errors.append(f"missing semicolon: {line}")
        if line.startswith("?"):
            errors.append(f"code-to-optimize line is not allowed: {line}")
        if line.startswith(":") or "_BGColor" in line or re.search(r":(?:Red|Blue|Black|Yellow)\b", line):
            errors.append(f"visualization code is not allowed: {line}")
        if re.search(r"[\u4e00-\u9fff]", line):
            errors.append(f"natural language line is not allowed: {line}")
        if not assignment_pattern.fullmatch(line):
            errors.append(f"invalid assignment pattern: {line}")

    return _ordered_unique(errors)


def _normalize_assignment_numeric_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", stripped, re.IGNORECASE):
            return stripped
    return None


def fallback_assignments_from_dict(variable_assignments: dict) -> tuple[str, list[str]]:
    if not isinstance(variable_assignments, dict):
        return "", ["variable_assignments is not a dict"]

    lines: list[str] = []
    warnings: list[str] = []

    for key, value in variable_assignments.items():
        value_text = _normalize_assignment_numeric_value(value)
        if value is None:
            warnings.append(f"missing assignment value for {key}")
            continue
        if value_text is None:
            warnings.append(f"invalid assignment value for {key}")
            continue
        lines.append(f"{key}={value_text};")

    return "\n".join(lines), _ordered_unique(warnings)


def _build_variable_assignments_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    error_lines = "\n".join(f"- {item}" for item in errors)
    return (
        f"{base_prompt}\n\n"
        "Previous output validation errors:\n"
        f"{error_lines}\n\n"
        "Fix only the Variable assignments code. "
        "Output only pure assignment statements."
    )


def _generate_variable_assignments_code(
    llm: Any,
    task_id: int | str,
    task_type: str,
    variable_assignments: dict,
) -> tuple[str, list[str]]:
    base_prompt = build_variable_assignments_agent_prompt(
        rules_text=VARIABLE_ASSIGNMENTS_RULES_TEXT,
        task_id=int(task_id) if isinstance(task_id, int) else task_id,
        task_type=task_type,
        variable_assignments=variable_assignments,
    )

    prompt = base_prompt
    last_errors: list[str] = []
    has_expected_values = any(_normalize_assignment_numeric_value(value) is not None for value in variable_assignments.values()) \
        if isinstance(variable_assignments, dict) else False

    for _ in range(3):
        raw_content = getattr(llm.invoke(prompt), "content", "")
        cleaned_code = clean_variable_assignments_code(str(raw_content))
        errors = validate_variable_assignments_code(cleaned_code)
        if has_expected_values and not cleaned_code.strip():
            errors.append("assignment code is empty but numeric assignments exist")
        if not errors:
            return cleaned_code, []
        last_errors = _ordered_unique(errors)
        prompt = _build_variable_assignments_retry_prompt(base_prompt, last_errors)

    raise ValueError(
        "variable_assignments generation failed: "
        + "; ".join(_ordered_unique(last_errors))
    )


def variable_assignments_agent_node(state: dict) -> dict:
    source_result = state.get("code_to_optimize_result") or state.get("task_blocks_result")
    if not isinstance(source_result, dict):
        raise ValueError("variable_assignments_agent_node requires source result")

    source_tasks = source_result.get("tasks", [])
    tasks = deepcopy(source_tasks) if isinstance(source_tasks, list) else []
    updated_result = {
        "tasks": tasks,
    }

    llm = _get_default_llm()

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue

        task_id = task.get("task_id", index + 1)
        task_type = str(task.get("task_type") or "unknown")
        variable_assignments_block = task.get("variable_assignments", {})
        if not isinstance(variable_assignments_block, dict):
            variable_assignments_block = {}

        generated_code = ""

        if not variable_assignments_block:
            generated_code = ""
        else:
            generated_code, _ = _generate_variable_assignments_code(
                llm=llm,
                task_id=task_id,
                task_type=task_type,
                variable_assignments=variable_assignments_block,
            )

        generated_code = compact_variable_assignments(generated_code)
        task["variable_assignments"] = {
            "code": generated_code,
        }

    result = {
        "variable_assignments_result": updated_result,
    }
    return result


def _looks_like_visualization_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped == "No need for visualization.":
        return True
    if stripped.startswith(("```", "Role", "Task", "Explanation", "Output", "Here")):
        return False
    if stripped.startswith(("#", "//", "import ", "def ", "print(")):
        return False
    if stripped.startswith(":"):
        return True
    return False


def _normalize_visualization_color_name(value: str) -> str:
    normalized = _normalize_color(value)
    if normalized:
        return normalized
    return value.strip()


def clean_multivectors_to_be_visualized_code(raw_text: str, required: bool) -> str:
    cleaned = (raw_text or "").replace("```gaalop", "").replace("```text", "").replace("```", "").strip()
    if not required:
        if "No need for visualization." in cleaned:
            return "No need for visualization."
        return "No need for visualization."

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _looks_like_visualization_line(line):
            continue
        if line == "No need for visualization.":
            continue
        if line.startswith("?"):
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+;?", line):
            continue
        if not line.startswith(":"):
            continue
        token = line[1:].rstrip(";").strip()
        if not token:
            continue
        if token.lower() in {
            "red", "blue", "green", "black", "yellow", "white", "gray", "grey", "orange", "purple", "cyan", "magenta"
        }:
            token = _normalize_visualization_color_name(token)
        line = f":{token};"
        lines.append(line)
    return "\n".join(lines)


def validate_multivectors_to_be_visualized_code(code: str, required: bool) -> list[str]:
    errors: list[str] = []
    normalized = (code or "").strip()

    if not required:
        if normalized != "No need for visualization.":
            errors.append("visualization-disabled code must be exactly 'No need for visualization.'")
        return errors

    if not normalized:
        return ["visualization code is empty"]
    if normalized == "No need for visualization.":
        return ["visualization is required but code says no visualization"]
    if "```" in normalized:
        errors.append("markdown code fence is not allowed")
    if "{" in normalized or "}" in normalized:
        errors.append("json-like braces are not allowed")
    if "#pragma" in normalized:
        errors.append("pragma is not allowed")
    if re.search(r"\bdef\b|\bimport\b|\bprint\b", normalized):
        errors.append("python syntax is not allowed")
    if "#" in normalized:
        errors.append("comments are not allowed")
    if re.search(r"\b(createPoint|sqrt|sin|cos|normal)\b", normalized):
        errors.append("calculation expressions are not allowed")
    if "^" in normalized or " . " in normalized or "." in normalized.replace("No need for visualization.", ""):
        dot_lines = [line for line in normalized.splitlines() if "." in line and line.strip() != "No need for visualization."]
        if dot_lines:
            errors.append("calculation operators are not allowed")

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    vis_pattern = re.compile(r":[A-Za-z_][A-Za-z0-9_]*;")
    for line in lines:
        if not line.endswith(";"):
            errors.append(f"missing semicolon: {line}")
        if not line.startswith(":"):
            errors.append(f"visualization line must start with colon: {line}")
        if line.startswith("?"):
            errors.append(f"code-to-optimize line is not allowed: {line}")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+;?", line):
            errors.append(f"assignment line is not allowed: {line}")
        if re.search(r"[\u4e00-\u9fff]", line):
            errors.append(f"natural language line is not allowed: {line}")
        if not vis_pattern.fullmatch(line):
            errors.append(f"invalid visualization pattern: {line}")

    return _ordered_unique(errors)


def fallback_visualization_from_objects(multivectors_to_be_visualized: dict) -> tuple[str, list[str]]:
    try:
        return render_visualization_code_from_objects(multivectors_to_be_visualized), []
    except ValueError as exc:
        return "No need for visualization.", [str(exc)]


def render_visualization_code_from_objects(multivectors_to_be_visualized: dict) -> str:
    payload = _apply_default_visualization_color_payload(multivectors_to_be_visualized)
    if not isinstance(payload, dict):
        return "No need for visualization."

    required = bool(payload.get("required", False))
    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        objects = []

    if not required or not objects:
        return "No need for visualization."

    lines: list[str] = []
    valid_name_found = False
    for item in objects:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        valid_name_found = True
        color = item.get("color")
        color_text = str(color).strip() if color is not None else ""
        if color is None or not color_text or color_text.lower() in {"null", "none"}:
            normalized_color = "Red"
        else:
            normalized_color = _normalize_visualization_color_name(color_text)
        lines.append(f":{normalized_color};")
        lines.append(f":{name};")

    if not valid_name_found:
        raise ValueError("Visualization required but no valid object name found.")

    rendered = "\n".join(lines).strip()
    if not rendered:
        raise ValueError("Visualization required but renderer produced empty code.")
    return rendered


def _build_multivectors_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    error_lines = "\n".join(f"- {item}" for item in errors)
    return (
        f"{base_prompt}\n\n"
        "Previous output validation errors:\n"
        f"{error_lines}\n\n"
        "Fix only the Multivectors to be visualized code. "
        "Output only pure visualization statements."
    )


def _generate_multivectors_to_be_visualized_code(
    llm: Any,
    task_id: int | str,
    task_type: str,
    operation: str,
    multivectors_to_be_visualized: dict,
) -> tuple[str, list[str]]:
    vis_block = _apply_default_visualization_color_payload(multivectors_to_be_visualized)
    required = bool(vis_block.get("required")) if isinstance(vis_block, dict) else False
    objects = vis_block.get("objects") if isinstance(vis_block, dict) and isinstance(vis_block.get("objects"), list) else []

    if not required or not objects:
        return "No need for visualization.", []

    try:
        rendered_code = render_visualization_code_from_objects(vis_block)
    except Exception as exc:
        objects_repr = json.dumps(objects, ensure_ascii=False)
        raise ValueError(
            "multivectors_to_be_visualized generation failed: "
            f"visualization code is empty for task_id={task_id}, operation={operation or 'unknown'}, objects={objects_repr}"
        ) from exc

    cleaned_rendered_code = clean_multivectors_to_be_visualized_code(rendered_code, required=True)
    rendered_errors = validate_multivectors_to_be_visualized_code(cleaned_rendered_code, required=True)
    if not rendered_errors and cleaned_rendered_code.strip():
        return cleaned_rendered_code, []

    if llm is None:
        llm = _get_default_llm()

    base_prompt = build_multivectors_to_be_visualized_agent_prompt(
        rules_text=MULTIVECTORS_TO_BE_VISUALIZED_RULES_TEXT,
        task_id=int(task_id) if isinstance(task_id, int) else task_id,
        task_type=task_type,
        multivectors_to_be_visualized=vis_block,
    )

    prompt = base_prompt
    last_errors: list[str] = list(rendered_errors)
    for _ in range(3):
        raw_content = getattr(llm.invoke(prompt), "content", "")
        cleaned_code = clean_multivectors_to_be_visualized_code(str(raw_content), required=required)
        if not cleaned_code.strip() and required and objects:
            cleaned_code = cleaned_rendered_code
        errors = validate_multivectors_to_be_visualized_code(cleaned_code, required=required)
        if not errors:
            return cleaned_code, []
        last_errors = _ordered_unique(errors)
        prompt = _build_multivectors_retry_prompt(base_prompt, last_errors)

    objects_repr = json.dumps(objects, ensure_ascii=False)
    raise ValueError(
        "multivectors_to_be_visualized generation failed: "
        f"{'; '.join(_ordered_unique(last_errors))} for task_id={task_id}, operation={operation or 'unknown'}, objects={objects_repr}"
    )


def multivectors_to_be_visualized_agent_node(state: dict) -> dict:
    source_result = (
        state.get("variable_assignments_result")
        or state.get("code_to_optimize_result")
        or state.get("task_blocks_result")
    )
    if not isinstance(source_result, dict):
        raise ValueError("multivectors_to_be_visualized_agent_node requires source result")

    source_tasks = source_result.get("tasks", [])
    tasks = deepcopy(source_tasks) if isinstance(source_tasks, list) else []
    updated_result = {
        "tasks": tasks,
    }

    llm = None

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue

        task_id = task.get("task_id", index + 1)
        task_type = str(task.get("task_type") or "unknown")
        operation = str(task.get("operation") or "").strip()
        vis_block = task.get("multivectors_to_be_visualized", {})
        if not isinstance(vis_block, dict):
            vis_block = {}
        vis_block = _apply_default_visualization_color_payload(vis_block)

        generated_code, _ = _generate_multivectors_to_be_visualized_code(
            llm=llm,
            task_id=task_id,
            task_type=task_type,
            operation=operation,
            multivectors_to_be_visualized=vis_block,
        )

        task["multivectors_to_be_visualized"] = {
            "code": generated_code,
        }

    result = {
        "multivectors_to_be_visualized_result": updated_result,
    }
    return result
