import ast
import json
import math
import re
from copy import deepcopy
from typing import Any, Callable

from ga_visagent.main_graph.operation_registry import (
    ALIAS_TO_OPERATION,
    get_default_output_type,
    get_operation_spec,
    get_operation_for_task_type,
    get_task_type_for_operation,
    normalize_operation_alias as registry_normalize_operation_alias,
)

OPERATION_ALIASES = dict(ALIAS_TO_OPERATION)

TASK_TYPE_ALIASES = {
    "construct_point": "construct_cga_point",
    "construct_cga_point": "construct_cga_point",
    "construct_line": "construct_cga_line_from_two_points",
    "construct_cga_line": "construct_cga_line_from_two_points",
    "construct_cga_line_from_two_points": "construct_cga_line_from_two_points",
    "construct_vector": "construct_vector",
    "construct_plane": "construct_cga_plane_from_three_points",
    "construct_cga_plane_from_three_points": "construct_cga_plane_from_three_points",
    "plane_from_three_points": "construct_cga_plane_from_three_points",
    "construct_plane_from_point_and_normal": "construct_cga_plane_from_point_and_normal",
    "construct_cga_plane_from_point_and_normal": "construct_cga_plane_from_point_and_normal",
    "plane_from_point_and_normal": "construct_cga_plane_from_point_and_normal",
    "reflect_point": "reflect_cga_point",
    "reflect_cga_point": "reflect_cga_point",
    "reflection": "reflect_cga_point",
    "reflect": "reflect_cga_point",
    "point_reflection": "reflect_cga_point",
    "rotate_object": "rotate_cga_object",
    "rotate_circle": "rotate_cga_object",
    "rotate_line": "rotate_cga_object",
    "rotate_point": "rotate_cga_object",
    "rotate_sphere": "rotate_cga_object",
    "rotation": "rotate_cga_object",
    "rotate_cga_object": "rotate_cga_object",
    "construct_rotor": "construct_rotor",
    "create_rotor": "construct_rotor",
    "rotor": "construct_rotor",
    "build_rotor": "construct_rotor",
    "point_distance": "compute_cga_point_distance",
    "compute_point_distance": "compute_cga_point_distance",
    "compute_cga_point_distance": "compute_cga_point_distance",
    "midpoint": "compute_midpoint",
    "compute_midpoint": "compute_midpoint",
    "middle_point": "compute_midpoint",
    "mid_point": "compute_midpoint",
    "geometric_product": "compute_geometric_product",
    "compute_geometric_product": "compute_geometric_product",
    "outer_product": "compute_outer_product",
    "wedge_product": "compute_outer_product",
    "compute_outer_product": "compute_outer_product",
    "inner_product": "compute_inner_product",
    "dot_product": "compute_inner_product",
    "compute_inner_product": "compute_inner_product",
    "norm": "compute_norm",
    "compute_norm": "compute_norm",
    "dual": "compute_dual",
    "compute_dual": "compute_dual",
    "meet": "compute_meet",
    "intersection": "compute_meet",
    "line_intersection": "compute_meet",
    "compute_meet": "compute_meet",
    "circle_from_three_points": "construct_cga_circle_from_three_points",
    "construct_cga_circle_from_three_points": "construct_cga_circle_from_three_points",
    "construct_sphere": "construct_cga_sphere",
    "construct_cga_sphere": "construct_cga_sphere",
    "point_pair_decomposition": "decompose_cga_point_pair",
    "decompose_point_pair": "decompose_cga_point_pair",
    "point_pair_decompose": "decompose_cga_point_pair",
    "extract_point_pair": "decompose_cga_point_pair",
    "split_point_pair": "decompose_cga_point_pair",
    "point_pair_to_points": "decompose_cga_point_pair",
    "compute_intersection_points": "decompose_cga_point_pair",
    "decompose_cga_point_pair": "decompose_cga_point_pair",
    "unknown": "unknown",
}

TASK_TYPE_TO_OPERATION = {
    "construct_cga_point": "construct_point",
    "construct_cga_line_from_two_points": "line_from_two_points",
    "construct_vector": "construct_vector",
    "construct_cga_plane_from_three_points": "plane_from_three_points",
    "construct_cga_plane_from_point_and_normal": "plane_from_point_and_normal",
    "reflect_cga_point": "reflect_point",
    "rotate_cga_object": "rotate_object",
    "construct_rotor": "construct_rotor",
    "compute_midpoint": "midpoint",
    "compute_geometric_product": "geometric_product",
    "construct_cga_circle_from_three_points": "circle_from_three_points",
    "construct_cga_sphere": "construct_sphere",
    "compute_cga_point_distance": "point_distance",
    "compute_outer_product": "outer_product",
    "compute_inner_product": "inner_product",
    "compute_norm": "norm",
    "compute_dual": "dual",
    "compute_meet": "meet",
    "decompose_cga_point_pair": "point_pair_decomposition",
}


def _parse_task_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def _normalize_color(value: Any) -> str | None:
    if value is None:
        return None
    normalized = {
        "red": "Red",
        "blue": "Blue",
        "black": "Black",
        "yellow": "Yellow",
        "green": "Green",
        "cyan": "Cyan",
    }.get(str(value).strip().lower())
    if normalized:
        return normalized
    text = str(value).strip()
    if not text:
        return None
    return text[:1].upper() + text[1:].lower()


def _normalize_task_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return TASK_TYPE_ALIASES.get(text, text)


def normalize_operation_name(value: Any, task_type: str) -> str:
    text = str(value or "").strip()
    if text:
        return registry_normalize_operation_alias(text)
    return get_operation_for_task_type(task_type) or TASK_TYPE_TO_OPERATION.get(task_type, "unknown")


def _normalize_symbol_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_plane_symbol(value: Any, fallback: str = "Pi") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    normalized = text
    for token in ("\\Pi", "\\pi", "Π", "π", "螤", "蟺"):
        normalized = normalized.replace(token, "Pi")
    if normalized.lower() == "pi":
        return "Pi"
    return normalized


def _normalize_depends_on(value: Any) -> list[int]:
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    else:
        items = [value]
    depends_on: list[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            depends_on.append(item)
            continue
        digits = re.findall(r"\d+", str(item))
        if digits:
            depends_on.append(int(digits[0]))
    return depends_on


def _normalize_coordinate_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return float(text)
    return text


def _normalize_coordinates_list(value: Any) -> list[Any] | None:
    if not isinstance(value, list):
        return None
    coordinates = [_normalize_coordinate_value(item) for item in value[:3]]
    if len(coordinates) != 3 or any(item is None for item in coordinates):
        return None
    return coordinates


def sanitize_symbol_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return "RotatedObject"
    match = re.fullmatch(r"([A-Za-z0-9_]+)[\'’`]+", text)
    if match:
        return f"{match.group(1)}_rotated"
    normalized = text.replace("'", "_rotated").replace("’", "_rotated").replace("`", "_rotated")
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "RotatedObject"


def _default_rotated_output_name(inputs: list[str]) -> str:
    input_name = str(inputs[0]).strip() if inputs else ""
    if not input_name:
        return "RotatedObject"
    base_name = sanitize_symbol_name(input_name)
    if base_name.endswith("_rotated"):
        return base_name
    return f"{base_name}_rotated"


def _infer_angle_unit(angle: Any, angle_unit: Any) -> str:
    explicit = str(angle_unit or "").strip().lower()
    if explicit in {"degree", "degrees", "deg"}:
        return "degree"
    if explicit in {"radian", "radians", "rad"}:
        return "radian"
    text = str(angle or "").strip().lower()
    if any(token in text for token in ("degree", "degrees", "deg", "°")):
        return "degree"
    if "pi" in text or "π" in text:
        return "radian"
    return "radian"


def _evaluate_angle_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_angle_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_angle_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _evaluate_angle_ast(node.left)
        right = _evaluate_angle_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError("unsupported angle expression")


def parse_rotation_angle_to_radians(angle, angle_unit: str | None = None) -> float:
    if isinstance(angle, bool):
        raise ValueError("boolean angle is invalid")
    if isinstance(angle, (int, float)):
        value = float(angle)
        if _infer_angle_unit(angle, angle_unit) == "degree":
            return value * math.pi / 180.0
        return value
    text = str(angle or "").strip()
    if not text:
        raise ValueError("empty angle")
    inferred_unit = _infer_angle_unit(text, angle_unit)
    normalized = text.lower().replace("π", "pi")
    for token in ("degrees", "degree", "deg", "radians", "radian", "rad", "°"):
        normalized = normalized.replace(token, "")
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("empty angle expression")
    if not re.fullmatch(r"[0-9pi\.\+\-\*/\(\)\s]+", normalized):
        raise ValueError("unsupported angle expression")
    try:
        parsed = ast.parse(normalized, mode="eval")
        value = _evaluate_angle_ast(parsed)
    except Exception as exc:
        raise ValueError("unsupported angle expression") from exc
    if inferred_unit == "degree":
        return float(value) * math.pi / 180.0
    return float(value)


def _normalize_rotation_axis(axis: Any) -> list[Any] | Any:
    normalized_axis = _normalize_coordinates_list(axis)
    if normalized_axis is not None:
        return normalized_axis
    if isinstance(axis, str):
        text = str(axis).strip().lower()
        compact = text.replace(" ", "")
        axis_mapping = {
            "x-axis": [1, 0, 0],
            "xaxis": [1, 0, 0],
            "x": [1, 0, 0],
            "y-axis": [0, 1, 0],
            "yaxis": [0, 1, 0],
            "y": [0, 1, 0],
            "z-axis": [0, 0, 1],
            "zaxis": [0, 0, 1],
            "z": [0, 0, 1],
        }
        if compact in axis_mapping:
            return axis_mapping[compact]
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if len(matches) == 3:
            return [_normalize_coordinate_value(item) for item in matches]
    return axis


def _normalize_vector_expression(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(
        r"(?<![\w.])(-?(?:\d+(?:\.\d*)?|\.\d+))\s*(e[123])",
        r"\1*\2",
        text,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _sanitize_symbol_for_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "value"
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", text)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "value"
    if sanitized[0].isdigit():
        sanitized = f"v_{sanitized}"
    return sanitized


def _default_reflect_point_output(inputs: list[str]) -> str:
    if len(inputs) >= 1:
        point_name = str(inputs[0]).strip()
        if point_name:
            return f"{point_name}_reflected"
    return "P_reflected"


def _looks_like_reflect_point_task(
    task_type: str,
    operation: str,
    object_specs: dict[str, Any],
    inputs: list[str],
    outputs: list[str],
) -> bool:
    if operation in {"reflect_point", "reflection", "reflect", "point_reflection"}:
        return True
    if task_type == "reflect_cga_point":
        return True
    if operation != "geometric_product" and task_type != "compute_geometric_product":
        return False
    if len(inputs) != 2:
        return False
    object_type = str(object_specs.get("type") or "").strip().lower()
    object_name = str(object_specs.get("name") or "").strip().lower()
    object_point = str(object_specs.get("point") or "").strip()
    object_mirror = str(object_specs.get("mirror") or "").strip()
    formula_text = str(object_specs.get("formula") or "").strip().lower()
    normalized_outputs = [str(item).strip().lower() for item in outputs if str(item).strip()]
    if object_type == "point":
        return True
    if object_point or object_mirror:
        return True
    if "reflect" in object_name:
        return True
    if any(name.endswith("_reflected") for name in normalized_outputs):
        return True
    if "reflect" in formula_text:
        return True
    if "m v m" in formula_text or formula_text.replace(" ", "") == "mvm":
        return True
    return False


def _extract_point_index(output_symbol: str) -> int | None:
    match = re.fullmatch(r"P(\d+)", str(output_symbol or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _infer_symbol_type_from_task(task: dict[str, Any]) -> str:
    object_specs = task.get("object_specs")
    if isinstance(object_specs, dict):
        object_type = str(object_specs.get("type") or "").strip()
        if object_type:
            return object_type
    operation = registry_normalize_operation_alias(str(task.get("operation") or "").strip())
    return get_default_output_type(operation) or "multivector"


def _normalize_visualization_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized_objects: list[dict[str, Any]] = []
    for obj in value:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        normalized_obj = {
            "name": name,
            "type": str(obj.get("type") or "").strip() or "point",
            "color": _normalize_color(obj.get("color")),
        }
        members = obj.get("members")
        if isinstance(members, list):
            normalized_members = [str(member).strip() for member in members if str(member).strip()]
            if normalized_members:
                normalized_obj["members"] = normalized_members
        normalized_objects.append(normalized_obj)
    return normalized_objects


def apply_default_visualization_color(visualization: Any) -> dict[str, Any]:
    if not isinstance(visualization, dict):
        return {"required": False, "objects": []}
    required = bool(visualization.get("required", False))
    raw_objects = visualization.get("objects")
    if not isinstance(raw_objects, list):
        raw_objects = []
    normalized_objects = _normalize_visualization_objects(raw_objects)
    if "required" not in visualization and normalized_objects:
        required = True
    updated_objects: list[dict[str, Any]] = []
    for obj in normalized_objects:
        updated_obj = deepcopy(obj)
        color = updated_obj.get("color")
        color_text = str(color).strip() if color is not None else ""
        if color is None or not color_text or color_text.lower() in {"null", "none"}:
            updated_obj["color"] = "Red"
        else:
            updated_obj["color"] = _normalize_color(color_text) or "Red"
        updated_objects.append(updated_obj)
    return {
        "required": required,
        "objects": updated_objects,
    }


def normalize_visualization_block(task: dict) -> dict:
    visualization = task.get("visualization")
    if not isinstance(visualization, dict):
        visualization = task.get("multivectors_to_be_visualized")
    normalized_visualization = apply_default_visualization_color(visualization)
    required = bool(normalized_visualization.get("required"))
    normalized_objects = (
        normalized_visualization.get("objects")
        if isinstance(normalized_visualization.get("objects"), list)
        else []
    )
    return {
        "required": required,
        "objects": normalized_objects,
    }


def is_visualization_only_task(task: dict) -> bool:
    if not isinstance(task, dict):
        return False
    task_type = str(task.get("task_type") or "").strip().lower()
    operation = str(task.get("operation") or "").strip().lower()
    visualization_only_names = {
        "visualization",
        "visualize",
        "visualize_objects",
        "point_set_visualization",
    }
    if task_type in visualization_only_names or operation in visualization_only_names:
        return True
    outputs = task.get("outputs")
    has_outputs = isinstance(outputs, list) and any(str(item).strip() for item in outputs)
    if has_outputs:
        return False
    visualization = normalize_visualization_block(task)
    if bool(visualization.get("required")) and visualization.get("objects"):
        return True
    raw_multivectors = task.get("multivectors_to_be_visualized")
    if isinstance(raw_multivectors, dict):
        normalized_multivectors = apply_default_visualization_color(raw_multivectors)
        if bool(normalized_multivectors.get("required")) and normalized_multivectors.get("objects"):
            return True
    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    members = object_specs.get("members")
    if str(object_specs.get("type") or "").strip().lower() == "point_set" and isinstance(members, list) and members:
        return True
    return False


def _extract_visualization_only_objects(task: dict) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    visualization = normalize_visualization_block(task)
    for obj in visualization.get("objects", []):
        if isinstance(obj, dict):
            collected.append(deepcopy(obj))
    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    point_set_type = str(object_specs.get("type") or "").strip().lower()
    members = object_specs.get("members")
    if point_set_type == "point_set" and isinstance(members, list) and members:
        normalized_members = [str(member).strip() for member in members if str(member).strip()]
        if normalized_members:
            collected.append(
                {
                    "name": str(object_specs.get("name") or "PointSet").strip() or "PointSet",
                    "type": "point_set",
                    "members": normalized_members,
                    "color": _normalize_color(object_specs.get("color")) or "Red",
                }
            )
    return collected


def merge_visualization_only_tasks(
    tasks: list[dict[str, Any]],
    visualization_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(tasks, list) or not tasks or not isinstance(visualization_tasks, list) or not visualization_tasks:
        return tasks
    output_to_task: dict[str, dict[str, Any]] = {}
    point_outputs: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
        object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
        object_type = str(object_specs.get("type") or "").strip()
        for output in outputs:
            output_name = str(output).strip()
            if not output_name:
                continue
            output_to_task[output_name] = task
            if object_type == "point":
                point_outputs.append(output_name)

    def attach_visualization(target_name: str, obj: dict[str, Any]) -> None:
        target_task = output_to_task.get(target_name)
        if not isinstance(target_task, dict):
            return
        target_visualization = (
            target_task.get("visualization")
            if isinstance(target_task.get("visualization"), dict)
            else {"required": False, "objects": []}
        )
        target_objects = target_visualization.get("objects") if isinstance(target_visualization.get("objects"), list) else []
        target_object_specs = target_task.get("object_specs") if isinstance(target_task.get("object_specs"), dict) else {}
        incoming_type = (
            str(obj.get("type") or "").strip()
            or str(target_object_specs.get("type") or "").strip()
            or "multivector"
        )
        incoming_color = _normalize_color(obj.get("color")) or "Red"

        existing_object = None
        for existing in target_objects:
            if isinstance(existing, dict) and str(existing.get("name") or "").strip() == target_name:
                existing_object = existing
                break

        if existing_object is None:
            target_objects.append(
                {
                    "name": target_name,
                    "type": incoming_type,
                    "color": incoming_color,
                }
            )
        else:
            existing_type = str(existing_object.get("type") or "").strip()
            if not existing_type:
                existing_object["type"] = incoming_type
            existing_color = str(existing_object.get("color") or "").strip()
            if not existing_color or existing_color.lower() in {"null", "none"}:
                existing_object["color"] = incoming_color

        target_task["visualization"] = {
            "required": True,
            "objects": target_objects,
        }

    for visualization_task in visualization_tasks:
        if not isinstance(visualization_task, dict):
            continue
        for obj in _extract_visualization_only_objects(visualization_task):
            if not isinstance(obj, dict):
                continue
            object_type = str(obj.get("type") or "").strip().lower()
            color = _normalize_color(obj.get("color")) or "Red"
            members = obj.get("members") if isinstance(obj.get("members"), list) else []
            if object_type == "point_set":
                expanded_members = [str(member).strip() for member in members if str(member).strip()]
                if not expanded_members:
                    expanded_members = point_outputs
                for member in expanded_members:
                    attach_visualization(
                        member,
                        {
                            "name": member,
                            "type": "point",
                            "color": color,
                        },
                    )
                continue
            target_name = str(obj.get("name") or "").strip()
            if target_name:
                attach_visualization(target_name, obj)
    return tasks


def _infer_output_type_from_prior_tasks(tasks: list[dict[str, Any]], symbol: Any) -> str:
    target_symbol = str(symbol or "").strip()
    if not target_symbol:
        return ""
    for task in reversed(tasks):
        if not isinstance(task, dict):
            continue
        outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
        normalized_outputs = [str(output).strip() for output in outputs if str(output).strip()]
        if target_symbol not in normalized_outputs:
            continue
        object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
        object_type = str(object_specs.get("type") or "").strip()
        if object_type == "point_pair_decomposition":
            return "point"
        if object_type:
            return object_type
        return _infer_symbol_type_from_task(task)
    return ""


def _looks_like_point_pair_output_symbol(name: Any) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if re.fullmatch(r"X\d+", text):
        return True
    lower_text = text.lower()
    return (
        lower_text in {"p_plus", "p_minus", "x_plus", "x_minus"}
        or "plus" in lower_text
        or "minus" in lower_text
        or "pm" in lower_text
    )


def _looks_like_computed_point_pair_construct_point(task: dict) -> bool:
    if not isinstance(task, dict):
        return False
    task_type = str(task.get("task_type") or "").strip()
    operation = str(task.get("operation") or "").strip()
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if task_type != "construct_cga_point" and operation != "construct_point":
        return False
    if not inputs or not outputs:
        return False
    if not any(_looks_like_point_pair_output_symbol(output) for output in outputs):
        return False
    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    serialized = json.dumps(
        {
            "inputs": inputs,
            "outputs": outputs,
            "object_specs": object_specs,
        },
        ensure_ascii=False,
    ).lower()
    if any(marker in serialized for marker in ("point pair", "sqrt", "denom", "einf", "p_pm", "p±", "intersection")):
        return True
    return any(str(symbol).strip().lower() in {"p", "pp", "point_pair"} for symbol in inputs)


def _merge_computed_point_pair_construct_point_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(tasks, list) or not tasks:
        return tasks
    grouped_indices: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        if not _looks_like_computed_point_pair_construct_point(task):
            continue
        inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
        object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
        point_pair_symbol = str(object_specs.get("point_pair") or (inputs[0] if inputs else "")).strip()
        if not point_pair_symbol:
            continue
        grouped_indices.setdefault(point_pair_symbol, []).append(index)

    if not grouped_indices:
        return tasks

    merged_tasks: list[dict[str, Any]] = []
    consumed_indices: set[int] = set()

    for index, task in enumerate(tasks):
        if index in consumed_indices:
            continue
        if not _looks_like_computed_point_pair_construct_point(task):
            merged_tasks.append(task)
            continue
        inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
        object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
        point_pair_symbol = str(object_specs.get("point_pair") or (inputs[0] if inputs else "")).strip()
        related_indices = grouped_indices.get(point_pair_symbol, [])
        available_indices = [item_index for item_index in related_indices if item_index not in consumed_indices]
        related_tasks = [tasks[item_index] for item_index in available_indices]

        if len(related_tasks) < 2:
            merged_tasks.append(task)
            continue

        combined_outputs: list[str] = []
        combined_depends_on: list[Any] = []
        combined_visualization_objects: list[dict[str, Any]] = []
        for related_index, related_task in zip(available_indices, related_tasks):
            consumed_indices.add(related_index)
            related_outputs = related_task.get("outputs") if isinstance(related_task.get("outputs"), list) else []
            for output in related_outputs:
                output_name = str(output).strip()
                if output_name and output_name not in combined_outputs:
                    combined_outputs.append(output_name)
            related_depends_on = related_task.get("depends_on") if isinstance(related_task.get("depends_on"), list) else []
            for dep in related_depends_on:
                if dep not in combined_depends_on:
                    combined_depends_on.append(dep)
            related_visualization = normalize_visualization_block(related_task)
            for obj in related_visualization.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                obj_name = str(obj.get("name") or "").strip()
                if not obj_name:
                    continue
                if obj_name not in {str(existing.get("name") or "").strip() for existing in combined_visualization_objects}:
                    combined_visualization_objects.append(deepcopy(obj))

        if len(combined_outputs) < 2:
            combined_outputs = ["X4", "X5"]

        merged_task = deepcopy(task)
        merged_task["task_type"] = "decompose_cga_point_pair"
        merged_task["operation"] = "point_pair_decomposition"
        merged_task["inputs"] = [point_pair_symbol] if point_pair_symbol else []
        merged_task["outputs"] = combined_outputs[:2]
        merged_task["depends_on"] = combined_depends_on
        merged_task["object_specs"] = {
            "name": "point_pair_decomposition",
            "type": "point_pair_decomposition",
            "point_pair": point_pair_symbol,
            "formula": "X_pm = (P ± sqrt(P.P)) / (einf.P)",
        }
        merged_task["visualization"] = {
            "required": bool(combined_visualization_objects),
            "objects": combined_visualization_objects if combined_visualization_objects else [
                {"name": combined_outputs[0], "type": "point", "color": "Yellow"},
                {"name": combined_outputs[1], "type": "point", "color": "Yellow"},
            ],
        }
        merged_tasks.append(merged_task)

    return merged_tasks


def normalize_common_task_fields(task: dict, index: int) -> dict:
    if not isinstance(task, dict):
        raise ValueError(f"task {index} is not a dict")

    task_id = _parse_task_id(task.get("task_id"))
    if task_id is None:
        task_id = index

    task_type = _normalize_task_type(task.get("task_type"))
    operation = normalize_operation_name(task.get("operation"), task_type)
    if not operation or operation == "unknown":
        raise ValueError(f"task {index} missing operation; task={task}")
    if task_type == "unknown":
        task_type = get_task_type_for_operation(operation) or task_type

    raw_object_specs = task.get("object_specs")
    if raw_object_specs is None:
        raw_object_specs = {}
    if not isinstance(raw_object_specs, dict):
        raw_object_specs = {}
    object_specs = deepcopy(raw_object_specs)

    inputs = _normalize_symbol_list(task.get("inputs"))
    outputs = _normalize_symbol_list(task.get("outputs"))
    if _looks_like_reflect_point_task(task_type, operation, object_specs, inputs, outputs):
        task_type = "reflect_cga_point"
        operation = "reflect_point"

    if operation == "construct_point" and not outputs:
        inferred_output = str(object_specs.get("name") or "").strip()
        if inferred_output:
            outputs = [inferred_output]
    if operation == "line_from_two_points" and not outputs:
        outputs = [str(object_specs.get("name") or "L").strip() or "L"]
    if operation == "point_distance" and not outputs:
        outputs = ["d2"]
    if operation == "midpoint" and not outputs:
        inferred_output = str(object_specs.get("name") or "M").strip() or "M"
        outputs = [inferred_output]
    if operation == "construct_vector" and not outputs:
        inferred_output = str(object_specs.get("name") or "").strip()
        if inferred_output:
            outputs = [inferred_output]
    if operation == "reflect_point" and not outputs:
        outputs = [_default_reflect_point_output(inputs)]
    if operation == "geometric_product" and not outputs:
        outputs = ["G"]
    if operation == "outer_product" and not outputs:
        outputs = ["M"]
    if operation == "inner_product" and not outputs:
        outputs = ["IP"]
    if operation == "norm" and not outputs:
        if len(inputs) == 1 and str(inputs[0]).strip():
            outputs = [f"Norm{str(inputs[0]).strip()}"]
        else:
            outputs = ["Norm"]
    if operation == "dual" and not outputs:
        if len(inputs) == 1 and str(inputs[0]).strip():
            outputs = [f"Dual{str(inputs[0]).strip()}"]
        else:
            outputs = ["DualResult"]
    if operation == "construct_rotor" and not outputs:
        inferred_output = str(object_specs.get("name") or "R").strip() or "R"
        outputs = [_sanitize_symbol_for_identifier(inferred_output)]
    if operation == "point_pair_decomposition" and not outputs:
        outputs = ["X4", "X5"]
    if operation == "meet" and not outputs:
        outputs = ["I"]
    if operation == "rotate_object":
        object_symbol = str(object_specs.get("object") or "").strip()
        rotor_symbol = str(object_specs.get("rotor") or "").strip()
        if not inputs and object_symbol and rotor_symbol:
            inputs = [object_symbol, rotor_symbol]
        elif not inputs and object_symbol:
            inputs = [object_symbol]
        if not outputs:
            outputs = [_default_rotated_output_name([object_symbol or (inputs[0] if inputs else "")])]
    if operation == "construct_sphere" and not outputs:
        inferred_output = str(object_specs.get("name") or "S").strip() or "S"
        outputs = [inferred_output]
    if (
        (
            operation == "plane_from_point_and_normal"
            or task_type == "construct_cga_plane_from_point_and_normal"
            or (
                task_type == "construct_cga_plane_from_three_points"
                and not inputs
                and isinstance(object_specs.get("point"), list)
                and isinstance(object_specs.get("normal"), list)
            )
        )
        and not outputs
    ):
        inferred_output = _normalize_plane_symbol(object_specs.get("name") or "Pi", fallback="Pi")
        outputs = [inferred_output]

    if not outputs:
        raise ValueError(
            f"task {index} missing outputs; task_type={task.get('task_type')}, "
            f"operation={task.get('operation')}, task={task}"
        )

    depends_on = _normalize_depends_on(task.get("depends_on"))
    visualization = normalize_visualization_block(task)
    normalized_task = {
        "task_id": task_id,
        "task_type": task_type,
        "operation": operation,
        "inputs": inputs,
        "outputs": outputs,
        "depends_on": depends_on,
        "object_specs": object_specs,
        "visualization": visualization,
    }

    if (
        task_type == "construct_cga_plane_from_three_points"
        and not inputs
        and isinstance(object_specs.get("point"), list)
        and isinstance(object_specs.get("normal"), list)
    ):
        normalized_task["task_type"] = "construct_cga_plane_from_point_and_normal"
        normalized_task["operation"] = "plane_from_point_and_normal"

    return normalized_task


def normalize_construct_point_task(
    normalized_task: dict[str, Any],
    raw_task: dict[str, Any],
    normalized_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    output_symbol = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    point_index = _extract_point_index(output_symbol)
    if point_index is None:
        return normalized_task
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    coordinates = _normalize_coordinates_list(object_specs.get("coordinates"))
    normalized_task["task_type"] = "construct_cga_point"
    normalized_task["operation"] = "construct_point"
    normalized_task["inputs"] = []
    normalized_task["outputs"] = [output_symbol]
    normalized_task["depends_on"] = []
    normalized_task["object_specs"] = {
        "name": output_symbol,
        "type": "point",
        "coordinates": coordinates if coordinates is not None else [],
    }
    return normalized_task


def normalize_construct_vector_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    normalized_task["task_type"] = "construct_vector"
    normalized_task["operation"] = "construct_vector"
    vector_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    vector_output = vector_output or str(object_specs.get("name") or "").strip()
    normalized_task["outputs"] = [vector_output] if vector_output else []
    normalized_task["inputs"] = []
    normalized_task["depends_on"] = []
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or vector_output).strip() or vector_output,
        "type": str(object_specs.get("type") or "vector").strip() or "vector",
        "expression": _normalize_vector_expression(object_specs.get("expression")),
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_geometric_product_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_geometric_product"
    normalized_task["operation"] = "geometric_product"
    gp_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    gp_output = gp_output or str(object_specs.get("name") or "G").strip() or "G"
    normalized_task["outputs"] = [gp_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 2 else [])
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or gp_output).strip() or gp_output,
        "type": str(object_specs.get("type") or "multivector").strip() or "multivector",
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or "*").strip() or "*",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_midpoint_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_midpoint"
    normalized_task["operation"] = "midpoint"
    midpoint_from = _normalize_symbol_list(object_specs.get("from"))
    if not inputs and len(midpoint_from) == 2:
        normalized_task["inputs"] = midpoint_from
        inputs = midpoint_from
    midpoint_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    midpoint_output = midpoint_output or str(object_specs.get("name") or "M").strip() or "M"
    normalized_task["outputs"] = [midpoint_output]
    midpoint_left = inputs[0] if len(inputs) >= 1 else "P1"
    midpoint_right = inputs[1] if len(inputs) >= 2 else "P2"
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or midpoint_output).strip() or midpoint_output,
        "type": str(object_specs.get("type") or "point").strip() or "point",
        "from": midpoint_from or (inputs if len(inputs) == 2 else []),
        "formula": (
            str(object_specs.get("formula") or f"{midpoint_output} = ({midpoint_left} + {midpoint_right}) / 2").strip()
            or f"{midpoint_output} = ({midpoint_left} + {midpoint_right}) / 2"
        ),
    }
    midpoint_visualization = normalize_visualization_block(raw_task)
    midpoint_required = bool(midpoint_visualization.get("required"))
    midpoint_objects = midpoint_visualization.get("objects") if isinstance(midpoint_visualization.get("objects"), list) else []
    if midpoint_required and not midpoint_objects and midpoint_output:
        midpoint_objects = [{"name": midpoint_output, "type": "point", "color": "Red"}]
    normalized_task["visualization"] = {
        "required": midpoint_required,
        "objects": midpoint_objects,
    }
    return normalized_task


def normalize_outer_product_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_outer_product"
    normalized_task["operation"] = "outer_product"
    outer_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    outer_output = outer_output or str(object_specs.get("name") or "M").strip() or "M"
    normalized_task["outputs"] = [outer_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) >= 2 else [])
    outer_type = str(object_specs.get("type") or "").strip() or "multivector"
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or outer_output).strip() or outer_output,
        "type": outer_type,
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or "^").strip() or "^",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_inner_product_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_inner_product"
    normalized_task["operation"] = "inner_product"
    inner_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    inner_output = inner_output or str(object_specs.get("name") or "IP").strip() or "IP"
    normalized_task["outputs"] = [inner_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 2 else [])
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or inner_output).strip() or inner_output,
        "type": str(object_specs.get("type") or "scalar").strip() or "scalar",
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or ".").strip() or ".",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_norm_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_norm"
    normalized_task["operation"] = "norm"
    norm_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    if not norm_output:
        norm_output = f"Norm{str(inputs[0]).strip()}" if len(inputs) == 1 and str(inputs[0]).strip() else "Norm"
    normalized_task["outputs"] = [norm_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 1 else [])
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or norm_output).strip() or norm_output,
        "type": str(object_specs.get("type") or "scalar").strip() or "scalar",
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or "sqrt_dot").strip() or "sqrt_dot",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_dual_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_dual"
    normalized_task["operation"] = "dual"
    dual_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    if not dual_output:
        dual_output = f"Dual{str(inputs[0]).strip()}" if len(inputs) == 1 and str(inputs[0]).strip() else "DualResult"
    normalized_task["outputs"] = [dual_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 1 else [])
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or dual_output).strip() or dual_output,
        "type": str(object_specs.get("type") or "multivector").strip() or "multivector",
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or "*").strip() or "*",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_point_pair_decomposition_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "decompose_cga_point_pair"
    normalized_task["operation"] = "point_pair_decomposition"
    point_pair_symbol = str(object_specs.get("point_pair") or "").strip()
    if not point_pair_symbol and inputs:
        point_pair_symbol = str(inputs[0]).strip()
    if not normalized_task["inputs"] and point_pair_symbol:
        normalized_task["inputs"] = [point_pair_symbol]

    pair_outputs = [str(symbol).strip() for symbol in normalized_task["outputs"] if str(symbol).strip()]
    pair_visualization = normalize_visualization_block(raw_task)
    if len(pair_outputs) < 2:
        pair_visualization_names = [
            str(obj.get("name") or "").strip()
            for obj in pair_visualization.get("objects", [])
            if isinstance(obj, dict) and str(obj.get("name") or "").strip()
        ]
        for name in pair_visualization_names:
            if name not in pair_outputs:
                pair_outputs.append(name)
    if len(pair_outputs) < 2:
        pair_outputs = ["X4", "X5"]
    normalized_task["outputs"] = pair_outputs[:2]
    normalized_task["object_specs"] = {
        "name": "point_pair_decomposition",
        "type": "point_pair_decomposition",
        "point_pair": point_pair_symbol,
        "formula": str(object_specs.get("formula") or "X_pm = (P ± sqrt(P.P)) / (einf.P)").strip()
        or "X_pm = (P ± sqrt(P.P)) / (einf.P)",
    }
    pair_visualization_required = bool(pair_visualization.get("required"))
    pair_visualization_objects = (
        pair_visualization.get("objects")
        if isinstance(pair_visualization.get("objects"), list)
        else []
    )
    if pair_visualization_required and not pair_visualization_objects:
        pair_visualization_objects = [
            {"name": normalized_task["outputs"][0], "type": "point", "color": "Yellow"},
            {"name": normalized_task["outputs"][1], "type": "point", "color": "Yellow"},
        ]
    normalized_task["visualization"] = {
        "required": pair_visualization_required,
        "objects": pair_visualization_objects,
    }
    return normalized_task


def normalize_meet_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_meet"
    normalized_task["operation"] = "meet"
    meet_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    meet_output = meet_output or str(object_specs.get("name") or "I").strip() or "I"
    normalized_task["outputs"] = [meet_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 2 else [])
    meet_type = str(object_specs.get("type") or "").strip() or "multivector"
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or meet_output).strip() or meet_output,
        "type": meet_type,
        "from": normalized_from,
        "operator": str(object_specs.get("operator") or "meet").strip() or "meet",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_reflect_point_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "reflect_cga_point"
    normalized_task["operation"] = "reflect_point"
    reflect_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    reflect_output = (
        reflect_output
        or str(object_specs.get("name") or _default_reflect_point_output(inputs)).strip()
        or _default_reflect_point_output(inputs)
    )
    normalized_task["outputs"] = [reflect_output]
    point_symbol = str(object_specs.get("point") or "").strip()
    mirror_symbol = str(object_specs.get("mirror") or "").strip()
    if not point_symbol and len(inputs) >= 1:
        point_symbol = str(inputs[0]).strip()
    if not mirror_symbol and len(inputs) >= 2:
        mirror_symbol = str(inputs[1]).strip()
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or reflect_output).strip() or reflect_output,
        "type": str(object_specs.get("type") or "point").strip() or "point",
        "point": point_symbol,
        "mirror": mirror_symbol,
        "formula": str(object_specs.get("formula") or "M v M").strip() or "M v M",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_rotate_object_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "rotate_cga_object"
    normalized_task["operation"] = "rotate_object"
    object_symbol = str(object_specs.get("object") or "").strip()
    if not object_symbol and inputs:
        object_symbol = str(inputs[0]).strip()
    rotor_symbol = str(object_specs.get("rotor") or "").strip()
    if not rotor_symbol and len(inputs) >= 2:
        rotor_symbol = str(inputs[1]).strip()
    if not normalized_task["inputs"]:
        if object_symbol and rotor_symbol:
            normalized_task["inputs"] = [object_symbol, rotor_symbol]
        elif object_symbol:
            normalized_task["inputs"] = [object_symbol]
    elif len(normalized_task["inputs"]) >= 2:
        normalized_task["inputs"] = [
            str(normalized_task["inputs"][0]).strip(),
            str(normalized_task["inputs"][1]).strip(),
        ]
        if not object_symbol:
            object_symbol = normalized_task["inputs"][0]
        if not rotor_symbol:
            rotor_symbol = normalized_task["inputs"][1]
    elif len(normalized_task["inputs"]) == 1:
        first_input_symbol = str(normalized_task["inputs"][0]).strip()
        if not object_symbol:
            object_symbol = first_input_symbol
        if rotor_symbol:
            normalized_task["inputs"] = [object_symbol, rotor_symbol]
        elif object_symbol:
            normalized_task["inputs"] = [object_symbol]

    rotate_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    if not rotate_output:
        rotate_output = str(object_specs.get("name") or "").strip()
    raw_rotate_output = rotate_output
    rotate_output = (
        sanitize_symbol_name(rotate_output)
        if rotate_output
        else _default_rotated_output_name(
            [object_symbol or (normalized_task["inputs"][0] if normalized_task["inputs"] else "")]
        )
    )
    normalized_task["outputs"] = [rotate_output]

    inferred_type = str(object_specs.get("type") or "").strip()
    original_operation = str(raw_task.get("operation") or "").strip()
    alias_type_mapping = {
        "rotate_circle": "circle",
        "rotate_line": "line",
        "rotate_point": "point",
        "rotate_sphere": "sphere",
    }
    if not inferred_type:
        inferred_type = (
            _infer_output_type_from_prior_tasks(normalized_tasks, object_symbol)
            or alias_type_mapping.get(original_operation, "multivector")
        )

    normalized_task["object_specs"] = {
        "name": sanitize_symbol_name(str(object_specs.get("name") or rotate_output).strip() or rotate_output),
        "type": inferred_type or "multivector",
        "object": object_symbol,
    }
    if rotor_symbol:
        normalized_task["object_specs"]["rotor"] = rotor_symbol
    if "axis" in object_specs or not rotor_symbol:
        normalized_task["object_specs"]["axis"] = _normalize_rotation_axis(object_specs.get("axis"))
    if "angle" in object_specs or not rotor_symbol:
        normalized_task["object_specs"]["angle"] = object_specs.get("angle")
        normalized_task["object_specs"]["angle_unit"] = _infer_angle_unit(
            object_specs.get("angle"),
            object_specs.get("angle_unit"),
        )
    if "axis_point" in object_specs:
        normalized_task["object_specs"]["axis_point"] = object_specs.get("axis_point")

    rotate_visualization = normalize_visualization_block(raw_task)
    rotate_objects: list[dict[str, Any]] = []
    for obj in rotate_visualization.get("objects", []):
        if not isinstance(obj, dict):
            continue
        updated_obj = deepcopy(obj)
        object_name = str(updated_obj.get("name") or "").strip()
        if object_name:
            sanitized_name = sanitize_symbol_name(object_name)
            if object_name == raw_rotate_output or sanitized_name == rotate_output:
                updated_obj["name"] = rotate_output
        rotate_objects.append(updated_obj)
    normalized_task["visualization"] = {
        "required": bool(rotate_visualization.get("required")),
        "objects": rotate_objects,
    }
    return normalized_task


def normalize_construct_rotor_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    normalized_task["task_type"] = "construct_rotor"
    normalized_task["operation"] = "construct_rotor"
    raw_rotor_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    rotor_output = _sanitize_symbol_for_identifier(
        raw_rotor_output or str(object_specs.get("name") or "R").strip() or "R"
    )
    normalized_task["outputs"] = [rotor_output]
    normalized_task["inputs"] = []
    normalized_task["depends_on"] = []
    normalized_task["object_specs"] = {
        "name": _sanitize_symbol_for_identifier(str(object_specs.get("name") or rotor_output).strip() or rotor_output),
        "type": str(object_specs.get("type") or "rotor").strip() or "rotor",
        "axis": _normalize_rotation_axis(object_specs.get("axis")),
        "angle": object_specs.get("angle"),
        "angle_unit": _infer_angle_unit(object_specs.get("angle"), object_specs.get("angle_unit")),
    }
    rotor_visualization = normalize_visualization_block(raw_task)
    rotor_objects: list[dict[str, Any]] = []
    for obj in rotor_visualization.get("objects", []):
        if not isinstance(obj, dict):
            continue
        updated_obj = deepcopy(obj)
        object_name = str(updated_obj.get("name") or "").strip()
        if object_name == raw_rotor_output:
            updated_obj["name"] = rotor_output
        rotor_objects.append(updated_obj)
    normalized_task["visualization"] = {
        "required": bool(rotor_visualization.get("required")),
        "objects": rotor_objects,
    }
    return normalized_task


def normalize_line_from_two_points_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    output_symbol = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    normalized_task["task_type"] = "construct_cga_line_from_two_points"
    normalized_task["operation"] = "line_from_two_points"
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or output_symbol).strip() or output_symbol,
        "type": str(object_specs.get("type") or "line").strip() or "line",
        "from": _normalize_symbol_list(object_specs.get("from")) or inputs[:2],
    }
    return normalized_task


def normalize_point_distance_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "compute_cga_point_distance"
    normalized_task["operation"] = "point_distance"
    normalized_task["outputs"] = normalized_task["outputs"] or ["d2"]
    distance_output = str(normalized_task["outputs"][0]).strip() or "d2"
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or distance_output).strip() or distance_output,
        "type": str(object_specs.get("type") or "scalar").strip() or "scalar",
        "from": _normalize_symbol_list(object_specs.get("from")) or inputs,
        "quantity": str(object_specs.get("quantity") or "squared_distance").strip() or "squared_distance",
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_circle_from_three_points_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "construct_cga_circle_from_three_points"
    normalized_task["operation"] = "circle_from_three_points"
    circle_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    circle_output = circle_output or str(object_specs.get("name") or "C").strip() or "C"
    normalized_task["outputs"] = [circle_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 3 else [])
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or circle_output).strip() or circle_output,
        "type": str(object_specs.get("type") or "circle").strip() or "circle",
        "from": normalized_from,
    }
    if "center" in object_specs:
        normalized_task["object_specs"]["center"] = _normalize_coordinates_list(object_specs.get("center")) or object_specs.get("center")
    if "radius" in object_specs:
        normalized_task["object_specs"]["radius"] = _normalize_coordinate_value(object_specs.get("radius"))
    if "plane" in object_specs:
        normalized_task["object_specs"]["plane"] = str(object_specs.get("plane") or "").strip()
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def normalize_plane_from_three_points_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    inputs = normalized_task.get("inputs") if isinstance(normalized_task.get("inputs"), list) else []
    normalized_task["task_type"] = "construct_cga_plane_from_three_points"
    normalized_task["operation"] = "plane_from_three_points"
    plane_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    plane_output = _normalize_plane_symbol(plane_output or object_specs.get("name") or "Pi", fallback="Pi")
    normalized_task["outputs"] = [plane_output]
    normalized_from = _normalize_symbol_list(object_specs.get("from")) or (inputs if len(inputs) == 3 else [])
    normalized_task["object_specs"] = {
        "name": _normalize_plane_symbol(object_specs.get("name") or plane_output, fallback=plane_output),
        "type": str(object_specs.get("type") or "plane").strip() or "plane",
        "from": normalized_from,
    }
    plane_visualization = normalize_visualization_block(raw_task)
    plane_objects = []
    for obj in plane_visualization.get("objects", []):
        if isinstance(obj, dict):
            updated_obj = deepcopy(obj)
            if str(updated_obj.get("type") or "").strip() == "plane":
                updated_obj["name"] = _normalize_plane_symbol(updated_obj.get("name") or plane_output, fallback=plane_output)
            plane_objects.append(updated_obj)
    normalized_task["visualization"] = {
        "required": bool(plane_visualization.get("required")),
        "objects": plane_objects,
    }
    return normalized_task


def normalize_plane_from_point_and_normal_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    normalized_task["task_type"] = "construct_cga_plane_from_point_and_normal"
    normalized_task["operation"] = "plane_from_point_and_normal"
    plane_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    plane_output = _normalize_plane_symbol(plane_output or object_specs.get("name") or "Pi", fallback="Pi")
    normalized_task["outputs"] = [plane_output]

    point = object_specs.get("point")
    if point is None:
        for key in ("point_on_plane", "through_point", "passing_point", "pointOnPlane", "origin"):
            if key in object_specs:
                point = object_specs.get(key)
                break
    normal = object_specs.get("normal")
    if normal is None:
        for key in ("normal_vector", "normalVec", "n", "direction_normal"):
            if key in object_specs:
                normal = object_specs.get(key)
                break

    normalized_point = _normalize_coordinates_list(point)
    normalized_normal = _normalize_coordinates_list(normal)
    normalized_task["object_specs"] = {
        "name": _normalize_plane_symbol(object_specs.get("name") or plane_output, fallback=plane_output),
        "type": str(object_specs.get("type") or "plane").strip() or "plane",
        "point": normalized_point if normalized_point is not None else point,
        "normal": normalized_normal if normalized_normal is not None else normal,
    }
    plane_visualization = normalize_visualization_block(raw_task)
    plane_objects = []
    for obj in plane_visualization.get("objects", []):
        if isinstance(obj, dict):
            updated_obj = deepcopy(obj)
            if str(updated_obj.get("name") or "").strip() in {"Π", "π", "\\Pi", "\\pi", "Pi"}:
                updated_obj["name"] = plane_output
            if str(updated_obj.get("type") or "").strip() == "plane":
                updated_obj["name"] = plane_output
            plane_objects.append(updated_obj)
    normalized_task["visualization"] = {
        "required": bool(plane_visualization.get("required")),
        "objects": plane_objects,
    }
    return normalized_task


def normalize_construct_sphere_task(normalized_task, raw_task, normalized_tasks):
    object_specs = normalized_task.get("object_specs") if isinstance(normalized_task.get("object_specs"), dict) else {}
    normalized_task["task_type"] = "construct_cga_sphere"
    normalized_task["operation"] = "construct_sphere"
    sphere_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
    sphere_output = sphere_output or str(object_specs.get("name") or "S").strip() or "S"
    normalized_task["outputs"] = [sphere_output]
    raw_center = object_specs.get("center")
    center = _normalize_coordinates_list(raw_center)
    center_symbol = str(raw_center or "").strip() if isinstance(raw_center, str) else ""
    if center_symbol and not normalized_task["inputs"]:
        normalized_task["inputs"] = [center_symbol]
    radius = _normalize_coordinate_value(object_specs.get("radius"))
    normalized_task["object_specs"] = {
        "name": str(object_specs.get("name") or sphere_output).strip() or sphere_output,
        "type": str(object_specs.get("type") or "sphere").strip() or "sphere",
        "center": center if center is not None else (center_symbol or raw_center),
        "radius": radius,
    }
    normalized_task["visualization"] = normalize_visualization_block(raw_task)
    return normalized_task


def default_normalizer(normalized_task, raw_task, normalized_tasks):
    return normalized_task


OPERATION_NORMALIZERS: dict[str, Callable[[dict[str, Any], dict[str, Any], list[dict[str, Any]]], dict[str, Any]]] = {
    "construct_point": normalize_construct_point_task,
    "line_from_two_points": normalize_line_from_two_points_task,
    "point_distance": normalize_point_distance_task,
    "construct_sphere": normalize_construct_sphere_task,
    "circle_from_three_points": normalize_circle_from_three_points_task,
    "plane_from_three_points": normalize_plane_from_three_points_task,
    "plane_from_point_and_normal": normalize_plane_from_point_and_normal_task,
    "construct_vector": normalize_construct_vector_task,
    "geometric_product": normalize_geometric_product_task,
    "outer_product": normalize_outer_product_task,
    "inner_product": normalize_inner_product_task,
    "norm": normalize_norm_task,
    "dual": normalize_dual_task,
    "meet": normalize_meet_task,
    "reflect_point": normalize_reflect_point_task,
    "rotate_object": normalize_rotate_object_task,
    "construct_rotor": normalize_construct_rotor_task,
    "point_pair_decomposition": normalize_point_pair_decomposition_task,
    "midpoint": normalize_midpoint_task,
}


def operation_aware_normalize_task_blocks_result(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("task decomposition result must be a dict")

    tasks = parsed.get("tasks")
    if not isinstance(tasks, list):
        nested = parsed.get("task_blocks_result")
        if isinstance(nested, dict):
            tasks = nested.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task decomposition result has no tasks")

    normalized_tasks: list[dict[str, Any]] = []
    visualization_only_tasks: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} is not a dict")

        if is_visualization_only_task(task):
            visualization_only_tasks.append(deepcopy(task))
            continue

        normalized_task = normalize_common_task_fields(task, index)
        operation = str(normalized_task.get("operation") or "").strip()
        normalizer = OPERATION_NORMALIZERS.get(operation, default_normalizer)
        operation_spec = get_operation_spec(operation)
        if operation_spec and operation_spec.normalizer_name:
            registry_normalizer = globals().get(operation_spec.normalizer_name)
            if callable(registry_normalizer):
                normalizer = registry_normalizer
        normalized_task = normalizer(normalized_task, task, normalized_tasks)
        normalized_tasks.append(normalized_task)

    if not normalized_tasks:
        raise ValueError("task decomposition result has no valid tasks")

    normalized_tasks = merge_visualization_only_tasks(normalized_tasks, visualization_only_tasks)
    normalized_tasks = _merge_computed_point_pair_construct_point_tasks(normalized_tasks)

    return {
        "tasks": normalized_tasks,
    }
