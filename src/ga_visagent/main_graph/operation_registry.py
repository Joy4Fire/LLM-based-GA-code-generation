from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OperationSpec:
    operation: str
    task_type: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_output_type: str = "multivector"
    allowed_output_types: tuple[str, ...] = ("multivector",)
    description: str = ""
    prompt_spec_key: Optional[str] = None
    normalizer_name: Optional[str] = None
    compiler_name: Optional[str] = None


OPERATION_REGISTRY: dict[str, OperationSpec] = {
    "construct_point": OperationSpec(
        operation="construct_point",
        task_type="construct_cga_point",
        aliases=("construct_cga_point", "create_point", "point_creation"),
        default_output_type="point",
        allowed_output_types=("point",),
        description="Construct a CGA point from coordinates.",
        prompt_spec_key="construct_point",
        normalizer_name="normalize_construct_point_task",
        compiler_name="build_construct_point_task_block",
    ),
    "line_from_two_points": OperationSpec(
        operation="line_from_two_points",
        task_type="construct_cga_line_from_two_points",
        aliases=("construct_line", "construct_cga_line_from_two_points", "line_from_points"),
        default_output_type="line",
        allowed_output_types=("line",),
        description="Construct a CGA line from two points.",
        prompt_spec_key="line_from_two_points",
        normalizer_name="normalize_line_from_two_points_task",
        compiler_name="build_line_from_two_points_task_block",
    ),
    "point_distance": OperationSpec(
        operation="point_distance",
        task_type="compute_cga_point_distance",
        aliases=("compute_point_distance", "compute_cga_point_distance"),
        default_output_type="scalar",
        allowed_output_types=("scalar",),
        description="Compute squared distance between two points.",
        prompt_spec_key="point_distance",
        normalizer_name="normalize_point_distance_task",
        compiler_name="build_point_distance_task_block",
    ),
    "midpoint": OperationSpec(
        operation="midpoint",
        task_type="compute_midpoint",
        aliases=("compute_midpoint", "middle_point", "mid_point"),
        default_output_type="point",
        allowed_output_types=("point",),
        description="Compute midpoint of two points.",
        prompt_spec_key="midpoint",
        normalizer_name="normalize_midpoint_task",
        compiler_name="build_midpoint_task_block",
    ),
    "construct_sphere": OperationSpec(
        operation="construct_sphere",
        task_type="construct_cga_sphere",
        aliases=("construct_cga_sphere", "create_sphere"),
        default_output_type="sphere",
        allowed_output_types=("sphere",),
        description="Construct a CGA sphere from center and radius.",
        prompt_spec_key="construct_sphere",
        normalizer_name="normalize_construct_sphere_task",
        compiler_name="build_construct_sphere_task_block",
    ),
    "circle_from_three_points": OperationSpec(
        operation="circle_from_three_points",
        task_type="construct_cga_circle_from_three_points",
        aliases=("construct_circle", "construct_cga_circle_from_three_points", "circle_from_points"),
        default_output_type="circle",
        allowed_output_types=("circle",),
        description="Construct a CGA circle from three points.",
        prompt_spec_key="circle_from_three_points",
        normalizer_name="normalize_circle_from_three_points_task",
        compiler_name="build_circle_from_three_points_task_block",
    ),
    "plane_from_three_points": OperationSpec(
        operation="plane_from_three_points",
        task_type="construct_cga_plane_from_three_points",
        aliases=("construct_cga_plane_from_three_points", "plane_from_points"),
        default_output_type="plane",
        allowed_output_types=("plane",),
        description="Construct a CGA plane from three points.",
        prompt_spec_key="plane_from_three_points",
        normalizer_name="normalize_plane_from_three_points_task",
        compiler_name="build_plane_from_three_points_task_block",
    ),
    "plane_from_point_and_normal": OperationSpec(
        operation="plane_from_point_and_normal",
        task_type="construct_cga_plane_from_point_and_normal",
        aliases=(
            "construct_plane",
            "construct_plane_from_point_and_normal",
            "construct_cga_plane_from_point_and_normal",
            "plane_point_normal",
        ),
        default_output_type="plane",
        allowed_output_types=("plane",),
        description="Construct a CGA plane from a point and a normal vector.",
        prompt_spec_key="plane_from_point_and_normal",
        normalizer_name="normalize_plane_from_point_and_normal_task",
        compiler_name="build_plane_from_point_and_normal_task_block",
    ),
    "construct_vector": OperationSpec(
        operation="construct_vector",
        task_type="construct_vector",
        aliases=("create_vector", "vector_creation"),
        default_output_type="vector",
        allowed_output_types=("vector",),
        description="Construct a vector expression.",
        prompt_spec_key="construct_vector",
        normalizer_name="normalize_construct_vector_task",
        compiler_name="build_construct_vector_task_block",
    ),
    "geometric_product": OperationSpec(
        operation="geometric_product",
        task_type="compute_geometric_product",
        aliases=("compute_geometric_product", "gp"),
        default_output_type="multivector",
        allowed_output_types=("multivector",),
        description="Compute geometric product of two symbols.",
        prompt_spec_key="geometric_product",
        normalizer_name="normalize_geometric_product_task",
        compiler_name="build_geometric_product_task_block",
    ),
    "outer_product": OperationSpec(
        operation="outer_product",
        task_type="compute_outer_product",
        aliases=("compute_outer_product", "wedge_product", "join"),
        default_output_type="multivector",
        allowed_output_types=("multivector", "point", "point_pair", "line", "circle", "sphere", "plane"),
        description="Compute outer/wedge product of two or more symbols.",
        prompt_spec_key="outer_product",
        normalizer_name="normalize_outer_product_task",
        compiler_name="build_outer_product_task_block",
    ),
    "inner_product": OperationSpec(
        operation="inner_product",
        task_type="compute_inner_product",
        aliases=("compute_inner_product", "dot_product"),
        default_output_type="scalar",
        allowed_output_types=("scalar",),
        description="Compute inner product of two symbols.",
        prompt_spec_key="inner_product",
        normalizer_name="normalize_inner_product_task",
        compiler_name="build_inner_product_task_block",
    ),
    "norm": OperationSpec(
        operation="norm",
        task_type="compute_norm",
        aliases=("compute_norm",),
        default_output_type="scalar",
        allowed_output_types=("scalar",),
        description="Compute the norm of a symbol.",
        prompt_spec_key="norm",
        normalizer_name="normalize_norm_task",
        compiler_name="build_norm_task_block",
    ),
    "dual": OperationSpec(
        operation="dual",
        task_type="compute_dual",
        aliases=("compute_dual",),
        default_output_type="multivector",
        allowed_output_types=("multivector", "point", "point_pair", "line", "circle", "sphere", "plane"),
        description="Compute the dual of a symbol.",
        prompt_spec_key="dual",
        normalizer_name="normalize_dual_task",
        compiler_name="build_dual_task_block",
    ),
    "meet": OperationSpec(
        operation="meet",
        task_type="compute_meet",
        aliases=("compute_meet", "intersection", "line_intersection"),
        default_output_type="multivector",
        allowed_output_types=("multivector", "point", "point_pair", "line", "circle", "sphere", "plane"),
        description="Compute meet/intersection of two geometric objects.",
        prompt_spec_key="meet",
        normalizer_name="normalize_meet_task",
        compiler_name="build_meet_task_block",
    ),
    "reflect_point": OperationSpec(
        operation="reflect_point",
        task_type="reflect_cga_point",
        aliases=("reflect", "reflection", "point_reflection", "reflect_cga_point"),
        default_output_type="point",
        allowed_output_types=("point",),
        description="Reflect a point with respect to a mirror object.",
        prompt_spec_key="reflect_point",
        normalizer_name="normalize_reflect_point_task",
        compiler_name="build_reflect_point_task_block",
    ),
    "rotate_object": OperationSpec(
        operation="rotate_object",
        task_type="rotate_cga_object",
        aliases=("rotate", "rotation", "rotate_point", "rotate_line", "rotate_circle", "rotate_sphere", "rotate_cga_object"),
        default_output_type="multivector",
        allowed_output_types=("multivector", "point", "line", "circle", "sphere", "plane", "point_pair"),
        description="Rotate a geometric object, with implicit axis-angle or explicit rotor.",
        prompt_spec_key="rotate_object",
        normalizer_name="normalize_rotate_object_task",
        compiler_name="build_rotate_object_task_block",
    ),
    "construct_rotor": OperationSpec(
        operation="construct_rotor",
        task_type="construct_rotor",
        aliases=("create_rotor", "rotor", "build_rotor"),
        default_output_type="rotor",
        allowed_output_types=("rotor",),
        description="Construct a rotor from axis and angle.",
        prompt_spec_key="construct_rotor",
        normalizer_name="normalize_construct_rotor_task",
        compiler_name="build_construct_rotor_task_block",
    ),
    "point_pair_decomposition": OperationSpec(
        operation="point_pair_decomposition",
        task_type="decompose_cga_point_pair",
        aliases=(
            "decompose_point_pair",
            "point_pair_decompose",
            "extract_point_pair",
            "split_point_pair",
            "point_pair_to_points",
            "compute_intersection_points",
            "decompose_cga_point_pair",
        ),
        default_output_type="point",
        allowed_output_types=("point",),
        description="Decompose a point pair into two points.",
        prompt_spec_key="point_pair_decomposition",
        normalizer_name="normalize_point_pair_decomposition_task",
        compiler_name="build_point_pair_decomposition_task_block",
    ),
}


def _normalize_key(value: str) -> str:
    return str(value or "").strip().replace(" ", "_").lower()


ALIAS_TO_OPERATION: dict[str, str] = {}
TASK_TYPE_TO_OPERATION: dict[str, str] = {}
for _operation, _spec in OPERATION_REGISTRY.items():
    ALIAS_TO_OPERATION[_normalize_key(_operation)] = _operation
    TASK_TYPE_TO_OPERATION[_normalize_key(_spec.task_type)] = _operation
    for _alias in _spec.aliases:
        ALIAS_TO_OPERATION[_normalize_key(_alias)] = _operation


def normalize_operation_alias(operation: str) -> str:
    normalized = _normalize_key(operation)
    if not normalized:
        return ""
    return ALIAS_TO_OPERATION.get(normalized, normalized)


def get_operation_spec(operation: str) -> OperationSpec | None:
    normalized = normalize_operation_alias(operation)
    if not normalized:
        return None
    return OPERATION_REGISTRY.get(normalized)


def get_task_type_for_operation(operation: str) -> str | None:
    spec = get_operation_spec(operation)
    if spec is None:
        return None
    return spec.task_type


def get_default_output_type(operation: str) -> str | None:
    spec = get_operation_spec(operation)
    if spec is None:
        return None
    return spec.default_output_type


def get_allowed_output_types(operation: str) -> tuple[str, ...]:
    spec = get_operation_spec(operation)
    if spec is None:
        return ()
    return spec.allowed_output_types


def get_operation_prompt_spec(operation: str) -> str | None:
    spec = get_operation_spec(operation)
    if spec is None:
        return None
    return spec.prompt_spec_key or spec.operation


def get_operation_aliases(operation: str) -> tuple[str, ...]:
    spec = get_operation_spec(operation)
    if spec is None:
        return ()
    return spec.aliases


def get_operation_for_task_type(task_type: str) -> str | None:
    normalized = _normalize_key(task_type)
    if not normalized:
        return None
    return TASK_TYPE_TO_OPERATION.get(normalized)


def get_supported_operations() -> tuple[str, ...]:
    return tuple(sorted(OPERATION_REGISTRY.keys()))
