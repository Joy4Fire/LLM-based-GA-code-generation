OPERATION_SPECS = {
    "construct_point": """
construct_point:
Use when the user creates a point like P1(0,0,0).
If the user asks to visualize a point or a point set:
- attach visualization to the construct_point task
- for a red point set, each point should use color Red
- do not create a standalone visualization task
Task:
{
  "task_type": "construct_cga_point",
  "operation": "construct_point",
  "inputs": [],
  "outputs": ["P1"],
  "object_specs": {"name": "P1", "type": "point", "coordinates": [0, 0, 0]}
}
Example with Red visualization:
{
  "task_type": "construct_cga_point",
  "operation": "construct_point",
  "outputs": ["P1"],
  "object_specs": {"name": "P1", "type": "point", "coordinates": [0, 0, 0]},
  "visualization": {"required": true, "objects": [{"name": "P1", "type": "point", "color": "Red"}]}
}
""".strip(),
    "line_from_two_points": """
line_from_two_points:
Use when the user creates a line from two points.
Task:
{
  "task_type": "construct_cga_line_from_two_points",
  "operation": "line_from_two_points",
  "inputs": ["P1", "P2"],
  "outputs": ["L"],
  "object_specs": {"name": "L", "type": "line", "from": ["P1", "P2"]}
}
""".strip(),
    "point_distance": """
point_distance:
Use when the user asks for distance between two points.
Task:
{
  "task_type": "compute_cga_point_distance",
  "operation": "point_distance",
  "inputs": ["P1", "P2"],
  "outputs": ["d2"],
  "object_specs": {"name": "d2", "type": "scalar", "from": ["P1", "P2"], "quantity": "squared_distance"}
}
""".strip(),
    "midpoint": """
midpoint:
Use when the user asks to calculate the midpoint / middle point between two points.
Examples:
- calculate midpoint of P1 and P2
- M = (P1 + P2) / 2
- midpoint between points P1(0,0,0) and P2(2,0,0)
Task:
{
  "task_type": "compute_midpoint",
  "operation": "midpoint",
  "inputs": ["P1", "P2"],
  "outputs": ["M"],
  "object_specs": {"name": "M", "type": "point", "from": ["P1", "P2"], "formula": "M = (P1 + P2) / 2"}
}
Rules:
- If P1 and P2 are defined in the same user input, create construct_point tasks first.
- midpoint inputs must be the two point symbols.
- midpoint output type is point.
- Do not use construct_point for the midpoint result, because it is computed from existing points.
- Do not output code_to_optimize.
- Do not output variable_assignments.
- Do not output multivectors_to_be_visualized.
""".strip(),
    "construct_sphere": """
construct_sphere:
Use when the user creates a sphere from center and radius.
construct_sphere supports two center forms:
- center as coordinates: {"center": [0, 0, 0], "radius": 0.5}
- center as an existing point symbol: {"center": "X1", "radius": 0.5}
If the user writes centers X1(0,0,0), X2(...), X3(...), prefer:
- construct_point X1
- construct_point X2
- construct_point X3
- construct_sphere S1 with center "X1"
- construct_sphere S2 with center "X2"
- construct_sphere S3 with center "X3"
Task:
{
  "task_type": "construct_cga_sphere",
  "operation": "construct_sphere",
  "inputs": [],
  "outputs": ["S"],
  "object_specs": {"name": "S", "type": "sphere", "center": [0, 0, 0], "radius": 1.0}
}
""".strip(),
    "circle_from_three_points": """
circle_from_three_points:
Use when the user asks for a circle through three points.
Task:
{
  "task_type": "construct_cga_circle_from_three_points",
  "operation": "circle_from_three_points",
  "inputs": ["P1", "P2", "P3"],
  "outputs": ["C"],
  "object_specs": {"name": "C", "type": "circle", "from": ["P1", "P2", "P3"]}
}
""".strip(),
    "plane_from_three_points": """
plane_from_three_points:
Use when the user creates a plane through three points.
Task:
{
  "task_type": "construct_cga_plane_from_three_points",
  "operation": "plane_from_three_points",
  "inputs": ["P1", "P2", "P3"],
  "outputs": ["Pi"],
  "object_specs": {"name": "Pi", "type": "plane", "from": ["P1", "P2", "P3"]}
}
""".strip(),
    "plane_from_point_and_normal": """
plane_from_point_and_normal:
Use when the user creates a plane from a point and a normal vector.
Task:
{
  "task_type": "construct_cga_plane_from_point_and_normal",
  "operation": "plane_from_point_and_normal",
  "inputs": [],
  "outputs": ["Pi"],
  "object_specs": {"name": "Pi", "type": "plane", "point": [0, 0, 0], "normal": [0, 0, 1]}
}
""".strip(),
    "construct_vector": """
construct_vector:
Use when the user defines a vector like A=e1 or A=2*e1+2*e2.
Task:
{
  "task_type": "construct_vector",
  "operation": "construct_vector",
  "inputs": [],
  "outputs": ["A"],
  "object_specs": {"name": "A", "type": "vector", "expression": "e1"}
}
""".strip(),
    "geometric_product": """
geometric_product:
Use when the user asks for geometric product of two symbols.
Task:
{
  "task_type": "compute_geometric_product",
  "operation": "geometric_product",
  "inputs": ["A", "B"],
  "outputs": ["G"],
  "object_specs": {"name": "G", "type": "multivector", "from": ["A", "B"], "operator": "*"}
}
""".strip(),
    "outer_product": """
outer_product:
Use when the user asks for outer product / wedge product like P1^P2 or S1^S2^S3.
outer_product can produce either a generic multivector or a concrete geometric object.
Examples:
- M = A ^ B -> type multivector
- P = L ^ Pi -> type point, if the user explicitly describes an intersection point using wedge formula
- C = P1 ^ P2 ^ P3 -> type circle
- S = P1 ^ P2 ^ P3 ^ P4 -> type sphere
- Pi = P1 ^ P2 ^ P3 ^ einf -> type plane
Do not force object_specs.type to multivector if the user clearly asks for
a point, circle, sphere, plane, line, or point pair.
Task:
{
  "task_type": "compute_outer_product",
  "operation": "outer_product",
  "inputs": ["P1", "P2"],
  "outputs": ["M"],
  "object_specs": {"name": "M", "type": "multivector", "from": ["P1", "P2"], "operator": "^"}
}
""".strip(),
    "inner_product": """
inner_product:
Use when the user asks for inner product / dot product of two symbols.
Task:
{
  "task_type": "compute_inner_product",
  "operation": "inner_product",
  "inputs": ["A", "B"],
  "outputs": ["IP"],
  "object_specs": {"name": "IP", "type": "scalar", "from": ["A", "B"], "operator": "."}
}
""".strip(),
    "norm": """
norm:
Use when the user asks for norm such as ||A|| or sqrt(A.A).
Task:
{
  "task_type": "compute_norm",
  "operation": "norm",
  "inputs": ["A"],
  "outputs": ["NormA"],
  "object_specs": {"name": "NormA", "type": "scalar", "from": ["A"], "operator": "sqrt_dot"}
}
""".strip(),
    "dual": """
dual:
Use when the user asks for the dual of one symbol, such as *P.
Task:
{
  "task_type": "compute_dual",
  "operation": "dual",
  "inputs": ["P"],
  "outputs": ["DualP"],
  "object_specs": {"name": "DualP", "type": "multivector", "from": ["P"], "operator": "*"}
}
""".strip(),
    "meet": """
meet:
Use when the user asks for intersection / meet of two geometric objects.
meet can produce either a generic multivector or a concrete geometric object.
Examples:
- X = meet(L1, L2) -> type point
- C = meet(S1, S2) -> type circle
- L = meet(Pi1, Pi2) -> type line
- M = meet(A, B) -> type multivector
Do not force object_specs.type to multivector if the user clearly asks for an
intersection point, line, circle, sphere, plane, or point pair.
Task:
{
  "task_type": "compute_meet",
  "operation": "meet",
  "inputs": ["L1", "L2"],
  "outputs": ["I"],
  "object_specs": {"name": "I", "type": "multivector", "from": ["L1", "L2"], "operator": "meet"}
}
""".strip(),
    "reflect_point": """
reflect_point:
Use when the user reflects a point across a plane.
Task:
{
  "task_type": "reflect_cga_point",
  "operation": "reflect_point",
  "inputs": ["P", "Pi"],
  "outputs": ["P_reflected"],
  "object_specs": {"name": "P_reflected", "type": "point", "point": "P", "mirror": "Pi", "formula": "M v M"}
}
""".strip(),
    "construct_rotor": """
construct_rotor:
Use when the user asks to create or construct a rotor from a rotation axis and rotation angle.
Examples:
- Create a rotor with rotation axis (6,8,9) and rotation angle 90 degree.
- Rotor representation R = cos(theta/2) - sin(theta/2) * B.
- R = cos(45 degree) - sin(45 degree)*(6/sqrt(181)e23 + 8/sqrt(181)e31 + 9/sqrt(181)e12)
Rules:
- Do not split rotor construction into separate sqrt, cos, or sin tasks.
- Do not put numeric constants such as 6, 8, 9, 181, or 45_deg in inputs.
- Axis and angle belong in object_specs.
- inputs must be [].
- If the user writes 90 degree, use angle=90 and angle_unit="degree".
- If the user writes pi/2 or π/2, use angle="pi/2" and angle_unit="radian".
- If the user writes R explicitly, output should be ["R"].
- If no output name is specified, default output is ["R"].
Task:
{
  "task_type": "construct_rotor",
  "operation": "construct_rotor",
  "inputs": [],
  "outputs": ["R"],
  "object_specs": {"name": "R", "type": "rotor", "axis": [6, 8, 9], "angle": 90, "angle_unit": "degree"}
}
""".strip(),
    "rotate_object": """
rotate_object:
Use when the user asks to rotate a geometric object such as point, line, circle, sphere, plane, or multivector.
Examples:
- rotate circle C by pi/4 around axis (0,0,1)
- rotate line L by 60 degrees around X-axis
- C' = R C ~R
rotate_object supports two modes:
Mode A: axis-angle rotation without an explicit rotor.
{
  "operation": "rotate_object",
  "inputs": ["P"],
  "outputs": ["P_rotated"],
  "object_specs": {"object": "P", "axis": [0, 0, 1], "angle": "pi/2", "angle_unit": "radian"}
}
Mode B: explicit rotor rotation.
If the user creates a rotor R and then applies P' = R P ~R, use:
{
  "operation": "rotate_object",
  "inputs": ["P", "R"],
  "outputs": ["P_rotated"],
  "object_specs": {"object": "P", "rotor": "R"}
}
Rules:
- For direct axis-angle rotation on an existing object, keep rotation as one rotate_object task.
- If the user explicitly creates rotor R and then applies P' = R P ~R, prefer explicit rotor mode.
- If output name is C', normalize it to C_rotated.
- If output name is P', normalize it to P_rotated.
- If output name is L', normalize it to L_rotated.
- If the user asks to visualize original and rotated objects, keep original visualization on the original task and rotated visualization on rotate_object.
- Do not force rotate_object to have only one input.
- In explicit rotor mode, the second input should be the rotor symbol and construct_rotor should define it first.
Task:
{
  "task_type": "rotate_cga_object",
  "operation": "rotate_object",
  "inputs": ["C"],
  "outputs": ["C_rotated"],
  "object_specs": {"name": "C_rotated", "type": "circle", "object": "C", "axis": [0, 0, 1], "angle": "pi/4", "angle_unit": "radian"}
}
""".strip(),
    "point_pair_decomposition": """
point_pair_decomposition:
Use when the user asks to decompose a point pair into two intersection points, such as X4 and X5, P_plus/P_minus, or P±.
Common context:
- M = S1 ^ S2 ^ S3
- P = *M
- X± = (P ± sqrt(P.P)) / (einf.P)
Rules:
- Do not use construct_point for X4 and X5.
- construct_point is only for coordinate-based point creation.
- X4 and X5 are computed points from the point pair P.
- The built-in symbol einf does not need to be defined as a task.
Task:
{
  "task_type": "decompose_cga_point_pair",
  "operation": "point_pair_decomposition",
  "inputs": ["P"],
  "outputs": ["X4", "X5"],
  "object_specs": {"name": "point_pair_decomposition", "type": "point_pair_decomposition", "point_pair": "P", "formula": "X_pm = (P ± sqrt(P.P)) / (einf.P)"}
}
    """.strip(),
}


def get_prompt_spec_for_operation(operation: str) -> str:
    from ga_visagent.main_graph.operation_registry import get_operation_prompt_spec, normalize_operation_alias

    normalized_operation = normalize_operation_alias(operation)
    spec_key = get_operation_prompt_spec(normalized_operation) or normalized_operation
    return OPERATION_SPECS.get(spec_key, "")
