import ast
import json
import math
import os
import re
import time
from copy import deepcopy
from functools import lru_cache
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None

from ga_visagent.legacy_agents.nodes import (
    code_to_optimize_agent_node,
    compact_code_to_optimize,
    compact_variable_assignments,
    multivectors_to_be_visualized_agent_node,
    variable_assignments_agent_node,
)
from ga_visagent.main_graph.operation_registry import (
    OPERATION_REGISTRY,
    get_allowed_output_types,
    get_default_output_type,
    get_operation_for_task_type,
    get_supported_operations,
    get_task_type_for_operation,
    normalize_operation_alias,
)
from ga_visagent.main_graph.operation_specs import OPERATION_SPECS, get_prompt_spec_for_operation
from ga_visagent.main_graph.state import MainGraphState
from ga_visagent.models.llm_setup import get_llm

DEBUG_VERBOSE = False
DEBUG_PROMPT = False
DEBUG_SEMANTIC_RETRY = False
MAX_TASK_DECOMPOSITION_SEMANTIC_RETRIES = 2
MAX_GAALOP_SCRIPT_REPAIR_RETRIES = 2
GAALOP_COMPILE_API_URL = "http://gacrac.gagis.cn:8080/api/v1/compile"
USE_LEGACY_NORMALIZER_FALLBACK = True
BUILTIN_SYMBOLS = {"e0", "e1", "e2", "e3", "einf"}
ALLOWED_OUTER_PRODUCT_TYPES = set(get_allowed_output_types("outer_product")) or {
    "multivector",
    "point",
    "point_pair",
    "line",
    "circle",
    "sphere",
    "plane",
}
ALLOWED_MEET_TYPES = set(get_allowed_output_types("meet")) or {
    "multivector",
    "point",
    "point_pair",
    "line",
    "circle",
    "sphere",
    "plane",
}
ALLOWED_ROTATE_OBJECT_TYPES = set(get_allowed_output_types("rotate_object")) or {
    "point",
    "line",
    "circle",
    "sphere",
    "plane",
    "multivector",
    "point_pair",
}
ALLOWED_DUAL_TYPES = set(get_allowed_output_types("dual")) or {
    "multivector",
}


def _debug_print(*args, **kwargs) -> None:
    if DEBUG_VERBOSE:
        print(*args, **kwargs)


INFORMATION_EXTRACTION_TEMPLATE = """
Role: You are an efficient information extraction assistant.
Task: Precisely extract three key pieces of information from the following user input: function name (function_name), target language (target_language), and target space (target_space).

# Extraction Rules:
1. Function name: Look for the function name specified by the user. If not explicitly specified, generate a meaningful CamelCase name based on the content of the user's request, e.g., CalculateIntersection.
2. Target language: Look for the target programming language specified by the user. If not explicitly specified, the default is "PYTHON". The range of values is: CLUCALC, JULIA, VERILOG, GAPP_DEBUGGER, CSHARP, RUST, JAVA, VIS2D, GAALET_OUTPUT, GAPP, COMPRESSED, VISUALIZER, GANJA, GAPP_OPENCL, PYTHON, MATLAB, DOT, MATHematica, CPP, LATEX.
3. Target space: Look for the geometric algebra space specified by the user. If not explicitly specified, return "unknown". The range of values is: ALGEBRA_2D, ALGEBRA_3D, ALGEBRA_2D_PGA, ALGEBRA_3D_PGA, ALGEBRA_CRA, ALGEBRA_STA, ALGEBRA_CGA, ALGEBRA_GAC, ALGEBRA_DCGA, ALGEBRA_CCGA, ALGEBRA_QGA, unknown.

# Output Format:
Must output a pure JSON object without markdown or explanations.
Example:
{
  "function_name": "CalculateIntersection",
  "target_language": "PYTHON",
  "target_space": "ALGEBRA_CGA"
}

---
User input:
"{user_input}"
---

JSON output:
""".strip()


TASK_DECOMPOSITION_TEMPLATE = """
Role:
You are a task decomposition assistant for a geometric algebra code generation main graph.

Task:
Read the original user input and split the complex request into minimal task blocks.

You only need to decompose the task itself.
Do not extract function_name.
Do not extract target_language.
Do not extract target_space.
Do not care about code language.
Do not care about target space.
These have already been handled by the previous node.

Output only this JSON structure:
{
  "tasks": [...]
}

Each task must contain:
1. task_id
2. task_type
3. code_to_optimize
4. variable_assignments
5. multivectors_to_be_visualized

Critical rule:
The code generation, variable assignments, and visualization information for the same geometric object must stay in the same task.
Do not split visualization into a separate task if it belongs to an object created by a task.

Rules:
1. Split complex tasks into minimal executable tasks.
2. Each created point must become its own separate task.
3. Never combine P1, P2, and P3 into one task.
4. code_to_optimize must be a JSON object, never a string.
5. multivectors_to_be_visualized must be a JSON object, never a list.
6. Do not output visualize(...), point_set, arrays of points, or any Python code.
7. For a red point set, each point task must visualize only that point in Red.
8. For point construction, use this exact expanded CGA formula pattern:
   Pi = ai*e1 + bi*e2 + ci*e3 + 0.5*(ai*ai + bi*bi + ci*ci)*einf + e0
9. Use einf, not e_inf.
10. Use * for multiplication, never **.
11. For point coordinates like P1(0,0,0), extract coordinate values into variable_assignments.
12. Replace concrete coordinate values with lowercase placeholders.
13. Use P1 -> a1,b1,c1; P2 -> a2,b2,c2; P3 -> a3,b3,c3.
14. Placeholder names must not be repeated across different points.
15. Formula must use placeholders, never concrete coordinate values.
16. variable_assignments contains concrete values.
17. parameters contains placeholders only, not basis vectors.
18. If visualization is required for a point set, put the visualization object in the same task that creates that point.
19. If no visualization is requested, required=false and objects=[].
20. task_type for creating points must be construct_cga_point, not visualization.
21. Do not output space.
22. Do not output target.
23. Do not output function_name.
24. Do not output target_language.
25. Do not output target_space.
26. Do not output missing_assignments.
27. Do not output warnings.
28. Output pure JSON only.
29. Do not output markdown or explanations.
30. Follow this example structure exactly:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_cga_point",
      "code_to_optimize": {
        "goal": "在 CGA 空间中生成点 P1，并将其作为红色点集的一部分可视化。",
        "formula": "P1 = a1*e1 + b1*e2 + c1*e3 + 0.5*(a1*a1 + b1*b1 + c1*c1)*einf + e0",
        "parameters": ["a1", "b1", "c1"],
        "output": "P1"
      },
      "variable_assignments": {
        "a1": 0,
        "b1": 0,
        "c1": 0
      },
      "multivectors_to_be_visualized": {
        "required": true,
        "objects": [
          {
            "name": "P1",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    }
  ]
}

User input:
{user_input}

JSON output:
""".strip()


TASK_DECOMPOSITION_REPAIR_TEMPLATE = """
Role:
You are repairing an invalid semantic task decomposition IR for a geometric algebra code generation graph.

The previous JSON was invalid according to the local validator.

Your task:
Fix the JSON so that it follows the schema and passes validation.

Original user input:
{user_input}

Selected operations:
{selected_operations}

Available operation rules:
{operation_rules}

Validator errors:
{validation_errors}

Invalid JSON:
{invalid_task_blocks_result}

Output requirements:
1. Output only pure JSON.
2. Output format must be:
   {{
     "tasks": [...]
   }}
3. Every task must contain only:
   - task_id
   - task_type
   - operation
   - inputs
   - outputs
   - depends_on
   - object_specs
   - visualization
4. Do not output code_to_optimize.
5. Do not output variable_assignments.
6. Do not output multivectors_to_be_visualized.
7. Do not output GAALOPScript.
8. Do not create standalone visualization tasks.
9. Visualization must be attached to the task that creates or computes the object.
10. construct_point is only for coordinate-created points.
11. Computed points must use the appropriate operation, not construct_point.
12. e0, e1, e2, e3, einf are built-in symbols and should not be constructed as tasks.
13. Preserve the original user intent.
14. Preserve colors and visualization requests.
15. Preserve dependency order.
16. Use existing operation names from the available operation rules when possible.

Return repaired JSON only:
""".strip()


GAALOP_SCRIPT_REPAIR_TEMPLATE = """
Role:
You are repairing GAALOPScript that failed backend compilation.

The semantic task decomposition and geometric intent are already fixed.
Only repair the script fields accepted by the Gaalop compile API.

Original user input:
{user_input}

Current compile request:
{request_payload}

Backend compile error:
{compile_error}

Output requirements:
1. Output only pure JSON.
2. Output format must be:
   {{
     "optimizeCode": "...",
     "variableAssignments": "...",
     "multivectorsVisualized": "..."
   }}
3. Do not output markdown.
4. Do not change functionName, algebraPlugins, codegenPlugins, outputMode, or optimization.
5. Preserve the original geometry and visualization intent.
6. optimizeCode must be GAALOPScript only.
7. variableAssignments must contain only assignments such as a1=0; b1=0;.
8. multivectorsVisualized must contain only visualization directives such as :Red; and :P; or be empty.
9. If code variables are intended as outputs, keep the leading ? prefix.
10. Do not introduce unsupported operations or Python/Java syntax.

Return repaired JSON only:
""".strip()


TASK_DECOMPOSITION_TEMPLATE = """
Role:
You are a semantic task decomposition assistant for a geometric algebra code generation main graph.

Task:
Read the original user input and decompose the complex request into pure semantic IR tasks.

You only need to output semantic IR.
Do not extract function_name.
Do not extract target_language.
Do not extract target_space.
Do not generate code templates.
Do not generate formulas.
Do not generate variable assignments.
Do not generate GAALOPScript.
These have already been handled by the previous node.

Output only this JSON structure:
{
  "tasks": [...]
}

Each task must contain only:
1. task_id
2. task_type
3. operation
4. inputs
5. outputs
6. depends_on
7. object_specs
8. visualization

Field meaning:
- task_id: integer task id.
- task_type: task category, such as construct_cga_point or construct_cga_line_from_two_points.
- operation: semantic operation, such as construct_point or line_from_two_points.
- inputs: symbols used by this task.
- outputs: symbols produced by this task.
- depends_on: task ids this task depends on.
- object_specs: semantic description of the generated object.
- visualization: visualization requirements for the object generated by this task.

Critical rules:
1. Do not extract function_name.
2. Do not extract target_language.
3. Do not extract target_space.
4. These have already been handled by the previous node.
5. Do not output code_to_optimize.
6. Do not output variable_assignments.
7. Do not output multivectors_to_be_visualized.
8. Do not output formula.
9. Do not output parameters.
10. Do not output GAALOPScript.
11. Split complex tasks into minimal executable semantic tasks.
12. The visualization information for an object must stay in the same task that creates that object.
13. Do not split visualization into a separate task if it belongs to an object created by a task.
14. For point coordinates like P1(0,0,0), extract coordinates into object_specs.coordinates.
15. For construct point tasks:
    - task_type must be construct_cga_point
    - operation must be construct_point
    - inputs must be []
    - outputs must be ["P1"], ["P2"], etc.
    - depends_on must be []
    - object_specs must include name, type, coordinates
16. For line from two points:
    - task_type must be construct_cga_line_from_two_points
    - operation must be line_from_two_points
    - inputs must be the two point symbols, such as ["P1", "P2"]
    - outputs must be the line symbol, such as ["L"]
    - depends_on must contain the task ids that define the input points
    - object_specs must include name, type, from
17. For point distance:
    - If the user asks to calculate distance between two points, create a task with:
      - task_type: compute_cga_point_distance
      - operation: point_distance
      - inputs: the two point symbols, such as ["P1", "P2"]
      - outputs: ["d2"] by default
      - depends_on: task ids that define the input points
      - object_specs:
        {
          "name": "d2",
          "type": "scalar",
          "from": ["P1", "P2"],
          "quantity": "squared_distance"
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
18. For vector construction:
    - If the user defines a vector such as A=e1, B=e2, A=2e1+2e2, or A=2*e1+2*e2, create a task with:
      - task_type: construct_vector
      - operation: construct_vector
      - inputs: []
      - outputs: ["A"] or ["B"]
      - depends_on: []
      - object_specs:
        {
          "name": "A",
          "type": "vector",
          "expression": "e1"
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
19. For geometric product:
    - If the user asks to calculate the geometric product of A and B, create a task with:
      - task_type: compute_geometric_product
      - operation: geometric_product
      - inputs: ["A", "B"]
      - outputs: ["G"] by default
      - depends_on: task ids that define A and B
      - object_specs:
        {
          "name": "G",
          "type": "multivector",
          "from": ["A", "B"],
          "operator": "*"
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
20. For outer product:
    - If the user asks to calculate the outer product / wedge product of two or more symbols, create a task with:
      - task_type: compute_outer_product
      - operation: outer_product
      - inputs: the operand symbols, such as ["P1", "P2"] or ["S1", "S2", "S3"]
      - outputs: ["M"] by default, unless the user specifies another output symbol
      - depends_on: task ids that define all input symbols
      - object_specs:
        {
          "name": "M",
          "type": "multivector",
          "from": ["P1", "P2"],
          "operator": "^"
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
21. For inner product:
    - If the user asks to calculate the inner product / dot product of two symbols, create a task with:
      - task_type: compute_inner_product
      - operation: inner_product
      - inputs: the two operand symbols, such as ["P1", "P2"] or ["A", "B"]
      - outputs: ["IP"] by default, unless the user specifies another output symbol
      - depends_on: task ids that define all input symbols
      - object_specs:
        {
          "name": "IP",
          "type": "scalar",
          "from": ["P1", "P2"],
          "operator": "."
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
22. For norm:
    - If the user asks to calculate the norm of a vector or multivector, create a task with:
      - task_type: compute_norm
      - operation: norm
      - inputs: the symbol whose norm should be computed, such as ["A"]
      - outputs: ["NormA"] by default, unless the user specifies another output symbol
      - depends_on: task ids that define the input symbol
      - object_specs:
        {
          "name": "NormA",
          "type": "scalar",
          "from": ["A"],
          "operator": "sqrt_dot"
        }
      - visualization:
        {
          "required": false,
          "objects": []
        }
23. For dual:
    - If the user asks to calculate the dual of a symbol, create a task with:
      - task_type: compute_dual
      - operation: dual
      - inputs: the symbol to be dualized, such as ["P"] or ["M"]
      - outputs: ["DualP"] by default for input P, or ["DualM"] for input M, unless the user specifies another output symbol
      - depends_on: task ids that define the input symbol
      - object_specs:
        {
          "name": "DualP",
          "type": "multivector",
          "from": ["P"],
          "operator": "*"
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
24. For meet:
    - If the user asks to calculate the meet or intersection of two geometric objects, create a task with:
      - task_type: compute_meet
      - operation: meet
      - inputs: the two operand symbols, such as ["L1", "L2"] or ["L", "Pi"]
      - outputs: ["I"] by default, unless the user specifies another output symbol
      - depends_on: task ids that define all input symbols
      - object_specs:
        {
          "name": "I",
          "type": "multivector",
          "from": ["L1", "L2"],
          "operator": "meet"
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
25. For point reflection:
    - If the user asks to reflect a point with respect to a plane, create a task with:
      - task_type: reflect_cga_point
      - operation: reflect_point
      - inputs: [point_symbol, mirror_plane_symbol]
      - outputs: ["P_reflected"] by default, unless the user specifies another output name
      - depends_on: task ids that define the point and the mirror plane
      - object_specs:
        {
          "name": "P_reflected",
          "type": "point",
          "point": "P",
          "mirror": "Pi",
          "formula": "M v M"
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
26. Built-in coordinate planes:
    - If user says y-z plane:
      - This is the plane x=0.
      - Create plane_from_point_and_normal with point [0,0,0], normal [1,0,0], output ["Pi"].
    - If user says x-z plane:
      - This is the plane y=0.
      - Create plane_from_point_and_normal with point [0,0,0], normal [0,1,0], output ["Pi"].
    - If user says x-y plane:
      - This is the plane z=0.
      - Create plane_from_point_and_normal with point [0,0,0], normal [0,0,1], output ["Pi"].
27. If the input symbol is defined in the same user request, create its construction task before the norm or dual task.
28. If the input operands are defined in the same user request, create their construction tasks before the meet task.
29. If the point or mirror plane is defined in the same user request, create its construction task before the reflect_point task.
30. Do not treat P1/P2/S1/S2/A/B/P/M/L1/L2/Pi/C/S as already defined unless previous tasks define them.
31. For generic "intersection" between two geometric objects, prefer operation="meet" unless there is already a more specific supported operation.
32. If the user writes A=e1, B=e2, or A=2e1+2e2, do not treat A or B as already defined inputs. Create construct_vector tasks before geometric_product, inner_product, or norm.
33. If the user says "calculate its dual *P, and visualize it as green", visualize the dual result, not the original object.
34. The vector expression should be stored in object_specs.expression.
35. Normalize vector expressions with implicit multiplication:
    - 2e1 -> 2*e1
    - 2e2 -> 2*e2
    - -3e3 -> -3*e3
36. Do not generate formulas for construct_vector, geometric_product, outer_product, inner_product, norm, dual, meet, or reflect_point in task_decomposition_node.
37. Do not decompose a point reflection request into geometric_product. Use reflect_point instead.
38. For sphere construction:
    - If the user asks to create a sphere with center coordinates and radius, create a task with:
      - task_type: construct_cga_sphere
      - operation: construct_sphere
      - inputs: []
      - outputs: ["S"] or ["S1"], ["S2"], etc.
      - depends_on: []
      - object_specs:
        {
          "name": "S",
          "type": "sphere",
          "center": [0, 0, 0],
          "radius": 1.0
        }
      - visualization:
        {
          "required": true,
          "objects": [
            {
              "name": "S",
              "type": "sphere",
              "color": "Blue"
            }
          ]
        }
28. For circle from three points:
    - If the user asks to construct or calculate a circle through three points, create a task with:
      - task_type: construct_cga_circle_from_three_points
      - operation: circle_from_three_points
      - inputs: the three point symbols, such as ["P1", "P2", "P3"]
      - outputs: ["C"] or ["C1"], ["C2"], etc.
      - depends_on: task ids that define the input points
      - object_specs:
        {
          "name": "C",
          "type": "circle",
          "from": ["P1", "P2", "P3"]
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
29. If the user asks to create a circle with center coordinates, radius, and lying in the XY plane, but does not provide three explicit points, decompose it into three constructed points and one circle_from_three_points task.
30. For center (cx, cy, cz), radius r, XY plane:
    - P1 = (cx + r, cy, cz)
    - P2 = (cx, cy + r, cz)
    - P3 = (cx - r, cy, cz)
31. Then create:
    - construct_point P1
    - construct_point P2
    - construct_point P3
    - circle_from_three_points C from P1, P2, P3
32. For plane from three points:
    - If the user asks to construct a plane through three points, create a task with:
      - task_type: construct_cga_plane_from_three_points
      - operation: plane_from_three_points
      - inputs: the three point symbols, such as ["P1", "P2", "P3"]
      - outputs: ["Pi"] by default, or ["Plane1"] if a safer name is needed
      - depends_on: task ids that define the input points
      - object_specs:
        {
          "name": "Pi",
          "type": "plane",
          "from": ["P1", "P2", "P3"]
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
33. For plane from point and normal:
    - If the user asks to construct a plane passing through a point with a normal vector, create a task with:
      - task_type: construct_cga_plane_from_point_and_normal
      - operation: plane_from_point_and_normal
      - inputs: []
      - outputs: ["Pi"] by default
      - depends_on: []
      - object_specs:
        {
          "name": "Pi",
          "type": "plane",
          "point": [x0, y0, z0],
          "normal": [n1, n2, n3]
        }
      - visualization:
        {
          "required": true/false,
          "objects": [...]
        }
34. Do not output the Unicode symbol Π as a code variable.
35. Use "Pi" instead of "Π".
36. First version computes squared distance d2, not sqrt distance d.
37. If user says visualize them as a red point set, each point task should have visualization.required=true and color=Red.
38. If user says visualize L in red, the line task should have visualization.required=true and color=Red.
39. Visualization color rule:
    - If the user explicitly specifies a color, preserve that color.
    - If the user requests visualization but does not specify a color, use "Red" as the default color.
    - Do not output null color for required visualization objects.
    - For "visualize all geometric elements", every generated geometric object should be visualized.
    - If no color is specified in "visualize all geometric elements", use "Red" for every object.
40. If no visualization is requested for a task, use:
    "visualization": {
      "required": false,
      "objects": []
    }
41. Output pure JSON only.
42. Do not output markdown or explanations.
43. Top-level output must only contain tasks.
44. Do not output space.
45. Do not output target.
46. Do not output function_name.
47. Do not output target_language.
48. Do not output target_space.
49. Do not output missing_assignments.
50. Do not output warnings.
51. Follow this example structure exactly:
{
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
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P1",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    }
  ]
}

Example for sphere:
Input:
In conformal space, create a sphere S with center at (0,0,0) and radius 1.0, and visualize it as blue. I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_cga_sphere",
      "operation": "construct_sphere",
      "inputs": [],
      "outputs": ["S"],
      "depends_on": [],
      "object_specs": {
        "name": "S",
        "type": "sphere",
        "center": [0, 0, 0],
        "radius": 1.0
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "S",
            "type": "sphere",
            "color": "Blue"
          }
        ]
      }
    }
  ]
}

Example for circle through three points:
Input:
In conformal space, create three points P1(0,0,0), P2(1,0,0), P3(0,1,0), calculate the circle passing through them, and visualize the points as red and the circle as blue. I need Python code.

Expected JSON:
{
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
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P1",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P2",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 3,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P3"],
      "depends_on": [],
      "object_specs": {
        "name": "P3",
        "type": "point",
        "coordinates": [0, 1, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P3",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 4,
      "task_type": "construct_cga_circle_from_three_points",
      "operation": "circle_from_three_points",
      "inputs": ["P1", "P2", "P3"],
      "outputs": ["C"],
      "depends_on": [1, 2, 3],
      "object_specs": {
        "name": "C",
        "type": "circle",
        "from": ["P1", "P2", "P3"]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "C",
            "type": "circle",
            "color": "Blue"
          }
        ]
      }
    }
  ]
}

Example for center + radius + XY plane circle:
Input:
In conformal space, create a circle C with center at (0,0,0), radius 1.0, lying in the XY plane. Visualize C as green. I need Python code.

Expected JSON:
{
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
        "coordinates": [1.0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [0, 1.0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P3"],
      "depends_on": [],
      "object_specs": {
        "name": "P3",
        "type": "point",
        "coordinates": [-1.0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 4,
      "task_type": "construct_cga_circle_from_three_points",
      "operation": "circle_from_three_points",
      "inputs": ["P1", "P2", "P3"],
      "outputs": ["C"],
      "depends_on": [1, 2, 3],
      "object_specs": {
        "name": "C",
        "type": "circle",
        "from": ["P1", "P2", "P3"],
        "center": [0, 0, 0],
        "radius": 1.0,
        "plane": "XY"
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "C",
            "type": "circle",
            "color": "Green"
          }
        ]
      }
    }
  ]
}

Example for plane through three points:
Input:
Create three non-collinear points P1(0,0,0), P2(1,0,0), P3(0,1,0). Create a plane Π passing through these points and visualize all geometric elements. I need Python code.

Expected JSON:
{
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
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P1",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P2",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 3,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P3"],
      "depends_on": [],
      "object_specs": {
        "name": "P3",
        "type": "point",
        "coordinates": [0, 1, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P3",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 4,
      "task_type": "construct_cga_plane_from_three_points",
      "operation": "plane_from_three_points",
      "inputs": ["P1", "P2", "P3"],
      "outputs": ["Pi"],
      "depends_on": [1, 2, 3],
      "object_specs": {
        "name": "Pi",
        "type": "plane",
        "from": ["P1", "P2", "P3"]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "Pi",
            "type": "plane",
            "color": "Red"
          }
        ]
      }
    }
  ]
}

Example for plane from point and normal:
Input:
In conformal space, create a plane Π passing through point (0,0,0) with a normal vector (0,0,1), and visualize it as yellow. I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_cga_plane_from_point_and_normal",
      "operation": "plane_from_point_and_normal",
      "inputs": [],
      "outputs": ["Pi"],
      "depends_on": [],
      "object_specs": {
        "name": "Pi",
        "type": "plane",
        "point": [0, 0, 0],
        "normal": [0, 0, 1]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "Pi",
            "type": "plane",
            "color": "Yellow"
          }
        ]
      }
    }
  ]
}

Example for vector construction and geometric product:
Input:
In conformal space, calculate the geometric product of two vectors A=e1 and B=e2. I need Python code.Calculation process: 1. Geometric product: $$ A * B $$

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_vector",
      "operation": "construct_vector",
      "inputs": [],
      "outputs": ["A"],
      "depends_on": [],
      "object_specs": {
        "name": "A",
        "type": "vector",
        "expression": "e1"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_vector",
      "operation": "construct_vector",
      "inputs": [],
      "outputs": ["B"],
      "depends_on": [],
      "object_specs": {
        "name": "B",
        "type": "vector",
        "expression": "e2"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "compute_geometric_product",
      "operation": "geometric_product",
      "inputs": ["A", "B"],
      "outputs": ["G"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "G",
        "type": "multivector",
        "from": ["A", "B"],
        "operator": "*"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

Example for outer product:
Input:
In conformal space, create points P1(1,0,0) and P2(0,1,0), calculate their outer product P1∧P2, and output the result. I need Python code.

Expected JSON:
{
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
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [0, 1, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "compute_outer_product",
      "operation": "outer_product",
      "inputs": ["P1", "P2"],
      "outputs": ["M"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "M",
        "type": "multivector",
        "from": ["P1", "P2"],
        "operator": "^"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

Example for point inner product:
Input:
In conformal space, create points P1(1,0,0) and P2(0,1,0), calculate their inner product P1 . P2, and output the result. I need Python code.

Expected JSON:
{
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
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [0, 1, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "compute_inner_product",
      "operation": "inner_product",
      "inputs": ["P1", "P2"],
      "outputs": ["IP"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "IP",
        "type": "scalar",
        "from": ["P1", "P2"],
        "operator": "."
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

Example for vector inner product:
Input:
Calculate the inner product of two vectors A=e1 and B=e2. I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_vector",
      "operation": "construct_vector",
      "inputs": [],
      "outputs": ["A"],
      "depends_on": [],
      "object_specs": {
        "name": "A",
        "type": "vector",
        "expression": "e1"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_vector",
      "operation": "construct_vector",
      "inputs": [],
      "outputs": ["B"],
      "depends_on": [],
      "object_specs": {
        "name": "B",
        "type": "vector",
        "expression": "e2"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "compute_inner_product",
      "operation": "inner_product",
      "inputs": ["A", "B"],
      "outputs": ["IP"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "IP",
        "type": "scalar",
        "from": ["A", "B"],
        "operator": "."
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

Example for norm:
Input:
In conformal space, calculate the norm of the vector A=2e1+2e2, and output the result. I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_vector",
      "operation": "construct_vector",
      "inputs": [],
      "outputs": ["A"],
      "depends_on": [],
      "object_specs": {
        "name": "A",
        "type": "vector",
        "expression": "2*e1 + 2*e2"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "compute_norm",
      "operation": "norm",
      "inputs": ["A"],
      "outputs": ["NormA"],
      "depends_on": [1],
      "object_specs": {
        "name": "NormA",
        "type": "scalar",
        "from": ["A"],
        "operator": "sqrt_dot"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

Example for dual:
Input:
In conformal space, create a point P(0,0,0), calculate its dual *P, and visualize it as green. I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P"],
      "depends_on": [],
      "object_specs": {
        "name": "P",
        "type": "point",
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "compute_dual",
      "operation": "dual",
      "inputs": ["P"],
      "outputs": ["DualP"],
      "depends_on": [1],
      "object_specs": {
        "name": "DualP",
        "type": "multivector",
        "from": ["P"],
        "operator": "*"
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "DualP",
            "type": "multivector",
            "color": "Green"
          }
        ]
      }
    }
  ]
}

Example fragment for dual with explicit output symbol:
Input fragment:
Take the dual of M to obtain P.

Expected dual task:
{
  "task_id": 2,
  "task_type": "compute_dual",
  "operation": "dual",
  "inputs": ["M"],
  "outputs": ["P"],
  "depends_on": [1],
  "object_specs": {
    "name": "P",
    "type": "multivector",
    "from": ["M"],
    "operator": "*"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}

Example for meet:
Input:
In conformal space, create two lines L1 from P1(0,0,0), P2(1,0,0) and L2 from P3(0,0,0), P4(0,1,0), then compute their intersection using meet and visualize it as yellow. I need Python code.

Expected JSON:
{
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
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "construct_cga_line_from_two_points",
      "operation": "line_from_two_points",
      "inputs": ["P1", "P2"],
      "outputs": ["L1"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "L1",
        "type": "line",
        "from": ["P1", "P2"]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 4,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P3"],
      "depends_on": [],
      "object_specs": {
        "name": "P3",
        "type": "point",
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 5,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P4"],
      "depends_on": [],
      "object_specs": {
        "name": "P4",
        "type": "point",
        "coordinates": [0, 1, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 6,
      "task_type": "construct_cga_line_from_two_points",
      "operation": "line_from_two_points",
      "inputs": ["P3", "P4"],
      "outputs": ["L2"],
      "depends_on": [4, 5],
      "object_specs": {
        "name": "L2",
        "type": "line",
        "from": ["P3", "P4"]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 7,
      "task_type": "compute_meet",
      "operation": "meet",
      "inputs": ["L1", "L2"],
      "outputs": ["I"],
      "depends_on": [3, 6],
      "object_specs": {
        "name": "I",
        "type": "multivector",
        "from": ["L1", "L2"],
        "operator": "meet"
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "I",
            "type": "multivector",
            "color": "Yellow"
          }
        ]
      }
    }
  ]
}

Example for point reflection:
Input:
In conformal space, create a point P at (1,0,0), reflect it with respect to the y-z plane, and visualize the original point (red) and the reflected point (blue). I need Python code.

Expected JSON:
{
  "tasks": [
    {
      "task_id": 1,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P"],
      "depends_on": [],
      "object_specs": {
        "name": "P",
        "type": "point",
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P",
            "type": "point",
            "color": "Red"
          }
        ]
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_plane_from_point_and_normal",
      "operation": "plane_from_point_and_normal",
      "inputs": [],
      "outputs": ["Pi"],
      "depends_on": [],
      "object_specs": {
        "name": "Pi",
        "type": "plane",
        "point": [0, 0, 0],
        "normal": [1, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "reflect_cga_point",
      "operation": "reflect_point",
      "inputs": ["P", "Pi"],
      "outputs": ["P_reflected"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "P_reflected",
        "type": "point",
        "point": "P",
        "mirror": "Pi",
        "formula": "M v M"
      },
      "visualization": {
        "required": true,
        "objects": [
          {
            "name": "P_reflected",
            "type": "point",
            "color": "Blue"
          }
        ]
      }
    }
  ]
}

Example for distance:
Input:
In conformal space, create points P1(0,0,0) and P2(1,0,0), then calculate the distance between them. I need Python code.

Expected JSON:
{
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
        "coordinates": [0, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 2,
      "task_type": "construct_cga_point",
      "operation": "construct_point",
      "inputs": [],
      "outputs": ["P2"],
      "depends_on": [],
      "object_specs": {
        "name": "P2",
        "type": "point",
        "coordinates": [1, 0, 0]
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    },
    {
      "task_id": 3,
      "task_type": "compute_cga_point_distance",
      "operation": "point_distance",
      "inputs": ["P1", "P2"],
      "outputs": ["d2"],
      "depends_on": [1, 2],
      "object_specs": {
        "name": "d2",
        "type": "scalar",
        "from": ["P1", "P2"],
        "quantity": "squared_distance"
      },
      "visualization": {
        "required": false,
        "objects": []
      }
    }
  ]
}

User input:
{user_input}

JSON output:
""".strip()


LEGACY_TASK_DECOMPOSITION_TEMPLATE = TASK_DECOMPOSITION_TEMPLATE

TASK_DECOMPOSITION_TEMPLATE = """
Role:
You are a semantic task decomposition assistant for a geometric algebra code generation main graph.

Task:
Read the user input and decompose it into pure semantic IR tasks.

Output only pure JSON:
{{
  "tasks": [...]
}}

Each task must contain only:
- task_id
- task_type
- operation
- inputs
- outputs
- depends_on
- object_specs
- visualization

Global rules:
1. Do not extract function_name, target_language, or target_space.
2. Do not output code_to_optimize.
3. Do not output variable_assignments.
4. Do not output multivectors_to_be_visualized.
5. Do not output GAALOPScript.
6. Do not output formulas as executable code.
7. Split complex requests into minimal executable semantic tasks.
8. Every input symbol used by a task must be produced by an earlier task.
9. If the user defines A=e1 or B=e2, create construct_vector tasks first.
10. If the user defines P1(0,0,0), create construct_point tasks first.
11. Visualization belongs to the task that creates or computes the visualized object.
12. If visualization is requested but no color is specified, use Red.
13. If visualization is not requested, use:
    {{"required": false, "objects": []}}
14. Use ASCII-safe variable names. Replace Π with Pi.
15. Preserve user-specified colors.
16. Do not use geometric_product for reflection requests; use reflect_point.
17. Use rotate_object for rotation requests on an existing geometric object.
18. If the user asks to create or construct a rotor, use construct_rotor.
19. Do not split rotor construction into separate sqrt, cos, sin, or normalization tasks.
20. Do not output rotate_circle, rotate_line, or rotate_point; use rotate_object instead.
21. If the user writes C', L', or P', normalize them to C_rotated, L_rotated, or P_rotated.
22. construct_point is only for coordinate-based point creation. Do not use construct_point for computed points from formulas.
23. For three-sphere intersection tasks, use construct_sphere for S1/S2/S3, outer_product for M=S1^S2^S3, dual for P=*M, and point_pair_decomposition for X4/X5 from P.
24. e0, e1, e2, e3, and einf are built-in CGA basis symbols and do not need their own definition tasks.
25. If X4 and X5 are computed from a point pair, put their visualization on the point_pair_decomposition task.
26. Numeric constants and angle constants are not input symbols; for construct_rotor, put axis and angle in object_specs and keep inputs as [].
27. Do not create standalone visualization tasks.
28. Visualization must be attached to the task that creates or computes the visualized object.
29. If the user says "visualize them as a red point set" or "visualize all points as red", attach Red visualization to each point task.
30. If the user says "visualize P1, P2, P3 as red", each corresponding construct_point task should carry its own Red visualization object.
31. Never output task_type="visualization" or operation="visualization".
32. If the user asks for a midpoint / middle point / mid point / 中点, use operation="midpoint".
33. midpoint is a computed point, so do not use construct_point for the midpoint result.
34. construct_point is only for explicitly coordinate-defined points.
35. If the user writes M=(P1+P2)/2, generate one midpoint task with outputs defaulting to ["M"].
36. If midpoint visualization is requested, attach it to the midpoint task.
37. If the user explicitly creates a rotor R and writes P'=R P ~R, use rotate_object with inputs [P, R] after construct_rotor defines R.
38. Do not force explicit rotor rotation into axis-angle rotate_object mode.
  
  Available operation rules:
  {operation_rules}

User input:
{user_input}

JSON output:
""".strip()


ALLOWED_LANGUAGES = {
    "CLUCALC",
    "JULIA",
    "VERILOG",
    "GAPP_DEBUGGER",
    "CSHARP",
    "RUST",
    "JAVA",
    "VIS2D",
    "GAALET_OUTPUT",
    "GAPP",
    "COMPRESSED",
    "VISUALIZER",
    "GANJA",
    "GAPP_OPENCL",
    "PYTHON",
    "MATLAB",
    "DOT",
    "MATHEMATICA",
    "CPP",
    "LATEX",
}


def build_information_extraction_prompt(user_input: str) -> str:
    return INFORMATION_EXTRACTION_TEMPLATE.replace("{user_input}", user_input)


def select_relevant_operations(user_input: str) -> list[str]:
    text = str(user_input or "")
    lower_text = text.lower()
    selected: list[str] = []

    def add_operation(name: str) -> None:
        normalized_name = normalize_operation_alias(name)
        if normalized_name in OPERATION_SPECS and normalized_name not in selected and len(selected) < 10:
            selected.append(normalized_name)

    add_operation("construct_point")

    if "distance" in lower_text:
        add_operation("point_distance")

    if (
        "midpoint" in lower_text
        or "mid point" in lower_text
        or "middle point" in lower_text
        or "center point between" in lower_text
        or "average point" in lower_text
        or "中点" in text
        or "m = (p1 + p2) / 2" in lower_text
        or "(p1 + p2)/2" in lower_text
        or "(p_{1} + p_{2})/2" in lower_text
    ):
        add_operation("midpoint")
        add_operation("construct_point")

    if (
        "line" in lower_text
        or "two points" in lower_text
        or "passing through" in lower_text
        or re.search(r"\bl\d+\b", lower_text)
    ):
        add_operation("line_from_two_points")

    if "sphere" in lower_text:
        add_operation("construct_sphere")
        add_operation("outer_product")

    if "circle" in lower_text:
        add_operation("circle_from_three_points")

    if (
        "two intersection points" in lower_text
        or "intersection points" in lower_text
        or "x4 and x5" in lower_text
        or "x4, x5" in lower_text
        or "point pair" in lower_text
        or "point pair decomposition" in lower_text
        or "decompose point pair" in lower_text
        or "three spheres" in lower_text
        or "sphere intersection" in lower_text
        or "sqrt(p" in lower_text
        or "p_{" in lower_text
        or "p_pm" in lower_text
        or "p±" in text
    ):
        add_operation("point_pair_decomposition")
        add_operation("construct_sphere")
        add_operation("outer_product")
        add_operation("dual")

    if (
        "plane" in lower_text
        or "\u03a0" in text
        or re.search(r"\bpi\b", lower_text)
    ):
        add_operation("plane_from_three_points")
        add_operation("plane_from_point_and_normal")

    if (
        "vector" in lower_text
        or re.search(r"\be[123]\b", lower_text)
        or re.search(r"\b[a-z]\s*=", lower_text)
    ):
        add_operation("construct_vector")

    if "geometric product" in lower_text:
        add_operation("construct_vector")
        add_operation("geometric_product")

    if (
        "outer product" in lower_text
        or "wedge product" in lower_text
        or "\u2227" in text
        or re.search(r"\b[A-Za-z][A-Za-z0-9_]*\s*\^\s*[A-Za-z][A-Za-z0-9_]*\b", text)
    ):
        add_operation("outer_product")

    if (
        "inner product" in lower_text
        or "dot product" in lower_text
        or "\u00b7" in text
        or "\u22c5" in text
        or re.search(r"\b[A-Z][A-Za-z0-9_]*\s*\.\s*[A-Z][A-Za-z0-9_]*\b", text)
    ):
        add_operation("inner_product")

    if "norm" in lower_text or "||" in text or "sqrt" in lower_text:
        add_operation("norm")
        add_operation("inner_product")

    if (
        "dual" in lower_text
        or re.search(r"\*[A-Za-z]", text)
        or re.search(r"[A-Za-z][A-Za-z0-9_]*\^\*", text)
        or "i^{-1}" in lower_text
    ):
        add_operation("dual")

    if "meet" in lower_text or "intersection" in lower_text or "intersect" in lower_text:
        add_operation("meet")
        add_operation("outer_product")
        add_operation("dual")

    if "reflect" in lower_text or "reflected" in lower_text or "mirror" in lower_text:
        add_operation("reflect_point")
        add_operation("plane_from_point_and_normal")

    rotor_construction_request = (
        "create a rotor" in lower_text
        or "construct a rotor" in lower_text
        or (
            "rotor" in lower_text
            and (
                "rotation axis" in lower_text
                or "rotation angle" in lower_text
                or "theta/2" in lower_text
                or "r = cos" in lower_text
                or "cos(45" in lower_text
                or "sin(45" in lower_text
                or "e23" in lower_text
                or "e31" in lower_text
                or "e12" in lower_text
            )
        )
    )
    if rotor_construction_request:
        add_operation("construct_rotor")

    explicit_rotor_rotation_request = (
        "~r" in lower_text
        or "\\tilde{r}" in lower_text
        or "p'=r p" in lower_text
        or "p' = r p" in lower_text
        or "r p" in lower_text
        or "rotation operation" in lower_text
    )

    if (
        "rotate" in lower_text
        or "rotated" in lower_text
        or ("rotation" in lower_text and not rotor_construction_request)
        or explicit_rotor_rotation_request
        or "around the axis" in lower_text
        or "around x-axis" in lower_text
        or "around y-axis" in lower_text
        or "around z-axis" in lower_text
    ):
        add_operation("rotate_object")
        if "circle" in lower_text:
            add_operation("circle_from_three_points")
        if "line" in lower_text:
            add_operation("line_from_two_points")
        if "sphere" in lower_text:
            add_operation("construct_sphere")
        add_operation("construct_vector")

    if rotor_construction_request and explicit_rotor_rotation_request:
        add_operation("construct_rotor")
        add_operation("rotate_object")

    if len(selected) <= 1:
        return [
            "construct_point",
            "line_from_two_points",
            "construct_sphere",
            "circle_from_three_points",
            "outer_product",
        ]

    return [name for name in selected[:10] if name in OPERATION_REGISTRY]


def _build_task_decomposition_operation_rules(selected_operations: list[str]) -> str:
    rules = [
        get_prompt_spec_for_operation(name).strip()
        for name in selected_operations
        if get_prompt_spec_for_operation(name).strip()
    ]
    if not rules:
        fallback_operations = [
            "construct_point",
            "line_from_two_points",
            "construct_sphere",
            "circle_from_three_points",
            "outer_product",
        ]
        rules = [OPERATION_SPECS[name].strip() for name in fallback_operations if name in OPERATION_SPECS]
    return "\n\n".join(rules)


def build_task_decomposition_prompt(
    user_input: str,
    selected_operations: list[str] | None = None,
) -> str:
    selected = selected_operations or select_relevant_operations(user_input)
    operation_rules = _build_task_decomposition_operation_rules(selected)
    return TASK_DECOMPOSITION_TEMPLATE.format(
        user_input=user_input,
        operation_rules=operation_rules,
    )


def build_task_decomposition_repair_prompt(
    *,
    user_input: str,
    invalid_task_blocks_result: dict,
    validation_errors: list[str],
    selected_operations: list[str],
    operation_rules: str,
) -> str:
    invalid_json_text = json.dumps(
        invalid_task_blocks_result,
        ensure_ascii=False,
        indent=2,
    )
    error_text = "\n".join(str(error).strip() for error in validation_errors if str(error).strip())
    selected_operations_text = ", ".join(selected_operations) if selected_operations else "(none)"
    return TASK_DECOMPOSITION_REPAIR_TEMPLATE.format(
        user_input=user_input,
        selected_operations=selected_operations_text,
        operation_rules=operation_rules,
        validation_errors=error_text or "(no validator errors provided)",
        invalid_task_blocks_result=invalid_json_text,
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").replace("```json", "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("no json object found")
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("parsed result is not a json object")
    return parsed


def is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text or "readtimeout" in text


def invoke_llm_with_retry(
    llm,
    prompt: str,
    *,
    node_name: str = "llm",
    max_retries: int = 3,
    base_sleep_seconds: float = 2.0,
):
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if not is_timeout_error(exc):
                raise
            last_error = exc
            if attempt >= max_retries - 1:
                raise
            sleep_seconds = base_sleep_seconds * (2 ** attempt)
            print(
                f"[{node_name}] LLM timeout, retry {attempt + 1}/{max_retries} after {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


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


def _extract_explicit_function_name(user_input: str) -> str | None:
    patterns = [
        r"function\s+name\s*(?:is|=|:)?\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"function\s+called\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"函数名称(?:为|是)?[:：]?\s*([A-Za-z_][A-Za-z0-9_]*)",
        r"函数名(?:称)?(?:为|是)?[:：]?\s*([A-Za-z_][A-Za-z0-9_]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _to_camel_case(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return "generatedFunction"
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def _infer_meaningful_function_name(user_input: str) -> str:
    lower_text = user_input.lower()
    if "distance" in lower_text and "point" in lower_text:
        return "CalculateDistanceBetweenPoints"
    if "intersection" in lower_text or "交集" in user_input:
        return "CalculateIntersection"
    if "sphere" in lower_text or "球" in user_input:
        return "CreateSphere"
    if "line" in lower_text or "直线" in user_input:
        return "CreateLine"
    if "circle" in lower_text or "圆" in user_input:
        return "CreateCircle"
    if "point" in lower_text or "点" in user_input:
        return "CreatePoint"
    return "generatedFunction"


def _normalize_function_name(value: Any, user_input: str) -> str:
    text = str(value or "").strip()
    if not text:
        explicit_name = _extract_explicit_function_name(user_input)
        if explicit_name:
            return explicit_name
        return _infer_meaningful_function_name(user_input)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    camel = _to_camel_case(text)
    return camel or "generatedFunction"


def _infer_target_language(user_input: str) -> str:
    upper_text = user_input.upper()
    if "JAVA" in upper_text:
        return "JAVA"
    if "PYTHON" in upper_text:
        return "PYTHON"
    if "C++" in user_input or "CPP" in upper_text:
        return "CPP"
    if "MATLAB" in upper_text:
        return "MATLAB"
    return "PYTHON"


def _normalize_target_language(value: Any, user_input: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return _infer_target_language(user_input)
    if text in {"JAVA", "PYTHON", "CPP", "MATLAB"}:
        return text
    if text in ALLOWED_LANGUAGES:
        return text
    if "JAVA" in text:
        return "JAVA"
    if "PYTHON" in text:
        return "PYTHON"
    return _infer_target_language(user_input)


def _infer_target_space(user_input: str) -> str:
    lower_text = user_input.lower()
    if "cga" in lower_text or "conformal space" in lower_text or "conformal" in lower_text or "共形空间" in user_input:
        return "ALGEBRA_CGA"
    if "pga" in lower_text or "projective" in lower_text:
        if "2d" in lower_text or "二维" in user_input:
            return "ALGEBRA_2D_PGA"
        if "3d" in lower_text or "三维" in user_input:
            return "ALGEBRA_3D_PGA"
        return "unknown"
    if "ega" in lower_text or "euclidean" in lower_text:
        return "ALGEBRA_3D"
    return "unknown"


def _normalize_target_space(value: Any, user_input: str) -> str:
    text = str(value or "").strip()
    upper_text = text.upper()
    if not text:
        return _infer_target_space(user_input)
    if upper_text == "CGA" or "CONFORMAL" in upper_text:
        return "ALGEBRA_CGA"
    if upper_text == "ALGEBRA_CGA":
        return "ALGEBRA_CGA"
    if upper_text in {"ALGEBRA_2D_PGA", "ALGEBRA_3D_PGA", "ALGEBRA_3D", "UNKNOWN"}:
        return upper_text.lower() if upper_text == "UNKNOWN" else upper_text
    if upper_text == "PGA":
        return _infer_target_space(user_input)
    return _infer_target_space(user_input)


def _rule_based_extract_information(user_input: str) -> dict[str, str]:
    return {
        "function_name": _normalize_function_name("", user_input),
        "target_language": _infer_target_language(user_input),
        "target_space": _infer_target_space(user_input),
    }


def information_extraction_node(state: MainGraphState) -> dict[str, Any]:
    user_input = state.get("user_input", "")
    raw_output = ""

    if not user_input.strip():
        raise ValueError("information_extraction_node requires non-empty user_input")

    llm = _get_default_llm()
    prompt = build_information_extraction_prompt(user_input=user_input)
    _debug_print("--- Node: Information Extraction ---")
    _debug_print("Information extraction prompt:")
    _debug_print(prompt)
    result = invoke_llm_with_retry(
        llm,
        prompt,
        node_name="information_extraction_node",
        max_retries=3,
        base_sleep_seconds=2.0,
    )
    raw_output = getattr(result, "content", str(result))
    _debug_print("Raw information extraction output:")
    _debug_print(raw_output)
    parsed = parse_json_object(raw_output)
    return {
        "function_name": _normalize_function_name(parsed.get("function_name"), user_input),
        "target_language": _normalize_target_language(parsed.get("target_language"), user_input),
        "target_space": _normalize_target_space(parsed.get("target_space"), user_input),
        "information_extraction_raw": raw_output,
    }


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
    mapping = {
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
        "midpoint": "compute_midpoint",
        "compute_midpoint": "compute_midpoint",
        "middle_point": "compute_midpoint",
        "mid_point": "compute_midpoint",
        "geometric_product": "compute_geometric_product",
        "compute_geometric_product": "compute_geometric_product",
        "outer_product": "compute_outer_product",
        "wedge_product": "compute_outer_product",
        "inner_product": "compute_inner_product",
        "dot_product": "compute_inner_product",
        "norm": "compute_norm",
        "dual": "compute_dual",
        "meet": "compute_meet",
        "intersection": "compute_meet",
        "line_intersection": "compute_meet",
        "construct_circle": "construct_cga_circle_from_three_points",
        "construct_cga_circle_from_three_points": "construct_cga_circle_from_three_points",
        "construct_sphere": "construct_cga_sphere",
        "construct_cga_sphere": "construct_cga_sphere",
        "compute_cga_point_distance": "compute_cga_point_distance",
        "compute_outer_product": "compute_outer_product",
        "compute_inner_product": "compute_inner_product",
          "compute_norm": "compute_norm",
          "compute_dual": "compute_dual",
          "compute_meet": "compute_meet",
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
    return mapping.get(text, text)


def _normalize_operation(value: Any, task_type: str) -> str:
    text = str(value or "").strip()
    if text:
        alias_mapping = {
            "construct_point": "construct_point",
            "construct_line": "line_from_two_points",
            "construct_cga_line_from_two_points": "line_from_two_points",
            "construct_vector": "construct_vector",
            "construct_plane": "plane_from_three_points",
            "construct_cga_plane_from_three_points": "plane_from_three_points",
            "construct_plane_from_point_and_normal": "plane_from_point_and_normal",
            "construct_cga_plane_from_point_and_normal": "plane_from_point_and_normal",
            "reflect_point": "reflect_point",
        "reflect_cga_point": "reflect_point",
        "reflection": "reflect_point",
            "reflect": "reflect_point",
            "point_reflection": "reflect_point",
            "rotate_object": "rotate_object",
            "rotate_circle": "rotate_object",
            "rotate_line": "rotate_object",
            "rotate_point": "rotate_object",
            "rotate_sphere": "rotate_object",
            "rotation": "rotate_object",
            "rotate_cga_object": "rotate_object",
            "construct_rotor": "construct_rotor",
            "create_rotor": "construct_rotor",
            "rotor": "construct_rotor",
            "build_rotor": "construct_rotor",
            "midpoint": "midpoint",
            "compute_midpoint": "midpoint",
            "middle_point": "midpoint",
            "mid_point": "midpoint",
        "geometric_product": "geometric_product",
            "compute_geometric_product": "geometric_product",
            "outer_product": "outer_product",
            "wedge_product": "outer_product",
            "compute_outer_product": "outer_product",
            "inner_product": "inner_product",
            "dot_product": "inner_product",
            "compute_inner_product": "inner_product",
            "norm": "norm",
            "compute_norm": "norm",
              "dual": "dual",
              "compute_dual": "dual",
              "meet": "meet",
              "intersection": "meet",
              "line_intersection": "meet",
              "compute_meet": "meet",
              "point_pair_decomposition": "point_pair_decomposition",
              "decompose_point_pair": "point_pair_decomposition",
              "point_pair_decompose": "point_pair_decomposition",
              "extract_point_pair": "point_pair_decomposition",
              "split_point_pair": "point_pair_decomposition",
              "point_pair_to_points": "point_pair_decomposition",
              "compute_intersection_points": "point_pair_decomposition",
              "decompose_cga_point_pair": "point_pair_decomposition",
              "circle_from_three_points": "circle_from_three_points",
              "construct_cga_circle_from_three_points": "circle_from_three_points",
              "construct_sphere": "construct_sphere",
              "construct_cga_sphere": "construct_sphere",
              "compute_cga_point_distance": "point_distance",
            "compute_point_distance": "point_distance",
        }
        return alias_mapping.get(text, text)

    mapping = {
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
          "decompose_point_pair": "point_pair_decomposition",
      }
    return mapping.get(task_type, "unknown")


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


def _normalize_plane_symbol_legacy(value: Any, fallback: str = "Pi") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    normalized = (
        text.replace("\\Pi", "Pi")
        .replace("Π", "Pi")
        .replace("π", "Pi")
        .replace("螤", "Pi")
    )
    if normalized.lower() == "pi":
        return "Pi"
    return normalized


def _normalize_plane_symbol(value: Any, fallback: str = "Pi") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback

    normalized = text
    for token in ("\\Pi", "\\pi", "\u03A0", "\u03C0"):
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

    match = re.fullmatch(r"([A-Za-z0-9_]+)[\'′]+", text)
    if match:
        return f"{match.group(1)}_rotated"

    normalized = text.replace("′", "_rotated").replace("'", "_rotated")
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


def _infer_angle_unit(angle: Any, angle_unit: Any) -> str:
    explicit = str(angle_unit or "").strip().lower()
    if explicit in {"degree", "degrees", "deg"}:
        return "degree"
    if explicit in {"radian", "radians", "rad"}:
        return "radian"

    text = str(angle or "").strip().lower()
    if any(token in text for token in ("degree", "degrees", "deg", "°", "掳", "ёу")):
        return "degree"
    if "pi" in text or "π" in text or "蟺" in text or "іа" in text:
        return "radian"
    return "radian"


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
    normalized = (
        text.lower()
        .replace("π", "pi")
        .replace("蟺", "pi")
        .replace("іа", "pi")
    )
    for token in ("degrees", "degree", "deg", "radians", "radian", "rad", "°", "掳", "ёу"):
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


def _derive_point_coordinates(index: int, variable_assignments: dict[str, Any]) -> list[Any] | None:
    keys = [f"a{index}", f"b{index}", f"c{index}"]
    coordinates = [_normalize_coordinate_value(variable_assignments.get(key)) for key in keys]
    if any(value is None for value in coordinates):
        return None
    return coordinates


def _build_point_formula(output_symbol: str, index: int) -> str:
    a_name = f"a{index}"
    b_name = f"b{index}"
    c_name = f"c{index}"
    return (
        f"{output_symbol} = {a_name}*e1 + {b_name}*e2 + {c_name}*e3 + "
        f"0.5*({a_name}*{a_name} + {b_name}*{b_name} + {c_name}*{c_name})*einf + e0"
    )


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


def normalize_ir_visualization_block(task: dict) -> dict:
    visualization = task.get("visualization")
    if not isinstance(visualization, dict):
        visualization = task.get("multivectors_to_be_visualized")
    normalized_visualization = apply_default_visualization_color(visualization)
    required = bool(normalized_visualization.get("required"))
    normalized_objects = normalized_visualization.get("objects") if isinstance(normalized_visualization.get("objects"), list) else []
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

    visualization = normalize_ir_visualization_block(task)
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
    visualization = normalize_ir_visualization_block(task)
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
        target_visualization = target_task.get("visualization") if isinstance(target_task.get("visualization"), dict) else {
            "required": False,
            "objects": [],
        }
        target_objects = target_visualization.get("objects") if isinstance(target_visualization.get("objects"), list) else []
        target_object_specs = target_task.get("object_specs") if isinstance(target_task.get("object_specs"), dict) else {}
        incoming_type = str(obj.get("type") or "").strip() or str(target_object_specs.get("type") or "").strip() or "multivector"
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
    return lower_text in {"p_plus", "p_minus", "x_plus", "x_minus"} or "plus" in lower_text or "minus" in lower_text or "pm" in lower_text


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
            related_visualization = normalize_ir_visualization_block(related_task)
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


def legacy_normalize_task_blocks_result(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("task decomposition result must be a dict")

    tasks = parsed.get("tasks")
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

        task_type = _normalize_task_type(task.get("task_type"))
        operation = _normalize_operation(task.get("operation"), task_type)
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
        if (operation == "point_distance" or task_type == "compute_cga_point_distance") and not outputs:
            outputs = ["d2"]
        if (
            operation in {"midpoint", "compute_midpoint", "middle_point", "mid_point"}
            or task_type == "compute_midpoint"
        ) and not outputs:
            inferred_output = str(object_specs.get("name") or "M").strip() or "M"
            outputs = [inferred_output]
        if (operation == "construct_vector" or task_type == "construct_vector") and not outputs:
            inferred_output = str(object_specs.get("name") or "").strip()
            if inferred_output:
                outputs = [inferred_output]
        if (
            operation in {"reflect_point", "reflection", "reflect", "point_reflection"}
            or task_type == "reflect_cga_point"
        ) and not outputs:
            outputs = [_default_reflect_point_output(inputs)]
        if (operation == "geometric_product" or task_type == "compute_geometric_product") and not outputs:
            outputs = ["G"]
        if (operation == "outer_product" or task_type == "compute_outer_product") and not outputs:
            outputs = ["M"]
        if (operation == "inner_product" or task_type == "compute_inner_product") and not outputs:
            outputs = ["IP"]
        if (operation == "norm" or task_type == "compute_norm") and not outputs:
            if len(inputs) == 1 and str(inputs[0]).strip():
                outputs = [f"Norm{str(inputs[0]).strip()}"]
            else:
                outputs = ["Norm"]
        if (operation == "dual" or task_type == "compute_dual") and not outputs:
            if len(inputs) == 1 and str(inputs[0]).strip():
                outputs = [f"Dual{str(inputs[0]).strip()}"]
            else:
                outputs = ["DualResult"]
        if (
            operation in {"construct_rotor", "create_rotor", "rotor", "build_rotor"}
            or task_type == "construct_rotor"
        ) and not outputs:
            inferred_output = str(object_specs.get("name") or "R").strip() or "R"
            outputs = [_sanitize_symbol_for_identifier(inferred_output)]
        if (
            operation in {
                "point_pair_decomposition",
                "decompose_point_pair",
                "point_pair_decompose",
                "extract_point_pair",
                "split_point_pair",
                "point_pair_to_points",
                "compute_intersection_points",
            }
            or task_type == "decompose_cga_point_pair"
        ) and not outputs:
            outputs = ["X4", "X5"]
        if (
            operation in {"meet", "intersection", "line_intersection"}
            or task_type == "compute_meet"
        ) and not outputs:
            outputs = ["I"]
        if (
            operation in {"rotate_object", "rotate_circle", "rotate_line", "rotate_point", "rotate_sphere", "rotation"}
            or task_type == "rotate_cga_object"
        ):
            object_symbol = str(object_specs.get("object") or "").strip()
            rotor_symbol = str(object_specs.get("rotor") or "").strip()
            if not inputs and object_symbol and rotor_symbol:
                inputs = [object_symbol, rotor_symbol]
            elif not inputs and object_symbol:
                inputs = [object_symbol]
            if not outputs:
                outputs = [_default_rotated_output_name([object_symbol or (inputs[0] if inputs else "")])]
        if (operation == "construct_sphere" or task_type == "construct_cga_sphere") and not outputs:
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
        visualization = normalize_ir_visualization_block(task)

        normalized_task = {
            "task_id": index,
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
            task_type = "construct_cga_plane_from_point_and_normal"
            normalized_task["task_type"] = task_type
            normalized_task["operation"] = "plane_from_point_and_normal"

        output_symbol = str(outputs[0]).strip()
        point_index = _extract_point_index(output_symbol) if task_type == "construct_cga_point" else None
        if point_index is not None:
            coordinates = _normalize_coordinates_list(object_specs.get("coordinates"))
            normalized_task["operation"] = "construct_point"
            normalized_task["inputs"] = []
            normalized_task["outputs"] = [output_symbol]
            normalized_task["depends_on"] = []
            normalized_task["object_specs"] = {
                "name": output_symbol,
                "type": "point",
                "coordinates": coordinates if coordinates is not None else [],
            }
        elif normalized_task["operation"] == "construct_vector" or task_type == "construct_vector":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "geometric_product" or task_type == "compute_geometric_product":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "midpoint" or task_type == "compute_midpoint":
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
            midpoint_visualization = normalize_ir_visualization_block(task)
            midpoint_required = bool(midpoint_visualization.get("required"))
            midpoint_objects = midpoint_visualization.get("objects") if isinstance(midpoint_visualization.get("objects"), list) else []
            if midpoint_required and not midpoint_objects and midpoint_output:
                midpoint_objects = [
                    {
                        "name": midpoint_output,
                        "type": "point",
                        "color": "Red",
                    }
                ]
            normalized_task["visualization"] = {
                "required": midpoint_required,
                "objects": midpoint_objects,
            }
        elif normalized_task["operation"] == "outer_product" or task_type == "compute_outer_product":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "inner_product" or task_type == "compute_inner_product":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "norm" or task_type == "compute_norm":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "dual" or task_type == "compute_dual":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "point_pair_decomposition" or task_type == "decompose_cga_point_pair":
              normalized_task["task_type"] = "decompose_cga_point_pair"
              normalized_task["operation"] = "point_pair_decomposition"
              point_pair_symbol = str(object_specs.get("point_pair") or "").strip()
              if not point_pair_symbol and inputs:
                  point_pair_symbol = str(inputs[0]).strip()
              if not normalized_task["inputs"] and point_pair_symbol:
                  normalized_task["inputs"] = [point_pair_symbol]

              pair_outputs = [str(symbol).strip() for symbol in normalized_task["outputs"] if str(symbol).strip()]
              pair_visualization = normalize_ir_visualization_block(task)
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
                  "formula": str(object_specs.get("formula") or "X_pm = (P ± sqrt(P.P)) / (einf.P)").strip() or "X_pm = (P ± sqrt(P.P)) / (einf.P)",
              }
              pair_visualization_required = bool(pair_visualization.get("required"))
              pair_visualization_objects = pair_visualization.get("objects") if isinstance(pair_visualization.get("objects"), list) else []
              if pair_visualization_required and not pair_visualization_objects:
                  pair_visualization_objects = [
                      {"name": normalized_task["outputs"][0], "type": "point", "color": "Yellow"},
                      {"name": normalized_task["outputs"][1], "type": "point", "color": "Yellow"},
                  ]
              normalized_task["visualization"] = {
                  "required": pair_visualization_required,
                  "objects": pair_visualization_objects,
              }
        elif normalized_task["operation"] == "meet" or task_type == "compute_meet":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "reflect_point" or task_type == "reflect_cga_point":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "rotate_object" or task_type == "rotate_cga_object":
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
                else _default_rotated_output_name([object_symbol or (normalized_task["inputs"][0] if normalized_task["inputs"] else "")])
            )
            normalized_task["outputs"] = [rotate_output]

            inferred_type = str(object_specs.get("type") or "").strip()
            original_operation = str(task.get("operation") or "").strip()
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
                normalized_task["object_specs"]["angle_unit"] = _infer_angle_unit(object_specs.get("angle"), object_specs.get("angle_unit"))
            if "axis_point" in object_specs:
                normalized_task["object_specs"]["axis_point"] = object_specs.get("axis_point")
            rotate_visualization = normalize_ir_visualization_block(task)
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
        elif normalized_task["operation"] == "construct_rotor" or task_type == "construct_rotor":
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
            rotor_visualization = normalize_ir_visualization_block(task)
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
        elif normalized_task["operation"] == "line_from_two_points" or task_type == "construct_cga_line_from_two_points":
            normalized_task["task_type"] = "construct_cga_line_from_two_points"
            normalized_task["operation"] = "line_from_two_points"
            normalized_task["object_specs"] = {
                "name": str(object_specs.get("name") or output_symbol).strip() or output_symbol,
                "type": str(object_specs.get("type") or "line").strip() or "line",
                "from": _normalize_symbol_list(object_specs.get("from")) or inputs[:2],
            }
        elif normalized_task["operation"] == "point_distance" or task_type == "compute_cga_point_distance":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "circle_from_three_points" or task_type == "construct_cga_circle_from_three_points":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)
        elif normalized_task["operation"] == "plane_from_three_points" or task_type == "construct_cga_plane_from_three_points":
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
            plane_visualization = normalize_ir_visualization_block(task)
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
        elif normalized_task["operation"] == "plane_from_point_and_normal" or task_type == "construct_cga_plane_from_point_and_normal":
            normalized_task["task_type"] = "construct_cga_plane_from_point_and_normal"
            normalized_task["operation"] = "plane_from_point_and_normal"
            plane_output = str(normalized_task["outputs"][0]).strip() if normalized_task["outputs"] else ""
            plane_output = _normalize_plane_symbol(plane_output or object_specs.get("name") or "Pi", fallback="Pi")
            normalized_task["outputs"] = [plane_output]
            point = _normalize_coordinates_list(object_specs.get("point"))
            normal = _normalize_coordinates_list(object_specs.get("normal"))
            normalized_task["object_specs"] = {
                "name": _normalize_plane_symbol(object_specs.get("name") or plane_output, fallback=plane_output),
                "type": str(object_specs.get("type") or "plane").strip() or "plane",
                "point": point if point is not None else object_specs.get("point"),
                "normal": normal if normal is not None else object_specs.get("normal"),
            }
            plane_visualization = normalize_ir_visualization_block(task)
            plane_objects = []
            for obj in plane_visualization.get("objects", []):
                if isinstance(obj, dict):
                    updated_obj = deepcopy(obj)
                    if str(updated_obj.get("name") or "").strip() in {"\u03A0", "\u03C0", "\\Pi", "\\pi"}:
                        updated_obj["name"] = plane_output
                    if str(updated_obj.get("type") or "").strip() == "plane":
                        updated_obj["name"] = plane_output
                    plane_objects.append(updated_obj)
            normalized_task["visualization"] = {
                "required": bool(plane_visualization.get("required")),
                "objects": plane_objects,
            }
        elif normalized_task["operation"] == "construct_sphere" or task_type == "construct_cga_sphere":
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
            normalized_task["visualization"] = normalize_ir_visualization_block(task)

        normalized_tasks.append(normalized_task)

    if not normalized_tasks:
        raise ValueError("task decomposition result has no valid tasks")

    normalized_tasks = merge_visualization_only_tasks(normalized_tasks, visualization_only_tasks)
    normalized_tasks = _merge_computed_point_pair_construct_point_tasks(normalized_tasks)

    return {
        "tasks": normalized_tasks,
    }


def normalize_task_blocks_result(parsed: dict) -> dict:
    from ga_visagent.main_graph.normalizers import operation_aware_normalize_task_blocks_result

    try:
        return operation_aware_normalize_task_blocks_result(parsed)
    except Exception as exc:
        if USE_LEGACY_NORMALIZER_FALLBACK:
            print(
                "[normalize_task_blocks_result] operation-aware failed, "
                f"fallback to legacy: {exc}"
            )
            return legacy_normalize_task_blocks_result(parsed)
        raise


def task_decomposition_node(state: MainGraphState) -> dict:
    user_input = state.get("user_input", "")
    if not user_input:
        raise ValueError("task_decomposition_node requires non-empty user_input")

    selected_operations = select_relevant_operations(user_input)
    operation_rules = _build_task_decomposition_operation_rules(selected_operations)
    prompt = build_task_decomposition_prompt(user_input=user_input, selected_operations=selected_operations)

    _debug_print("--- Node: Task Decomposition ---")
    if DEBUG_PROMPT:
        print(f"[task_decomposition_node] Selected operations: {selected_operations}")
        print(f"[task_decomposition_node] Task decomposition prompt length: {len(prompt)} chars")

    llm = _get_default_llm()
    for attempt in range(MAX_TASK_DECOMPOSITION_SEMANTIC_RETRIES + 1):
        response = invoke_llm_with_retry(
            llm,
            prompt,
            node_name="task_decomposition_node",
            max_retries=3,
            base_sleep_seconds=2.0,
        )
        raw_output = getattr(response, "content", str(response))

        _debug_print("Raw task decomposition output:")
        _debug_print(raw_output)

        if not str(raw_output).strip():
            raise ValueError("task_decomposition_node LLM output is empty")

        try:
            parsed = parse_json_object(str(raw_output))
        except Exception as exc:
            raise ValueError(f"task_decomposition_node JSON parse failed: {exc}") from exc

        normalized = normalize_task_blocks_result(parsed)
        validation_result = validate_task_blocks_result(
            normalized,
            raise_on_error=False,
        )
        if validation_result["valid"]:
            if DEBUG_SEMANTIC_RETRY and attempt > 0:
                print(
                    f"[task_decomposition_node] semantic retry succeeded after {attempt} repair(s)."
                )
            return {
                "task_blocks_result": normalized,
                "task_decomposition_validation_result": validation_result,
                "task_decomposition_semantic_retry_count": attempt,
            }

        if attempt >= MAX_TASK_DECOMPOSITION_SEMANTIC_RETRIES:
            raise ValueError(
                "task_decomposition_node semantic retry failed after "
                f"{MAX_TASK_DECOMPOSITION_SEMANTIC_RETRIES} retries:\n"
                + "\n".join(validation_result["errors"])
            )

        print(
            "[task_decomposition_node] semantic retry "
            f"{attempt + 1}/{MAX_TASK_DECOMPOSITION_SEMANTIC_RETRIES} because validation failed."
        )
        if DEBUG_SEMANTIC_RETRY:
            print("[task_decomposition_node] errors:")
            for error in validation_result["errors"]:
                print(f"- {error}")

        prompt = build_task_decomposition_repair_prompt(
            user_input=user_input,
            invalid_task_blocks_result=normalized,
            validation_errors=validation_result["errors"],
            selected_operations=selected_operations,
            operation_rules=operation_rules,
        )

    raise AssertionError("unreachable")


def _parse_task_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return None


def _infer_symbol_type_from_task(task: dict[str, Any]) -> str:
    object_specs = task.get("object_specs")
    if isinstance(object_specs, dict):
        object_type = str(object_specs.get("type") or "").strip()
        if object_type:
            return object_type

    operation = normalize_operation_alias(str(task.get("operation") or "").strip())
    return get_default_output_type(operation) or "multivector"


def _infer_output_symbol_type(task: dict[str, Any], symbol_table: dict[str, dict[str, Any]]) -> str:
    operation = normalize_operation_alias(str(task.get("operation") or "").strip())
    if operation == "point_pair_decomposition":
        return "point"

    object_specs = task.get("object_specs")
    if isinstance(object_specs, dict):
        object_type = str(object_specs.get("type") or "").strip()
        if object_type == "point_pair_decomposition":
            return "point"
        if object_type:
            return object_type

    if operation == "rotate_object":
        inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
        if inputs:
            input_symbol = str(inputs[0]).strip()
            input_type = str(symbol_table.get(input_symbol, {}).get("type") or "").strip()
            if input_type:
                return input_type

    return _infer_symbol_type_from_task(task)


def validate_task_blocks_result(
    task_blocks_result: dict,
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    tasks = task_blocks_result.get("tasks", []) if isinstance(task_blocks_result, dict) else []
    errors: list[str] = []
    warnings: list[str] = []
    symbol_table: dict[str, dict[str, Any]] = {}
    defined_symbols_before_current: set[str] = set()

    if not isinstance(tasks, list) or not tasks:
        errors.append("task_ir_validator_node requires non-empty tasks")

    required_fields = [
        "task_id",
        "task_type",
        "operation",
        "inputs",
        "outputs",
        "depends_on",
        "object_specs",
        "visualization",
    ]

    parsed_tasks: list[tuple[int, dict[str, Any]]] = []
    seen_task_ids: set[int] = set()

    if isinstance(tasks, list):
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                errors.append(f"Task at index {index} is not a dict")
                continue

            task_id_value = task.get("task_id")
            task_id = _parse_task_id(task_id_value)
            if task_id is None:
                errors.append(f"Task at index {index} missing required field: task_id")
                continue
            if task_id in seen_task_ids:
                errors.append(f"Duplicate task_id: {task_id}")
            else:
                seen_task_ids.add(task_id)

            for field_name in required_fields:
                if field_name not in task:
                    errors.append(f"Task {task_id} missing required field: {field_name}")

            if not isinstance(task.get("inputs"), list):
                errors.append(f"Task {task_id} field inputs must be a list")
            if not isinstance(task.get("outputs"), list):
                errors.append(f"Task {task_id} field outputs must be a list")
            if not isinstance(task.get("depends_on"), list):
                errors.append(f"Task {task_id} field depends_on must be a list")
            if not isinstance(task.get("object_specs"), dict):
                errors.append(f"Task {task_id} field object_specs must be a dict")
            if not isinstance(task.get("visualization"), dict):
                errors.append(f"Task {task_id} field visualization must be a dict")

            parsed_tasks.append((task_id, task))

    task_id_set = {task_id for task_id, _ in parsed_tasks}
    ordered_tasks = sorted(parsed_tasks, key=lambda item: item[0])

    for task_id, task in ordered_tasks:
        inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
        outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
        depends_on = task.get("depends_on") if isinstance(task.get("depends_on"), list) else []
        object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
        visualization = task.get("visualization") if isinstance(task.get("visualization"), dict) else {}

        for symbol in outputs:
            normalized_symbol = str(symbol).strip()
            if not normalized_symbol:
                continue
            if normalized_symbol in symbol_table:
                errors.append(f"Output symbol {normalized_symbol} is defined multiple times.")
            else:
                symbol_table[normalized_symbol] = {
                    "type": _infer_output_symbol_type(task, symbol_table),
                    "defined_by": task_id,
                    "operation": normalize_operation_alias(str(task.get("operation") or "").strip()),
                }

        for symbol in inputs:
            normalized_symbol = str(symbol).strip()
            if not normalized_symbol or normalized_symbol in BUILTIN_SYMBOLS:
                continue
            if normalized_symbol not in defined_symbols_before_current:
                errors.append(
                    f"Task {task_id} input symbol {normalized_symbol} is not defined by previous tasks."
                )

        for dep in depends_on:
            dep_id = _parse_task_id(dep)
            if dep_id is None or dep_id not in task_id_set:
                errors.append(f"Task {task_id} depends_on unknown task_id: {dep}")
                continue
            if dep_id == task_id:
                errors.append(f"Task {task_id} cannot depend on itself.")
            elif dep_id > task_id:
                errors.append(f"Task {task_id} depends on future task {dep_id}.")

        required = bool(visualization.get("required")) if "required" in visualization else False
        objects = visualization.get("objects") if isinstance(visualization.get("objects"), list) else []
        if not required and objects:
            warnings.append(f"Task {task_id} visualization required=false but objects is not empty.")

        if required:
            current_outputs = {str(symbol).strip() for symbol in outputs if str(symbol).strip()}
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name") or "").strip()
                if not name:
                    continue
                if name not in current_outputs and name not in defined_symbols_before_current:
                    errors.append(f"Task {task_id} visualization object {name} is not defined.")

        task_type = str(task.get("task_type") or "").strip()
        operation = normalize_operation_alias(str(task.get("operation") or "").strip())
        construct_point_task_type = get_task_type_for_operation("construct_point") or "construct_cga_point"
        construct_vector_task_type = get_task_type_for_operation("construct_vector") or "construct_vector"
        midpoint_task_type = get_task_type_for_operation("midpoint") or "compute_midpoint"
        geometric_product_task_type = get_task_type_for_operation("geometric_product") or "compute_geometric_product"
        outer_product_task_type = get_task_type_for_operation("outer_product") or "compute_outer_product"
        inner_product_task_type = get_task_type_for_operation("inner_product") or "compute_inner_product"
        norm_task_type = get_task_type_for_operation("norm") or "compute_norm"
        dual_task_type = get_task_type_for_operation("dual") or "compute_dual"
        point_pair_decomposition_task_type = (
            get_task_type_for_operation("point_pair_decomposition") or "decompose_cga_point_pair"
        )
        meet_task_type = get_task_type_for_operation("meet") or "compute_meet"
        reflect_point_task_type = get_task_type_for_operation("reflect_point") or "reflect_cga_point"
        rotate_object_task_type = get_task_type_for_operation("rotate_object") or "rotate_cga_object"
        construct_rotor_task_type = get_task_type_for_operation("construct_rotor") or "construct_rotor"
        point_distance_task_type = get_task_type_for_operation("point_distance") or "compute_cga_point_distance"
        circle_from_three_points_task_type = (
            get_task_type_for_operation("circle_from_three_points") or "construct_cga_circle_from_three_points"
        )
        plane_from_three_points_task_type = (
            get_task_type_for_operation("plane_from_three_points") or "construct_cga_plane_from_three_points"
        )
        plane_from_point_and_normal_task_type = (
            get_task_type_for_operation("plane_from_point_and_normal") or "construct_cga_plane_from_point_and_normal"
        )
        construct_sphere_task_type = get_task_type_for_operation("construct_sphere") or "construct_cga_sphere"

        if task_type == construct_point_task_type or operation == "construct_point":
            if operation != "construct_point":
                errors.append(f"Task {task_id} operation must be construct_point.")
            if inputs:
                errors.append(f"Task {task_id} construct_point inputs must be [].")
            if len(outputs) != 1:
                errors.append(f"Task {task_id} construct_point must have exactly one output.")
            output_name = str(outputs[0]).strip() if outputs else ""
            if not output_name:
                errors.append(f"Task {task_id} construct_point output is empty.")
            if str(object_specs.get("type") or "").strip() != "point":
                errors.append(f"Task {task_id} construct_point object_specs.type must be point.")
            if output_name and str(object_specs.get("name") or "").strip() != output_name:
                errors.append(f"Task {task_id} object_specs.name does not match outputs[0].")
            coordinates = object_specs.get("coordinates")
            if not isinstance(coordinates, list) or len(coordinates) != 3:
                warnings.append(f"Task {task_id} construct_point coordinates should be a length-3 list.")
        if task_type == construct_vector_task_type or operation == "construct_vector":
            if task_type != construct_vector_task_type:
                errors.append(f"Task {task_id} task_type must be {construct_vector_task_type}.")
            if operation != "construct_vector":
                errors.append(f"Task {task_id} operation must be construct_vector.")
            if inputs:
                errors.append(f"Task {task_id} construct_vector inputs must be empty.")
            if len(outputs) != 1:
                errors.append(f"Task {task_id} construct_vector requires one output symbol.")
            output_name = str(outputs[0]).strip() if outputs else ""
            if str(object_specs.get("type") or "").strip() != "vector":
                errors.append(f"Task {task_id} construct_vector object_specs.type must be vector.")
            if output_name and str(object_specs.get("name") or "").strip() != output_name:
                errors.append(f"Task {task_id} construct_vector object_specs.name must match outputs[0].")
            expression = _normalize_vector_expression(object_specs.get("expression"))
            if not expression:
                errors.append(f"Task {task_id} construct_vector requires object_specs.expression.")
        if task_type == midpoint_task_type or operation == "midpoint":
            if task_type != midpoint_task_type:
                errors.append(f"Task {task_id} task_type must be {midpoint_task_type}.")
            if operation != "midpoint":
                errors.append(f"Task {task_id} operation must be midpoint.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} midpoint requires exactly two point inputs.")
            if not outputs:
                errors.append(f"Task {task_id} midpoint requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "point":
                errors.append(f"Task {task_id} midpoint object_specs.type must be point.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} midpoint object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} midpoint object_specs.from must match inputs.")
            normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
            if len(normalized_inputs) == 2:
                input_types = [
                    str(symbol_table.get(symbol, {}).get("type") or "").strip()
                    for symbol in normalized_inputs
                ]
                if any(input_type and input_type != "point" for input_type in input_types):
                    errors.append(f"Task {task_id} midpoint inputs must be points.")
        if task_type == geometric_product_task_type or operation == "geometric_product":
            if task_type != geometric_product_task_type:
                errors.append(f"Task {task_id} task_type must be {geometric_product_task_type}.")
            if operation != "geometric_product":
                errors.append(f"Task {task_id} operation must be geometric_product.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} geometric_product requires exactly two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} geometric_product requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "multivector":
                errors.append(f"Task {task_id} geometric_product object_specs.type must be multivector.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} geometric_product object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} geometric_product object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != "*":
                errors.append(f"Task {task_id} geometric_product operator must be '*'.")
        if task_type == outer_product_task_type or operation == "outer_product":
            if task_type != outer_product_task_type:
                errors.append(f"Task {task_id} task_type must be {outer_product_task_type}.")
            if operation != "outer_product":
                errors.append(f"Task {task_id} operation must be outer_product.")
            if len(inputs) < 2:
                errors.append(f"Task {task_id} outer_product requires at least two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} outer_product requires one output symbol.")
            object_type = str(object_specs.get("type") or "").strip()
            if object_type not in ALLOWED_OUTER_PRODUCT_TYPES:
                errors.append(
                    f"Task {task_id} outer_product object_specs.type must be one of {sorted(ALLOWED_OUTER_PRODUCT_TYPES)}."
                )
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} outer_product object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} outer_product object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != "^":
                errors.append(f"Task {task_id} outer_product operator must be '^'.")
        if task_type == inner_product_task_type or operation == "inner_product":
            if task_type != inner_product_task_type:
                errors.append(f"Task {task_id} task_type must be {inner_product_task_type}.")
            if operation != "inner_product":
                errors.append(f"Task {task_id} operation must be inner_product.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} inner_product requires exactly two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} inner_product requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "scalar":
                errors.append(f"Task {task_id} inner_product object_specs.type must be scalar.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} inner_product object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} inner_product object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != ".":
                errors.append(f"Task {task_id} inner_product operator must be '.'.")
        if task_type == norm_task_type or operation == "norm":
            if task_type != norm_task_type:
                errors.append(f"Task {task_id} task_type must be {norm_task_type}.")
            if operation != "norm":
                errors.append(f"Task {task_id} operation must be norm.")
            if len(inputs) != 1:
                errors.append(f"Task {task_id} norm requires exactly one input.")
            if not outputs:
                errors.append(f"Task {task_id} norm requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "scalar":
                errors.append(f"Task {task_id} norm object_specs.type must be scalar.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} norm object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} norm object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != "sqrt_dot":
                errors.append(f"Task {task_id} norm operator must be 'sqrt_dot'.")
        if task_type == dual_task_type or operation == "dual":
            if task_type != dual_task_type:
                errors.append(f"Task {task_id} task_type must be {dual_task_type}.")
            if operation != "dual":
                errors.append(f"Task {task_id} operation must be dual.")
            if len(inputs) != 1:
                errors.append(f"Task {task_id} dual requires exactly one input.")
            if not outputs:
                errors.append(f"Task {task_id} dual requires one output symbol.")
            object_type = str(object_specs.get("type") or "").strip()
            if object_type and object_type not in ALLOWED_DUAL_TYPES:
                errors.append(
                    f"Task {task_id} dual object_specs.type must be one of {sorted(ALLOWED_DUAL_TYPES)}."
                )
            if str(object_specs.get("type") or "").strip() != "multivector":
                errors.append(f"Task {task_id} dual object_specs.type must be multivector.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} dual object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} dual object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != "*":
                errors.append(f"Task {task_id} dual operator must be '*'.")
        if task_type == point_pair_decomposition_task_type or operation == "point_pair_decomposition":
            if task_type != point_pair_decomposition_task_type:
                errors.append(f"Task {task_id} task_type must be {point_pair_decomposition_task_type}.")
            if operation != "point_pair_decomposition":
                errors.append(f"Task {task_id} operation must be point_pair_decomposition.")
            if len(inputs) != 1:
                errors.append(f"Task {task_id} point_pair_decomposition requires exactly one point pair input.")
            if len(outputs) < 2:
                errors.append(f"Task {task_id} point_pair_decomposition requires two output point symbols.")
            if str(object_specs.get("type") or "").strip() != "point_pair_decomposition":
                errors.append(f"Task {task_id} point_pair_decomposition object_specs.type must be point_pair_decomposition.")
            expected_point_pair = str(inputs[0]).strip() if len(inputs) >= 1 else ""
            if str(object_specs.get("point_pair") or "").strip() != expected_point_pair:
                errors.append(f"Task {task_id} point_pair_decomposition object_specs.point_pair must match inputs[0].")
        if task_type == meet_task_type or operation == "meet":
            if task_type != meet_task_type:
                errors.append(f"Task {task_id} task_type must be {meet_task_type}.")
            if operation != "meet":
                errors.append(f"Task {task_id} operation must be meet.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} meet requires exactly two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} meet requires one output symbol.")
            object_type = str(object_specs.get("type") or "").strip()
            if object_type not in ALLOWED_MEET_TYPES:
                errors.append(
                    f"Task {task_id} meet object_specs.type must be one of {sorted(ALLOWED_MEET_TYPES)}."
                )
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} meet object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} meet object_specs.from must match inputs.")
            if str(object_specs.get("operator") or "").strip() != "meet":
                errors.append(f"Task {task_id} meet operator must be 'meet'.")
        if task_type == reflect_point_task_type or operation == "reflect_point":
            if task_type != reflect_point_task_type:
                errors.append(f"Task {task_id} task_type must be {reflect_point_task_type}.")
            if operation != "reflect_point":
                errors.append(f"Task {task_id} operation must be reflect_point.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} reflect_point requires exactly two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} reflect_point requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "point":
                errors.append(f"Task {task_id} reflect_point object_specs.type must be point.")
            expected_point = str(inputs[0]).strip() if len(inputs) >= 1 else ""
            expected_mirror = str(inputs[1]).strip() if len(inputs) >= 2 else ""
            if str(object_specs.get("point") or "").strip() != expected_point:
                errors.append(f"Task {task_id} reflect_point object_specs.point must match inputs[0].")
            if str(object_specs.get("mirror") or "").strip() != expected_mirror:
                errors.append(f"Task {task_id} reflect_point object_specs.mirror must match inputs[1].")
        if task_type == rotate_object_task_type or operation == "rotate_object":
            if task_type != rotate_object_task_type:
                errors.append(f"Task {task_id} task_type must be {rotate_object_task_type}.")
            if operation != "rotate_object":
                errors.append(f"Task {task_id} operation must be rotate_object.")
            if len(inputs) not in {1, 2}:
                errors.append(
                    f"Task {task_id} rotate_object requires one object input, or object plus rotor input."
                )
            if not outputs:
                errors.append(f"Task {task_id} rotate_object requires one output symbol.")
            expected_object = str(inputs[0]).strip() if len(inputs) >= 1 else ""
            if str(object_specs.get("object") or "").strip() != expected_object:
                errors.append(f"Task {task_id} rotate_object object_specs.object must match inputs[0].")
            if len(inputs) == 1:
                axis = object_specs.get("axis")
                if axis is None or axis == "":
                    errors.append(f"Task {task_id} rotate_object implicit mode requires object_specs.axis.")
                elif not isinstance(axis, list) or len(axis) != 3:
                    errors.append(f"Task {task_id} rotate_object requires object_specs.axis with length 3.")
                else:
                    normalized_axis = _normalize_coordinates_list(axis)
                    if (
                        normalized_axis is None
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in normalized_axis)
                    ):
                        errors.append(f"Task {task_id} rotate_object requires object_specs.axis with length 3.")
                    elif all(float(value) == 0.0 for value in normalized_axis):
                        errors.append(f"Task {task_id} rotate_object axis vector cannot be zero.")

                if "angle" not in object_specs or object_specs.get("angle") in {None, ""}:
                    errors.append(f"Task {task_id} rotate_object implicit mode requires object_specs.angle.")
                else:
                    try:
                        parse_rotation_angle_to_radians(
                            object_specs.get("angle"),
                            str(object_specs.get("angle_unit") or "").strip() or None,
                        )
                    except Exception:
                        errors.append(f"Task {task_id} rotate_object angle cannot be parsed.")
            elif len(inputs) == 2:
                expected_rotor = str(inputs[1]).strip()
                actual_rotor = str(object_specs.get("rotor") or "").strip()
                if not actual_rotor:
                    errors.append(f"Task {task_id} rotate_object explicit rotor mode requires object_specs.rotor.")
                elif actual_rotor != expected_rotor:
                    errors.append(f"Task {task_id} rotate_object object_specs.rotor must match inputs[1].")
                rotor_type = str(symbol_table.get(expected_rotor, {}).get("type") or "").strip()
                if rotor_type and rotor_type != "rotor":
                    errors.append(f"Task {task_id} rotate_object second input must be a rotor.")

            object_type = str(object_specs.get("type") or "").strip()
            if object_type and object_type not in ALLOWED_ROTATE_OBJECT_TYPES:
                errors.append(
                    f"Task {task_id} rotate_object object_specs.type must be one of {sorted(ALLOWED_ROTATE_OBJECT_TYPES)}."
                )
        if task_type == construct_rotor_task_type or operation == "construct_rotor":
            if task_type != construct_rotor_task_type:
                errors.append(f"Task {task_id} task_type must be {construct_rotor_task_type}.")
            if operation != "construct_rotor":
                errors.append(f"Task {task_id} operation must be construct_rotor.")
            if inputs:
                errors.append(f"Task {task_id} construct_rotor inputs must be [].")
            if not outputs:
                errors.append(f"Task {task_id} construct_rotor requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "rotor":
                errors.append(f"Task {task_id} construct_rotor object_specs.type must be rotor.")

            axis = object_specs.get("axis")
            if not isinstance(axis, list) or len(axis) != 3:
                errors.append(f"Task {task_id} construct_rotor requires object_specs.axis with length 3.")
            else:
                normalized_axis = _normalize_coordinates_list(axis)
                if (
                    normalized_axis is None
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in normalized_axis)
                ):
                    errors.append(f"Task {task_id} construct_rotor requires object_specs.axis with length 3.")
                elif all(float(value) == 0.0 for value in normalized_axis):
                    errors.append(f"Task {task_id} construct_rotor axis vector cannot be zero.")

            if "angle" not in object_specs or object_specs.get("angle") in {None, ""}:
                errors.append(f"Task {task_id} construct_rotor requires object_specs.angle.")
            else:
                try:
                    parse_rotation_angle_to_radians(
                        object_specs.get("angle"),
                        str(object_specs.get("angle_unit") or "").strip() or None,
                    )
                except Exception:
                    errors.append(f"Task {task_id} construct_rotor angle cannot be parsed.")
        if task_type == point_distance_task_type or operation == "point_distance":
            if task_type != point_distance_task_type:
                errors.append(f"Task {task_id} task_type must be {point_distance_task_type}.")
            if operation != "point_distance":
                errors.append(f"Task {task_id} operation must be point_distance.")
            if len(inputs) != 2:
                errors.append(f"Task {task_id} point_distance requires exactly two inputs.")
            if not outputs:
                errors.append(f"Task {task_id} point_distance requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "scalar":
                errors.append(f"Task {task_id} point_distance object_specs.type must be scalar.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} point_distance object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} point_distance object_specs.from must match inputs.")
            if required:
                warnings.append(f"Task {task_id} point_distance usually does not require visualization.")
        if task_type == circle_from_three_points_task_type or operation == "circle_from_three_points":
            if task_type != circle_from_three_points_task_type:
                errors.append(f"Task {task_id} task_type must be {circle_from_three_points_task_type}.")
            if operation != "circle_from_three_points":
                errors.append(f"Task {task_id} operation must be circle_from_three_points.")
            if len(inputs) != 3:
                errors.append(f"Task {task_id} circle_from_three_points requires exactly three inputs.")
            if not outputs:
                errors.append(f"Task {task_id} circle_from_three_points requires one output symbol.")
            if str(object_specs.get("type") or "").strip() != "circle":
                errors.append(f"Task {task_id} circle_from_three_points object_specs.type must be circle.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} circle_from_three_points object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} circle_from_three_points object_specs.from must match inputs.")
        if task_type == plane_from_three_points_task_type or operation == "plane_from_three_points":
            if task_type != plane_from_three_points_task_type:
                errors.append(f"Task {task_id} task_type must be {plane_from_three_points_task_type}.")
            if operation != "plane_from_three_points":
                errors.append(f"Task {task_id} operation must be plane_from_three_points.")
            if len(inputs) != 3:
                errors.append(f"Task {task_id} plane_from_three_points requires exactly three inputs.")
            if not outputs:
                errors.append(f"Task {task_id} plane_from_three_points requires one output symbol.")
            output_name = str(outputs[0]).strip() if outputs else ""
            if output_name in {"Π", "π", "\\Pi", "螤"}:
                errors.append(f"Task {task_id} plane_from_three_points output should use Pi instead of Π.")
            if str(object_specs.get("type") or "").strip() != "plane":
                errors.append(f"Task {task_id} plane_from_three_points object_specs.type must be plane.")
            object_from = object_specs.get("from")
            if object_from is None:
                warnings.append(f"Task {task_id} plane_from_three_points object_specs.from should match inputs.")
            else:
                normalized_from = _normalize_symbol_list(object_from)
                normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                if normalized_from != normalized_inputs:
                    errors.append(f"Task {task_id} plane_from_three_points object_specs.from must match inputs.")
        if task_type == plane_from_point_and_normal_task_type or operation == "plane_from_point_and_normal":
            if task_type != plane_from_point_and_normal_task_type:
                errors.append(f"Task {task_id} task_type must be {plane_from_point_and_normal_task_type}.")
            if operation != "plane_from_point_and_normal":
                errors.append(f"Task {task_id} operation must be plane_from_point_and_normal.")
            if not outputs:
                errors.append(f"Task {task_id} plane_from_point_and_normal requires one output symbol.")
            output_name = str(outputs[0]).strip() if outputs else ""
            if output_name in {"\u03A0", "\u03C0", "\\Pi", "\\pi"}:
                errors.append(f"Task {task_id} plane_from_point_and_normal output should use Pi instead of Π.")
            if str(object_specs.get("type") or "").strip() != "plane":
                errors.append(f"Task {task_id} plane_from_point_and_normal object_specs.type must be plane.")
            point = object_specs.get("point")
            if not isinstance(point, list) or len(point) != 3:
                errors.append(f"Task {task_id} plane_from_point_and_normal requires object_specs.point with length 3.")
            normal = object_specs.get("normal")
            if not isinstance(normal, list) or len(normal) != 3:
                errors.append(f"Task {task_id} plane_from_point_and_normal requires object_specs.normal with length 3.")
            elif all(_normalize_coordinate_value(value) == 0 for value in normal):
                errors.append(f"Task {task_id} plane_from_point_and_normal normal vector cannot be zero.")
        if task_type == construct_sphere_task_type or operation == "construct_sphere":
            if task_type != construct_sphere_task_type:
                errors.append(f"Task {task_id} task_type must be {construct_sphere_task_type}.")
            if operation != "construct_sphere":
                errors.append(f"Task {task_id} operation must be construct_sphere.")
            if not outputs:
                errors.append(f"Task {task_id} construct_sphere requires one output symbol.")
            output_name = str(outputs[0]).strip() if outputs else ""
            if str(object_specs.get("type") or "").strip() != "sphere":
                errors.append(f"Task {task_id} construct_sphere object_specs.type must be sphere.")
            center = object_specs.get("center")
            if isinstance(center, list):
                normalized_center = _normalize_coordinates_list(center)
                if (
                    normalized_center is None
                    or len(normalized_center) != 3
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in normalized_center)
                ):
                    errors.append(
                        f"Task {task_id} construct_sphere requires object_specs.center with length 3 or a defined point symbol."
                    )
            elif isinstance(center, str):
                center_symbol = str(center).strip()
                if not center_symbol:
                    errors.append(
                        f"Task {task_id} construct_sphere requires object_specs.center as coordinates or point symbol."
                    )
                else:
                    if center_symbol not in defined_symbols_before_current:
                        errors.append(
                            f"Task {task_id} construct_sphere center symbol {center_symbol} is not defined by previous tasks."
                        )
                    else:
                        center_symbol_type = str(symbol_table.get(center_symbol, {}).get("type") or "").strip()
                        if center_symbol_type != "point":
                            errors.append(
                                f"Task {task_id} construct_sphere center symbol {center_symbol} must be a point."
                            )
                    normalized_inputs = [str(symbol).strip() for symbol in inputs if str(symbol).strip()]
                    if center_symbol not in normalized_inputs:
                        errors.append(
                            f"Task {task_id} construct_sphere inputs must include center symbol {center_symbol}."
                        )
            else:
                errors.append(
                    f"Task {task_id} construct_sphere requires object_specs.center as coordinates or point symbol."
                )
            radius = object_specs.get("radius")
            if isinstance(radius, bool) or not isinstance(radius, (int, float)):
                errors.append(f"Task {task_id} construct_sphere requires numeric object_specs.radius.")
            if output_name and str(object_specs.get("name") or "").strip() != output_name:
                errors.append(f"Task {task_id} construct_sphere object_specs.name must match outputs[0].")

        for symbol in outputs:
            normalized_symbol = str(symbol).strip()
            if normalized_symbol:
                defined_symbols_before_current.add(normalized_symbol)

    result = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "symbol_table": symbol_table,
        "defined_symbols": sorted(defined_symbols_before_current),
    }

    if raise_on_error and errors:
        raise ValueError("Task IR validation failed:\n" + "\n".join(errors))

    return result


def task_ir_validator_node(state: MainGraphState) -> dict:
    task_blocks_result = state.get("validated_task_blocks_result") or state.get("task_blocks_result") or {}
    tasks = task_blocks_result.get("tasks", []) if isinstance(task_blocks_result, dict) else []
    validation_result = validate_task_blocks_result(task_blocks_result, raise_on_error=True)

    return {
        "validated_task_blocks_result": {
            "tasks": deepcopy(tasks),
        },
        "task_ir_validation_result": validation_result,
    }


def infer_point_placeholder_index(output_name: str, task_id: int) -> int:
    match = re.fullmatch(r"P(\d+)", str(output_name or "").strip())
    if match:
        return int(match.group(1))
    return task_id


def infer_sphere_placeholder_index(output_name: str, task_id: int) -> int:
    normalized = str(output_name or "").strip()
    if normalized == "S":
        return task_id
    match = re.fullmatch(r"S(\d+)", normalized)
    if match:
        return int(match.group(1))
    return task_id


def normalize_visualization_block(task: dict) -> dict:
    visualization = normalize_ir_visualization_block(task)
    required = bool(visualization.get("required"))
    normalized_objects = visualization.get("objects") if isinstance(visualization.get("objects"), list) else []
    return {
        "required": required,
        "objects": normalized_objects if required else [],
    }


def build_construct_point_task_block(task: dict) -> dict:
    task_id = _parse_task_id(task.get("task_id")) or 0
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if not outputs:
        raise ValueError("construct_point requires one output symbol")

    output_name = str(outputs[0]).strip()
    if not output_name:
        raise ValueError("construct_point output symbol is empty")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    coordinates = object_specs.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 3:
        raise ValueError("construct_point requires object_specs.coordinates with length 3")

    index = infer_point_placeholder_index(output_name, task_id)
    a_name = f"a{index}"
    b_name = f"b{index}"
    c_name = f"c{index}"

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Construct CGA point {output_name}.",
        "formula": (
            f"{output_name} = {a_name}*e1 + {b_name}*e2 + {c_name}*e3 + "
            f"0.5*({a_name}*{a_name} + {b_name}*{b_name} + {c_name}*{c_name})*einf + e0"
        ),
        "parameters": [a_name, b_name, c_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {
        a_name: coordinates[0],
        b_name: coordinates[1],
        c_name: coordinates[2],
    }
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_construct_vector_task_block(task: dict) -> dict:
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if not outputs:
        raise ValueError("construct_vector requires one output symbol")

    output_name = str(outputs[0]).strip()
    if not output_name:
        raise ValueError("construct_vector output symbol is empty")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    expression = _normalize_vector_expression(object_specs.get("expression"))
    if not expression:
        raise ValueError("construct_vector requires object_specs.expression")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Construct vector {output_name}.",
        "formula": f"{output_name} = {expression}",
        "parameters": [],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_norm_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 1:
        raise ValueError("norm requires exactly one input")
    if not outputs:
        raise ValueError("norm requires one output symbol")

    input_name = str(inputs[0]).strip()
    output_name = str(outputs[0]).strip()
    if not input_name or not output_name:
        raise ValueError("norm requires non-empty input and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute norm {output_name} of {input_name}.",
        "formula": f"{output_name} = sqrt({input_name} . {input_name})",
        "parameters": [input_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_dual_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 1:
        raise ValueError("dual requires exactly one input")
    if not outputs:
        raise ValueError("dual requires one output symbol")

    input_name = str(inputs[0]).strip()
    output_name = str(outputs[0]).strip()
    if not input_name or not output_name:
        raise ValueError("dual requires non-empty input and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute dual {output_name} of {input_name}.",
        "formula": f"{output_name} = *{input_name}",
        "parameters": [input_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_point_pair_decomposition_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 1:
        raise ValueError("point_pair_decomposition requires exactly one point pair input")
    if len(outputs) < 2:
        raise ValueError("point_pair_decomposition requires two output point symbols")

    point_pair_name = str(inputs[0]).strip()
    first_output = str(outputs[0]).strip()
    second_output = str(outputs[1]).strip()
    if not point_pair_name or not first_output or not second_output:
        raise ValueError("point_pair_decomposition requires non-empty input and outputs")

    updated_task = deepcopy(task)
    updated_task["outputs"] = [first_output, second_output]
    if isinstance(updated_task.get("object_specs"), dict):
        updated_task["object_specs"]["name"] = "point_pair_decomposition"
        updated_task["object_specs"]["type"] = "point_pair_decomposition"
        updated_task["object_specs"]["point_pair"] = point_pair_name
        updated_task["object_specs"]["formula"] = (
            str(updated_task["object_specs"].get("formula") or "X_pm = (P ± sqrt(P.P)) / (einf.P)").strip()
            or "X_pm = (P ± sqrt(P.P)) / (einf.P)"
        )
    updated_task["code_to_optimize"] = {
        "goal": f"Decompose point pair {point_pair_name} into {first_output} and {second_output}.",
        "formula": (
            f"PP = {point_pair_name} . {point_pair_name}; "
            f"sqrt_PP = sqrt(PP); "
            f"denom = einf . {point_pair_name}; "
            f"{first_output} = ({point_pair_name} + sqrt_PP) * (1 / denom); "
            f"{second_output} = ({point_pair_name} - sqrt_PP) * (1 / denom)"
        ),
        "parameters": [point_pair_name],
        "output": f"{first_output}, {second_output}",
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_meet_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("meet requires exactly two inputs")
    if not outputs:
        raise ValueError("meet requires one output symbol")

    left = str(inputs[0]).strip()
    right = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not left or not right or not output_name:
        raise ValueError("meet requires non-empty inputs and output")

    left_id = _sanitize_symbol_for_identifier(left)
    right_id = _sanitize_symbol_for_identifier(right)
    output_id = _sanitize_symbol_for_identifier(output_name)
    dual_left = f"dual_{left_id}"
    dual_right = f"dual_{right_id}"
    tmp_name = f"tmp_{output_id}"

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": (
            f"Compute meet {output_name} of {left} and {right}. "
            f"Keep {dual_left}, {dual_right}, {tmp_name}, and {output_name} as explicit GAALOPScript output variables."
        ),
        "formula": (
            f"{dual_left} = *{left}\n"
            f"{dual_right} = *{right}\n"
            f"{tmp_name} = {dual_left} ^ {dual_right}\n"
            f"{output_name} = *{tmp_name}"
        ),
        "parameters": [left, right],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_reflect_point_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("reflect_point requires exactly two inputs")
    if not outputs:
        raise ValueError("reflect_point requires one output symbol")

    point_name = str(inputs[0]).strip()
    mirror_name = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not point_name or not mirror_name or not output_name:
        raise ValueError("reflect_point requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Reflect point {point_name} with respect to {mirror_name}.",
        "formula": f"{output_name} = {mirror_name} * {point_name} * {mirror_name}",
        "parameters": [point_name, mirror_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_line_from_two_points_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("line_from_two_points requires exactly two inputs")
    if not outputs:
        raise ValueError("line_from_two_points requires at least one output")

    p1 = str(inputs[0]).strip()
    p2 = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not p1 or not p2 or not output_name:
        raise ValueError("line_from_two_points requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Construct CGA line {output_name} from {p1} and {p2}.",
        "formula": f"{output_name} = {p1} ^ {p2} ^ einf",
        "parameters": [p1, p2],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_construct_sphere_task_block(task: dict) -> dict:
    task_id = _parse_task_id(task.get("task_id")) or 0
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if not outputs:
        raise ValueError("construct_sphere requires one output symbol")

    output_name = str(outputs[0]).strip()
    if not output_name:
        raise ValueError("construct_sphere output symbol is empty")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    center = object_specs.get("center")
    radius = object_specs.get("radius")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise ValueError("construct_sphere requires numeric object_specs.radius")

    index = infer_sphere_placeholder_index(output_name, task_id)
    radius_value_name = f"r{index}v"
    radius_symbol = f"r{index}"

    updated_task = deepcopy(task)
    if isinstance(center, list) and len(center) == 3:
        x_name = f"x{index}"
        y_name = f"y{index}"
        z_name = f"z{index}"
        center_symbol = f"C{index}"
        updated_task["code_to_optimize"] = {
            "goal": f"Construct CGA sphere {output_name}.",
            "formula": (
                f"{center_symbol} = createPoint({x_name}, {y_name}, {z_name}); "
                f"{radius_symbol} = {radius_value_name}; "
                f"{output_name} = {center_symbol} - 0.5 * ({radius_symbol} * {radius_symbol}) * einf"
            ),
            "parameters": [x_name, y_name, z_name, radius_value_name],
            "output": output_name,
        }
        updated_task["variable_assignments"] = {
            x_name: center[0],
            y_name: center[1],
            z_name: center[2],
            radius_value_name: radius,
        }
    elif isinstance(center, str) and str(center).strip():
        center_symbol = str(center).strip()
        updated_task["code_to_optimize"] = {
            "goal": f"Construct CGA sphere {output_name}.",
            "formula": (
                f"{radius_symbol} = {radius_value_name}; "
                f"{output_name} = {center_symbol} - 0.5 * ({radius_symbol} * {radius_symbol}) * einf"
            ),
            "parameters": [radius_value_name],
            "output": output_name,
        }
        updated_task["variable_assignments"] = {
            radius_value_name: radius,
        }
    else:
        raise ValueError("construct_sphere requires object_specs.center as coordinates or point symbol")
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_construct_rotor_task_block(task: dict) -> dict:
    task_id = _parse_task_id(task.get("task_id")) or 1
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if inputs:
        raise ValueError("construct_rotor inputs must be []")
    if not outputs:
        raise ValueError("construct_rotor requires one output symbol")

    output_name = _sanitize_symbol_for_identifier(str(outputs[0]).strip())
    if not output_name:
        raise ValueError("construct_rotor output symbol is empty")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    axis = _normalize_coordinates_list(object_specs.get("axis"))
    if axis is None or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in axis):
        raise ValueError("construct_rotor requires object_specs.axis with length 3")
    if all(float(value) == 0.0 for value in axis):
        raise ValueError("construct_rotor axis vector cannot be zero")

    angle_value = parse_rotation_angle_to_radians(
        object_specs.get("angle"),
        str(object_specs.get("angle_unit") or "").strip() or None,
    )

    ax_name = f"ax{task_id}"
    ay_name = f"ay{task_id}"
    az_name = f"az{task_id}"
    angle_name = f"angle{task_id}"
    axis_norm_name = f"axis_norm{task_id}"
    axis_b_name = f"axisB{task_id}"

    updated_task = deepcopy(task)
    updated_task["outputs"] = [output_name]
    if isinstance(updated_task.get("object_specs"), dict):
        updated_task["object_specs"]["name"] = output_name
        updated_task["object_specs"]["type"] = "rotor"
    updated_task["code_to_optimize"] = {
        "goal": f"Construct rotor {output_name} from axis and angle.",
        "formula": (
            f"{axis_norm_name} = sqrt({ax_name}*{ax_name} + {ay_name}*{ay_name} + {az_name}*{az_name}); "
            f"{axis_b_name} = ({ax_name}/{axis_norm_name})*(e2^e3) + ({ay_name}/{axis_norm_name})*(e3^e1) + ({az_name}/{axis_norm_name})*(e1^e2); "
            f"{output_name} = cos({angle_name}*0.5) - sin({angle_name}*0.5) * {axis_b_name}"
        ),
        "parameters": [ax_name, ay_name, az_name, angle_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {
        ax_name: axis[0],
        ay_name: axis[1],
        az_name: axis[2],
        angle_name: angle_value,
    }
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_rotate_object_task_block(task: dict) -> dict:
    task_id = _parse_task_id(task.get("task_id")) or 0
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) not in {1, 2}:
        raise ValueError("rotate_object requires one object input, or object plus rotor input")
    if not outputs:
        raise ValueError("rotate_object requires one output symbol")

    input_name = str(inputs[0]).strip()
    output_name = sanitize_symbol_name(str(outputs[0]).strip())
    if not input_name or not output_name:
        raise ValueError("rotate_object requires non-empty input and output")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    updated_task = deepcopy(task)
    updated_task["outputs"] = [output_name]
    if isinstance(updated_task.get("object_specs"), dict):
        updated_task["object_specs"]["name"] = output_name
        updated_task["object_specs"]["object"] = input_name

    explicit_rotor_symbol = str(object_specs.get("rotor") or "").strip()
    if not explicit_rotor_symbol and len(inputs) >= 2:
        explicit_rotor_symbol = str(inputs[1]).strip()

    if explicit_rotor_symbol:
        if isinstance(updated_task.get("object_specs"), dict):
            updated_task["object_specs"]["rotor"] = explicit_rotor_symbol
        updated_task["code_to_optimize"] = {
            "goal": f"Rotate object {input_name} with rotor {explicit_rotor_symbol}.",
            "formula": f"{output_name} = {explicit_rotor_symbol} * {input_name} * ~{explicit_rotor_symbol}",
            "parameters": [input_name, explicit_rotor_symbol],
            "output": output_name,
        }
        updated_task["variable_assignments"] = {}
    else:
        axis = _normalize_coordinates_list(object_specs.get("axis"))
        if axis is None or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in axis):
            raise ValueError("rotate_object requires object_specs.axis with length 3")
        if all(float(value) == 0.0 for value in axis):
            raise ValueError("rotate_object axis vector cannot be zero")

        angle_value = parse_rotation_angle_to_radians(
            object_specs.get("angle"),
            str(object_specs.get("angle_unit") or "").strip() or None,
        )

        ax_name = f"ax{task_id}"
        ay_name = f"ay{task_id}"
        az_name = f"az{task_id}"
        angle_name = f"angle{task_id}"
        axis_b_name = f"axisB{task_id}"
        rotor_name = f"R{task_id}"
        updated_task["code_to_optimize"] = {
            "goal": f"Rotate object {input_name} to {output_name}.",
            "formula": (
                f"{axis_b_name} = {ax_name}*(e2^e3) + {ay_name}*(e3^e1) + {az_name}*(e1^e2); "
                f"{rotor_name} = cos({angle_name}*0.5) - sin({angle_name}*0.5) * {axis_b_name}; "
                f"{output_name} = {rotor_name} * {input_name} * ~{rotor_name}"
            ),
            "parameters": [input_name, ax_name, ay_name, az_name, angle_name],
            "output": output_name,
        }
        updated_task["variable_assignments"] = {
            ax_name: axis[0],
            ay_name: axis[1],
            az_name: axis[2],
            angle_name: angle_value,
        }
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(updated_task)
    return updated_task


def build_circle_from_three_points_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 3:
        raise ValueError("circle_from_three_points requires exactly three inputs")
    if not outputs:
        raise ValueError("circle_from_three_points requires one output symbol")

    p1 = str(inputs[0]).strip()
    p2 = str(inputs[1]).strip()
    p3 = str(inputs[2]).strip()
    output_name = str(outputs[0]).strip()
    if not p1 or not p2 or not p3 or not output_name:
        raise ValueError("circle_from_three_points requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Construct CGA circle {output_name} from {p1}, {p2} and {p3}.",
        "formula": f"{output_name} = {p1} ^ {p2} ^ {p3}",
        "parameters": [p1, p2, p3],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_plane_from_three_points_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 3:
        raise ValueError("plane_from_three_points requires exactly three inputs")
    if not outputs:
        raise ValueError("plane_from_three_points requires one output symbol")

    p1 = str(inputs[0]).strip()
    p2 = str(inputs[1]).strip()
    p3 = str(inputs[2]).strip()
    output_name = _normalize_plane_symbol(outputs[0], fallback="Pi")
    if not p1 or not p2 or not p3 or not output_name:
        raise ValueError("plane_from_three_points requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["outputs"] = [output_name]
    if isinstance(updated_task.get("object_specs"), dict):
        updated_task["object_specs"]["name"] = output_name
    updated_task["code_to_optimize"] = {
        "goal": f"Construct CGA plane {output_name} from {p1}, {p2} and {p3}.",
        "formula": f"{output_name} = {p1} ^ {p2} ^ {p3} ^ einf",
        "parameters": [p1, p2, p3],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    vis_block = normalize_visualization_block(task)
    plane_objects = []
    for obj in vis_block.get("objects", []):
        if isinstance(obj, dict):
            updated_obj = deepcopy(obj)
            if str(updated_obj.get("type") or "").strip() == "plane":
                updated_obj["name"] = output_name
            plane_objects.append(updated_obj)
    updated_task["multivectors_to_be_visualized"] = {
        "required": bool(vis_block.get("required")),
        "objects": plane_objects if vis_block.get("required") else [],
    }
    return updated_task


def build_plane_from_point_and_normal_task_block(task: dict) -> dict:
    task_id = _parse_task_id(task.get("task_id")) or 1
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if not outputs:
        raise ValueError("plane_from_point_and_normal requires one output symbol")

    output_name = _normalize_plane_symbol(outputs[0], fallback="Pi")
    if not output_name:
        raise ValueError("plane_from_point_and_normal requires a non-empty output symbol")

    object_specs = task.get("object_specs") if isinstance(task.get("object_specs"), dict) else {}
    point = object_specs.get("point")
    normal = object_specs.get("normal")
    if not isinstance(point, list) or len(point) != 3:
        raise ValueError("plane_from_point_and_normal requires object_specs.point with length 3")
    if not isinstance(normal, list) or len(normal) != 3:
        raise ValueError("plane_from_point_and_normal requires object_specs.normal with length 3")

    x_name = f"x{task_id}"
    y_name = f"y{task_id}"
    z_name = f"z{task_id}"
    nx_name = f"nx{task_id}"
    ny_name = f"ny{task_id}"
    nz_name = f"nz{task_id}"
    d_name = f"d{task_id}"

    updated_task = deepcopy(task)
    updated_task["outputs"] = [output_name]
    if isinstance(updated_task.get("object_specs"), dict):
        updated_task["object_specs"]["name"] = output_name
    updated_task["code_to_optimize"] = {
        "goal": f"Construct CGA plane {output_name} from point and normal.",
        "formula": (
            f"{d_name} = -({nx_name}*{x_name} + {ny_name}*{y_name} + {nz_name}*{z_name}); "
            f"{output_name} = {nx_name}*e1 + {ny_name}*e2 + {nz_name}*e3 + {d_name}*einf"
        ),
        "parameters": [x_name, y_name, z_name, nx_name, ny_name, nz_name],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {
        x_name: point[0],
        y_name: point[1],
        z_name: point[2],
        nx_name: normal[0],
        ny_name: normal[1],
        nz_name: normal[2],
    }
    vis_block = normalize_visualization_block(task)
    plane_objects = []
    for obj in vis_block.get("objects", []):
        if isinstance(obj, dict):
            updated_obj = deepcopy(obj)
            if str(updated_obj.get("type") or "").strip() == "plane":
                updated_obj["name"] = output_name
            plane_objects.append(updated_obj)
    updated_task["multivectors_to_be_visualized"] = {
        "required": bool(vis_block.get("required")),
        "objects": plane_objects if vis_block.get("required") else [],
    }
    return updated_task


def build_point_distance_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("point_distance requires exactly two inputs")
    if not outputs:
        raise ValueError("point_distance requires one output symbol")

    p1 = str(inputs[0]).strip()
    p2 = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not p1 or not p2 or not output_name:
        raise ValueError("point_distance requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute squared distance {output_name} between {p1} and {p2}.",
        "formula": f"{output_name} = -2 * ({p1} . {p2}) / ((-einf . {p1}) * (-einf . {p2}))",
        "parameters": [p1, p2],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_midpoint_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("midpoint requires exactly two point inputs")
    if not outputs:
        raise ValueError("midpoint requires one output symbol")

    left = str(inputs[0]).strip()
    right = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not left or not right or not output_name:
        raise ValueError("midpoint requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute midpoint {output_name} of {left} and {right}.",
        "formula": f"{output_name} = ({left} + {right}) * 0.5",
        "parameters": [left, right],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_geometric_product_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("geometric_product requires exactly two inputs")
    if not outputs:
        raise ValueError("geometric_product requires one output symbol")

    left = str(inputs[0]).strip()
    right = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not left or not right or not output_name:
        raise ValueError("geometric_product requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute geometric product {output_name} of {left} and {right}.",
        "formula": f"{output_name} = {left} * {right}",
        "parameters": [left, right],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_outer_product_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) < 2:
        raise ValueError("outer_product requires at least two inputs")
    if not outputs:
        raise ValueError("outer_product requires one output symbol")

    normalized_inputs = [str(item).strip() for item in inputs if str(item).strip()]
    output_name = str(outputs[0]).strip()
    if len(normalized_inputs) < 2 or not output_name:
        raise ValueError("outer_product requires non-empty inputs and output")

    joined_inputs = " ^ ".join(normalized_inputs)
    if len(normalized_inputs) == 2:
        goal_suffix = f"{normalized_inputs[0]} and {normalized_inputs[1]}"
    else:
        goal_suffix = ", ".join(normalized_inputs[:-1]) + f" and {normalized_inputs[-1]}"

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute outer product {output_name} of {goal_suffix}.",
        "formula": f"{output_name} = {joined_inputs}",
        "parameters": normalized_inputs,
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def build_inner_product_task_block(task: dict) -> dict:
    inputs = task.get("inputs") if isinstance(task.get("inputs"), list) else []
    outputs = task.get("outputs") if isinstance(task.get("outputs"), list) else []
    if len(inputs) != 2:
        raise ValueError("inner_product requires exactly two inputs")
    if not outputs:
        raise ValueError("inner_product requires one output symbol")

    left = str(inputs[0]).strip()
    right = str(inputs[1]).strip()
    output_name = str(outputs[0]).strip()
    if not left or not right or not output_name:
        raise ValueError("inner_product requires non-empty inputs and output")

    updated_task = deepcopy(task)
    updated_task["code_to_optimize"] = {
        "goal": f"Compute inner product {output_name} of {left} and {right}.",
        "formula": f"{output_name} = {left} . {right}",
        "parameters": [left, right],
        "output": output_name,
    }
    updated_task["variable_assignments"] = {}
    updated_task["multivectors_to_be_visualized"] = normalize_visualization_block(task)
    return updated_task


def operation_to_task_block(task: dict) -> dict:
    operation = normalize_operation_alias(str(task.get("operation") or "").strip())
    task_type = str(task.get("task_type") or "").strip()
    if not operation:
        operation = get_operation_for_task_type(task_type) or ""

    if operation == "construct_point":
        return build_construct_point_task_block(task)
    if operation == "construct_vector":
        return build_construct_vector_task_block(task)
    if operation == "construct_rotor":
        return build_construct_rotor_task_block(task)
    if operation == "plane_from_three_points":
        return build_plane_from_three_points_task_block(task)
    if operation == "plane_from_point_and_normal":
        return build_plane_from_point_and_normal_task_block(task)
    if operation == "circle_from_three_points":
        return build_circle_from_three_points_task_block(task)
    if operation == "construct_sphere":
        return build_construct_sphere_task_block(task)
    if operation == "line_from_two_points":
        return build_line_from_two_points_task_block(task)
    if operation == "point_distance":
        return build_point_distance_task_block(task)
    if operation == "midpoint":
        return build_midpoint_task_block(task)
    if operation == "geometric_product":
        return build_geometric_product_task_block(task)
    if operation == "outer_product":
        return build_outer_product_task_block(task)
    if operation == "inner_product":
        return build_inner_product_task_block(task)
    if operation == "norm":
        return build_norm_task_block(task)
    if operation == "dual":
        return build_dual_task_block(task)
    if operation == "point_pair_decomposition":
        return build_point_pair_decomposition_task_block(task)
    if operation == "meet":
        return build_meet_task_block(task)
    if operation == "reflect_point":
        return build_reflect_point_task_block(task)
    if operation == "rotate_object":
        return build_rotate_object_task_block(task)
    if not operation and task_type == "construct_cga_point":
        return build_construct_point_task_block(task)
    if not operation and task_type == "construct_vector":
        return build_construct_vector_task_block(task)
    if not operation and task_type == "construct_rotor":
        return build_construct_rotor_task_block(task)
    if not operation and task_type == "construct_cga_plane_from_three_points":
        return build_plane_from_three_points_task_block(task)
    if not operation and task_type == "construct_cga_plane_from_point_and_normal":
        return build_plane_from_point_and_normal_task_block(task)
    if not operation and task_type == "construct_cga_circle_from_three_points":
        return build_circle_from_three_points_task_block(task)
    if not operation and task_type == "construct_cga_sphere":
        return build_construct_sphere_task_block(task)
    if (not operation and task_type == "construct_cga_line_from_two_points") or operation == "construct_line":
        return build_line_from_two_points_task_block(task)
    if not operation and task_type == "compute_cga_point_distance":
        return build_point_distance_task_block(task)
    if not operation and task_type == "compute_midpoint":
        return build_midpoint_task_block(task)
    if not operation and task_type == "compute_geometric_product":
        return build_geometric_product_task_block(task)
    if not operation and task_type == "compute_outer_product":
        return build_outer_product_task_block(task)
    if not operation and task_type == "compute_inner_product":
        return build_inner_product_task_block(task)
    if not operation and task_type == "compute_norm":
        return build_norm_task_block(task)
    if not operation and task_type == "compute_dual":
        return build_dual_task_block(task)
    if not operation and task_type == "decompose_cga_point_pair":
        return build_point_pair_decomposition_task_block(task)
    if not operation and task_type == "compute_meet":
        return build_meet_task_block(task)
    if not operation and task_type == "reflect_cga_point":
        return build_reflect_point_task_block(task)
    if not operation and task_type == "rotate_cga_object":
        return build_rotate_object_task_block(task)

    raise ValueError(
        f"Unsupported operation: {operation}. "
        f"Supported operations: {list(get_supported_operations())}"
    )


def operation_to_task_block_node(state: MainGraphState) -> dict:
    source_result = state.get("validated_task_blocks_result") or state.get("task_blocks_result") or {}
    tasks = source_result.get("tasks", []) if isinstance(source_result, dict) else []
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("operation_to_task_block_node requires non-empty tasks")

    converted_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("operation_to_task_block_node requires task items to be dict")
        converted_tasks.append(operation_to_task_block(task))

    return {
        "operation_task_blocks_result": {
            "tasks": converted_tasks,
        }
    }


def _sort_tasks_by_task_id(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(task: dict[str, Any]) -> tuple[int, int]:
        task_id = task.get("task_id")
        if isinstance(task_id, int):
            return (0, task_id)
        if isinstance(task_id, str):
            digits = re.findall(r"\d+", task_id)
            if digits:
                return (0, int(digits[0]))
        return (1, 0)

    return sorted(tasks, key=sort_key)


def topological_sort_tasks(tasks: list[dict]) -> list[dict]:
    if not isinstance(tasks, list):
        raise ValueError("topological_sort_tasks requires a list of tasks")

    task_map: dict[int, dict] = {}
    indegree: dict[int, int] = {}
    graph: dict[int, list[int]] = {}

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not a dict")

        task_id = _parse_task_id(task.get("task_id"))
        if task_id is None:
            raise ValueError(f"Task at index {index} missing valid task_id")
        if task_id in task_map:
            raise ValueError(f"Duplicate task_id: {task_id}")

        task_map[task_id] = task
        indegree[task_id] = 0
        graph[task_id] = []

    for task_id, task in task_map.items():
        depends_on = task.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list):
            raise ValueError(f"Task {task_id} field depends_on must be a list")

        normalized_depends_on: list[int] = []
        for dep in depends_on:
            dep_id = _parse_task_id(dep)
            if dep_id is None or dep_id not in task_map:
                raise ValueError(f"Task {task_id} depends_on unknown task_id: {dep}")
            normalized_depends_on.append(dep_id)

        for dep_id in normalized_depends_on:
            graph[dep_id].append(task_id)
            indegree[task_id] += 1

    queue = sorted([task_id for task_id, degree in indegree.items() if degree == 0])
    sorted_ids: list[int] = []

    while queue:
        current = queue.pop(0)
        sorted_ids.append(current)

        for next_id in sorted(graph[current]):
            indegree[next_id] -= 1
            if indegree[next_id] == 0:
                queue.append(next_id)
                queue.sort()

    if len(sorted_ids) != len(task_map):
        raise ValueError("Cycle detected in task dependencies")

    return [task_map[task_id] for task_id in sorted_ids]


def subtask_dispatcher_node(state: MainGraphState) -> dict:
    task_source = (
        state.get("operation_task_blocks_result")
        or state.get("validated_task_blocks_result")
        or state.get("task_blocks_result")
        or {}
    )
    tasks = task_source.get("tasks", []) if isinstance(task_source, dict) else []
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("subtask_dispatcher_node requires non-empty tasks")

    sorted_tasks = topological_sort_tasks([task for task in tasks if isinstance(task, dict)])
    execution_order = [_parse_task_id(task.get("task_id")) for task in sorted_tasks]
    execution_order = [task_id for task_id in execution_order if task_id is not None]

    _debug_print("--- Node: Subtask Dispatcher ---")
    _debug_print(f"Total subtasks: {len(sorted_tasks)}")
    _debug_print(f"Subtask execution order: {execution_order}")

    processed_tasks: list[dict[str, Any]] = []
    for task in sorted_tasks:
        task_id = task.get("task_id")
        task_type = task.get("task_type")
        _debug_print(f"Executing subtask {task_id}: {task_type}")

        try:
            single_task_state = {
                "task_blocks_result": {
                    "tasks": [deepcopy(task)],
                }
            }
            single_task_state.update(code_to_optimize_agent_node(single_task_state))
            single_task_state.update(variable_assignments_agent_node(single_task_state))
            single_task_state.update(multivectors_to_be_visualized_agent_node(single_task_state))

            final_result = single_task_state.get("multivectors_to_be_visualized_result", {})
            final_tasks = final_result.get("tasks", []) if isinstance(final_result, dict) else []
            if not isinstance(final_tasks, list) or not final_tasks:
                raise ValueError(f"subtask {task_id} did not produce a final task result")

            processed_task = final_tasks[0]
            if not isinstance(processed_task, dict):
                raise ValueError(f"subtask {task_id} final task result is invalid")

            processed_tasks.append(processed_task)
            _debug_print(f"Subtask {task_id} completed.")
        except Exception:
            raise

    _debug_print("All subtasks completed.")
    return {
        "subtask_execution_order": execution_order,
        "subtask_results": {
            "tasks": processed_tasks,
        }
    }


def _join_non_empty_blocks(blocks: list[str]) -> str:
    normalized = [block.strip() for block in blocks if isinstance(block, str) and block.strip()]
    return "\n".join(normalized)


def _normalize_optimize_code_block_for_gaalop(code: str) -> str:
    if not isinstance(code, str):
        return ""

    normalized_lines: list[str] = []
    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("?"):
            line = "?" + line[1:].replace("?", "")
        else:
            line = line.replace("?", "")
        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def final_code_assembler_node(state: MainGraphState) -> dict:
    subtask_results = state.get("subtask_results", {})
    tasks = subtask_results.get("tasks", []) if isinstance(subtask_results, dict) else []
    if not isinstance(tasks, list):
        tasks = []

    code_blocks: list[str] = []
    assignment_blocks: list[str] = []
    visualization_blocks: list[str] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue
        code_block = task.get("code_to_optimize", {})
        if isinstance(code_block, dict):
            code = compact_code_to_optimize(str(code_block.get("code") or "").strip())
            code = _normalize_optimize_code_block_for_gaalop(code)
            if code:
                code_blocks.append(code)

        assignment_block = task.get("variable_assignments", {})
        if isinstance(assignment_block, dict):
            assignment_code = compact_variable_assignments(str(assignment_block.get("code") or "").strip())
            if assignment_code and assignment_code != "No variable assignments.":
                assignment_blocks.append(assignment_code)

        visualization_block = task.get("multivectors_to_be_visualized", {})
        if isinstance(visualization_block, dict):
            visualization_code = str(visualization_block.get("code") or "").strip()
            if visualization_code:
                visualization_blocks.append(visualization_code)

    code_to_optimize_block = _join_non_empty_blocks(code_blocks)

    if assignment_blocks:
        variable_assignments_block = _join_non_empty_blocks(assignment_blocks)
    else:
        variable_assignments_block = "No variable assignments."

    real_visualization_blocks = [
        block.strip()
        for block in visualization_blocks
        if block.strip() and block.strip() != "No need for visualization."
    ]
    if real_visualization_blocks:
        visualization_block = _join_non_empty_blocks(real_visualization_blocks)
    else:
        visualization_block = "No need for visualization."

    final_code = (
        "Code to optimize:\n"
        f"{code_to_optimize_block}\n\n"
        "Variable assignments:\n"
        f"{variable_assignments_block}\n\n"
        "Multivectors to be visualized:\n"
        f"{visualization_block}"
    )

    return {
        "final_code": final_code,
    }


def split_final_code_sections(final_code: str) -> dict[str, str]:
    text = str(final_code or "")
    code_header = "Code to optimize:"
    assignments_header = "Variable assignments:"
    visualization_header = "Multivectors to be visualized:"

    code_start = text.find(code_header)
    assignments_start = text.find(assignments_header)
    visualization_start = text.find(visualization_header)

    if code_start == -1 or assignments_start == -1 or visualization_start == -1:
        raise ValueError("final_code is missing required section headers")
    if not (code_start < assignments_start < visualization_start):
        raise ValueError("final_code section headers are in invalid order")

    optimize_code = text[code_start + len(code_header) : assignments_start].strip()
    variable_assignments = text[
        assignments_start + len(assignments_header) : visualization_start
    ].strip()
    multivectors_visualized = text[visualization_start + len(visualization_header) :].strip()

    if variable_assignments == "No variable assignments.":
        variable_assignments = ""
    if multivectors_visualized == "No need for visualization.":
        multivectors_visualized = ""

    return {
        "optimizeCode": optimize_code,
        "variableAssignments": variable_assignments,
        "multivectorsVisualized": multivectors_visualized,
    }


def has_visualization_code(multivectors_visualized: str) -> bool:
    if multivectors_visualized is None:
        return False

    text = str(multivectors_visualized).strip()
    if not text:
        return False

    no_visualization_values = {
        "no need for visualization",
        "no need for visualization.",
        "no visualization needed",
        "no visualization needed.",
    }
    if text.lower() in no_visualization_values:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if any(line.startswith(":") for line in lines):
        return True

    return True


def resolve_optimization_settings(user_input: str) -> dict[str, bool]:
    text = str(user_input or "").strip().lower()

    def is_enabled(name: str) -> bool:
        negative_patterns = [
            rf"\b(?:no|without|disable|disabled|turn off|turn_off|do not use|don't use)\s+{name}\b",
            rf"\b{name}\s+(?:disabled|off)\b",
        ]
        for pattern in negative_patterns:
            if re.search(pattern, text):
                return False
        return bool(re.search(rf"\b{name}\b", text))

    return {
        "maxima": is_enabled("maxima"),
        "cse": is_enabled("cse"),
    }


def resolve_visual_plugin(target_space: str) -> str:
    normalized = str(target_space or "").strip().upper()
    if normalized in {"ALGEBRA_2D", "ALGEBRA_2D_PGA"}:
        return "INSERTER_2D"
    return "INSERTER"


def gaalop_request_builder_node(state: MainGraphState) -> dict:
    final_code = str(state.get("final_code") or "").strip()
    if not final_code:
        raise ValueError("gaalop_request_builder_node requires final_code")

    function_name = str(state.get("function_name") or "GeneratedFunction").strip() or "GeneratedFunction"
    target_language = str(state.get("target_language") or "JAVA").strip().upper() or "JAVA"
    raw_target_space = str(state.get("target_space") or "").strip()
    if raw_target_space and raw_target_space.lower() != "unknown":
        target_space = raw_target_space
    else:
        target_space = "ALGEBRA_CGA"

    sections = split_final_code_sections(final_code)
    multivectors_visualized = sections["multivectorsVisualized"]
    visualization_enabled = has_visualization_code(multivectors_visualized)
    output_mode = "CODE_AND_VISUALIZATION" if visualization_enabled else "CODE_ONLY"
    optimization_settings = resolve_optimization_settings(state.get("user_input") or "")

    gaalop_request_result = {
        "visualizationEnabled": visualization_enabled,
        "outputMode": output_mode,
        "codegenPlugins": target_language,
        "algebraPlugins": target_space,
        "optimization": optimization_settings,
        "script": {
            "multivectorsVisualized": multivectors_visualized,
            "optimizeCode": sections["optimizeCode"],
            "variableAssignments": sections["variableAssignments"],
            "functionName": function_name,
        },
    }

    return {
        "gaalop_request_result": gaalop_request_result,
    }


def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def call_gaalop_compile_api(
    payload: dict,
    *,
    api_url: str = GAALOP_COMPILE_API_URL,
    timeout: int = 120,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required to call the Gaalop compile API. Install it with: pip install requests")

    safe_payload = make_json_safe(payload)
    response = requests.post(
        api_url,
        json=safe_payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response_text = response.text
    response_json: Any = None
    try:
        response_json = response.json()
    except Exception:
        response_json = None

    success = 200 <= response.status_code < 300
    if isinstance(response_json, dict) and "statusCode" in response_json:
        success = success and str(response_json.get("statusCode")) == "200"

    return {
        "success": success,
        "api_url": api_url,
        "http_status": response.status_code,
        "response_headers": dict(response.headers),
        "response_text": response_text,
        "response_json": response_json,
    }


def _compile_error_text(result: dict[str, Any]) -> str:
    response_json = result.get("response_json")
    if isinstance(response_json, dict):
        message_parts = []
        for key in ("message", "error", "detail", "stackTrace"):
            value = response_json.get(key)
            if value:
                message_parts.append(str(value))
        if message_parts:
            return "\n".join(message_parts)
    return str(result.get("response_text") or result.get("http_status") or "Unknown Gaalop compile error")


def build_gaalop_script_repair_prompt(
    *,
    user_input: str,
    request_payload: dict,
    compile_result: dict[str, Any],
) -> str:
    return GAALOP_SCRIPT_REPAIR_TEMPLATE.format(
        user_input=user_input,
        request_payload=json.dumps(make_json_safe(request_payload), ensure_ascii=False, indent=2),
        compile_error=_compile_error_text(compile_result),
    )


def _normalize_repaired_script_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def repair_gaalop_script_with_llm(
    state: MainGraphState,
    request_payload: dict,
    compile_result: dict[str, Any],
) -> dict[str, str]:
    prompt = build_gaalop_script_repair_prompt(
        user_input=str(state.get("user_input") or ""),
        request_payload=request_payload,
        compile_result=compile_result,
    )
    llm = _get_default_llm()
    result = invoke_llm_with_retry(
        llm,
        prompt,
        node_name="gaalop_compile_node",
        max_retries=3,
        base_sleep_seconds=2.0,
    )
    parsed = parse_json_object(str(getattr(result, "content", "")))
    script = parsed.get("script") if isinstance(parsed.get("script"), dict) else parsed
    if not isinstance(script, dict):
        raise ValueError("GAALOP script repair result must be a JSON object")

    optimize_code = _normalize_repaired_script_value(script.get("optimizeCode"))
    if not optimize_code:
        raise ValueError("GAALOP script repair result missing optimizeCode")

    optimize_code = compact_code_to_optimize(optimize_code)
    optimize_code = _normalize_optimize_code_block_for_gaalop(optimize_code)
    variable_assignments = compact_variable_assignments(
        _normalize_repaired_script_value(script.get("variableAssignments"))
    )
    if variable_assignments == "No variable assignments.":
        variable_assignments = ""
    multivectors_visualized = _normalize_repaired_script_value(script.get("multivectorsVisualized"))
    if multivectors_visualized.lower() in {
        "no need for visualization",
        "no need for visualization.",
        "no visualization needed",
        "no visualization needed.",
    }:
        multivectors_visualized = ""

    return {
        "optimizeCode": optimize_code,
        "variableAssignments": variable_assignments,
        "multivectorsVisualized": multivectors_visualized,
    }


def _final_code_from_request(request_payload: dict) -> str:
    script = request_payload.get("script", {}) if isinstance(request_payload, dict) else {}
    optimize_code = str(script.get("optimizeCode") or "").strip() if isinstance(script, dict) else ""
    variable_assignments = str(script.get("variableAssignments") or "").strip() if isinstance(script, dict) else ""
    multivectors_visualized = str(script.get("multivectorsVisualized") or "").strip() if isinstance(script, dict) else ""
    return (
        "Code to optimize:\n"
        f"{optimize_code}\n\n"
        "Variable assignments:\n"
        f"{variable_assignments or 'No variable assignments.'}\n\n"
        "Multivectors to be visualized:\n"
        f"{multivectors_visualized or 'No need for visualization.'}"
    )


def _apply_repaired_script_to_request(request_payload: dict, repaired_script: dict[str, str]) -> dict:
    updated = deepcopy(request_payload)
    script = updated.setdefault("script", {})
    if not isinstance(script, dict):
        raise ValueError("gaalop request script must be a dict")

    script["optimizeCode"] = repaired_script["optimizeCode"]
    script["variableAssignments"] = repaired_script["variableAssignments"]
    script["multivectorsVisualized"] = repaired_script["multivectorsVisualized"]

    visualization_enabled = has_visualization_code(script["multivectorsVisualized"])
    updated["visualizationEnabled"] = visualization_enabled
    updated["outputMode"] = "CODE_AND_VISUALIZATION" if visualization_enabled else "CODE_ONLY"
    return updated


def gaalop_compile_node(state: MainGraphState) -> dict:
    request_payload = state.get("gaalop_request_result")
    if not isinstance(request_payload, dict):
        raise ValueError("gaalop_compile_node requires gaalop_request_result")

    current_request = deepcopy(request_payload)
    attempts: list[dict[str, Any]] = []
    repaired_scripts: list[dict[str, str]] = []

    for attempt in range(MAX_GAALOP_SCRIPT_REPAIR_RETRIES + 1):
        compile_result = call_gaalop_compile_api(current_request)
        attempts.append(compile_result)

        if compile_result.get("success"):
            return {
                "gaalop_request_result": current_request,
                "gaalop_compile_result": compile_result,
                "gaalop_compile_attempts": attempts,
                "gaalop_script_repair_count": attempt,
                "gaalop_script_repairs": repaired_scripts,
                "final_code": _final_code_from_request(current_request),
            }

        if attempt >= MAX_GAALOP_SCRIPT_REPAIR_RETRIES:
            raise ValueError(
                "Gaalop compile failed after "
                f"{MAX_GAALOP_SCRIPT_REPAIR_RETRIES} script repair retries: "
                f"{_compile_error_text(compile_result)}"
            )

        print(
            "[gaalop_compile_node] compile failed, "
            f"repair {attempt + 1}/{MAX_GAALOP_SCRIPT_REPAIR_RETRIES} with backend error feedback."
        )
        repaired_script = repair_gaalop_script_with_llm(state, current_request, compile_result)
        repaired_scripts.append(repaired_script)
        current_request = _apply_repaired_script_to_request(current_request, repaired_script)

    raise RuntimeError("unreachable gaalop_compile_node state")
