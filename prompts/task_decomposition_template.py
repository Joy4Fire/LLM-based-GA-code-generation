from langchain_core.prompts import PromptTemplate

# Task decomposition template: Decomposes the user's request into structured sub-tasks.
TASK_DECOMPOSITION_TEMPLATE = """
Role: You are a GAALOPScript expert, and your task is a geometric algebra task decomposer.
Input: {user_input}
Output: An ordered list of sub-tasks in JSON format.

Task Decomposition Rules:
1. Identify the left-hand side of the formula as the final goal.
2. Scan all variables on the right-hand side.
3. Create sub-tasks for each undefined variable (excluding predefined constants like e∞).
4. Sort by dependency: basic variables → composite objects.
5. Decompose to the most basic points or rotors. Common geometric algebra objects are represented by letters: P for points, L for lines, C for circles, S for spheres, ρ for radii, π for planes, PP for point pairs, N for rotors.
6. Carefully check if the task description contains any specific coordinates, radii, angles, or other numerical values.
   - If yes, the "whether contains specific values" field must clearly list these values.
   - If no, this field must be "no".

Here are some reference examples:
Example 1:
Input: In conformal space, create a sphere S with center at (1,1,1) and radius 2.0, and visualize it as yellow. I need Python Code. Calculation process: 1. Sphere representation: $$ S=C-\\frac{{1}}{{2}}r^{{2}}e_{{\\infty}} $$
Output: ```json
[
  {{
    "Id": 1,
    "Task Description": "Define the sphere center as the point (1, 1, 1) in the conformal space as the point C.",
    "Final variable name": "C",
    "The name of the dependent variable": [],
    "whether contains specific values": "yes, center coordinates: (1, 1, 1)"
  }},
  {{
    "Id": 2,
    "Task Description": "Compute the scalar term (1/2) * r², where the radius r = 2.0.",
    "Final variable name": "scalar_term",
    "The name of the dependent variable": [],
    "whether contains specific values": "yes, r = 2.0"
  }},
  {{
    "Id": 3,
    "Task Description": "Multiply the scalar term by the conformal basis vector e_infinity to obtain (1/2)*r²*e_infinity.",
    "Final variable name": "radial_term",
    "The name of the dependent variable": ["scalar_term"],
    "whether contains specific values": "no"
  }},
  {{
    "Id": 4,
    "Task Description": "Construct the sphere S using the conformal geometric algebra formula S = C - (1/2)*r²*e_infinity.",
    "Final variable name": "S",
    "The name of the dependent variable": ["C", "radial_term"],
    "whether contains specific values": "no"
  }},
  {{
    "Id": 5,
    "Task Description": "Render the sphere S in 3D visualization as yellow by assigning the color attribute 'yellow' to the geometric object S.",
    "Final variable name": "S",
    "The name of the dependent variable": ["S"],
    "whether contains specific values": "yes, color value: yellow"
  }}
]
```

Output Format:The output format is an array containing each decomposed sub-task, where each sub-task must satisfy the following format:
- Id: [sub-task serial number]
- Task Description: [detailed task description, e.g., create a point]
- Final variable name: [final variable name]
- The name of the dependent variable: [dependent variable names]
- whether contains specific values: "[If yes, must list all specific values of variables, e.g.: ' yes，X_1 (0,0,0), r1=0.5'. If no, then ' no ']

"""
task_decomposition_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=TASK_DECOMPOSITION_TEMPLATE
)
