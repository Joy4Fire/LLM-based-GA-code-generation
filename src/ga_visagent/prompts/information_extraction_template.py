INFORMATION_EXTRACTION_TEMPLATE = """
Role: You are an efficient information extraction assistant.
Task: Precisely extract three key pieces of information from the following user input: function name (function_name), target language (target_language), and target space (target_space).

# Extraction Rules:
1. **Function name**: Look for the function name specified by the user. If not explicitly specified, generate a meaningful CamelCase name based on the content of the user's request, e.g., `calculateIntersection`.
2. **Target language**: Look for the target programming language specified by the user. If not explicitly specified, the default is "PYTHON". The range of values is: CLUCALC, JULIA, VERILOG, GAPP_DEBUGGER, CSHARP, RUST, JAVA, VIS2D, GAALET_OUTPUT, GAPP, COMPRESSED, VISUALIZER, GANJA, GAPP_OPENCL, PYTHON, MATLAB, DOT, MATHematica, CPP, LATEX.
3. **Target space**: Look for the geometric algebra space specified by the user. If not explicitly specified, return "unknown". The range of values is: ALGEBRA_2D, ALGEBRA_3D, ALGEBRA_2D_PGA, ALGEBRA_3D_PGA, ALGEBRA_CRA, ALGEBRA_STA, ALGEBRA_CGA, ALGEBRA_GAC, ALGEBRA_DCGA, ALGEBRA_CCGA, ALGEBRA_QGA, unknown.

# Output Format:
Must output a pure JSON object without markdown or explanations.
Example:
{
  "function_name": "calculateIntersection",
  "target_language": "PYTHON",
  "target_space": "ALGEBRA_CGA"
}

---
User input:
"{user_input}"
---

JSON output:
""".strip()


def build_information_extraction_prompt(user_input: str) -> str:
    return INFORMATION_EXTRACTION_TEMPLATE.replace("{user_input}", user_input)
