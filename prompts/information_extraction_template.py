from langchain_core.prompts import PromptTemplate

# Information extraction template, which guides the LLM to extract structured information from user input
INFORMATION_EXTRACTION_TEMPLATE = """
Role: You are an efficient information extraction assistant.
Task: Precisely extract three key pieces of information from the following user input: function name (function_name), target language (target_language), and target space (target_space).

# Extraction Rules:
1.  **Function name**: Look for the function name specified by the user. If not explicitly specified, generate a meaningful CamelCase name based on the content of the user's request, e.g., `calculateIntersection`.
2.  **Target language**: Look for the target programming language specified by the user. If not explicitly specified, the default is "JAVA". The range of values is: CLUCALC, JULIA, VERILOG, GAPP_DEBUGGER, CSHARP, RUST, JAVA, VIS2D, GAALET_OUTPUT, GAPP, COMPRESSED, VISUALIZER, GANJA, GAPP_OPENCL, PYTHON, MATLAB, DOT, MATHematica, CPP, LATEX
3.  **Target space**: Look for the geometric algebra space specified by the user. If not explicitly specified, the default is "ALGEBRA_3D_PGA". The range of values is: ALGEBRA_2D, ALGEBRA_3D, ALGEBRA_2D_PGA, ALGEBRA_3D_PGA, ALGEBRA_CRA, ALGEBRA_STA, ALGEBRA_CGA, ALGEBRA_GAC, ALGEBRA_DCGA, ALGEBRA_CCGA, ALGEBRA_QGA 

# Output Format:
Must output in the format of a pure JSON object without any additional explanations.
Example:
{{
    "function_name": "calculateIntersection",
    "target_language": "JAVA",
    "target_space": "ALGEBRA_3D_PGA"
}}

---
User input:
"{user_input}"
---

JSON output:
"""
information_extraction_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=INFORMATION_EXTRACTION_TEMPLATE
)