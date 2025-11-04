from langchain_core.prompts import PromptTemplate

# 4. Assignment code generation template
ASSIGNMENT_CODE_GENERATION_TEMPLATE = """
You are a code assignment assistant among GAALOPScript experts, responsible for generating variable assignment code based on analysis results.
Analysis results: {assignment_result}
# Generation rules:
- Generate all assignment statements based on the "specific values corresponding to placeholders".
- Each assignment statement must end with a semicolon (e.g.: `a1=0;`).
- Group related assignment statements together, which can be placed in one line or multiple lines to ensure readability.
- Do not output any additional explanations or headings, only output the assignment code.

Assignment code:
"""
assignment_code_generation_prompt = PromptTemplate(
    input_variables=["assignment_result"],
    template=ASSIGNMENT_CODE_GENERATION_TEMPLATE
)