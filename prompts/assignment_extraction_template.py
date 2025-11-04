from langchain_core.prompts import PromptTemplate

# 3. Assignment structure extraction template
# Extracts placeholder variables and their corresponding numerical values.
ASSIGNMENT_EXTRACTION_TEMPLATE = """
You are a GAALOPScript expert, focusing on the following core tasks:
Input data:
    User input: {analysis_result}
    Code main body: {vd_code}
Processing flow:
    1. Scan {analysis_result} for valid numerical parameters
    2. If no valid numerical values are detected (do not make up values independently) → immediately output: No visualization needed
    3. If valid numerical values are detected, process and output according to the following requirements
Requirements when numerical values are detected:
    1. Value extraction: Precisely parse specific parameters of geometric objects (coordinates, angles, radii, etc.) from subtask descriptions;
    2. Variable mapping: Identify placeholder variables in the code template (e.g., x1, theta, rho) and establish a strict correspondence with numerical values;
    3. Standardized output: Generate non-redundant, uniformly formatted assignment statements in the following format (no need to integrate previously output results, this part is independent):
        Involved placeholders: [all placeholders]
        Specific values corresponding to placeholders: [placeholder=corresponding value, ....]
    4. Only output the standardized results without any comments or additional descriptions
"""
assignment_extraction_prompt = PromptTemplate(
    input_variables=["analysis_result", "vd_code"],
    template=ASSIGNMENT_EXTRACTION_TEMPLATE
)