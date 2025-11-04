# nodes.py
import json
from langchain_core.runnables import RunnablePassthrough

from agents.state import AgentState
# Import all prompt templates
from prompts.assignment_code_generation_template import assignment_code_generation_prompt
from prompts.assignment_extraction_template import assignment_extraction_prompt
from prompts.code_generation_template import code_generation_prompt
from prompts.code_integration_template import code_integration_prompt
from prompts.information_extraction_template import information_extraction_prompt
from prompts.task_decomposition_template import task_decomposition_prompt
from prompts.visualization_code_generation_template import visualization_code_generation_prompt


from models.llm_setup import get_llm
from utils.Gaalop import generate_gaalop_code

# Initialize LLM (Language Model) with specified parameters
llm = get_llm(
    llm_type="openai",
    model="XXX",
    api_key="sk-XXX",
    base_url="http://XXX:XXX/v1"
)


# --- Define node functions for each Agent ---
def analyze_task_node(state: AgentState) -> dict:
    """
    Node 1: Task Analysis
    Decomposes user input into structured subtasks using the task decomposition prompt.
    Returns the analysis result to update the agent state.
    """
    print("--- Node: Task Analysis ---")
    user_input = state['user_input']
    chain = task_decomposition_prompt | llm
    result = chain.invoke({"user_input": user_input})
    return {"analysis_result": result.content}


def generate_code_node(state: AgentState) -> dict:
    """
    Node 2: Generate Main Code
    Generates GAALOPScript main body (with placeholders) based on analysis results.
    Cleans the output by removing code block markers before returning.
    """
    print("--- Node: Generate Main Code ---")
    analysis_result = state['analysis_result']
    chain = code_generation_prompt | llm
    result = chain.invoke({"analysis_result": analysis_result})
    # Clean up code by removing possible code block wrappers
    clean_code = result.content.strip().replace("```gaalopscript", "").replace("```", "").strip()
    return {"generated_code": clean_code}


def extract_assignments_node(state: AgentState) -> dict:
    """
    Node 3: Extract Assignment Structure
    Extracts placeholder variables and their corresponding values from analysis results
    and generated code, preparing for assignment statement generation.
    """
    print("--- Node: Extract Assignment Structure ---")
    analysis_result = state['analysis_result']
    generated_code = state['generated_code']
    chain = assignment_extraction_prompt | llm
    result = chain.invoke({"analysis_result": analysis_result, "vd_code": generated_code})
    return {"assignment_result": result.content}


def generate_assignment_code_node(state: AgentState) -> dict:
    """
    Node 4: Generate Assignment Code
    Generates specific variable assignment statements based on extracted assignment results.
    Returns empty string if no visualization is needed or results are invalid.
    """
    print("--- Node: Generate Assignment Code ---")
    assignment_result = state['assignment_result']
    # Check for special cases where no assignment is needed
    if "No visualization needed" in assignment_result or "null" in assignment_result:
        return {"assignment_code": ""}
    chain = assignment_code_generation_prompt | llm
    result = chain.invoke({"assignment_result": assignment_result})
    return {"assignment_code": result.content.strip()}


def generate_visualization_code_node(state: AgentState) -> dict:
    """
    Node 5: Generate Visualization Code
    Creates visualization code for geometric algebra objects based on analysis results.
    Returns empty string if visualization is not required.
    """
    print("--- Node: Generate Visualization Code ---")
    analysis_result = state['analysis_result']
    chain = visualization_code_generation_prompt | llm
    result = chain.invoke({"analysis_result": analysis_result})
    # Return empty if visualization is not needed
    if "No visualization needed" in result.content:
        return {"visualization_code": ""}
    return {"visualization_code": result.content.strip()}


def integrate_code_node(state: AgentState) -> dict:
    """
    Node 6: Integrate Final Code
    (Fixed) Reuses LLM with an unambiguous, context-rich prompt to combine code components.
    Merges main code, assignments, and visualization code into a complete script.
    """
    print("--- Node: Integrate Final Code (Using LLM) ---")
    generated_code = state.get('generated_code', "").strip()
    assignment_code = state.get('assignment_code', "").strip()
    visualization_code = state.get('visualization_code', "").strip()

    # Invoke LLM for controlled final integration
    chain = code_integration_prompt | llm
    result = chain.invoke({
        "generated_code": generated_code,
        "assignment_code": assignment_code,
        "visualization_code": visualization_code
    })

    # Clean up possible extra explanations from LLM
    final_output = result.content.strip()
    return {"final_code": final_output}


# ----------------------------------------------------
# New first agent: Information Extractor
# ----------------------------------------------------
def extract_information_node(state: AgentState) -> dict:
    """
    Agent 1: Extract structured task parameters from user input.
    Extracts function name, target language, and target geometric algebra space
    using the information extraction prompt, returning parsed JSON data.
    """
    print("--- Agent 1: Information Extraction ---")
    user_input = state['user_input']

    chain = information_extraction_prompt | llm
    result = chain.invoke({"user_input": user_input})

    try:
        # Clean and parse JSON string from LLM response
        clean_json_str = result.content.strip().replace("```json", "").replace("```", "").strip()
        extracted_info = json.loads(clean_json_str)

        print(f"Extracted information: {extracted_info}")

        # Update state with extracted information for subsequent nodes
        return {
            "function_name": extracted_info.get("function_name"),
            "target_language": extracted_info.get("target_language"),
            "target_space": extracted_info.get("target_space")
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Information extraction failed, unable to parse JSON - {e}")
        # Can set an error flag on failure
        return {}


def call_gaalop_api_node(state: AgentState) -> dict:
    """
    Node 7: Call external Gaalop API tool to generate final code.
    Uses integrated GAALOPScript, function name, target language, and target space
    to invoke the API and return the response.
    """
    print("--- Node: Call Gaalop API ---")
    gaalop_script = state.get('final_code')
    function_name = state.get('function_name')
    target_language = state.get('target_language')
    target_space = state.get('target_space')  # <-- Read new field

    # Validate required parameters
    if not all([gaalop_script, function_name, target_language, target_space]):
        print("Error: Cannot call API because GAALOPScript, function name, target language, or target space is empty.")
        return {"api_response_code": {"error": "Missing key parameters."}}

    # Call tool with new parameters
    api_response = generate_gaalop_code(
        gaalop_script=gaalop_script,
        function_name=function_name,
        target_language=target_language,
        target_space=target_space  # <-- Pass new parameter
    )

    return {"api_response_code": api_response}


# --- Define condition routing function ---
def router_node(state: AgentState) -> str:
    """
    Router Node: Determines workflow path based on presence of numerical values.
    Returns "with_assignment" if specific values are detected, otherwise "without_assignment".
    """
    print("--- Node: Router ---")
    analysis_str = state['analysis_result']
    try:
        clean_json_str = analysis_str.strip().replace("```json", "").replace("```", "").strip()
        analysis_json = json.loads(clean_json_str)
        for task in analysis_json:
            if not isinstance(task, dict):
                continue
            # Use English key consistent with translated prompts
            value = task.get("whether contains specific values")
            if isinstance(value, bool) and value:
                print(">>> Decision: Boolean 'True' detected, executing full workflow.")
                return "with_assignment"
            if isinstance(value, str) and value.lower().strip() != "No":
                print(">>> Decision: Specific value description detected, executing full workflow.")
                return "with_assignment"
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f">>> Decision Warning: Error parsing analysis results ({e}), defaulting to simplified workflow.")
    print(">>> Decision: No numerical values detected, executing simplified workflow.")
    return "without_assignment"