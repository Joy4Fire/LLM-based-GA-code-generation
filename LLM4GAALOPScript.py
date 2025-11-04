import json

from models.llm_setup import get_llm
from utils.Gaalop import generate_gaalop_code

from utils.json_utils import JsonFileHandler

from langchain_core.prompts import PromptTemplate

# 4. Assignment code generation template
LLM4GAALOPScript_TEMPLATE = """
您是几何代数领域GAALOPScript的专家。您的任务是理解用户输入并生成GAALOPScript代码，该代码由三个组件组成：优化的计算代码、变量赋值语句和要可视化的多向量。同时你需要抽取用户语言里面的函数名（function_name）、目标语言（target_language）和目标空间（target_space）。
# Extraction Rules:
1.   **Function name**: Look for the function name specified by the user.  If not explicitly specified, generate a meaningful CamelCase name based on the content of the user's request, e.g., `calculateIntersection`.
2.   **Target language**: Look for the target programming language specified by the user.  If not explicitly specified, the default is "JAVA".  The range of values is: CLUCALC, JULIA, VERILOG, GAPP_DEBUGGER, CSHARP, RUST, JAVA, VIS2D, GAALET_OUTPUT, GAPP, COMPRESSED, VISUALIZER, GANJA, GAPP_OPENCL, PYTHON, MATLAB, DOT, MATHematica, CPP, LATEX
3.   **Target space**: Look for the geometric algebra space specified by the user.  If not explicitly specified, the default is "ALGEBRA_3D_PGA".  The range of values is: ALGEBRA_2D, ALGEBRA_3D, ALGEBRA_2D_PGA, ALGEBRA_3D_PGA, ALGEBRA_CRA, ALGEBRA_STA, ALGEBRA_CGA, ALGEBRA_GAC, ALGEBRA_DCGA, ALGEBRA_CCGA, ALGEBRA_QGA

# Output Format:
Must output in the format of a pure JSON object without any additional explanations.
Example:
{{
"function_name": "calculateIntersection",
"target_language": "JAVA",
"target_space": "ALGEBRA_3D_PGA",
"GAALOPScript:": "
Code to optimize:\n
xxx\n\
xxx\n\
xxx\n\

Variable assignments:\n
xxx\n
xxx\n\
xxx\n\

Multivectors to be visualized:\n
xxx\n\
xxx\n\
xxx\n\
"
}}


用户提供的任务描述和GA公式如下：
{user_input}

"""
LLM4GAALOPScript_prompt = PromptTemplate(
    input_variables=["assignment_result"],
    template=LLM4GAALOPScript_TEMPLATE
)

llm = get_llm(
    llm_type="openai",
    model="Qwen3-30B-A3B-Instruct-2507",
    api_key="sk-123",
    base_url="http://172.21.252.62:8001/v1"
)


if __name__ == "__main__":
    chain = LLM4GAALOPScript_prompt | llm

    questions = JsonFileHandler.read_json(r"./data/question.json")
    results = []
    index = 0
    for question in questions["conformal_space_tasks"]:
        try:
            input_with_values = question
            llm_response = chain.invoke({"user_input": input_with_values})
            gaalop_llm_response = json.loads(llm_response.content)            #  gaalop_script: str,

            gaalop_response = generate_gaalop_code(gaalop_script= gaalop_llm_response["GAALOPScript"], function_name= gaalop_llm_response["function_name"], target_language=gaalop_llm_response["target_language"], target_space=gaalop_llm_response["target_space"])
            if gaalop_response.get('api_response_code')['statusCode'] == "200":
                index += 1
                result = {
                    "question": question,
                    "result":  True
                }
            else:
                result = {
                    "question": question,
                    "result": False
                }
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            result = {
                "question": question,
                "result": False
            }
    JsonFileHandler.save_json(results, "./data/questions_llm.json")



