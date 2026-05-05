import json


GAALOPSCRIPT_RULES_TEXT = """
1. Output only pure GAALOPScript code.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Do not output variable assignments with explicit numeric values.
7. Do not output visualization code.
8. Every non-empty line must end with semicolon.
9. The output must contain at least one ? variable.
10. Use only the formula, parameters, output, goal from the current code_to_optimize block.
11. If formula is complete and already contains a left-hand output such as P1 = ..., rewrite it as ?P1 = ...;
12. If formula has no left-hand output but output exists, generate ?{output} = {formula};
13. Do not replace placeholders such as a1, b1, c1 with numeric values.
14. Do not use Python syntax, Java syntax, JSON, markdown, or natural language.
15. Do not output pragma or visualization statements.
16. Valid examples:
    ?P1 = a1*e1 + b1*e2 + c1*e3 + 0.5*(a1*a1 + b1*b1 + c1*c1)*einf + e0;
    ?L = P1 ^ P2 ^ einf;
    ?P = *M;
    ?X4 = -(P + sqrt(P . P)) / (einf . P);
17. Forbidden examples:
    a1=1;
    :Red;
    :P1;
    #pragma output P1
    def foo():
""".strip()


CODE_TO_OPTIMIZE_AGENT_TEMPLATE = """
Role:
You are a GAALOPScript Code to optimize generator.

Task:
Generate only the Code to optimize GAALOPScript code for the current task.

You must strictly follow the rules below.

# GAALOPScript Code Generation Rules:
{rules_text}

# Context:
task_id: {task_id}
task_type: {task_type}

# Current code_to_optimize block:
{code_to_optimize_json}

# Output Requirements:
1. Output only pure GAALOPScript code.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Do not output variable assignments.
7. Do not output visualization code.
8. Every non-empty line must end with semicolon.
9. The output must contain at least one ? variable.
10. Use only the formula, parameters, output, goal from the current code_to_optimize block.

GAALOPScript code:
""".strip()


def build_code_to_optimize_agent_prompt(
    rules_text: str,
    task_id: int | str,
    task_type: str,
    code_to_optimize: dict,
) -> str:
    prompt = CODE_TO_OPTIMIZE_AGENT_TEMPLATE
    prompt = prompt.replace("{rules_text}", rules_text)
    prompt = prompt.replace("{task_id}", str(task_id))
    prompt = prompt.replace("{task_type}", str(task_type))
    prompt = prompt.replace(
        "{code_to_optimize_json}",
        json.dumps(code_to_optimize, ensure_ascii=False, indent=2),
    )
    return prompt
