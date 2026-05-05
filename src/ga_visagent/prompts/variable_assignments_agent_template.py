import json


VARIABLE_ASSIGNMENTS_RULES_TEXT = """
1. Output only pure Variable assignments code.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Every non-empty line must end with semicolon.
7. Only generate plain assignment statements in the form variable=value;
8. Do not output any line starting with ?.
9. Do not output any line starting with :.
10. Do not output visualization code.
11. Do not output Code to optimize.
12. Skip null values.
13. If all values are null or invalid, output an empty string.
14. Numeric strings such as "1" or "0.5" may be emitted as 1 or 0.5.
15. Invalid values such as booleans, objects, arrays, or non-numeric strings must not be emitted.
16. Keep the input variable order when possible.
17. Valid examples:
    a1=1;
    b1=1;
    c1=1;
    r1v=0.5;
18. Forbidden examples:
    ?P1 = ...;
    :Red;
    a1=null;
    print(a1);
""".strip()


VARIABLE_ASSIGNMENTS_AGENT_TEMPLATE = """
Role:
You are a GAALOPScript Variable assignments generator.

Task:
Generate only the Variable assignments code for the current task.

You must strictly follow the rules below.

# Variable assignments Generation Rules:
{rules_text}

# Context:
task_id: {task_id}
task_type: {task_type}

# Current variable_assignments block:
{variable_assignments_json}

# Output Requirements:
1. Output only pure assignment statements.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Do not output Code to optimize.
7. Do not output visualization code.
8. Do not output any line starting with ?.
9. Do not output any line starting with :.
10. Every non-empty line must end with semicolon.
11. Use only the current variable_assignments block.

Variable assignments code:
""".strip()


def build_variable_assignments_agent_prompt(
    rules_text: str,
    task_id: int | str,
    task_type: str,
    variable_assignments: dict,
) -> str:
    prompt = VARIABLE_ASSIGNMENTS_AGENT_TEMPLATE
    prompt = prompt.replace("{rules_text}", rules_text)
    prompt = prompt.replace("{task_id}", str(task_id))
    prompt = prompt.replace("{task_type}", str(task_type))
    prompt = prompt.replace(
        "{variable_assignments_json}",
        json.dumps(variable_assignments, ensure_ascii=False, indent=2),
    )
    return prompt
