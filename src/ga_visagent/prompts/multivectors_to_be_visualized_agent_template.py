import json


MULTIVECTORS_TO_BE_VISUALIZED_RULES_TEXT = """
1. Output only pure visualization code.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Do not output Code to optimize.
7. Do not output Variable assignments.
8. If visualization is not required, output exactly: No need for visualization.
9. If visualization is required, output only lines in the form :Color; or :Variable;
10. Every non-empty visualization line must end with semicolon.
11. If an object has a color, output the color line first and then the object line.
12. If an object has no color but visualization is required, use Red as the default color and output the color line first.
13. Do not invent objects.
14. Do not output any line starting with ?.
15. Do not output assignment statements such as a1=1;.
16. Keep the object order from the input.
17. Normalize common color names such as red -> Red and BLUE -> Blue.
18. Valid examples:
    :Red;
    :P1;
    :Blue;
    :S1;
19. Forbidden examples:
    ?P1 = ...;
    a1=1;
    createPoint(...);
    No visualization needed.
""".strip()


MULTIVECTORS_TO_BE_VISUALIZED_AGENT_TEMPLATE = """
Role:
You are a GAALOPScript Multivectors to be visualized generator.

Task:
Generate only the Multivectors to be visualized code for the current task.

You must strictly follow the rules below.

# Multivectors to be visualized Generation Rules:
{rules_text}

# Context:
task_id: {task_id}
task_type: {task_type}

# Current multivectors_to_be_visualized block:
{multivectors_to_be_visualized_json}

# Output Requirements:
1. Output only pure visualization statements.
2. Do not output JSON.
3. Do not output markdown.
4. Do not output explanations.
5. Do not output comments.
6. Do not output Code to optimize.
7. Do not output Variable assignments.
8. Do not output any line starting with ?.
9. Do not output assignment statements like a1=1;.
10. Every non-empty visualization line must end with semicolon.
11. If visualization is not required, output exactly: No need for visualization.
12. If a required visualization object has no color, use Red as the default color.
13. Use only the current multivectors_to_be_visualized block.

Multivectors to be visualized code:
""".strip()


def build_multivectors_to_be_visualized_agent_prompt(
    rules_text: str,
    task_id: int | str,
    task_type: str,
    multivectors_to_be_visualized: dict,
) -> str:
    prompt = MULTIVECTORS_TO_BE_VISUALIZED_AGENT_TEMPLATE
    prompt = prompt.replace("{rules_text}", rules_text)
    prompt = prompt.replace("{task_id}", str(task_id))
    prompt = prompt.replace("{task_type}", str(task_type))
    prompt = prompt.replace(
        "{multivectors_to_be_visualized_json}",
        json.dumps(multivectors_to_be_visualized, ensure_ascii=False, indent=2),
    )
    return prompt
