import requests
import json


def generate_gaalop_code(
    gaalop_script: str,
    function_name: str,
    target_language: str = "JAVA",
    target_space: str = "ALGEBRA_3D_PGA"
) -> dict | None:
    """
    Parses a formatted GAALOPScript string, sends it to the Gaalop API,
    and returns the generated code.

   Parameters:
    gaalop_script: A string containing the full script (with header).
    function_name: The name to be used for generating the function.
    target_language: The target language for code generation (e.g., "JAVA", "CPP").
    target_space: The target geometric algebra space (e.g., "ALGEBRA_3D_PGA").


    Returns:
        The API's JSON response (as a dictionary), or None if an error occurs.
    """
    # Target API endpoint (replace if necessary)
    api_url = "http://XXX:XXX/gaalop/codeGenerate"

    # 从result字符串中提取各部分
    def extract_sections(script_str: str) -> tuple[str, str, str]:
        # ... (此内部函数无需改动)
        try:
            script_str = script_str.replace('\r\n', '\n')
            parts = script_str.split("Variable assignments:")
            code_part = parts[0].replace("Code to optimize:", "").strip()
            parts2 = parts[1].split("Multivectors to be visualized:")
            assignment_part = parts2[0].strip()
            visualization_part = parts2[1].strip()
            if not assignment_part:
                assignment_part = ""
            if not visualization_part or "无需可视化" in visualization_part:
                visualization_part = ""
            return code_part, assignment_part, visualization_part
        except Exception as e:
            print(f"解析GAALOPScript各部分时出错: {e}")
            return "", "", ""

    # 提取各部分内容
    optimizeCode, variableAssignments, multivectorsVisualized = extract_sections(gaalop_script)

    # 构造请求头
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # 构造请求数据（JSON格式）
    payload = {
        "gaalopPlugins": {
            "algebraPlugins": target_space.upper(),
            "codegenPlugins": target_language.upper(),
            "optPlugins": "TBA",
            "visualPlugins": "GANJA"
        },
        "gaalopScript": {
            "optimizeCode": optimizeCode,
            "variableAssignments": variableAssignments,
            "multivectorsVisualized": multivectorsVisualized
        },
        "functionName": function_name,
        "visualization": True
    }

    try:
        # 发送POST请求
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),  # 将字典转换为JSON字符串
            timeout=30  # 设置超时时间（秒）
        )

        # 检查响应状态
        if response.status_code == 200:
            print("请求成功！")
            print("响应内容:")
            print(json.dumps(response.json(), indent=2))  # 美化输出JSON响应
            return response.json()  # 返回响应内容
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"请求发生异常: {str(e)}")
        return None
    except json.JSONDecodeError:
        print("响应不是有效的JSON格式")
        return None

