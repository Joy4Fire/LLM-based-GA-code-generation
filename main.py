from agents.graph import create_graph

from utils.json_utils import JsonFileHandler

if __name__ == "__main__":
    # 创建可执行的 LangGraph 应用
    app = create_graph()

    # input_with_values = """
    # 创建一个点，并将其渲染成蓝绿色
    # """

    # input_with_values = """
    # # 在共形几何代数（CGA）空间中，定义四个点 P1、P2、P3 和 P4。计算经过这四个点的圆。
    # # 计算流程为：
    # # 1.  **在共形空间中定义四个点**：分别定义点 `P1`, `P2`, `P3`, `P4`。每个点都表示为基向量 `e1`, `e2`, `e3`（代表三维欧氏空间）, `einf`（代表无穷远点）和 `e0`（代表原点）的线性组合。
    # # 2.  **计算四点的外积**：计算这四个点的外积 `S = P1^P2^P3^P4`。在5D共形几何代数中，四个点的外积定义了一个经过这四个点的圆。
    # # 3.  **归一化结果**：将计算出的外积 `S` 进行归一化，得到最终的圆的表示 `C`。
    # # """

    # input_with_values = """
    # In conformal space, create three spheres 𝑆1, 𝑆2, 𝑆3 with centers at 𝑋_1 (1,1,2), 𝑋_2 (0,0.45,0), 𝑋_3 (0,0.45,0.2) and radii of 0.5, 0.4, and 0.3, respectively, 𝑆1, 𝑆2, 𝑆3 are visualized in blue, red, and black, respectively. Finally, calculate the intersection points 𝑋_4 and 𝑋_5 of the three balls and visualize them in yellow. I need Python code. 计算流程为：
    # 1、共形空间中球的表示：S=C-1/2r^2e_\\infty
    # 2、计算三球交集：M=S_1\\land S_2\\land S_3
    # 3、取对偶得到点对：P=P=M^\\ast=MI^{-1}
    # 4、分解点对得到两个交点：X_\\pm=-\\frac{P\\pm\\sqrt{P\\cdot P}}{e_\\infty\\cdot P}
    # """
    questions = JsonFileHandler.read_json(r"./data/question.json")
    results = []
    index = 0
    for question in questions["conformal_space_tasks"]:
        try:
            input_with_values = question

            # 输入初始状态
            initial_state = {"user_input": input_with_values}

            # 执行图
            final_state = app.invoke(initial_state)

            print("\n" + "-" * 20 + " result " + "-" * 20)
            print(final_state.get('final_code', 'The final code was not generated.'))
            print("-" * 55 + "\n\n")

            if final_state.get('api_response_code')['statusCode'] == "200":
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
            print(question)
            result = {
                "question": question,
                "result": False
            }
            result = {
                "question": question,
                "result": False
            }
    JsonFileHandler.save_json(results, "./data/questions_GAVisAgents.json")
    print(index/40)
