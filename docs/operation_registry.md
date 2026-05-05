# main_graph Operation Registry

本文根据当前项目实现整理，信息来源以以下文件为准：

- [main_graph/operation_specs.py](/E:/大模型/LLM-based-GA/main_graph/operation_specs.py)
- [main_graph/normalizers.py](/E:/大模型/LLM-based-GA/main_graph/normalizers.py)
- [main_graph/nodes.py](/E:/大模型/LLM-based-GA/main_graph/nodes.py)
- [test.py](/E:/大模型/LLM-based-GA/test.py)
- [data/question.json](/E:/大模型/LLM-based-GA/data/question.json)

本文只记录当前仓库里**已经实现并进入 `operation_to_task_block` 分发**的 operation，不引入任何未实现能力。

## 总览

当前已支持的 operation 共 19 个：

1. `construct_point`
2. `line_from_two_points`
3. `point_distance`
4. `midpoint`
5. `construct_sphere`
6. `circle_from_three_points`
7. `plane_from_three_points`
8. `plane_from_point_and_normal`
9. `construct_vector`
10. `geometric_product`
11. `outer_product`
12. `inner_product`
13. `norm`
14. `dual`
15. `meet`
16. `reflect_point`
17. `construct_rotor`
18. `rotate_object`
19. `point_pair_decomposition`

## 全局约定

### 1. 标准任务字段

主图 IR 的标准字段集合为：

```json
{
  "task_id": 1,
  "task_type": "...",
  "operation": "...",
  "inputs": [],
  "outputs": [],
  "depends_on": [],
  "object_specs": {},
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

### 2. normalize 层的统一行为

当前 `normalize_task_blocks_result(...)` 已重构为 operation-aware 结构，入口仍保留在 [main_graph/nodes.py](/E:/大模型/LLM-based-GA/main_graph/nodes.py)，具体 normalizer 在 [main_graph/normalizers.py](/E:/大模型/LLM-based-GA/main_graph/normalizers.py)。

统一行为包括：

- operation alias 统一
- `task_type` alias 统一
- `inputs` / `outputs` / `depends_on` / `object_specs` / `visualization` 缺省补全
- 符号名 sanitize
- plane 输出符号统一成 `Pi`
- `P'` / `C'` / `L'` 统一为 `P_rotated` / `C_rotated` / `L_rotated`
- visualization-only task 识别与合并
- `point_set` 可视化展开

当前仍保留 `legacy_normalize_task_blocks_result(...)` 作为兼容兜底；若 operation-aware normalize 抛异常，会回退到 legacy normalizer。该 fallback 只作用于同一份 LLM JSON 的 normalize，不是本地伪造任务。

### 3. validator 的全局约束

在 [main_graph/nodes.py](/E:/大模型/LLM-based-GA/main_graph/nodes.py) 中，`validate_task_blocks_result(...)` / `task_ir_validator_node(...)` 会统一执行：

- `task_id` 唯一性检查
- `depends_on` 不得依赖未来 task
- `inputs` 必须由前序 task 定义
- `visualization.required=true` 时，被可视化对象必须是当前输出或已定义符号

内置 CGA 符号：

```python
BUILTIN_SYMBOLS = {"e0", "e1", "e2", "e3", "einf"}
```

这些符号不需要前序任务定义。

### 4. 测试覆盖说明

当前仓库可见的测试入口主要是：

- [test.py](/E:/大模型/LLM-based-GA/test.py)：批量读取 [data/question.json](/E:/大模型/LLM-based-GA/data/question.json) 中 40 个问题执行 `run_main_graph(...)`

当前仓库中未看到独立的 `.py` 单测源码文件；因此本文的“相关测试”主要引用：

- `test.py` 批量回归
- `data/question.json` 中的题号示例

---

## 对象构造类

### `construct_point`

- operation 名称：`construct_point`
- task_type：`construct_cga_point`
- 语义说明：根据显式坐标构造 CGA 点。

标准 IR 示例：

```json
{
  "task_id": 1,
  "task_type": "construct_cga_point",
  "operation": "construct_point",
  "inputs": [],
  "outputs": ["P1"],
  "depends_on": [],
  "object_specs": {
    "name": "P1",
    "type": "point",
    "coordinates": [0, 0, 0]
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "P1", "type": "point", "color": "Red"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 占位参数名按点名或 `task_id` 推导，例如 `a1/b1/c1`
  - 公式为 `P = a*e1 + b*e2 + c*e3 + 0.5*(a*a+b*b+c*c)*einf + e0`
- 最终 GAALOPScript 示例：

```text
?P1=a1*e1+b1*e2+c1*e3+0.5*(a1*a1+b1*b1+c1*c1)*einf+e0;
```

- normalize 注意事项：
  - `inputs` 必须归一为 `[]`
  - `outputs` 缺失时可由 `object_specs.name` 补
  - `object_specs.type` 归一为 `point`
  - 视觉化必须附着在点任务本身，不能生成独立 `visualization` task
- validator 校验规则：
  - `operation` 必须是 `construct_point`
  - `inputs` 必须为空
  - 必须恰好 1 个输出
  - `object_specs.type` 必须为 `point`
  - `object_specs.name` 必须等于 `outputs[0]`
  - `coordinates` 非 length-3 时当前是 warning，不是 hard error
- 常见 alias：
  - `construct_cga_point`
  - `create_point`
  - `point_creation`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #1、#8、#13、#24、#29

### `construct_vector`

- operation 名称：`construct_vector`
- task_type：`construct_vector`
- 语义说明：构造向量表达式，如 `A=e1` 或 `A=2*e1+2*e2`。

标准 IR 示例：

```json
{
  "task_id": 1,
  "task_type": "construct_vector",
  "operation": "construct_vector",
  "inputs": [],
  "outputs": ["A"],
  "depends_on": [],
  "object_specs": {
    "name": "A",
    "type": "vector",
    "expression": "2*e1+2*e2"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 直接把 `object_specs.expression` 写入 `formula`
- 最终 GAALOPScript 示例：

```text
?A=2*e1+2*e2;
```

- normalize 注意事项：
  - 归一 task_type 为 `construct_vector`
  - `object_specs.type` 为 `vector`
  - 支持把 `2e1` 规范化成 `2*e1`
- validator 校验规则：
  - `inputs` 必须为空
  - 必须恰好 1 个输出
  - `object_specs.type` 必须为 `vector`
  - `object_specs.name` 必须等于 `outputs[0]`
  - `object_specs.expression` 必须非空
- 常见 alias：
  - `create_vector`
  - `vector_creation`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #7、#11

### `line_from_two_points`

- operation 名称：`line_from_two_points`
- task_type：`construct_cga_line_from_two_points`
- 语义说明：由两个点构造 CGA 直线。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "construct_cga_line_from_two_points",
  "operation": "line_from_two_points",
  "inputs": ["P1", "P2"],
  "outputs": ["L"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "L",
    "type": "line",
    "from": ["P1", "P2"]
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "L", "type": "line", "color": "Green"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`L = P1 ^ P2 ^ einf`
- 最终 GAALOPScript 示例：

```text
?L=P1^P2^einf;
```

- normalize 注意事项：
  - `object_specs.from` 缺失时用 `inputs`
  - `object_specs.type` 默认 `line`
  - `outputs` 缺失时可由 `object_specs.name` 或默认符号补齐
- validator 校验规则：
  - 当前 validator 主要依赖通用 `inputs` 已定义检查
  - 编译阶段要求 `inputs` 长度必须为 2，且输出非空
  - 语义上应提供 `line` 类型和 `from` 字段
- 常见 alias：
  - `construct_line`
  - `construct_cga_line_from_two_points`
  - `line_from_points`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #3、#20、#23、#29、#40

### `construct_sphere`

- operation 名称：`construct_sphere`
- task_type：`construct_cga_sphere`
- 语义说明：由球心和半径构造球。球心既可直接给坐标，也可引用已构造点符号。

标准 IR 示例：

```json
{
  "task_id": 2,
  "task_type": "construct_cga_sphere",
  "operation": "construct_sphere",
  "inputs": ["X1"],
  "outputs": ["S1"],
  "depends_on": [1],
  "object_specs": {
    "name": "S1",
    "type": "sphere",
    "center": "X1",
    "radius": 0.5
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "S1", "type": "sphere", "color": "Blue"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 若 `center` 是坐标：
    - 先生成 `Ck = createPoint(xk, yk, zk)`
    - 再生成 `Sk = Ck - 0.5*(rk*rk)*einf`
  - 若 `center` 是已有点符号：
    - 直接用该符号代入
- 最终 GAALOPScript 示例：

```text
?C1=createPoint(x1,y1,z1);
?r1=r1v;
?S=C1-0.5*(r1*r1)*einf;
```

或

```text
?r1=r1v;
?S1=X1-0.5*(r1*r1)*einf;
```

- normalize 注意事项：
  - `center` 允许 `[x,y,z]` 或点符号字符串
  - 若 `center` 是字符串且 `inputs` 为空，会补上该输入
  - 不会强制把中心改成坐标
- validator 校验规则：
  - `task_type` 必须是 `construct_cga_sphere`
  - `object_specs.type` 必须为 `sphere`
  - `center` 必须是 length-3 坐标，或一个已定义且类型为 `point` 的符号
  - 当 `center` 是符号时，`inputs` 必须包含该符号
  - `radius` 必须是数值
  - `object_specs.name` 必须等于 `outputs[0]`
- 常见 alias：
  - `construct_cga_sphere`
  - `create_sphere`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #2、#15、#19、#21、#31

### `circle_from_three_points`

- operation 名称：`circle_from_three_points`
- task_type：`construct_cga_circle_from_three_points`
- 语义说明：由三点构造圆。

标准 IR 示例：

```json
{
  "task_id": 4,
  "task_type": "construct_cga_circle_from_three_points",
  "operation": "circle_from_three_points",
  "inputs": ["P1", "P2", "P3"],
  "outputs": ["C"],
  "depends_on": [1, 2, 3],
  "object_specs": {
    "name": "C",
    "type": "circle",
    "from": ["P1", "P2", "P3"]
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "C", "type": "circle", "color": "Blue"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`C = P1 ^ P2 ^ P3`
- 最终 GAALOPScript 示例：

```text
?C=P1^P2^P3;
```

- normalize 注意事项：
  - `object_specs.from` 缺失时用 `inputs`
  - `object_specs.type` 默认 `circle`
  - 可附带 `center` / `radius` / `plane` 附加信息，但编译规则仍以 wedge 为主
- validator 校验规则：
  - 必须恰好 3 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `circle`
  - `object_specs.from` 应与 `inputs` 一致
- 常见 alias：
  - `construct_circle`
  - `construct_cga_circle_from_three_points`
  - `circle_from_points`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #5、#26

### `plane_from_three_points`

- operation 名称：`plane_from_three_points`
- task_type：`construct_cga_plane_from_three_points`
- 语义说明：由三点构造平面。

标准 IR 示例：

```json
{
  "task_id": 4,
  "task_type": "construct_cga_plane_from_three_points",
  "operation": "plane_from_three_points",
  "inputs": ["P1", "P2", "P3"],
  "outputs": ["Pi"],
  "depends_on": [1, 2, 3],
  "object_specs": {
    "name": "Pi",
    "type": "plane",
    "from": ["P1", "P2", "P3"]
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "Pi", "type": "plane", "color": "Blue"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`Pi = P1 ^ P2 ^ P3 ^ einf`
- 最终 GAALOPScript 示例：

```text
?Pi=P1^P2^P3^einf;
```

- normalize 注意事项：
  - 输出符号和 `object_specs.name` 会统一经过 plane sanitize，异常符号会规约到 `Pi`
  - `visualization` 中 plane 对象名也会一起更新
- validator 校验规则：
  - 必须恰好 3 个输入
  - 必须 1 个输出
  - 输出不应继续使用 `Π`、`\Pi`、`螤` 等异常符号，应该改为 `Pi`
  - `object_specs.type` 必须为 `plane`
  - `object_specs.from` 应与 `inputs` 一致
- 常见 alias：
  - `construct_cga_plane_from_three_points`
  - `plane_from_points`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #25、#33

### `plane_from_point_and_normal`

- operation 名称：`plane_from_point_and_normal`
- task_type：`construct_cga_plane_from_point_and_normal`
- 语义说明：由平面上一点和法向量构造平面。

标准 IR 示例：

```json
{
  "task_id": 1,
  "task_type": "construct_cga_plane_from_point_and_normal",
  "operation": "plane_from_point_and_normal",
  "inputs": [],
  "outputs": ["Pi"],
  "depends_on": [],
  "object_specs": {
    "name": "Pi",
    "type": "plane",
    "point": [0, 0, 0],
    "normal": [0, 0, 1]
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "Pi", "type": "plane", "color": "Yellow"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 先算 `d = -(n·x)`
  - 再构造 `Pi = nx*e1 + ny*e2 + nz*e3 + d*einf`
- 最终 GAALOPScript 示例：

```text
?d1=-(nx1*x1+ny1*y1+nz1*z1);
?Pi=nx1*e1+ny1*e2+nz1*e3+d1*einf;
```

- normalize 注意事项：
  - point 字段别名支持：
    - `point`
    - `point_on_plane`
    - `through_point`
    - `passing_point`
    - `pointOnPlane`
    - `origin`
  - normal 字段别名支持：
    - `normal`
    - `normal_vector`
    - `normalVec`
    - `n`
    - `direction_normal`
  - 输出符号会规约成 `Pi`
- validator 校验规则：
  - 必须 1 个输出
  - `object_specs.type` 必须为 `plane`
  - `object_specs.point` 必须是 length-3 list
  - `object_specs.normal` 必须是 length-3 list
  - `normal` 不能为零向量
- 常见 alias：
  - `construct_plane`
  - `construct_plane_from_point_and_normal`
  - `construct_cga_plane_from_point_and_normal`
  - `plane_point_normal`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #6、#15、#27、#34

### `construct_rotor`

- operation 名称：`construct_rotor`
- task_type：`construct_rotor`
- 语义说明：由旋转轴和角度构造 rotor。

标准 IR 示例：

```json
{
  "task_id": 1,
  "task_type": "construct_rotor",
  "operation": "construct_rotor",
  "inputs": [],
  "outputs": ["R"],
  "depends_on": [],
  "object_specs": {
    "name": "R",
    "type": "rotor",
    "axis": [0, 0, 1],
    "angle": "pi/2",
    "angle_unit": "radian"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 先算轴模长 `axis_norm{k}`
  - 再生成单位双向量 `axisB{k}`
  - 最后生成 `R = cos(angle/2) - sin(angle/2) * axisB`
- 最终 GAALOPScript 示例：

```text
?axis_norm1=sqrt(ax1*ax1+ay1*ay1+az1*az1);
?axisB1=(ax1/axis_norm1)*(e2^e3)+(ay1/axis_norm1)*(e3^e1)+(az1/axis_norm1)*(e1^e2);
?R=cos(angle1*0.5)-sin(angle1*0.5)*axisB1;
```

- normalize 注意事项：
  - `inputs` 一律清空为 `[]`
  - `outputs` 缺失时默认 `["R"]`
  - `object_specs.type` 归一为 `rotor`
  - `axis` 保留为三维向量
  - `angle_unit` 会由 `degree` / `radian` 语义推断
  - 不会把 `construct_rotor` 改写成 `rotate_object`
  - 复合 rotor（如 `Ry(30)Rx(60)`）的统一标准 schema 在当前实现中未单独展开，细节待确认
- validator 校验规则：
  - `inputs` 必须为空
  - 必须 1 个输出
  - `object_specs.type` 必须为 `rotor`
  - `axis` 必须是 length-3 数值向量
  - `axis` 不能为零向量
  - `angle` 必须存在且能被解析成弧度
- 常见 alias：
  - `create_rotor`
  - `rotor`
  - `build_rotor`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #24、#39

---

## 代数操作类

### `geometric_product`

- operation 名称：`geometric_product`
- task_type：`compute_geometric_product`
- 语义说明：计算两个符号的几何积。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "compute_geometric_product",
  "operation": "geometric_product",
  "inputs": ["A", "B"],
  "outputs": ["G"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "G",
    "type": "multivector",
    "from": ["A", "B"],
    "operator": "*"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`G = A * B`
- 最终 GAALOPScript 示例：

```text
?G=A*B;
```

- normalize 注意事项：
  - `object_specs.from` 缺失时用 `inputs`
  - `object_specs.operator` 默认 `*`
  - `object_specs.type` 默认 `multivector`
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `multivector`
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `*`
- 常见 alias：
  - `compute_geometric_product`
  - `gp`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #7

### `outer_product`

- operation 名称：`outer_product`
- task_type：`compute_outer_product`
- 语义说明：计算 wedge / join。既可表示通用 multivector，也可表示明确几何对象。

标准 IR 示例：

```json
{
  "task_id": 5,
  "task_type": "compute_outer_product",
  "operation": "outer_product",
  "inputs": ["L", "Pi"],
  "outputs": ["P"],
  "depends_on": [3, 4],
  "object_specs": {
    "name": "P",
    "type": "point",
    "from": ["L", "Pi"],
    "operator": "^"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 把所有输入用 `^` 串联
- 最终 GAALOPScript 示例：

```text
?P=L^Pi;
```

或

```text
?M=S1^S2^S3;
```

- normalize 注意事项：
  - `object_specs.operator` 默认 `^`
  - `object_specs.from` 缺失时用 `inputs`
  - 若已有 `object_specs.type`，必须保留
  - 允许类型：
    - `multivector`
    - `point`
    - `point_pair`
    - `line`
    - `circle`
    - `sphere`
    - `plane`
  - 只有缺失时才默认 `multivector`
- validator 校验规则：
  - 至少 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须属于允许集合
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `^`
- 常见 alias：
  - `compute_outer_product`
  - `wedge_product`
  - `join`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #15、#17、#19、#27、#34、#40

### `inner_product`

- operation 名称：`inner_product`
- task_type：`compute_inner_product`
- 语义说明：计算两个符号的内积 / 点积。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "compute_inner_product",
  "operation": "inner_product",
  "inputs": ["P1", "P2"],
  "outputs": ["IP"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "IP",
    "type": "scalar",
    "from": ["P1", "P2"],
    "operator": "."
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`IP = P1 . P2`
- 最终 GAALOPScript 示例：

```text
?IP=P1.P2;
```

- normalize 注意事项：
  - `object_specs.operator` 默认 `.`
  - `object_specs.from` 缺失时用 `inputs`
  - 当前标准类型为 `scalar`
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `scalar`
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `.`
- 常见 alias：
  - `compute_inner_product`
  - `dot_product`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #10

### `norm`

- operation 名称：`norm`
- task_type：`compute_norm`
- 语义说明：计算 `sqrt(A.A)` 形式的范数。

标准 IR 示例：

```json
{
  "task_id": 2,
  "task_type": "compute_norm",
  "operation": "norm",
  "inputs": ["A"],
  "outputs": ["NormA"],
  "depends_on": [1],
  "object_specs": {
    "name": "NormA",
    "type": "scalar",
    "from": ["A"],
    "operator": "sqrt_dot"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`NormA = sqrt(A . A)`
- 最终 GAALOPScript 示例：

```text
?NormA=sqrt(A.A);
```

- normalize 注意事项：
  - `object_specs.type` 默认 `scalar`
  - `object_specs.from` 缺失时用 `inputs`
  - `operator` 默认 `sqrt_dot`
- validator 校验规则：
  - 必须恰好 1 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `scalar`
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `sqrt_dot`
- 常见 alias：
  - `compute_norm`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #11

### `dual`

- operation 名称：`dual`
- task_type：`compute_dual`
- 语义说明：计算一个对象的 dual，即 `*A`。

标准 IR 示例：

```json
{
  "task_id": 2,
  "task_type": "compute_dual",
  "operation": "dual",
  "inputs": ["P"],
  "outputs": ["DualP"],
  "depends_on": [1],
  "object_specs": {
    "name": "DualP",
    "type": "multivector",
    "from": ["P"],
    "operator": "*"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`DualP = *P`
- 最终 GAALOPScript 示例：

```text
?DualP=*P;
```

- normalize 注意事项：
  - `object_specs.operator` 默认 `*`
  - `object_specs.from` 缺失时用 `inputs`
  - operation-aware normalizer 会保留已有 `object_specs.type`
  - 但当前 validator 仍要求 `object_specs.type == "multivector"`；因此实际推荐标准 IR 仍写 `multivector`
- validator 校验规则：
  - 必须恰好 1 个输入
  - 必须 1 个输出
  - `object_specs.type` 当前必须为 `multivector`
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `*`
- 常见 alias：
  - `compute_dual`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #14、#21

---

## 几何计算类

### `point_distance`

- operation 名称：`point_distance`
- task_type：`compute_cga_point_distance`
- 语义说明：计算两点的平方距离。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "compute_cga_point_distance",
  "operation": "point_distance",
  "inputs": ["P1", "P2"],
  "outputs": ["d2"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "d2",
    "type": "scalar",
    "from": ["P1", "P2"],
    "quantity": "squared_distance"
  },
  "visualization": {
    "required": false,
    "objects": []
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`d2 = -2 * (P1 . P2) / ((-einf . P1) * (-einf . P2))`
- 最终 GAALOPScript 示例：

```text
?d2=-2*(P1.P2)/((-einf.P1)*(-einf.P2));
```

- normalize 注意事项：
  - 默认输出名可为 `d2`
  - `quantity` 默认 `squared_distance`
  - 通常不需要 visualization
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `scalar`
  - `object_specs.from` 应与 `inputs` 一致
  - 若要求 visualization，当前会给 warning
- 常见 alias：
  - `compute_point_distance`
  - `compute_cga_point_distance`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #4

### `midpoint`

- operation 名称：`midpoint`
- task_type：`compute_midpoint`
- 语义说明：计算两点中点，结果仍是 point，不应误写成 `construct_point`。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "compute_midpoint",
  "operation": "midpoint",
  "inputs": ["P1", "P2"],
  "outputs": ["M"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "M",
    "type": "point",
    "from": ["P1", "P2"],
    "formula": "M = (P1 + P2) / 2"
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "M", "type": "point", "color": "Yellow"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 统一编译成 `M = (P1 + P2) * 0.5`
- 最终 GAALOPScript 示例：

```text
?M=(P1+P2)*0.5;
```

- normalize 注意事项：
  - alias 会统一到 `midpoint`
  - 缺省输出名默认 `M`
  - `object_specs.type` 默认 `point`
  - `object_specs.from` 缺失时用 `inputs`
  - 不会改写成 `construct_point`
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `point`
  - `object_specs.from` 应与 `inputs` 一致
  - 若前序符号已知类型，则两个输入应为 `point`
- 常见 alias：
  - `compute_midpoint`
  - `middle_point`
  - `mid_point`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #13

### `meet`

- operation 名称：`meet`
- task_type：`compute_meet`
- 语义说明：计算两个几何对象的 meet / 交。

标准 IR 示例：

```json
{
  "task_id": 7,
  "task_type": "compute_meet",
  "operation": "meet",
  "inputs": ["L1", "L2"],
  "outputs": ["X"],
  "depends_on": [3, 6],
  "object_specs": {
    "name": "X",
    "type": "point",
    "from": ["L1", "L2"],
    "operator": "meet"
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "X", "type": "point", "color": "Yellow"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 先 dual 两个输入
  - 再做 wedge
  - 最后 dual 回来
- 最终 GAALOPScript 示例：

```text
?dual_L1=*L1;
?dual_L2=*L2;
?tmp_X=dual_L1^dual_L2;
?X=*tmp_X;
```

- normalize 注意事项：
  - `object_specs.operator` 默认 `meet`
  - `object_specs.from` 缺失时用 `inputs`
  - 若已有 `object_specs.type` 必须保留
  - 允许类型：
    - `multivector`
    - `point`
    - `point_pair`
    - `line`
    - `circle`
    - `sphere`
    - `plane`
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须属于允许集合
  - `object_specs.from` 应与 `inputs` 一致
  - `operator` 必须为 `meet`
- 常见 alias：
  - `compute_meet`
  - `intersection`
  - `line_intersection`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #29

### `point_pair_decomposition`

- operation 名称：`point_pair_decomposition`
- task_type：`decompose_cga_point_pair`
- 语义说明：把点对 `P` 分解成两个交点，例如 `X4`、`X5`。

标准 IR 示例：

```json
{
  "task_id": 9,
  "task_type": "decompose_cga_point_pair",
  "operation": "point_pair_decomposition",
  "inputs": ["P"],
  "outputs": ["X4", "X5"],
  "depends_on": [8],
  "object_specs": {
    "name": "point_pair_decomposition",
    "type": "point_pair_decomposition",
    "point_pair": "P",
    "formula": "X_pm = (P ± sqrt(P.P)) / (einf.P)"
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "X4", "type": "point", "color": "Yellow"},
      {"name": "X5", "type": "point", "color": "Yellow"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - `PP = P . P`
  - `sqrt_PP = sqrt(PP)`
  - `denom = einf . P`
  - `X4 = (P + sqrt_PP) * (1 / denom)`
  - `X5 = (P - sqrt_PP) * (1 / denom)`
- 最终 GAALOPScript 示例：

```text
?PP=P.P;
?sqrt_PP=sqrt(PP);
?denom=einf.P;
?X4=(P+sqrt_PP)*(1/denom);
?X5=(P-sqrt_PP)*(1/denom);
```

- normalize 注意事项：
  - alias 会统一到 `point_pair_decomposition`
  - `task_type` 统一为 `decompose_cga_point_pair`
  - `object_specs.point_pair` 缺失时用 `inputs[0]`
  - `visualization.required=true` 且 objects 为空时，可根据输出补 Yellow
  - 会把明显错误拆成 `construct_point` 的 `X4/X5` 任务尽量并回该 operation
- validator 校验规则：
  - 必须恰好 1 个输入
  - 至少 2 个输出
  - `object_specs.type` 必须为 `point_pair_decomposition`
  - `object_specs.point_pair` 必须等于 `inputs[0]`
- 常见 alias：
  - `decompose_point_pair`
  - `point_pair_decompose`
  - `extract_point_pair`
  - `split_point_pair`
  - `point_pair_to_points`
  - `compute_intersection_points`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #21

---

## 几何变换类

### `reflect_point`

- operation 名称：`reflect_point`
- task_type：`reflect_cga_point`
- 语义说明：用镜像对象反射点。

标准 IR 示例：

```json
{
  "task_id": 3,
  "task_type": "reflect_cga_point",
  "operation": "reflect_point",
  "inputs": ["P", "Pi"],
  "outputs": ["P_reflected"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "P_reflected",
    "type": "point",
    "point": "P",
    "mirror": "Pi",
    "formula": "M v M"
  },
  "visualization": {
    "required": true,
    "objects": [
      {"name": "P_reflected", "type": "point", "color": "Blue"}
    ]
  }
}
```

- operation_to_task_block 编译规则：
  - 公式：`P_reflected = Pi * P * Pi`
- 最终 GAALOPScript 示例：

```text
?P_reflected=Pi*P*Pi;
```

- normalize 注意事项：
  - 能把明显像“反射点”的错误 `geometric_product` 任务纠正为 `reflect_point`
  - 缺省输出名可补为 `P_reflected`
  - `object_specs.point` / `mirror` 可由 `inputs` 回填
  - `formula` 缺失时补 `M v M`
- validator 校验规则：
  - 必须恰好 2 个输入
  - 必须 1 个输出
  - `object_specs.type` 必须为 `point`
  - `object_specs.point` 必须等于 `inputs[0]`
  - `object_specs.mirror` 必须等于 `inputs[1]`
- 常见 alias：
  - `reflect`
  - `reflection`
  - `point_reflection`
  - `reflect_cga_point`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #12

### `rotate_object`

- operation 名称：`rotate_object`
- task_type：`rotate_cga_object`
- 语义说明：旋转点、线、圆、球、平面或一般 multivector。当前实现支持两种输入模式。

模式 A：隐式 rotor 模式

```json
{
  "task_id": 2,
  "task_type": "rotate_cga_object",
  "operation": "rotate_object",
  "inputs": ["P"],
  "outputs": ["P_rotated"],
  "depends_on": [1],
  "object_specs": {
    "name": "P_rotated",
    "type": "point",
    "object": "P",
    "axis": [0, 0, 1],
    "angle": "pi/2",
    "angle_unit": "radian"
  }
}
```

模式 B：显式 rotor 模式

```json
{
  "task_id": 3,
  "task_type": "rotate_cga_object",
  "operation": "rotate_object",
  "inputs": ["P", "R"],
  "outputs": ["P_rotated"],
  "depends_on": [1, 2],
  "object_specs": {
    "name": "P_rotated",
    "type": "point",
    "object": "P",
    "rotor": "R"
  }
}
```

- operation_to_task_block 编译规则：
  - 模式 A：
    - 生成 `axisB{k}`
    - 生成内部 rotor `R{k}`
    - 生成 `Output = R{k} * Input * ~R{k}`
  - 模式 B：
    - 直接生成 `Output = R * P * ~R`
- 最终 GAALOPScript 示例：

隐式模式：

```text
?axisB9=ax9*(e2^e3)+ay9*(e3^e1)+az9*(e1^e2);
?R9=cos(angle9*0.5)-sin(angle9*0.5)*axisB9;
?P_rotated=R9*P*~R9;
```

显式 rotor 模式：

```text
?P_rotated=R*P*~R;
```

- normalize 注意事项：
  - 支持 `inputs=["P"]` 和 `inputs=["P","R"]`
  - 若 `object_specs.object` 缺失，会从 `inputs[0]` 回填
  - 若显式 rotor 模式缺少 `object_specs.rotor`，会从 `inputs[1]` 回填
  - `P'` / `P1'` / `C'` / `L'` 会 sanitize 成 `*_rotated`
  - 若 `object_specs.type` 缺失，优先继承被旋转对象类型
  - 不会把显式 rotor 模式压回单输入
- validator 校验规则：
  - `inputs` 长度必须为 1 或 2
  - 必须 1 个输出
  - `object_specs.object` 必须等于 `inputs[0]`
  - 若是隐式模式：
    - 必须有 `axis`
    - `axis` 必须是 length-3 非零向量
    - 必须有可解析的 `angle`
  - 若是显式 rotor 模式：
    - 必须有 `object_specs.rotor`
    - `object_specs.rotor` 必须等于 `inputs[1]`
    - 第二个输入若已知类型，应为 `rotor`
  - 当前允许输出类型：
    - `point`
    - `line`
    - `circle`
    - `sphere`
    - `plane`
    - `multivector`
    - `point_pair`
- 常见 alias：
  - `rotate`
  - `rotation`
  - `rotate_point`
  - `rotate_line`
  - `rotate_circle`
  - `rotate_sphere`
  - `rotate_cga_object`
- 相关测试 / question 示例：
  - `test.py` + `question.json` #9、#18、#24、#32

---

## 实现现状补充

### 1. visualization-only task 不是正式 operation

当前系统明确**不把 visualization 作为独立 operation**。如果 LLM 生成了：

```json
{
  "task_type": "visualization",
  "operation": "visualization",
  "visualization": {
    "required": true,
    "objects": [...]
  }
}
```

normalize 层会：

- 识别为 visualization-only task
- 把对象合并回对应的构造/计算任务
- 不让该 task 进入最终 `tasks`

这属于 normalize 兼容逻辑，不属于正式 operation。

### 2. dual 的 concrete type 现状

当前实现里，`dual` 的：

- `operation_specs` 写法是标准 `multivector`
- `build_dual_task_block(...)` 也按通用 dual 编译
- `validator` 目前仍严格要求 `object_specs.type == "multivector"`

因此虽然 normalize 层对某些具体类型有保留行为，**当前对外可依赖的标准写法仍应使用 `multivector`**。若未来要全面允许 `point_pair` 等 dual 输出类型，需要同步放宽 validator。

### 3. 当前可见测试形态

当前仓库中可见的源码级测试入口主要是：

- [test.py](/E:/大模型/LLM-based-GA/test.py)

它会读取 [data/question.json](/E:/大模型/LLM-based-GA/data/question.json) 的 40 个样例做完整主图回归。文档中的“相关测试”因此主要以 question 编号表示，而非单独 `.py` 单测文件。
