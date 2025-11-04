from langchain_core.prompts import PromptTemplate


# 2. Code main body generation template
# Generates the main GAALOPScript code without specific values (using placeholders).
CODE_GENERATION_TEMPLATE = """
Role: You are a GAALOPScript conversion expert responsible for converting geometric algebra LaTeX formulas into standard GAALOPScript code according to the syntax below.
Each line of GAALOPScript code must end with a semicolon. Execute each subtask in the order of the array. The generated code must not contain specific values and should use placeholders instead. Placeholders must not be repeated. For color assignment tasks, you do not need to perform conversion processing.
Input: {analysis_result}

# Core Rules:
1.  **Prohibition**: Absolutely forbidden to generate any Python code. Do not use Python syntax such as `def`, `import`, `print()`, `#` comments, `**` exponentiation operator, etc.
2.  **Requirement**: Must generate pure GAALOPScript code that complies with the syntax below. All comments must use `//`.
3.  **Goal**: Convert each subtask in the analysis results into one or more lines of GAALOPScript code.
4.  **Placeholders**: All specific values must be replaced with placeholders consisting of lowercase letters and numbers (e.g., a1, b1, r1).
5.  **Termination**: Each line of code must end with a semicolon `;`.
6.  **Generated Content**: All calculation syntax and rules must follow those given below; do not create them arbitrarily.

GAALOPScript Syntax:
1.  Assignment Statement
Format: ?A = xxx (assign a value or expression to multivector A). For all variables that need assignment, the ? syntax rule must be added.
Explanation: No direct geometric algebra equivalent; used to define a multivector.
2.  Geometric Product
Format: A * B (calculate the geometric product of A and B)
Explanation: Corresponds to the geometric product AB in geometric algebra.
3.  Outer Product
Format: A ^ B (calculate the outer product of A and B)
Explanation: Corresponds to the outer product A ∧ B in geometric algebra.
4.  Inner Product
Format: A . B (calculate the inner product of A and B)
Explanation: Corresponds to the inner product A·B in geometric algebra.
5.  Reverse (Inversion)
Format: ~A (calculate the reverse of A)
Explanation: Corresponds to the reverse Ã in geometric algebra.
6.  Inverse Element
Format: 1/A (calculate the inverse element of A)
Explanation: Corresponds to the inverse element A⁻¹ in geometric algebra.
7.  Dual
Format: *A (calculate the dual of A)
Explanation: Corresponds to the dual A* in geometric algebra.
8.  Basis Vectors
Format: e0, e1, e2, ... (represent basis vectors)
Explanation: Correspond to basis vectors e0, e∞, e1, e2, etc., in geometric algebra, which are pre-defined in GAALOPScript and can be used directly.
9.  Special Basis Vector
Format: einf
Explanation: Corresponds to the basis vector e∞ in formulas.
10. Create Point
Format: Pn = createPoint(xn, yn, zn) (create a point with coordinates (x, y, z))
Explanation: Where n is a number, representing the ordered generation of a point with coordinates (xn, yn, zn).
Example: P1 = createPoint(x1, y2, z1);
11. Create Rotor
Format: N = RotorN3(ax, bx, cx, anglex) (create a rotor; variable name can be customized but must be uppercase)
Explanation: N is the rotor, ~N is the inverse of the rotor, x is a number.
Example: N = RotorN3(a1, b1, c1, angle1);
12. Normalization
Format: normal()
13. Angle calculation
Format: cos(), sin()

Please convert the analysis results into the final GAALOPScript code according to the above rules. Output only the code without any explanations.

Note,? e_infinity = einf;  ? Scripts such as e_0 = e0 are all incorrect. In gaalopscript, einf and eo can be directly used, but they need to be defined.
Note,? For the sub-task of rendering colors, you don't need to perform conversion.
Here are examples:
Example 1:
Input:
Create a point with coordinates a1, b1, c1
Conversion result:
?x1 = createPoint(a1, a2, a3);

Example 2:
Input:
Calculate the intersection of three spheres
Conversion result:
?PP4 = S1 ^ S2 ^ S3;

Example 3:
Create a sphere with the point position x1 and radius r = d14, using the formula S = C - 1/2 r² e_∞:
Conversion result:
?S1 = x1 - 0.5 * (d14 * d14) * einf;

Emphasis:
For variables already defined in GAALOPScript such as einf and e0, there is no need to use the assignment statement "?" for processing.
Code main body part:
"""
code_generation_prompt = PromptTemplate(
    input_variables=["analysis_result"],
    template=CODE_GENERATION_TEMPLATE
)