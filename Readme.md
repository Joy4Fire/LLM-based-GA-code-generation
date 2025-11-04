# GAALOPScript Agent Code Generator Based on LangGraph

This project is an advanced Multi-Agent system built with [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). Its core functionality is to understand geometric algebra problems described by users in natural language, collaborate through a series of agents to gradually generate intermediate language GAALOPScript, and finally call an external API to convert it into executable code in specified programming languages (such as Java, C++).

## Core Features

- **Natural Language Understanding**: The system can automatically extract key task parameters from user descriptions in natural language, such as function names, target languages, and target spaces.
- **Multi-Agent Collaborative Workflow**: The entire process is divided into multiple independent agents, each responsible for a specialized task and collaborating through LangGraph:
  1. **Information Extraction Agent**: Parses user intentions.
  2. **Task Decomposition Agent**: Breaks down complex problems into structured subtasks.
  3. **Code Generation Agent**: Converts subtasks into GAALOPScript code templates without specific values.
  4. **Assignment and Visualization Agent**: Handles specific numerical values and visualization instructions.
  5. **Code Integration Agent**: Integrates all code snippets into a complete GAALOPScript script.
  6. **Tool Calling Agent**: Uses the generated script to call external APIs to complete the final code conversion.
- **Conditional Logic and Dynamic Routing**: The system has a built-in router that can dynamically determine the workflow based on analysis results. If the user request does not contain specific values, the system will automatically skip the assignment and visualization steps and execute a simplified symbolic calculation process.
- **External Tool Calling**: Demonstrates how agents can use external tools (Gaalop API in this project) to complete tasks that they cannot accomplish themselves, realizing end-to-end generation from abstract instructions to specific code.


## Project Structure

```
langgraph_gaalop_agent/
├── agents                    # Agent definition files, including graph.py - defines and builds the LangGraph graph (workflow), state.py - defines the graph's state object (AgentState), nodes.py - stores all executable node agents in the graph
├── data                      # Test data
├── models                    # Construction code for large language models
├── Prompt                    # Stores all prompt templates used by agents
├── utils                     # Access code for GAALOP API and code for reading and saving JSON files
├── main.py                   # Main program entry for running and testing agents
├── LLM4GAALOPScript.py       # GPT-4o comparison experiment
└── requirements.txt          # Project dependencies
```

## Installation and Running

### 1. Environment Preparation
- Ensure you have Python 3.9 or higher installed.

### 2. Clone the Project
```bash
cd langgraph_gaalop_agent
```

### 3. Create and Activate a Virtual Environment
- **Windows**:
  
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Core Configuration (Important!)
Before running the project, you **must** configure the following two items:

#### **A. LLM Service Configuration**

1. Configure in the models directory of the project.

   API key for connecting to your large language model service
   
   OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   
   API endpoint address of your large language model service
   
   OPENAI_API_BASE="http://your-llm-provider-ip:port/v1"
   
   Your large language model service model
   
   OPENAI_API_BASE="gpt-4o"


#### **B. Gaalop API Interface Configuration**

1. Open the `utils/Gaalop.py` file.
2. Find the definition of the `api_url` variable:
    ```python
    # tools.py
    
    # Target API address (replace if necessary)
    api_url = "http://XXX:XXX/gaalop/codeGenerate"
    ```
3. **Please replace it with the actual address of your own deployed Gaalop service.**

### 6. Run the Project
After completing the above configuration, simply run the main program:
```bash
python main.py
```
You will see logs of each agent running in the terminal, as well as the final code returned from the Gaalop API.


## 7. Explanation of Matching Between Code and the Five Cycles

1. **Representation**
   - nodes.py: Contains the `extract_information_node` function, responsible for extracting key information such as geometric algebra space (target_space) from user input, establishing the mapping from geographical space to mathematical space
   - state.py: Defines the `AgentState` data structure to store the representation form of geometric objects
2. **Reasoning**
   - nodes.py: The `analyze_task_node` function implements task analysis, performing logical reasoning based on GA operators
   - task_decomposition_template.py: Decomposes tasks according to geometric algebra rules, handling dependencies between objects (such as the reasoning chain of "basic variables → composite objects")
3. **Generation**
   - nodes.py: `generate_code_node` generates the main GAALOPScript code, `generate_assignment_code_node` generates variable assignment code
   - visualization_code_generation_template.py: Generates visualization code to complete the rendering of geometric objects
   - assignment_code_generation_template.py: Generates copy code for GA code
   - code_generation_template.py: Implements code generation operations in mathematical space through GAALOPScript rules
   - code_integration_template.py: Integrates various parts of the code to generate a complete executable script
4. **Synthesis/Validation**
   - nodes.py: `call_gaalop_api_node` verifies parameter integrity when calling the Gaalop API (such as checking whether GAALOPScript, function names, etc., exist)
   - Gaalop.py: The `generate_gaalop_code` function parses the script structure to ensure the code complies with syntax rules
   - assignment_extraction_template.py: Verifies the validity of values when extracting variable assignments to avoid invalid values
5. **Computing**
   - Gaalop.py: Calls the Gaalop computing engine through the API to perform the conversion from algebraic expressions to executable code