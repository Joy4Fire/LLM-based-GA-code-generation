# LLM-based Geometric Algebra Code Generation

This repository implements a LangGraph-based prototype for generating geometric algebra (GA) computation code from natural-language task descriptions. It focuses on the natural-language-to-GAALOPScript path of a broader Geometric Algebra Geospatial Model (GGM) workflow: user intent is converted into a structured multivector-oriented intermediate representation, validated as a task graph, translated into GAALOPScript sections, and packaged as a GAALOP-compatible generation request.

The current implementation mainly supports conformal geometric algebra (CGA) tasks such as points, lines, planes, circles, spheres, distances, inner/outer/geometric products, duals, meets, reflections, rotations, rotors, midpoints, and point-pair decomposition. The README therefore describes both:

- the implemented code-generation pipeline in this repository
- how this pipeline corresponds to the five GGM stages: representation, reasoning, generation, synthesis, and computation

## Reproducibility Quick Start

Install dependencies:

```bash
uv sync
```

Run GA-VisAgent over the 40 CGA tasks:

```bash
python scripts/run_ga_visagent.py
```

Run the GPT-4o baseline prompt over the same dataset:

```bash
python scripts/run_gpt4o_baseline.py
```

Run the local compile-retry test:

```bash
python tests/test_gaalop_compile_retry.py
```

Summarize saved result files:

```bash
python scripts/evaluate_results.py
```

The scripts read `data/question.json` and write reproducibility outputs under `results/`.

## Current Capabilities

- **Natural-language task parsing**: extracts `function_name`, `target_language`, and `target_space` from user input.
- **Semantic task decomposition**: converts a complex request into minimal semantic IR tasks with dependencies.
- **Operation-aware validation**: checks task IDs, dependency order, input symbol availability, output consistency, object types, and visualization references.
- **GA operation registry**: defines the supported operations and maps aliases to canonical operations.
- **Task-block compilation**: converts semantic IR into task blocks containing:
  - `code_to_optimize`
  - `variable_assignments`
  - `multivectors_to_be_visualized`
- **Subtask dispatching**: topologically sorts dependent tasks and runs the code, assignment, and visualization generation agents per task.
- **GAALOP request construction**: assembles the final code sections and builds a request payload for GAALOP-style code generation.

## Repository Structure

```text
LLM-based-GA-code-generation/
├── data/
│   └── question.json                         # Regression questions for CGA tasks
├── docs/
│   └── operation_registry.md                 # Detailed operation registry notes
├── examples/
│   ├── debug/
│   │   └── gaalop_request_payload.json       # Example/debug request payload
│   ├── run_single_task_agents.py             # Single-task development example
│   └── to_code.py                            # Interactive conversion example
├── scripts/
│   ├── run_ga_visagent.py                    # Batch GA-VisAgent experiment
│   ├── run_gpt4o_baseline.py                 # GPT-4o baseline experiment
│   ├── run_single.py                         # Single-task sub-agent runner
│   └── evaluate_results.py                   # Summarize experiment outputs
├── src/
│   └── ga_visagent/
│       ├── legacy_agents/                    # Earlier single-task agents
│       ├── main_graph/                       # Main graph, validation, compilation, dispatching
│       ├── models/                           # LLM factory
│       └── prompts/                          # Prompt templates
├── tests/
│   └── test_gaalop_compile_retry.py
└── results/
    └── ga_visagent/                          # Saved GA-VisAgent regression outputs
```

## Pipeline

The implemented main graph is defined in `src/ga_visagent/main_graph/graph.py`:

```text
user_input
  -> information_extraction_node
  -> task_decomposition_node
  -> task_ir_validator_node
  -> operation_to_task_block_node
  -> subtask_dispatcher_node
  -> final_code_assembler_node
  -> gaalop_request_builder_node
```

The output of this process is a `gaalop_request_result` object containing:

- `codegenPlugins`: target language
- `algebraPlugins`: target algebra space
- `optimization`: requested optimization flags such as `maxima` and `cse`
- `script.optimizeCode`: generated GAALOPScript expressions
- `script.variableAssignments`: numeric assignments
- `script.multivectorsVisualized`: visualization statements
- `script.functionName`: generated or extracted function name

## Relation to the Five GGM Stages

The following mapping aligns the current codebase with the five-stage formulation in the paper text.

### 1. Representation

In the general GGM formulation, heterogeneous geospatial data are mapped into a multivector space:

```text
E: D -> M
```

where `D` denotes geospatial data and `M` denotes the multivector space. A complete GGM system can organize vector, raster/field, and network data through a hierarchical scene model and a multivector-tree-like data structure.

In the current repository, this stage is implemented as a natural-language-to-semantic-IR representation layer:

- `information_extraction_node` extracts the target function, language, and algebra space.
- `task_decomposition_node` converts the user request into semantic tasks.
- `MainGraphState` stores the structured representation across the graph.
- Each task describes a multivector-oriented entity or operation using fields such as `operation`, `inputs`, `outputs`, `object_specs`, and `visualization`.

Example semantic IR:

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

This corresponds to a practical subset of `E: D -> M`: natural-language descriptions of CGA scenes are mapped into computation-ready multivector task representations.

### 2. Reasoning

In the general GGM formulation, the reasoning stage applies GA operators and semantic/spatial constraints:

```text
R(M, Phi_sem, Phi_sp) -> Theta
```

where `Phi_sem` and `Phi_sp` represent semantic and spatial rule sets, and `Theta` denotes actionable parameters for generation.

In the current codebase, reasoning is implemented by:

- `operation_registry.py`, which defines canonical GA operations, aliases, output types, and compiler mappings.
- `operation_specs.py`, which gives prompt-facing rules for selecting operations.
- `normalizers.py` and normalization logic in `nodes.py`, which repair and standardize LLM-produced task IR.
- `validate_task_blocks_result` and `task_ir_validator_node`, which enforce dependency, symbol, type, and visualization constraints.

The repository therefore realizes a rule-based and LLM-assisted reasoning layer. It does not yet implement the full geospatial constraint sets for raster fields, network topology, or physical simulation, but it does formalize many CGA semantic and spatial rules, such as:

- a line from two points depends on both point-construction tasks
- a plane from three points requires three prior point symbols
- a meet operation requires two already-defined geometric objects
- visualization objects must refer to current or previously defined outputs
- built-in CGA basis symbols such as `e0`, `e1`, `e2`, `e3`, and `einf` do not require construction tasks

### 3. Generation

In the general GGM formulation, generation is modeled as:

```text
G_theta: (M, Theta) -> M'
```

where the generator produces a new or transformed multivector representation.

In this repository, generation is implemented as code-generation over validated multivector task blocks:

- `operation_to_task_block_node` converts semantic IR into executable task blocks.
- `build_*_task_block` functions in `nodes.py` compile operations into formulas and assignment placeholders.
- `subtask_dispatcher_node` runs task-level generation in dependency order.
- Prompt templates under `src/ga_visagent/prompts/` generate:
  - GAALOPScript code to optimize
  - variable assignment code
  - multivector visualization statements
- `final_code_assembler_node` merges all task outputs into the final GAALOPScript sections.

For geometry-oriented scenes, the implemented generator uses GA formulas such as:

```text
P = x*e1 + y*e2 + z*e3 + 0.5*(x*x + y*y + z*z)*einf + e0
L = P1 ^ P2 ^ einf
C = P1 ^ P2 ^ P3
Pi = P1 ^ P2 ^ P3 ^ einf
P_rotated = R * P * ~R
```

This corresponds to the geometry-oriented branch of `G_theta`. Field-data learners such as Clifford convolutions and natural-language interfaces for broader GA script generation are architectural extensions rather than complete modules in the present codebase.

### 4. Synthesis

In the general GGM formulation, synthesis compares generated outputs with reference data and domain constraints through a deviation measure:

```text
Delta = sum_i w_i Delta_i
||Delta|| = sqrt(<Delta * reverse(Delta)>_0)
```

The full paper formulation uses this discrepancy for iterative refinement across geometry, topology, metric relations, physical constraints, field structures, networks, and trajectories.

In the current repository, synthesis is implemented in a narrower validation-and-assembly sense:

- `task_ir_validator_node` checks whether generated semantic IR satisfies local structural and semantic constraints.
- Repair prompts are used when task decomposition violates the validator.
- `final_code_assembler_node` synthesizes generated code, assignments, and visualization sections into a single GAALOPScript artifact.
- `gaalop_request_builder_node` converts the synthesized artifact into a structured request.
- `scripts/run_ga_visagent.py` runs batch regression over the examples in `data/question.json`, producing outputs under `results/ga_visagent/`.

The current implementation does not compute a multivector deviation metric `Delta` or perform iterative numerical convergence. If the README needs to match the paper exactly, this should be described as future work or as an intended extension rather than as an implemented feature.

### 5. Computation

In the general GGM formulation, computation translates validated multivectors into executable code for target hardware:

```text
F: (M', H) -> C
O: M' -> IR
E: (IR, H) -> C
C = E o O
```

In this repository, the computation stage is represented by GAALOPScript assembly and GAALOP request construction:

- `final_code_assembler_node` produces the high-level algebraic script sections.
- `gaalop_request_builder_node` builds the request payload with target language, algebra plugin, optimization settings, visualization mode, and script content.
- The generated request is suitable for a GAALOP-style optimization and code-generation backend.

The current repository prepares the computation request but does not include a complete deployed GAALOP server, GPU/OpenCL/FPGA runtime, Clifford layer runtime, or hardware execution platform. Those belong to the broader full-stack architecture described in the paper text.

## Installation

Use Python 3.10 or higher. The project uses `uv` for environment creation and dependency locking.

```bash
uv sync
```

This creates `.venv/` and installs the dependencies declared in `pyproject.toml`.

On Windows, if `uv run` cannot access its cache, activate the generated environment or call its Python executable directly:

```bash
.venv\Scripts\activate
python scripts/run_gpt4o_baseline.py --help
```

or:

```bash
.venv\Scripts\python.exe scripts/run_gpt4o_baseline.py --help
```

The visible code imports LangChain-compatible packages such as `langchain_openai` and `langchain_ollama`, depending on the selected LLM backend.

## LLM Configuration

The LLM factory is defined in `src/ga_visagent/models/llm_setup.py` and supports:

- `ollama`
- `lm_studio`
- `openai`

The graph currently calls the local default LLM through helper logic in `src/ga_visagent/main_graph/nodes.py`. Configure the model, API key, and base URL according to your runtime environment before running end-to-end generation.

Default environment variables:

```bash
GA_VISAGENT_LLM_TYPE=lm_studio
GA_VISAGENT_LLM_MODEL=Qwen/Qwen3.6-27B
GA_VISAGENT_LLM_API_KEY=local
GA_VISAGENT_LLM_BASE_URL=http://localhost:1234/v1
GA_VISAGENT_LLM_TIMEOUT=120
GA_VISAGENT_LLM_MAX_RETRIES=2
```

Override these variables for your local OpenAI-compatible endpoint. Do not commit private endpoints or API keys.

## Running

Run a default example:

```bash
python scripts/run_single.py
```

Run with a custom prompt:

```bash
python examples/to_code.py "In conformal space, create points P1(0,0,0) and P2(1,0,0), then construct line L from P1 and P2. Visualize L in red. I need Python code."
```

Run the batch regression examples:

```bash
python scripts/run_ga_visagent.py
```

Run the GPT-4o baseline using the prompt template reported in the paper, with
an added GAALOP Web request JSON schema so the model returns a callable payload:

```bash
$env:OPENAI_API_KEY="your_api_key"
python scripts/run_gpt4o_baseline.py
```

The GPT-4o outputs are written to `results/gpt4o_baseline/`. Each output asks for the same request structure produced by `gaalop_request_builder_node`:

```json
{
  "visualizationEnabled": true,
  "outputMode": "CODE_AND_VISUALIZATION",
  "codegenPlugins": "PYTHON",
  "algebraPlugins": "ALGEBRA_CGA",
  "optimization": {
    "maxima": false,
    "cse": false
  },
  "script": {
    "optimizeCode": "...",
    "variableAssignments": "...",
    "multivectorsVisualized": "...",
    "functionName": "GeneratedFunction"
  }
}
```

The script records API-call success and JSON-parse success separately from GAALOPScript correctness; the paper's reported success rate is based on whether the generated GAALOPScript can execute correctly.

## Paper-Reported Accuracy

The paper `GA-VisAgent: A Multi-Agent application for code generation and visualization in interactive learning` evaluates 40 CGA-space code-generation and visualization tasks. Table 5 reports the following comparison:

| Model | Total Tasks | Success Tasks | Success Rate (%) |
| --- | ---: | ---: | ---: |
| GA-VisAgent | 40 | 36 | 90 |
| GPT-4o | 40 | 8 | 20 |

According to the paper, GA-VisAgent improves the GPT-4o baseline by 70 percentage points on this dataset.

## README Writing Notes for the Paper Alignment

When using this repository as supplementary material for the five-stage GGM description, the README should avoid claiming that every part of the full paper architecture is implemented. A precise wording is:

> This repository implements a prototype instantiation of the GGM pipeline for natural-language-driven CGA code generation. It covers representation, reasoning, generation, and request-level computation for vector-geometry tasks, while raster/field data, network data, Clifford learners, deviation-norm feedback synthesis, and hardware-specific execution backends are retained as architectural extensions.

This keeps the README consistent with the code and with the broader theoretical description.
