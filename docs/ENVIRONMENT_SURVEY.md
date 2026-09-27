# Survey of existing environments and engines

Checked September 25, 2026. Research was done with web search, PyPI, crates.io, and project pages. GitHub's API was blocked during the survey, so some last-commit dates could not be confirmed. Those are marked "unverified".

## Bottom line

No existing environment can be adopted as Semantic World. Every candidate fails at least one core requirement: 3D, open source with no proprietary engine, a symbolic or propositional observation channel, language between agents, and support for both world regimes.

No maintained, open simulator combines novel compositional object kinds, a taxonomy over those kinds, and a generated lexicon. That gap is real, and filling that gap is a contribution in its own right.

We should therefore build our own core, and borrow heavily. Many environments solve one piece of our problem well, and their designs are worth copying.

## Recommended architecture

- **Core in Rust.** A headless simulation core, using the ECS (entity-component-system) part of the Bevy engine on its own, pinned to one version. Compositional objects become entity hierarchies, with part-of relations. Stepping is synchronous with a fixed timestep. Each world has its own seeded random number generator, so runs are deterministic.
- **Physics only where needed.** The scripted (survival) world uses no physics: navigation plus needs, weather, and seasons written as ECS systems. The generative world uses the Rapier physics library for stacking, support, collision, and spatial queries. Rapier has a cross-platform determinism mode.
- **Python API through PyO3 and maturin.** A batch object steps many worlds per call. Observations come back as NumPy arrays, alongside a symbolic state and query API.
- **Our own core API first, then thin adapters.** Gymnasium (single agent) and PettingZoo (multiple agents) assume a reward-driven reinforcement-learning loop. Symbolic planners and Bayesian models need structured queries instead. So the core API serves every model family, and the Gymnasium and PettingZoo interfaces are thin layers on top.
- **Viewing.** The core streams compact state snapshots over a websocket. A browser viewer renders the snapshots, either with three.js or with a Bevy viewer compiled to WebAssembly. The Rerun visualization tool handles debugging, logging, and replay from the first milestone. Rendered camera observations for agents are an optional add-on, kept out of the core loop.
- **Training harness.** PufferLib (fast vectorized training, MIT license) is worth trying as the reinforcement-learning side.

Why Rust rather than C++: memory safety matters in a codebase touched by rotating students and RAs; maturin makes Python packaging nearly painless; and the core, the physics library, and the viewer would share one language and one set of types. A C++ stack (flecs + Jolt + nanobind) is equally viable if we prefer C++.

### Alternatives considered

- **Extend Luanti (formerly Minetest), possibly through Craftium.** This is the fastest route to the survival world. Luanti is an open-source voxel engine with a C++ core, a headless server, Lua scripting, native multiplayer and chat. Craftium adds Gymnasium and PettingZoo APIs and reports roughly 2,700 steps per second. The weakness: everything is made of voxels, which fits the generative world's multi-part objects badly. Using Luanti would likely mean two engines, one per regime, which weakens the shared-contract design.
- **Godot 4.x as the core.** Godot gives an editor, a renderer, and web export immediately. The costs: one process per environment over a socket, weaker batching and determinism, a headless mode with no rendering, and a stagnant reinforcement-learning plugin. Godot is the best off-the-shelf fallback.

### Main risks

- Bevy ships breaking releases every three to six months. Mitigation: pin versions, and keep the ECS behind our own API.
- A custom viewer is real work. Rerun and three.js lower that cost early on.
- Rendered observations at scale need GPU rendering on the cluster.
- Deterministic physics rules out some of Rapier's speed optimizations.
- Grad students and RAs will need time to ramp up on Rust.

## Survival, open-ended, and multi-agent environments (mostly 2D)

| Environment | Stack | License | Status | Multi-agent | Needs | Symbolic observations | Verdict |
|---|---|---|---|---|---|---|---|
| Crafter | Python, 2D | MIT | Dormant since Dec 2023 | No | Food, drink, energy, health; day/night | Semantic grid | Borrow ideas |
| Craftax | JAX, 2D | MIT | Active (Jun 2026) | No (forks yes) | Food, drink, energy, health, mana | Symbolic tensor | Borrow ideas |
| Alem (2026) | JAX, 2D | MIT | Very active | Yes | Food, water, and more | Symbolic, pixel, and text | Borrow ideas |
| Neural MMO 2 | Python, 2D | MIT | Ended Aug 2024 | Yes, 100+ | Food, water, health | Entity tables | Borrow ideas |
| Neural MMO 3 | C, 2D (in PufferLib) | MIT code, restricted art | Active | Yes, 1000+ | Health; seasons | Integer features | Borrow ideas |
| Melting Pot / DMLab2D | C++ and Lua, 2D | Apache-2.0 | Active | Yes | Per scenario | Customizable | Borrow ideas |
| XLand-MiniGrid | JAX, 2D | Apache-2.0 | Low activity | No | None | Grid plus rule grammar | Borrow ideas |
| Overcooked-AI | Python and JavaScript, 2D | MIT | Low activity | Two agents | None | Yes | Rule out |
| NetHack LE / MiniHack | C and Python, 2D | NetHack GPL / Apache-2.0 | Active / dormant | No | Hunger, health | Glyphs plus text | Rule out |

## 3D embodied and game-based environments

| Environment | Stack | License problem | Status | Multi-agent | Symbolic observations | Verdict |
|---|---|---|---|---|---|---|
| Luanti | C++17 core, Lua mods | None (LGPL-2.1) | Very active | Native | Full access via Lua; no channel built in | Extend (survival world only) |
| Craftium | Luanti fork plus Python | None (LGPL-2.1) | Small academic project | PettingZoo | None built in | Extend or borrow ideas |
| MineRL, MineDojo, Malmo, MineStudio | Minecraft (Java) | Proprietary Minecraft | Dormant or abandoned | Mostly no | Limited | Rule out |
| MineLand | Minecraft plus JavaScript | Proprietary Minecraft | Archived Jun 2026 | Yes, 48 agents; physical needs | Limited | Borrow ideas |
| AI2-THOR / ProcTHOR | Unity | Proprietary Unity | Slow public development | Yes | Excellent object metadata | Borrow ideas |
| ThreeDWorld | Unity | Closed engine source | Long-term support only | Yes | Rich object data | Rule out |
| Habitat 3.x | C++ core, Python | Some datasets non-commercial | Meta has stopped development | Yes | Semantic annotations | Borrow ideas |
| BEHAVIOR-1K / OmniGibson | Isaac Sim | Proprietary Isaac Sim; RTX GPU | Very active | Not a focus | Best in class (BDDL predicates) | Borrow ideas |
| SimWorld (2025) | Unreal Engine 5 | Unreal license terms | Active | Aims at 1000+ | Structured observations | Borrow ideas |

## Engines, physics, and infrastructure

| Option | What it is | Verdict |
|---|---|---|
| Bevy / bevy_ecs | Rust game engine and ECS; headless mode; compiles to the web | Strong candidate (ECS and viewer) |
| Rapier | Rust physics; determinism mode; JavaScript and new Python bindings | Strong candidate (Rust path) |
| Jolt | C++ physics; deterministic; used in Godot | Strong candidate (C++ path) |
| flecs / EnTT | C and C++ ECS libraries; flecs has built-in relationships and hierarchies | Possible (C++ path) |
| PyO3 + maturin | Rust-to-Python bindings and packaging | Strong candidate |
| nanobind | C++-to-Python bindings, faster than pybind11 | Strong candidate (C++ path) |
| Godot 4.x | Open-source game engine | Possible (fallback) |
| Unity + ML-Agents | Proprietary engine | Rule out (conflicts with open source) |
| MuJoCo / MJX | Robotics physics | Possible, optional backend for the generative world |
| Genesis, Newton, Isaac | GPU robotics simulators | Rule out as core |
| Bullet / PyBullet, PhysX | Older or heavier physics | Rule out |
| Gymnasium, PettingZoo | Agent interface standards | Strong candidates, as thin adapters |
| PufferLib | Fast vectorized RL training | Possible training harness |
| Rerun | 3D logging and web viewer | Strong candidate for debugging and replay |
| three.js / Babylon.js | Browser 3D rendering | Strong candidates for the web viewer |

## Language, compositional, and propositional resources

| Resource | What it contributes | Verdict |
|---|---|---|
| SHRDLU revivals | Dialog types: reference, anaphora, "why" questions, commands | Borrow ideas |
| Blocks World PDDL, PlanBench | Propositional state and action format; Mystery Blocksworld renames predicates, and LLM performance collapses | Adopt the format |
| CLEVR | Scene graph → program → question → answer, with answer balancing | Borrow ideas |
| Super-CLEVR | Objects with parts and part-level attributes ("the bus with a red wheel") | Borrow ideas |
| CLEVRER | Event annotations; descriptive, explanatory, predictive, counterfactual questions | Borrow ideas |
| Kubric | Procedural 3D scene and video rendering (Blender plus PyBullet) | Extend, optional rendering path |
| PHYRE, Physion | Generalization splits; hidden properties revealed by behavior | Borrow ideas |
| BabyAI / MiniGrid | Instruction grammar with a verifier; Gymnasium API | Borrow ideas |
| XLand-MiniGrid | Generated rule sets ("if A near B then C") | Borrow ideas |
| SCAN, gSCAN, ReaSCAN | Compositional generalization splits; ReaSCAN's distractors make every word necessary | Borrow ideas |
| TextWorld, ALFWorld | Facts-plus-rules state; one world with text and visual views | Borrow ideas |
| ScienceWorld | Shows how real-world knowledge biases comparisons toward LLMs | Borrow ideas |
| EGG | Emergent-communication games and compositionality metrics (archived Aug 2026) | Borrow ideas |
| MEWL | Nine few-shot word-learning tasks drawn from developmental psychology | Borrow ideas |
| DevBench | Scores models by similarity to children's response patterns | Borrow ideas |

## Ideas to borrow, by workstream

### Contracts and engine

- **One canonical symbolic state, with every modality derived from that state.** Pixels, propositions, and text are all views of the same state (the ALFWorld and TextWorld pattern). This matches our observation-channel design.
- **A PDDL-compatible propositional layer.** If our facts and action schemas are PDDL-compatible, off-the-shelf classical planners become symbolic baselines for free. That matters because the symbolic planner is in our first comparison.
- **BDDL-style predicates over object states** (from BEHAVIOR) and AI2-THOR's object metadata as templates for the propositional channel.
- **Melting Pot's component-based entity definitions**, which let new entities and rules be declared as data.
- **Symbolic, pixel, and text observation channels, plus agent messaging** (from Alem).
- **A fast C or Rust core behind a thin Python binding** (from Neural MMO 3 and PufferLib).

### Survival world

- **Crafter's need dynamics** are a clean minimal model to start from: separate hunger, thirst, and fatigue counters, slower need decay during sleep, and a day/night cycle. Our version adds temperature.
- **Neural MMO's multi-agent resources**: regrowth, depletion, respawning, and death with many agents.
- **Luanti's biome heat and humidity values**, as a model for regional temperature.

### Generative world

- **Rule generation in the style of XLand-MiniGrid.** Sample a grammar of rules (deterministic or probabilistic) linking features, parts, and kinds to capacities and interactions. Rule complexity and stochasticity then become controlled experimental variables.
- **Super-CLEVR's part-attribute schema** for objects like "the blorg with red arms".
- **Compositional splits as built-in generator options**: primitive addition (a new "blorg"), held-out combinations of attributes and kinds, depth and length, and within-template versus cross-template splits.
- **ReaSCAN's distractor generation**, so that every word in a referring expression is necessary. This defeats bag-of-words shortcuts.
- **CLEVR's question balancing** and **CLEVRER's four question types** for the question-answering and prediction tasks.
- **The lexicon as an experimental manipulation.** Mystery Blocksworld and ScienceWorld both show that familiar English names give LLMs an advantage from prior knowledge. Switching between English, novel, and mixed labels over an identical world is the cleanest control for that bias. This matches the decision to make the lexicon configurable.
- **MEWL's nine word-learning tasks** as a checklist for the word-learning benchmark.

### Models and evaluation

- **Melting Pot's held-out-population evaluation**: testing agents against partners they did not train with.
- **EGG's compositionality metrics** (such as topographic similarity) for the emergent-language setting.
- **DevBench's method** of scoring models by similarity to children's response patterns, if developmental comparisons become a priority.

## Decisions this survey suggests

- Choose Rust for the core (recommended), unless there is a reason to prefer C++.
- Use one engine for both regimes, rather than extending Luanti for the survival world only.
- Make the propositional layer PDDL-compatible.
- Design our own core API first, with Gymnasium and PettingZoo as adapters.

## Sources

Survival, open-ended, and multi-agent environments:

- Crafter — https://github.com/danijar/crafter
- Craftax — https://github.com/MichaelTMatthews/Craftax ; https://arxiv.org/abs/2402.16801
- Alem — https://github.com/alem-world/alem-env ; https://arxiv.org/abs/2606.08340
- Multi-agent Craftax — https://github.com/BaselOmari/MA-Craftax ; https://arxiv.org/abs/2511.04904
- JaxLife — https://github.com/luchris429/JaxLife ; https://arxiv.org/abs/2409.00853
- Neural MMO — https://github.com/NeuralMMO/environment
- PufferLib — https://github.com/PufferAI/PufferLib
- Melting Pot — https://github.com/google-deepmind/meltingpot ; https://github.com/google-deepmind/lab2d
- XLand-MiniGrid — https://github.com/dunnolab/xland-minigrid
- Overcooked-AI — https://github.com/HumanCompatibleAI/overcooked_ai
- NetHack Learning Environment — https://github.com/NetHack-LE/nle ; MiniHack — https://github.com/samvelyan/minihack

3D environments:

- Luanti — https://github.com/luanti-org/luanti ; https://blog.luanti.org/2026/05/24/5.16-released/
- Craftium — https://github.com/mikelma/craftium ; https://arxiv.org/abs/2407.03969
- MineRL — https://github.com/minerllabs/minerl ; MineDojo — https://github.com/MineDojo/MineDojo ; Malmo — https://github.com/microsoft/malmo
- MineLand — https://github.com/cocacola-lab/MineLand ; MineStudio — https://github.com/CraftJarvis/MineStudio
- AI2-THOR — https://github.com/allenai/ai2thor ; ProcTHOR — https://github.com/allenai/procthor
- ThreeDWorld — https://github.com/threedworld-mit/tdw
- Habitat — https://github.com/facebookresearch/habitat-sim ; https://github.com/facebookresearch/habitat-lab
- BEHAVIOR-1K — https://github.com/StanfordVL/BEHAVIOR-1K
- SimWorld — https://github.com/maitrix-org/SimWorld ; https://arxiv.org/abs/2512.01078

Engines and infrastructure:

- Godot — https://endoflife.date/godot ; Godot RL Agents — https://github.com/edbeeching/godot_rl_agents
- Unity ML-Agents — https://github.com/Unity-Technologies/ml-agents/releases
- Bevy — https://bevy.org/news/bevy-0-19/ ; https://github.com/bevyengine/bevy/blob/main/examples/app/headless.rs
- Rapier — https://crates.io/crates/rapier3d ; https://rapier.rs/docs/user_guides/rust/determinism/
- Jolt — https://github.com/jrouwe/JoltPhysics
- MuJoCo — https://github.com/google-deepmind/mujoco/releases ; Genesis — https://github.com/Genesis-Embodied-AI/Genesis ; Newton — https://github.com/newton-physics/newton
- PyO3 — https://crates.io/crates/pyo3 ; maturin — https://pypi.org/project/maturin/ ; nanobind — https://pypi.org/project/nanobind/
- flecs — https://ajmmertens.medium.com/flecs-4-1-is-out-fab4f32e36f6 ; EnTT — https://github.com/skypjack/entt/releases
- Rerun — https://rerun.io/docs/reference/cli
- Gymnasium — https://pypi.org/project/gymnasium/ ; PettingZoo — https://pettingzoo.farama.org/

Language, compositional, and propositional resources:

- SHRDLU revivals — https://github.com/garfix/blocks-world ; https://github.com/tsgouros/www-shrdlu
- PDDL generators — https://github.com/AI-Planning/pddl-generators ; PlanBench — https://github.com/karthikv792/LLMs-Planning ; https://arxiv.org/html/2409.13373v1
- CLEVR — https://github.com/facebookresearch/clevr-dataset-gen ; Super-CLEVR — https://github.com/Lizw14/Super-CLEVR ; CLEVRER — https://github.com/chuangg/CLEVRER
- Kubric — https://github.com/google-research/kubric ; PHYRE — https://github.com/facebookresearch/phyre ; Physion — https://github.com/cogtoolslab/physics-benchmarking-neurips2021
- MiniGrid / BabyAI — https://minigrid.farama.org/
- SCAN — https://github.com/brendenlake/SCAN ; gSCAN — https://github.com/LauraRuis/groundedSCAN ; ReaSCAN — https://github.com/frankaging/Reason-SCAN
- TextWorld — https://github.com/microsoft/TextWorld ; ALFWorld — https://github.com/alfworld/alfworld ; ScienceWorld — https://github.com/allenai/ScienceWorld
- EGG — https://github.com/facebookresearch/EGG
- MEWL — https://arxiv.org/abs/2306.00503 ; DevBench — https://arxiv.org/abs/2406.10215
