# Contracts

Draft, September 27, 2026. Proposed choices are marked **Proposed**. Items marked **Proposed** are the working design for milestone 1 (decided 2026-09-27). A **Proposed** item that turns out wrong during the build goes back to Jon as a question. Choices already made elsewhere are marked **Decided**, with the source.

## Why contracts

The project has three workstreams: engine, world, and models. The workstreams stay independent only if the boundaries between them are fixed. A contract is a written, versioned description of one boundary: what crosses the boundary, in what format, and when. Code on either side of a contract can change freely as long as the contract holds.

This document defines ten contracts:

1. Parameter schema: how entity types are declared, with defaults and ranges.
2. World rules: how actions change the world.
3. Agent interface: what an agent receives and returns.
4. Internal state and reward: how need levels reach the agent, and where reward is computed.
5. Propositional format: the symbolic description of the world.
6. Time: ticks, decisions, and clocks.
7. Logs: what is recorded, and in what format.
8. Viewer protocol: how state reaches a viewer.
9. Task format: how tasks and subskills are defined.
10. Experiment configuration: how one run or one batch of runs is specified.

At the end, the document tests the contracts on paper against two toy worlds: a tiny survival world and a tiny blocks world. The tests found gaps, which are listed with the tests.

## Principles shared by all contracts

- **One canonical state.** The engine holds one true world state. Pixels, sounds, propositions, and text are views computed from the canonical state. No view is ever the source of truth. (**Proposed**, as recommended in the environment survey.)
- **Engine-neutral data.** Worlds, entity types, tasks, and experiments are data files plus named rule modules. The engine never hard-codes a particular world.
- **No reward from the world.** The world reports need levels. Reward, if any, is computed on the agent's side. (**Decided**, 2026-09-26.)
- **Everything reproducible.** Every run is fully determined by its experiment configuration, its seeds, and the versions of the code and the contracts.
- **Versioned.** Each contract carries a version number. A change that breaks old files or old agents raises the major version. Every log records the contract versions used to produce the log. (**Proposed**.)
- **Swappable parts are registered by name.** Sensors, actuators, minds, rule modules, reward functions, and view generators are registered under names. Data files refer to the names. Adding a new part never requires changing a data file format. (**Proposed**. The design follows Melting Pot, where every game object is a list of named components with parameters.)

## 1. Parameter schema

### What the parameter schema covers

Every entity type is built in code from parameters (**Decided**, 2026-09-26). Files hold only default values and allowed ranges. The parameter schema defines the shape of those files, and the shape of a trait.

### Traits

Every numeric or categorical setting is a trait. A trait declaration has these fields:

| Field | Meaning | Required |
|---|---|---|
| `default` | The default value. | Yes |
| `range` | The allowed range: `[min, max]` for numbers, a list of options for categories. | Yes |
| `units` | Physical units, for example `kg`, `m/s`, or `per_hour`. | Numbers only |
| `variation` | How individuals vary when sampled: a distribution name and its parameters, for example `{normal: {sd: 5}}`. Values are clipped to `range`. | No (default: no variation) |
| `heritable` | Whether offspring inherit the trait. | No (default: `false`) |
| `mutable` | Whether the trait can mutate between generations. | No (default: `false`) |
| `visible` | Whether other agents can perceive the trait. | No (default: `false`) |

A trait written as a bare value is shorthand for a trait with that default, no variation, and a range equal to that single value. For example, `mass: 70` is shorthand for `mass: {default: 70, range: [70, 70], units: kg}`. The shorthand keeps simple files simple.

### Entity types

An entity type file names up to four components: body, sensors, actuators, and mind (**Decided**, 2026-09-26). Each component is a registered module name plus trait values for that module's parameters.

```yaml
type: human
version: 1
body:
  plan: biped                       # body plan: sets the parts list and the collision shapes
  model: human                      # asset name, looked up in the asset manifest
  mass: {default: 70, range: [40, 110], units: kg, variation: {normal: {sd: 10}}, visible: true}
  height: {default: 1.7, range: [1.4, 2.0], units: m, variation: {normal: {sd: 0.08}}, visible: true}
  insulation: {default: 0.2, range: [0, 1]}
  needs:
    hunger: {rise_per_day: {default: 0.5, range: [0, 5]}, at_max: {health_drain_per_hour: 0.5}}
    thirst: {rise_per_day: {default: 1.0, range: [0, 5]}, at_max: {health_drain_per_hour: 1.0}}
    fatigue: {rise_per_day: {default: 1.0, range: [0, 5]}, at_max: {collapse: {duration_hours: 2}}}   # full need format: addition A1
sensors:
  eyes: {resolution: [64, 64], fov_deg: 90, range_m: 50, color: true}
  interoception: {needs: [hunger, thirst, fatigue], noise_sd: 0.0}
  touch: {detail: whole_body}
actuators:
  legs: {}
  arms_and_hands: {}
  mouth: {}
mind: {module: random}
```

The engine builds an entity by starting from the type's defaults, applying any overrides from the experiment configuration, and then sampling each trait that has a `variation`. The sampled values are logged for every individual.

A new type can be declared on the fly in an experiment configuration, as a named override of an existing type:

```yaml
type: short_human
extends: human
body: {height: {default: 1.5}}
```

### Validation

The engine validates every file against a schema before a run starts. A missing required field, an unknown module name, or a value outside its range stops the run with an error. (**Proposed**: the schema is written once as Rust types, and a JSON Schema is generated from the Rust types for Python tools and editors.)

### File syntax

**Decided** (2026-09-27): YAML. YAML handles nested structures more readably than TOML, allows comments, and is what Crafter and BEHAVIOR use. YAML's known pitfalls (for example, `no` being read as false) are avoided by validating against the schema and by using a strict YAML parser.

## 2. World rules

### What the world-rules contract covers

World rules say what each action does, and what happens in the world without any action (plants regrowing, needs rising, water flowing). The rules are the ground truth of the world. The rules also generate the PDDL domain given to the planner agent.

### Scripted world

**Proposed**: every action in the scripted world is a rule with preconditions and effects over the canonical state. The format borrows from TextWorld's logic files and from Crafter's `data.yaml`.

```yaml
actions:
  eat:
    actuator: mouth
    args: {target: {has: edible}}
    pre:
      - within(self, target, 1.0)
    effects:
      - change: {entity: self, need: hunger, by: "-target.nutrition"}
      - consume: {entity: target, amount: 1}
    duration: {default: 2.0, units: s}
    energy_cost: {default: 0.001}
    sound: {event: chewing, loudness: 0.2}
  drink:
    actuator: mouth
    args: {target: {has: drinkable}}
    pre:
      - within(self, target, 1.0)
    effects:
      - change: {entity: self, need: thirst, by: -0.3}
    duration: {default: 3.0, units: s}

processes:
  regrow:
    applies_to: {has: regrows}
    every: {default: 600, units: s}
    effects:
      - restore: {entity: this, amount: 1, up_to: this.max_amount}
```

Rules whose effects cannot be written as data, such as pathfinding or predator chase behavior, are registered rule modules written in Rust. A data file names a rule module and passes the module's parameters. Every rule module must still declare which parts of the state it reads and writes, so the log and the propositional view stay complete.

Action failure follows the impossible-actions option (**Decided**, 2026-09-26). With masking on, an action whose preconditions fail is removed from the action mask. With masking off, the agent may choose the action, the action fails, and the failure is reported to the agent as an event (and optionally a proprioceptive signal).

### Generative world

In the generative world, physics (Rapier) decides what actions do (**Decided**, 2026-09-25). There are no hand-written effects for pushing, stacking, or toppling. Rules still exist for things physics does not cover: joining and separating parts, object behaviors (intrinsic motion, reactions to other objects, reactions to the agent), and scheduled changes. Object behaviors are registered modules with parameters, like the scripted world's rule modules.

Because physics has no preconditions and effects, the PDDL domain for the generative world cannot be generated from the rules. See the propositional-format section.

## 3. Agent interface

### Shape of the interface

**Decided** (2026-09-25): our own core API, with Gymnasium and PettingZoo as thin adapters. **Decided** (2026-09-27): the core API is multi-agent and parallel. Every call takes and returns dictionaries keyed by agent ID. A single-agent world is the case of one key.

```python
world = semantic_world.make(config)          # config: an experiment configuration (contract 10)
obs, info = world.reset(seed=0)              # obs: {agent_id: Observation}
while not world.done:
    actions = {aid: brains[aid].act(obs[aid]) for aid in world.agents_awaiting_action}
    obs, events, info = world.step(actions)
```

`step` returns observations, events, and info. `step` does not return reward, because the world gives no reward (**Decided**). The Gymnasium and PettingZoo adapters fill the reward slot by calling the agent's reward function (contract 4). The adapters exist so standard RL libraries run without changes.

Agents that are scripted inside the engine (animals, NPCs, scripted partners) use the same interface. Scripted agents run in Rust for speed, but their inputs and outputs follow the same manifest. A scripted agent can therefore be swapped for a Python model without changing the world.

### Observations

An observation is a set of named blocks, one per sensor, plus a flat vector made by concatenating all the blocks (**Decided**, 2026-09-26: the input vector and the sensor manifest).

```python
Observation(
    blocks={"eyes": uint8[64, 64, 3], "interoception": float32[3], "touch": float32[4], ...},
    vector=float32[N],          # all non-image blocks, concatenated in manifest order
    props=None or [...],        # propositional sensor, if switched on (contract 5)
    text=None or "...",         # heard speech as symbols, if that option is on
    mask=None or bool[...],     # action mask, if impossible actions are masked
)
```

**Decided** (2026-09-27): images stay as separate blocks and are not flattened into the vector. A 64×64 color image is 12,288 numbers. Flattening images would force every model to deal with them. Models that want everything as one vector can flatten the image blocks themselves.

### Sensor manifest

The sensor manifest is published once per agent at `reset`. The manifest lists each block's name, shape, type, range, units, and position in the flat vector, plus a human-readable label for each element.

```yaml
agent: human_0
blocks:
  - {name: eyes, shape: [64, 64, 3], dtype: uint8, in_vector: false}
  - {name: interoception, shape: [3], dtype: float32, range: [0, 1], offset: 0,
     labels: [hunger, thirst, fatigue]}
  - {name: touch, shape: [4], dtype: float32, offset: 3,
     labels: [intensity, dir_x, dir_y, dir_z]}
```

### Actions

**Proposed**: an action is a hybrid: one discrete choice of action type, plus that action type's arguments. Arguments can be continuous (a direction, a speed, a force) or discrete (a target). An action manifest, published with the sensor manifest, lists every action type and its argument slots.

Targets need care. Naming a target by its entity ID (`eat(bush_3)`) gives the agent object identity for free, which is one of the built-in representations the project wants to control. **Decided** (2026-09-27): targets can be given in two ways, and which ways an agent may use is an experiment setting:

- **Egocentric.** A target is whatever is in a given direction or at a given point in the visual field ("eat what is in front of me"). Agents that see only pixels use egocentric targets.
- **By ID.** A target is a named entity. Only agents that have the propositional sensor, and so already know the IDs, can target by ID.

### Timing

Agents act at decision points, not necessarily every tick (contract 6). `world.agents_awaiting_action` lists which agents must act before the next step. An agent in the middle of a voluntary durative action (for example, eating for two seconds) is still asked to act at every decision point: `noop` continues the action, and any other action interrupts the action (addition A4). A collapsed agent is not asked to act until the collapse ends.

## 4. Internal state and reward

**Decided** (2026-09-26): need levels arrive as simple sensory values through interoception. Reward is homeostatic: an agent is rewarded for lowering its needs, through a nonlinear function that levels off.

**Proposed**:

- The world's only job is to report need levels, through the interoception block. Interoception can be exact or noisy, as a trait.
- Reward functions live in an agent-side library, `semantic_world.reward`, registered by name. A reward function receives the agent's previous and current interoception blocks, plus the agent's set points, and returns a number.
- The default reward function is homeostatic drive reduction, following Keramati and Gutkin (2014): drive is a distance from the set points, `D(h) = (Σ_i |h*_i − h_i|^n)^(1/m)`, and reward is the drop in drive, `r_t = D(h_t) − D(h_{t+1})`. The exponents `n` and `m` control how strongly large deficits dominate. The exact curve is still open in `FIRST_STUDY.md`.
- Set points are traits of the mind, so set points can later evolve, as in Dynamica.
- **Decided** (2026-09-27): the agent-side reward library includes intrinsic rewards, such as curiosity or prediction error, alongside homeostatic reward. Intrinsic rewards give agents a reason to act in worlds without needs.
- **Decided** (2026-09-27): an optional **instruction channel** delivers goals to an agent, as a proposition, as symbols, or as speech from a teacher agent. A reward function may use the instruction, for example by rewarding the agent for satisfying the goal. The world still gives no reward.
- The reward an agent computed is logged with the agent's internals, so reward can be analyzed even though the world never produced reward.

## 5. Propositional format

### Facts

**Decided** (2026-09-25): the propositional layer is PDDL-compatible. **Proposed**: a fact is a predicate applied to arguments, written in PDDL style: `(on block_3 block_7)`, `(near human_0 bush_3)`, `(hunger human_0 0.62)`. Numeric values use PDDL 2.1 numeric fluents.

Object names follow BEHAVIOR's pattern of type plus index: `bush_3`, `wolf_1`. For the generative world, names are instance IDs only (`obj_17`), because the world has no given categories. The world's hidden ground-truth kinds are logged separately for analysis and are never shown to agents.

### Three uses of propositions

1. **The propositional sensor.** An agent with the propositional sensor receives the facts within the sensor's range, at the sensor's level of detail.
2. **Logs.** The full propositional state is logged at every decision point, for analysis.
3. **Planning.** The planner agent receives a PDDL domain and a PDDL problem.

### Where facts come from

- **Scripted world.** Facts are read directly from the canonical state, because the scripted world's rules are already written over symbolic state.
- **Generative world.** Facts are computed by registered detectors with thresholds, for example `on(a, b)` when `a` touches `b` and `a`'s center is above `b`'s top surface. Detectors are the "principled way to name things the world never labeled" from `TODO_world.md`. Detector thresholds are parameters, and are logged.

### The PDDL domain

For the scripted world, the PDDL domain is generated automatically from the action rules in contract 2. Stochastic effects and processes (plants regrowing, a wolf wandering) cannot be expressed in plain PDDL. **Proposed**: the generated domain covers the deterministic actions only, and the planner agent replans when the world surprises the agent. PPDDL (probabilistic PDDL) is a later option.

For the generative world, a hand-written PDDL domain over the detector predicates is needed. The hand-written domain is an approximation of the physics, and the approximation is itself an experimental variable.

## 6. Time

**Decided** (2026-09-25): the clock is configurable, with synchronous stepping by default and an optional real-time budget.

**Proposed**:

- **Tick.** The world advances in fixed ticks of simulated time. The default tick is 0.1 s of simulated time. Physics in the generative world may take several substeps per tick. (**Decided**, 2026-09-27.)
- **Decision interval.** Agents are asked for an action every `k` ticks (default 2, so every 0.2 s), or when a durative action ends. (**Decided**, 2026-09-27.)
- **Durative actions.** Actions have durations (contract 2). A durative action can be interrupted by the agent at any decision point, and by the world (for example, by an attack).
- **Synchronous mode.** The world waits for every agent awaiting action. Simulated time runs as fast as the compute allows.
- **Real-time mode.** Each decision point has a wall-clock budget. An agent that has not answered in time repeats its last action, or does nothing. Which of the two is an option, and the default is to repeat the last action. (**Decided**, 2026-09-27.)
- **World calendar.** Simulated time maps to days and seasons through a world parameter (for example, one simulated day equals 24 simulated minutes). Scheduled world changes are given in calendar time.

## 7. Logs

**Proposed**: every run writes one run folder:

```
runs/<run_id>/
  run.yaml             # the full resolved experiment configuration, seeds, code commit, contract versions
  manifests/           # sensor and action manifests for every agent
  individuals.parquet  # sampled trait values for every entity
  state.parquet        # canonical state snapshots, every K ticks (K is a setting)
  events.parquet       # every event: actions, action failures, sound events, births, deaths, speech
  props.parquet        # propositional state at each decision point (optional)
  agents/<agent_id>/   # observations, actions, reward, and internals (values, prediction errors, policy probabilities)
  checkpoints/         # world checkpoints and agent checkpoints at scheduled points
  recording.rrd        # optional Rerun recording for viewing and debugging
```

- **Tables in Apache Parquet.** (**Decided**, 2026-09-27.) Parquet files load directly into Python (pandas, polars) and R, which the analysis needs. Rust writes Parquet through Apache Arrow.
- **Images stored separately.** Rendered observations are large, so storing images is a setting: off, every `k`-th frame, or all frames. Images can always be re-rendered from state snapshots, because rendering is a view.
- **Agent internals.** A mind reports internals as named arrays with a manifest, in the same way sensors report blocks. Internals are logged at a rate set in the experiment configuration. (**Decided**, 2026-09-26: agent internals are loggable and viewable.)
- **Replay.** State snapshots plus the event log are enough to replay a run in a viewer. Exact re-simulation from a checkpoint and seed is also possible, because the engine is deterministic.
- **Large logs** stay out of the repository. They live on the cluster, in cloud storage, or in Box.

## 8. Viewer protocol

**Decided** (README): the simulation core streams state to a browser. **Proposed**, as recommended in the environment survey: state streams over a websocket to a three.js or Bevy-WebAssembly viewer, and Rerun is used for debugging and replay.

**Proposed**: the viewer protocol has four message types:

| Message | When | Contents |
|---|---|---|
| `init` | On connect | World size, terrain, the asset manifest entries in use, and every entity's type, model, and trait values that affect appearance. |
| `frame` | Every `k` ticks | Tick number, simulated time, and every changed entity's position, rotation, animation state, and visible trait changes. |
| `events` | With each frame | Events since the last frame: sounds, speech, actions, births, deaths. |
| `internals` | On request, for selected agents | Need levels, current action, and the internals the mind reports. |

- Messages are encoded in MessagePack, which is compact and supported in Rust, Python, and JavaScript. JSON is available as a fallback for debugging. (**Decided**, 2026-09-27.)
- A live run and a replay of a log send the same messages, so one viewer handles both.
- Viewers only receive. Viewers never change the world. A later "god mode" for demonstrations would be a separate, authenticated channel.

## 9. Task format

The world gives no reward, so a task is not a reward signal. A task is one of two things:

1. **An evaluation criterion.** A condition checked by the experimenter to measure success, such as survival time, whether a tool was made, or whether a tower of three blocks was built. The agent never sees the criterion.
2. **An instruction.** A goal given to the agent through a sense, for example a spoken instruction or a goal proposition. Instructions are needed in worlds without needs, such as the generative world (see the paper tests).

**Decided** (2026-09-26): tasks are compositions of named subskills, with controllable delays. **Proposed** format, borrowing BEHAVIOR's BDDL goal conditions:

```yaml
subskills:
  get_stone: {goal: "(holding ?agent ?s) (is stone ?s)"}
  get_stick: {goal: "(holding ?agent ?k) (is stick ?k)"}
  make_axe:  {goal: "(holding ?agent ?a) (is axe ?a)", requires: [get_stone, get_stick]}
  fell_tree: {goal: "(felled ?t)", requires: [make_axe]}

task:
  name: fell_a_tree
  goal: fell_tree
  delays:                               # time between the step that pays off and the payoff
    make_axe->fell_tree: {default: 0, range: [0, 3600], units: s}
  success: {within: {default: 7200, units: s}}
```

Goals are written as propositions (contract 5), so the same goal condition works in both worlds and can be checked from logs after a run.

## 10. Experiment configuration

An experiment configuration specifies everything needed to reproduce a run or a batch of runs.

```yaml
experiment: first_study_pilot
version: 1
world:
  regime: scripted
  map: {generator: meadow, size_m: [200, 200], params: {bushes: 20, water_sources: 3}}
  calendar: {minutes_per_day: 24}
  schedule:                              # scheduled world changes
    - {at_day: 30, change: {bush_regrow_every_s: 1200}}
population:
  - {type: human, count: 1, mind: {module: ppo_lstm, params: {...}}, overrides: {}}
  - {type: wolf, count: 2}
views: {propositional_sensor: false, speech_as_symbols: false}
options:
  impossible_actions: masked             # masked | attempt_and_fail
  touch_detail: whole_body
  death: ends_lifetime                   # ends_lifetime | persistent_society
clock: {mode: synchronous, tick_s: 0.1, decision_every_ticks: 2}
lifetime: {default_days: 60}
checkpoints: {every_days: 10}
logging: {state_every_ticks: 10, images: every_10th, internals: every_decision}
conditions:                              # the experimental design: one run per cell per seed
  mind: [random, scripted_optimal, planner_given, planner_learned, ppo_lstm, world_model]
  observation: [pixels, state_vector, propositions]
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

A batch runner expands `conditions` × `seeds` into runs, and every run writes its fully resolved configuration to `run.yaml`.

## Additions (decided 2026-09-27)

These additions fill gaps that the first build specification (`docs/specs/MILESTONE_1.md`) needed filled. The additions were found by a review of that specification and approved on 2026-09-27. Where an addition changes an earlier part of this document, the earlier part has been updated to match.

### A1. Needs (extends contract 1)

A body declares its needs as a mapping. Each need is a generic mechanism with these fields, all of them traits:

```yaml
needs:
  hunger:
    initial: 0.0
    rise_per_day: 0.5                      # rise rate in normal conditions
    rise_when: always                      # always | night_outside_shelter
    rise_multipliers: {running: 1.0}       # multiply the rise rate while a state holds
    fall_per_day: 0.0                      # passive fall rate, when the rise condition does not hold
    fall_multipliers: {}                   # multiply the fall rate while a state holds
    insulation_reduces_rise: false         # if true, rise rate × (1 − body.insulation)
    at_max: {health_drain_per_hour: 0.5}   # what happens at 1.0: health drain, or collapse
```

`at_max` holds either `health_drain_per_hour` or `collapse: {duration_hours}`. The states usable in multipliers are `running`, `asleep`, and `in_shelter`. Need values are clamped to the range 0 to 1. "At max" means a value of 1.0 after clamping. The `lethal_at` and `collapse_at` fields in the contract 1 example are replaced by `at_max`.

The body also declares `health: {initial: 1.0, recover_per_hour: 0.05}`. Health drains from all needs at max add together. Health recovers only while no need is at max. Health is clamped to the range 0 to 1. The agent dies when health reaches 0.

### A2. Tags, stocks, and placeholders (extends contract 1)

```yaml
body:
  tags: [edible, regrows]                  # free-form labels that rules select on
  stock: {berries: {initial: 3, max: 3}}   # countable amounts held by the entity
  provides: {hunger: -0.25}                # need changes per unit consumed
  placeholder: {shape: sphere, size: [1.2, 1.2, 1.2], color: "#2e7d32"}
  solid: true                              # whether the entity blocks movement
```

In milestone 1, `placeholder` replaces `model`. The `model` field is optional and is not validated against an asset manifest. Placeholder shapes are `capsule`, `sphere`, `cylinder`, `cone`, `box`, `disc`, and `shelter` (a box with one open side). A shape may be a list of parts with offsets, so a tree is a cylinder plus a cone. `size` is the full extent in meters along x, y, and z.

### A3. Rule expressions (extends contract 2)

Preconditions are a list of calls to registered predicates, all of which must hold:

| Predicate | Meaning |
|---|---|
| `within(a, b, d)` | The gap between the surfaces of `a` and `b` (distance between centers minus both radii on the ground plane) is at most `d` meters. |
| `facing(a, b, deg)` | The direction from `a` to `b` is within `deg` degrees of `a`'s facing. |
| `has_tag(x, tag)` | `x` carries the tag. |
| `stock_at_least(x, name, n)` | `x` holds at least `n` of the stock. |
| `awake(a)`, `asleep(a)`, `collapsed(a)` | The agent's sleep state. |

Effects are a list, each with one of these forms:

| Effect | Meaning |
|---|---|
| `change_need: {entity, need, by}` | Add `by` to a need, once. |
| `change_need_rate: {entity, need, per_second}` | Add `per_second` × elapsed time to a need, every tick while the action lasts. |
| `take_stock: {entity, name, amount}` | Remove stock. |
| `add_stock: {entity, name, amount, up_to}` | Add stock, up to a limit. |
| `set_state: {entity, state, value}` | Set a state such as `asleep`. |

Arguments may be literals or references: `self`, `target`, `this`, and a field path on one of those (`target.provides.hunger`, `this.stock.berries.max`). There is no arithmetic beyond an optional leading minus sign. New predicates and effects are added as registered modules, never as special cases in the engine.

### A4. Durative actions (clarifies contracts 3 and 6)

An agent in a voluntary durative action (`eat`, `drink`, `sleep`) stays in `agents_awaiting_action` at every decision point. Choosing `noop` continues the current action. Choosing any other action interrupts the current action and starts the new one. A collapsed agent is not asked to act until the collapse ends. Contract 3's sentence "an agent in the middle of a durative action is not asked to act until the action ends or is interrupted" is replaced by this rule.

### A5. Conditions in experiment configurations (clarifies contract 10)

Each key under `conditions` is a dotted path into the configuration, and each value is the list of settings to cross. For example, `population.0.mind.module: [random, scripted_optimal]` and `options.impossible_actions: [masked, attempt_and_fail]`. The batch runner crosses all conditions with all seeds. Mind module names resolve through a registry in `semantic_world.agents`.

## Paper tests

### Test 1: tiny survival world

**World.** A 50 m × 50 m meadow. One human. Five berry bushes, each holding three berries, regrowing one berry every ten minutes. One pond. One wolf that wanders and chases the human when the human is within 10 m. Needs: hunger and thirst. Death at a need level of 1.0, or after three wolf bites.

**Walkthrough.**

- *Parameter schema.* Human, wolf, berry bush, and pond are each expressible as a type file. Berries are an amount on the bush, not separate entities, as in Crafter. **Works.**
- *World rules.* `eat`, `drink`, and `regrow` are data rules. Wolf wandering and chasing need a rule module (a scripted mind). The wolf's bite is an action whose effect is an injury. **Works**, but the test showed that an injury needs a home in the body: **gap 1**, see below.
- *Agent interface.* Observations: eyes, interoception, touch, and pain. Actions: move (continuous direction and speed), turn, eat (target), drink (target). **Works.**
- *Internal state and reward.* Hunger and thirst come through interoception. The homeostatic reward function computes reward on the agent side. **Works.**
- *Propositional format.* `(near human_0 bush_3)`, `(berries bush_3 2)`, `(hunger human_0 0.4)`, `(near human_0 wolf_1)`. The generated PDDL domain covers `eat` and `drink`. Regrowth and the wolf are not in the domain, so the planner must replan. **Works, with the known PDDL limit.**
- *Time.* Eating is a 2 s durative action, and a wolf attack interrupts eating. **Works.**
- *Task format.* Evaluation criterion: days survived. **Works.**
- *Logs and viewer.* **Work** as specified.

### Test 2: tiny blocks world

**World.** A table. Three blocks of different sizes and colors. One manipulator agent: a hand that can move, grasp, and release. Physics on. No needs.

**Walkthrough.**

- *Parameter schema.* Blocks are nonliving bodies with size, color, mass, and friction traits. The manipulator is a body with a hand actuator. **Works.**
- *World rules.* Physics handles pushing, stacking, and falling. Grasping needs a rule: a grasp succeeds if the hand touches the block and the grip force exceeds a threshold. **Works.**
- *Agent interface.* Observations: eyes, touch per part (the hand), and proprioception (hand position and grip state). Actions are continuous: move the hand by a vector, set grip. **Works**, and the test confirms that the hybrid action design is needed: the survival world is mostly discrete choices, the blocks world is almost entirely continuous.
- *Internal state and reward.* No needs, so interoception is empty and homeostatic reward is always zero. **Gap 2**, see below.
- *Propositional format.* Detectors compute `(on obj_1 obj_2)`, `(clear obj_3)`, `(holding hand obj_1)`. A hand-written PDDL domain gives the classic blocks-world actions. **Works.**
- *Time.* Physics substeps inside each 0.1 s tick. **Works.**
- *Task format.* "Build a tower of three" as `(on obj_1 obj_2) (on obj_2 obj_3)`, checked by detectors. **Works as an evaluation criterion.** As an instruction to the agent, the task needs a channel: **gap 2** again.

### Gaps found

1. **Injury and health have no defined place.** Need levels are defined, but injuries are not a need. **Decided** (2026-09-27): the body's physiology has a `health` state and a list of injuries per body part. Health is reported to the agent through pain and interoception. Death conditions are written over needs and health together.
2. **Worlds without needs give an agent no reason to act.** In the blocks world, homeostatic reward is always zero. **Decided** (2026-09-27): both an **instruction channel** and intrinsic rewards. The instruction channel is an optional sense that delivers a goal to the agent (as a proposition, as symbols, or as speech from a teacher agent). Reward functions may then use the instruction, for example a reward for satisfying the goal. Intrinsic rewards, such as curiosity or prediction error, are part of the same agent-side reward library. The world still gives no reward: the reward function stays on the agent's side, and the agent's side decides how to use the instruction. See contract 4.
3. **Targets by ID leak object identity.** Found while writing the agent interface; addressed by the egocentric-or-ID option in contract 3.
4. **Stochastic and ongoing processes are outside plain PDDL.** Addressed for now by replanning (contract 5).

## Open questions

None at the moment. All ten open questions from the first draft were answered on 2026-09-27.
