# Milestone 1: the first working slice

Draft, September 27, 2026. This document is the build specification for the first working version of Semantic World. The specification is written for handoff to Claude Code, or to any developer, and should be complete without access to any other planning material.

How this specification relates to the other documents:

- `docs/CONTRACTS.md` defines the interfaces and file formats. The section "Contract additions for milestone 1" below fills gaps in the contracts that milestone 1 needed filled. The additions were approved on 2026-09-27, and the same text is in `CONTRACTS.md` under "Additions". Items marked **Proposed** in `CONTRACTS.md` are the working design for milestone 1.
- `docs/ENTITY_DEFINITIONS.md` explains the design of entities. Its YAML examples predate the contracts and are illustrative only. Where `ENTITY_DEFINITIONS.md` and `CONTRACTS.md` differ on file format, `CONTRACTS.md` wins.
- This specification defines the scope, the world's content, the numbers, and the build order.

## Goal

Build the smallest end-to-end version of the scripted survival world:

- a Rust simulation core that runs headless and deterministically;
- a Python API that follows the agent-interface contract;
- one tiny survival world, defined entirely in data files;
- placeholder shapes instead of real 3D models;
- rendered vision (64×64 color images from each agent's eyes);
- two baseline agents: a random agent and a scripted-optimal agent;
- logs in Parquet, and optional Rerun recordings;
- a batch runner that runs an experiment configuration across conditions and seeds.

Milestone 1 is done when a small validation run shows that the scripted-optimal agent survives far longer than the random agent, across ten seeds, with every run reproducible from its configuration and seed. That run is the first check of the first study's central question: does the world tell agents apart?

## Out of scope for milestone 1

These items are designed in the documents but are built later. Code for milestone 1 must not block them, and must not build them yet:

- the generative (blocks) world and Rapier physics;
- predators, injuries, and other animals;
- ears, smell, taste, pain, and per-part touch;
- speech, language, and the instruction channel;
- intrinsic rewards (a stub only);
- the web viewer and the viewer protocol (Rerun is the only viewer in milestone 1);
- real 3D models, the asset manifest, and glTF loading;
- the planner agents, the RL agent, and the world-model agent;
- PDDL domain generation (the propositional sensor is in scope; generating a PDDL domain is not);
- evolution, births, and persistent societies;
- scheduled world changes and checkpoints;
- real-time clock mode (synchronous mode only).

Configuration keys for out-of-scope features (for example `schedule`, `checkpoints`, `death: persistent_society`, `speech_as_symbols: true`, or `clock: {mode: realtime}`) must parse, then be rejected with a clear error that says the feature is not in milestone 1.

## Technology

Versions were checked on 2026-09-27. Pin minor versions in `Cargo.toml` and `pyproject.toml`. Upgrade deliberately, in a separate commit.

| Purpose | Choice | Version |
|---|---|---|
| Core language | Rust, edition 2024, current stable toolchain | stable |
| Entity-component system | `bevy_ecs`, used on its own, without the rest of Bevy | 0.19 |
| Math | `glam` | 0.33 |
| Random numbers | `rand` and `rand_chacha` (ChaCha8Rng) | 0.10 and 0.10 |
| Seed derivation | `sha2` | 0.11 |
| Data files | `serde` with `serde-saphyr` for YAML (`serde_yaml` is deprecated) | serde-saphyr 1.3 |
| JSON Schema from Rust types | `schemars` | 1.2 |
| Offscreen rendering | `wgpu` | 30 |
| Logs | `arrow` and `parquet` | 60 |
| Recordings | `rerun`, behind a Cargo feature named `rerun` | 0.38 |
| Python bindings | `pyo3` with the `abi3-py311` feature | 0.29 |
| NumPy arrays from Rust | `numpy` (rust-numpy), which must match the `pyo3` version | 0.29 |
| Python build | `maturin` | 1.15 |
| Python | 3.11 or later, with `numpy`, `pyarrow`, `polars`, `pytest`, and `ruff` | — |
| Python adapters | `gymnasium`, `pettingzoo` | 1.3, 1.27 |
| Python recordings | `rerun-sdk`, matching the Rust `rerun` version | 0.38 |

Rapier (`rapier3d` 0.36) is part of the planned stack but is not used in milestone 1.

## Repository layout

```
Cargo.toml                    # Cargo workspace
pyproject.toml                # Python package, built by maturin
crates/
  sw-schema/                  # Rust types for every data file; YAML loading; validation; JSON Schema export
  sw-core/                    # the world: ECS setup, clock, calendar, seeded random streams, building entities, needs and health, movement, the step loop, state hashing
  sw-rules/                   # the rule engine: actions, preconditions, effects, processes, detectors
  sw-sense/                   # sensors (except eyes): interoception, touch, proprioception, propositional sensor; manifests
  sw-render/                  # headless wgpu renderer for eyes; placeholder shapes
  sw-log/                     # Parquet writers; optional Rerun recording
  sw-py/                      # PyO3 bindings, compiled as the Python module semantic_world._core
python/semantic_world/
  __init__.py                 # make(), World, VecWorld, Observation
  reward.py                   # agent-side reward library (contract 4)
  agents/__init__.py          # Mind base class and the registry of agents by name
  agents/random_agent.py
  agents/scripted_optimal.py
  adapters/gymnasium_env.py   # single-agent Gymnasium adapter, plus the discrete-action wrapper
  adapters/pettingzoo_env.py  # parallel PettingZoo adapter
  run.py                      # batch runner: python -m semantic_world.run <experiment.yaml>
  analysis/summary.py         # validation summary for the definition of done
data/
  types/                      # default-value files for entity types: human, berry_bush, pond, shelter, tree
  rules/                      # actions, processes, and detectors for the survival world
  worlds/                     # world settings: map, calendar, light
  experiments/                # experiment configurations: m1_smoke.yaml, m1_validation.yaml
schemas/                      # JSON Schemas generated from sw-schema (committed, regenerated by a command)
tests/                        # Python tests; Rust tests live inside each crate
docs/
```

Crate and module names are recommendations. Keep the separation they express. The schema crate knows nothing about the engine. Engine crates contain mechanisms, not content: a need is a generic mechanism, and "hunger" exists only in `data/`. The same holds for predicates: the proposition vocabulary is declared in `data/rules/`, and each predicate is computed by a named, generic detector.

## Contract additions for milestone 1

The contracts left some formats open that milestone 1 must fix. These additions were **decided on 2026-09-27**, and are also recorded in `CONTRACTS.md` under "Additions".

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

## The tiny survival world

Everything in this section lives in `data/` files, not in Rust code. Every number is a default, declared as a trait or a world parameter with a range.

### Time and the calendar

Two time scales are used, and every number below says which one it uses:

- **Simulated seconds** govern fast, physical things: movement speed and action durations. One tick is 0.1 simulated seconds. Agents act every 2 ticks, so every 0.2 simulated seconds (contract 6).
- **The calendar** governs slow things: day length, needs, health, and regrowth. One calendar day is 1,440 simulated seconds (14,400 ticks, 7,200 decision points). One calendar hour is 1/24 of a day, which is 60 simulated seconds.

The day starts at dawn. Daylight holds for the first 60% of the day (14.4 calendar hours). Night holds for the rest (9.6 calendar hours). The light level rises and falls linearly over 1 calendar hour at dawn and at dusk, and the transition is centered on the boundary. It is **night** whenever the light level is below 0.5. The renderer uses the light level, so night images are darker. Every run starts at tick 0, at the start of dawn's transition.

### Space

- A flat, square meadow, 100 m × 100 m, centered on the origin, with walls at the edges that agents cannot pass.
- The world is 3D, but the ground is a flat plane at height 0 in milestone 1. The y axis points up.
- Movement is kinematic, with no physics. Every solid entity is a circle on the ground plane for blocking and distances. A moving agent that would overlap a solid circle is stopped at contact.

### Entities

| Type | Count | Placeholder | Blocking radius | Role |
|---|---|---|---|---|
| `human` | 1 to 8 | tan capsule, 1.7 m tall, 0.5 m wide | 0.25 m | The agent. Insulation 0.2. |
| `berry_bush` | 20 | green sphere, 1.2 m across; red when berries remain | 0.6 m | Food. Stock of 3 berries. Each berry lowers hunger by 0.25. |
| `pond` | 3 | flat blue disc, 3 m radius | not solid | Water. Agents can walk into ponds in milestone 1. |
| `shelter` | 2 | brown shelter shape, 3 m × 2.5 m × 3 m, open on one side | walls only, 0.2 m thick | Warmth and a better place to sleep. |
| `tree` | 30 | brown cylinder (0.3 m radius, 3 m tall) plus a dark green cone | 0.3 m | Obstacle and landmark. |

Placement uses the `map` random stream. Every object's center is at least 4 m from every other object's center and at least 3 m from the meadow's edge. Each shelter gets a random rotation, and its open side faces the way the rotation points. An agent is **inside** a shelter when the agent's center is within the shelter's footprint. Agents start at random positions at least 5 m from any object, with a random facing, drawn from the `map` stream. All needs start at 0, and health starts at 1.

### Needs

| Need | Rises | Falls | At max (1.0) |
|---|---|---|---|
| Hunger | 0.5 per day, always | Only by eating | Health drains 0.5 per calendar hour. |
| Thirst | 1.0 per day, always | Only by drinking | Health drains 1.0 per calendar hour. |
| Fatigue | 1.0 per day while awake; ×2 while running | 4.0 per day while asleep; ×2 inside a shelter | Collapse: forced sleep for 2 calendar hours, which cannot be interrupted. |
| Cold | 4.0 per day at night outside a shelter, × (1 − insulation) | 4.0 per day in daylight or inside a shelter | Health drains 1.0 per calendar hour. |

With insulation 0.2, cold rises by 0.133 per calendar hour at night outside a shelter. Cold therefore reaches 1.0 about 7.5 hours into the 9.6-hour night, so an agent that ignores shelter loses health every night.

Health recovers 0.05 per calendar hour while no need is at max. Death ends that agent's lifetime. Other agents continue. The run ends when every agent is dead, or when the configured lifetime ends.

Several agents can share the world. Agents do not interact in milestone 1, except by competing for berries and blocking each other's movement.

### Processes

- **Regrowth.** Every berry bush adds 1 berry, up to 3, every 10 calendar hours (600 simulated seconds), counted per bush from the start of the run.

### Actions

Actions follow contract 2 and the rule expressions in addition A3. Each action is one discrete type plus arguments (contract 3).

| Action | Arguments | Precondition | Effect |
|---|---|---|---|
| `noop` | — | — | Nothing, or continue the current durative action (addition A4). |
| `move` | `direction` (radians relative to facing, −π to π), `speed` (0 to 1) | Awake | For one decision interval (0.2 s), turn to face `direction`, then move at velocity `v`. `v = 1.4 × speed / 0.6` m/s for speed up to 0.6 (walking), and `v = 3.5` m/s for speed above 0.6 (running). |
| `turn` | `angle` (radians, −π/4 to π/4) | Awake | Rotate the facing by `angle`. |
| `eat` | `target` (optional) | `within(self, target, 1.0)`, `facing(self, target, 45)`, `has_tag(target, edible)`, `stock_at_least(target, berries, 1)`, awake | Durative, 2 simulated seconds. At the end: take 1 berry and lower hunger by 0.25. Interrupted eating has no effect. |
| `drink` | `target` (optional) | `within(self, target, 1.0)`, `facing(self, target, 45)`, `has_tag(target, drinkable)`, awake | Durative, up to 10 simulated seconds. Lowers thirst by 0.1 per simulated second. |
| `sleep` | — | Awake | Durative, until interrupted or until fatigue reaches 0. |

- **Targets.** Without a `target`, `eat` and `drink` act on the nearest object in front of the agent that meets the preconditions (egocentric targeting). With a `target` ID, the ID is used. Only agents with the propositional sensor may pass an ID (decided 2026-09-27). An ID from any other agent is an error.
- **Impossible actions.** Both options from the contracts are implemented: `masked` (the observation carries an action mask) and `attempt_and_fail` (the action fails, the agent loses that decision interval, and a `failed` event is reported). The option is set in the experiment configuration.
- **Action mask.** `bool[number of action types]`, in action-manifest order. The mask covers action types only, not arguments. `move` and `turn` are unmasked while the agent is awake.
- **Energy.** Every action has an `energy_cost` field in the rule files, set to 0 in milestone 1. Fatigue carries the cost of activity instead.

### Sensors

| Block | Shape | Contents |
|---|---|---|
| `eyes` | 64 × 64 × 3, uint8 | Rendered view from a camera 1.6 m above the ground, level, looking along the agent's facing. 90° horizontal field of view, 50 m range. |
| `interoception` | 5, float32 | Hunger, thirst, fatigue, cold, health. Optional Gaussian noise (trait `noise_sd`, default 0), clipped to 0 to 1. |
| `touch` | 4, float32 | Whole-body contact: intensity, then contact direction as a unit vector in the agent's frame. Intensity is 1 while the agent is blocked by a solid, and 0 otherwise. All zeros when not touching anything. |
| `proprioception` | 6, float32 | Current speed divided by 3.5, asleep (0 or 1), collapsed (0 or 1), heading as sine and cosine, and whether the last action failed (0 or 1). |

The propositional sensor is not a block. Its facts go in `Observation.props` (contract 3), never in `blocks` or `vector`. The propositional sensor is off by default. When on, the sensor reports facts about entities within 20 m.

Proposition vocabulary for milestone 1, declared in `data/rules/`:

- `(near ?agent ?obj)`: the surface gap is at most 3 m.
- `(visible ?agent ?obj)`: the object's center is within the field of view and range, and a line of sight from the camera to the center is not blocked by a tree trunk or shelter wall.
- `(is ?obj <type>)`: the object's type.
- `(berries ?bush n)`, `(inside ?agent ?shelter)`, `(asleep ?agent)`, `(collapsed ?agent)`, and `(night)`.
- Numeric fluents `(hunger ?agent v)`, `(thirst ?agent v)`, `(fatigue ?agent v)`, `(cold ?agent v)`, and `(health ?agent v)`, rounded to 2 decimal places.

Entity IDs are the type name plus an index, counting from 0 within each type in the order entities are created: `human_0`, `berry_bush_3`, `pond_1`.

The sensor manifest (contract 3) lists every block, with labels. The action manifest lists every action type, in a fixed order, with its argument slots and their ranges.

### Rendering

- Rendering is a view of the canonical state. The simulation never depends on rendering, and rendering can be switched off entirely.
- Build a small offscreen renderer on `wgpu`, not Bevy's renderer. The renderer draws flat-shaded placeholder shapes, the ground plane, one directional light scaled by the light level, and a sky color that follows the light level.
- Render all agents' views for one decision point in one batch, for speed.
- The renderer must run without a display. On machines without a GPU, including continuous-integration machines, `wgpu` must fall back to a software adapter (for example, llvmpipe or lavapipe on Linux). Tests that need rendering must pass on the software adapter.
- Rendering must be deterministic for a given state on a given machine. Exact pixel equality across different GPUs is not required.

## Agents

Both agents live in Python and implement the mind interface as an abstract base class, `semantic_world.agents.Mind`: `act(observation) -> action`, with optional `learn`, `report_internals`, `save`, and `load`. Later agents follow the same shape.

- **Random agent.** Chooses uniformly among action types, then draws uniform arguments. With masking on, chooses only among unmasked action types. Uses a random generator seeded from `world.agent_seed(agent_id)`. The random agent is the floor.
- **Scripted-optimal agent.** A hand-coded policy with full knowledge. The scripted-optimal agent may read the full world state through a privileged call, `world.debug_state()`, which must be clearly marked as privileged and must never be used by any learning agent. The policy: if health is falling, serve the need that is at max; otherwise serve the highest need above 0.3; at night, go to the nearest shelter and sleep there; walk to the nearest object that serves the chosen need, turn to face the object, and act until the need is below 0.1. Ties are broken by entity ID. The scripted-optimal agent is the ceiling. It does not need to be truly optimal, only competent enough that a good learner could approach it.

## Python API

The API follows contract 3. In outline:

```python
import semantic_world as sw

world = sw.make("data/experiments/m1_smoke.yaml", overrides={"seed": 3})
obs, info = world.reset(seed=3)                 # obs: {agent_id: Observation}
manifests = world.manifests()                   # sensor and action manifests per agent
brains = {aid: sw.agents.create("random", manifests[aid], seed=world.agent_seed(aid)) for aid in world.agents}
while not world.done:
    actions = {aid: brains[aid].act(obs[aid]) for aid in world.agents_awaiting_action}
    obs, events, info = world.step(actions)
```

- `Observation.blocks` is a dict of NumPy arrays. `Observation.vector` concatenates all non-image blocks, in manifest order. Images are never in the vector (decided 2026-09-27).
- Arrays cross from Rust to Python without extra copies where practical, through rust-numpy.
- `world.step` returns no reward (contract 4).
- `sw.VecWorld(config, n)` runs `n` worlds in parallel threads behind one call, with each world's seed derived from the master seed and the world's index. `VecWorld` is built in stage 10.

### Reward library

`semantic_world.reward` provides:

- `homeostatic`: drive reduction following Keramati and Gutkin (2014), as in contract 4. The drive is `D(h) = (Σ_i |h*_i − h_i|^n)^(1/m)`, and the reward is `D(h_t) − D(h_{t+1})`. The needs used are hunger, thirst, fatigue, and cold, each with a set point of 0. Health is not a need and is not used. The exponents `n` and `m` are parameters, with defaults given in the experiment configuration. **The default values of `n` and `m` are provisional** (decided 2026-09-27): use `n = 4` and `m = 3`, mark them as provisional in the code and in `run.yaml`, and expect them to change with the first study.
- `intrinsic`: a stub that raises `NotImplementedError`.

The Gymnasium and PettingZoo adapters compute reward by calling the reward function named in the experiment configuration.

### Adapters

- The Gymnasium adapter wraps one agent. It uses a `Dict` observation space and a `Dict` action space: `type` as `Discrete`, and each argument as a `Box`.
- A wrapper flattens actions into a small discrete set, because many RL libraries need discrete actions: noop, walk forward, run forward, turn left 45°, turn right 45°, eat, drink, sleep.
- The PettingZoo adapter follows PettingZoo's parallel API.
- Gymnasium's and PettingZoo's own API checkers must pass.

## Determinism

- One master seed per run, a 64-bit unsigned integer.
- Every source of randomness uses its own stream, named with a fixed string (for example, `"map"`, `"traits"`, `"rules"`, `"agent:human_0"`). A stream's seed is SHA-256 of the master seed (8 bytes, little-endian) followed by the stream name (UTF-8). The 32-byte digest seeds a `ChaCha8Rng`. Adding a new stream never changes an existing stream.
- Python agents get their randomness from `world.agent_seed(agent_id)`, which returns the same 32-byte digest for the stream `"agent:<id>"`, as an integer. Python seeds `numpy.random.default_rng` with that integer.
- No iteration over hash maps or hash sets anywhere simulation results depend on the order. Use ordered collections, or sort by entity ID.
- The core is single-threaded within one world. Parallelism comes from running many worlds at once.
- **State hash.** The hash covers every simulation component of every entity, visited in entity-ID order, plus the tick number, with floating-point values hashed by their bit patterns. The hash does not cover random-stream positions or rendering. Tests compare state hashes.

## Logs

Follow contract 7. In milestone 1, write:

- `run.yaml`: the fully resolved configuration, all seeds, the git commit hash (with a flag if the working tree had uncommitted changes), the Rust and Python package versions, and the contract version numbers.
- `manifests/`, `individuals.parquet`, `state.parquet` (every 10 ticks by default), and `events.parquet`.
- `agents/<agent_id>/`: non-image observations, actions, and agent-side reward and internals. The Python runner writes these tables, because reward and internals are computed in Python.
- Images: off by default, with a setting to store every `k`th frame.
- `recording.rrd`, when the `rerun` Cargo feature is on: the ground plane, entity positions, each agent's needs as time series, and each agent's current image.

## Build stages

Build in this order. Each stage ends with its tests passing locally and a commit. Each stage should be small enough for one focused working session. Jon pushes to GitHub and opens pull requests. Coding sessions do not push.

1. **Workspace skeleton.** Cargo workspace, `pyproject.toml`, empty crates, `maturin develop` working, a Python smoke test importing `semantic_world._core`. A GitHub Actions workflow that runs the full check list from `CLAUDE.md`, including `cargo clippy` and `cargo test` with `--features rerun`. *Accept:* the full check list passes locally.
2. **Schema.** Rust types for every data file in the contracts and in the additions above. Strict YAML loading. Validation with clear error messages that name the file and the field. The `extends` mechanism for derived types. Sampling of trait variation from the `traits` stream. Rejection of out-of-scope features. A command that writes JSON Schemas to `schemas/`. *Accept:* the YAML examples in `CONTRACTS.md` parse into the schema types (parsing only: the examples name types and modules, such as `wolf` and `ppo_lstm`, that milestone 1 does not define, so the examples are not resolved); all files in `data/` load and resolve; a set of deliberately broken files each fail with the right error.
3. **Core world.** Build entities from types in `bevy_ecs`. Map generation. The clock, ticks, decision points, calendar, and light level. The generic needs and health mechanism (addition A1), driven by the data files. Kinematic movement and blocking. State snapshots and the state hash. *Accept:* a headless run of 10 days with `noop` agents shows every need following its configured rate, and agents dying on the first night, when cold drains health (cold reaches 1.0 about 21.9 calendar hours into the run, and death follows 1 hour later); two runs with the same seed give identical state hashes at every day boundary; two runs with different seeds differ.
4. **Rules.** The rule engine for actions, processes, and detectors (addition A3), loaded from `data/rules/`. Durative actions and interruption (addition A4). Both impossible-action options. Events. *Accept:* unit tests for every action's precondition, effect, duration, interruption, and failure; a berry bush regrows on schedule; a masked action never appears as available when its precondition fails.
5. **Sensors and the Python API.** Interoception, touch, proprioception, the propositional sensor, and the sensor and action manifests. The PyO3 bindings and the Python `World` class. *Accept:* Python tests for `reset`, `step`, manifests, the action mask, observation shapes, `props`, and `agent_seed`; the random agent runs for 1 day from Python.
6. **Eyes.** The `wgpu` offscreen renderer and the placeholder shapes. *Accept:* images render on the software adapter; a test places an agent facing a bush and checks that the center of the image is mostly green; night images are darker than day images; sample images are written to `runs/samples/` for Jon to look at.
7. **Logs and recordings.** Parquet logs and the `rerun` feature. *Accept:* a run's logs load in polars; re-simulating from `run.yaml` reproduces the logged state hashes at every logged tick; a recording opens in the Rerun viewer.
8. **Agents, adapters, and the runner.** The reward library, the agent registry, the random and scripted-optimal agents, the Gymnasium and PettingZoo adapters, and the batch runner. *Accept:* Gymnasium's and PettingZoo's API checkers pass; the batch runner expands conditions × seeds and writes one run folder per run.
9. **Validation run (definition of done).** `data/experiments/m1_validation.yaml`: 1 agent per world; conditions `mind: [random, scripted_optimal]` × `impossible_actions: [masked, attempt_and_fail]`; 10 seeds each; lifetime 30 days; rendering off, except for one scripted-optimal run kept for viewing. `analysis/summary.py` reports, for each condition, the mean and spread of days survived, and the fraction of time with any need above 0.8. *Accept:* in both impossible-action conditions, the scripted-optimal agent survives far longer than the random agent on every seed (the scripted-optimal agent should reach the 30-day limit on most seeds, and the random agent should die within about 2 days); every run reproduces exactly from its `run.yaml`.
10. **Speed check and VecWorld.** Add `VecWorld`. Measure, and record in `docs/BENCHMARKS.md`, agent-steps per second, where one agent-step is one agent's decision: headless without eyes, and with eyes, on the developer's laptop. *Targets*, to be revised once measured: at least 100,000 agent-steps per second without eyes, and at least 2,000 with eyes.

## Questions for Jon

When a question comes up that this specification and the contracts do not answer, and the answer would change the contracts, the world's content, or what agents are given, stop and ask. Write the question, the options, and a recommendation in `docs/proposals/`, one file per question, rather than deciding silently. Small engineering choices inside a crate need no question.
