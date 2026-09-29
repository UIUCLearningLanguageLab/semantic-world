# Semantic World

An artificial world for comparing cognitive models.

**Status:** early design. The design documents are in `docs/`. No simulation code yet.

## Purpose

Semantic World is a simulated world in which different cognitive models can be placed "in the head" of an agent. We want to see what each agent does, and how well each agent does it. The questions are which kinds of models learn which kinds of tasks best, how those models work, and how to design better ones.

The candidate models are deliberately broad: neural networks of many kinds, reinforcement-learning agents, symbolic planners and cognitive architectures, Bayesian models, choice and decision models, and more biologically plausible models.

## Design principles

The project has three workstreams, kept as independent as possible, so a choice in one does not force a rewrite of the others:

1. **Engine** — the technical implementation: simulation core, headless execution, rendering, web viewing, and logging.
2. **World** — the content: what exists, what can happen, what agents need, and what language agents speak.
3. **Models** — the cognitive models placed in agents, and the questions tested with them.

Explicit contracts sit between the workstreams:

- **World ↔ engine.** A world is defined as data plus rules. The engine runs any world and hard-codes none.
- **Engine ↔ agent.** Every agent receives observations and returns actions through one interface. Our own core API serves reinforcement-learning agents, planners, and Bayesian models alike. Gymnasium and PettingZoo are thin adapters on top.
- **Engine ↔ viewer.** The simulation runs without a display. Viewers are optional clients.

A few more principles run through the design:

- **One canonical symbolic state.** Pixels, propositions, and text are all generated as views of the same world state. Which view a model receives is an experimental variable.
- **PDDL-compatible propositions.** The symbolic description of the world uses PDDL-compatible facts and action schemas, so classical planners can read the world directly.
- **The world hands out no reward.** The world reports each agent's internal state, such as hunger or thirst. Each agent, or each experiment, defines its own reward from that internal state. Different theories of reward can therefore be swapped in.
- **Agent internals are visible.** Value estimates, prediction errors, and predictions can be logged and shown in the viewer.

## Two kinds of world

- **A scripted world**, in the style of the game Dawn of Man: a hunter-gatherer world with procedural movement and no real physics. Agents have needs such as hunger, thirst, fatigue, and body temperature, and must find food, water, rest, and shelter.
- **A generative world**, in the style of Blocks World and SHRDLU: compositional objects made of parts, with features, behaviors, and a taxonomy of kinds. Objects are described in propositions, and the vocabulary for describing them is configurable (English words, novel words, or a mix).

## Entities

Humans and animals are defined by four modular components: a body, sensors, actuators, and a nervous system. The nervous system is the cognitive model. Plants and nonliving entities have a body only. Entity types are built from parameters, and every parameter leaves room for genetics and evolution later. See `docs/ENTITY_DEFINITIONS.md`.

## Planned architecture

- A headless simulation core written in **Rust**, using Bevy's entity-component system on its own.
- The **Rapier** physics library for the generative world only. The scripted world uses no physics.
- A **Python API** through PyO3, so models written in Python can connect.
- A browser viewer that receives state streamed from the compute machine, plus **Rerun** for debugging and replay.
- Development on laptops; heavy runs on a campus cluster and in the cloud.

The reasoning behind these choices is in `docs/ENVIRONMENT_SURVEY.md`.

## Documents

- `docs/specs/MILESTONE_1.md` — the build specification for the first working version: scope, world content, and build stages.
- `docs/CONTRACTS.md` — the contracts between the engine, the world, the agents, and the viewer: interfaces and file formats.
- `CONTRIBUTING.md` — how to set up and make a change. `CLAUDE.md` holds the same rules for Claude Code sessions.
- `docs/ENVIRONMENT_SURVEY.md` — a survey of existing environments, engines, and benchmarks, with a browsable version in `docs/environment_survey.html`.
- `docs/ENTITY_DEFINITIONS.md` — how bodies, sensors, actuators, and nervous systems are defined.
- `docs/specs/` — draft specifications for later work: developing organisms for the generative world, and a full wave simulation of sound.

## License

Apache License 2.0. See `LICENSE`.

## Contact

Semantic World is developed in the Learning and Language Lab at the University of Illinois Urbana-Champaign, led by Jon Willits.
