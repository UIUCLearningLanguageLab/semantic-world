# Semantic World

An artificial world for comparing cognitive models.

**Status:** early build. The simulation engine is being built in stages against `docs/specs/MILESTONE_1.md`: stages 1 to 5 (the workspace, the schema, the core world, the rules, and the sensors with the Python API) are built. The world-content and language programs are complete: the taxonomy generator, the word-form pipeline, and the corpus generator, each with a user guide in `docs/guides/`. The git log names the stages done so far.

## What works now

Semantic World runs worlds in two modes. The 3D simulation runs entities in space and time, and is still being built. The disembodied simulation generates a world's entities and events as propositions, without an active simulation. The disembodied simulation works now, in a first form, and its outputs are usable for training and testing models.

The disembodied simulation is a chain of three Python programs, which need no Rust:

1. **The taxonomy generator** makes the world: categories, instances, and their feature vectors, under rules we control (`docs/guides/TAXONOMY.md`).
2. **The corpus generator** states propositions about the world's rules and events, and realizes the propositions as documents in an artificial language, with test sets. Sentences can use conceptual labels or spoken words (`docs/guides/CORPUS.md`).
3. **The word-form pipeline** makes a spoken word for every word of the lexicon, with audio from many voices and sound embeddings (`docs/guides/WORDFORMS.md`).

The commands for the full chain are in `docs/guides/README.md`, under "The full chain". The chain ran at default scale on October 2, 2026: 10,000 documents with 92,464 sentences, spoken words for 173 content lexemes with five sound embeddings, and 14,736 test items. Each guide states what its program does not do yet.

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

Humans and animals are defined by four modular components: a body, sensors, actuators, and a mind. The mind is the cognitive model. Plants and nonliving entities have a body only. Entity types are built from parameters, and every parameter leaves room for genetics and evolution later. See `docs/ENTITY_DEFINITIONS.md`.

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
- `docs/ENTITY_DEFINITIONS.md` — how bodies, sensors, actuators, and minds are defined.
- `docs/specs/` — draft specifications for later work: developing organisms for the generative world, and a full wave simulation of sound.
- `docs/specs/TAXONOMY_GENERATOR.md`, `TAXONOMY_RELATIONS.md`, `WORDFORM_PIPELINE.md`, and `CORPUS_GENERATOR.md` — the specifications of the world-content and language programs: categories, features, and relations; spoken word forms and sound embeddings; and documents in an artificial language. `CONNECTED_SPEECH.md` plans spoken sentences, and is not built yet. `WORLD_AND_LANGUAGE.md` specifies the refactor that puts both simulation modes under one world model of state and change, and rebuilds the corpus on that model. It is drafted, and not built yet.
- `docs/guides/` — user guides for the taxonomy generator, the word-form pipeline, and the corpus generator.
- `docs/DECISIONS.md` — every design decision of the world-content and language programs, with who proposed it and who decided it.

## License

Apache License 2.0. See `LICENSE`.

## Contact

Semantic World is developed in the Learning and Language Lab at the University of Illinois Urbana-Champaign, led by Jon Willits.
