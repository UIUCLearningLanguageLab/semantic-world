# World model and language: the refactor

Draft, October 6, 2026. Written in the planning conversation, for handoff to Claude Code. Not yet built.

## What this specification does

This specification brings the disembodied simulation under one world model of state and change. It then rebuilds the corpus generator on that model. The 3D engine already works with state, actions with preconditions and effects, and time (contracts 2, 5, and 6 in `docs/CONTRACTS.md`). Until now, the disembodied pipeline has had static entities and events that change nothing (CG.4). After the refactor, both simulation modes read one definition of a world.

The refactor is built in four phases. Each phase ends with a working corpus:

- **(a)** static and fluent one-place facts, event types with preconditions and effects, inertia, and time steps;
- **(b)** comparisons derived from scalars (CG.65 and CG.66);
- **(c)** stored fluent relations, their profiles, and locations;
- **(d)** sorts, derived relations, and parts.

Phases (a) and (b) are specified in full. Phases (c) and (d) are specified as a design with open questions. Each of the two gets its detailed sections, reviewed by Jon, before its build starts.

Every taxonomy run, world run, and corpus is regenerated. No output of the refactor needs to match an output made before the refactor.

### Decisions this specification builds on

| ID | Decision |
| --- | --- |
| REL.16 | The world is defined by each entity's base vector (free IS and HAS features, and scalars), feature rules, and event types. CAN features and projections are derived values: computed, cached with the identity of the rule set, never an input, and never edited. What a model sees is an experiment setting. What the language names is a setting of the language. |
| REL.17 | An event type has participant roles, each with a requirement over the features of whatever fills the role. CAN features become one-place event types. Verbs become two-place event types. On the world side, "event type" replaces "verb". |
| REL.18 | Rules are stored in Boolean form and computed as two-layer threshold matrices. Each run exports the matrices, and a test checks that the two forms agree. |
| REL.19 | Static relations stay featural. Stored relations exist only as fluents, created and destroyed by events. |
| CG.59 to CG.64 | The fixes of the proposition check: `ALL` and `NEC(ALL(...))`; no `GEN`; rule statements are `NEC(ALL(...))`; `MOST` means more than half; the assertion is separated from the descriptions that identify referents; aspect belongs to the report. |
| CG.65 to CG.67 | Relative gradable adjectives name their comparison class and threshold; comparatives compare two values; "near" is a stored fluent relation, added with phase (c). |
| Box `DECISIONS.md`, October 6 | One world model for both simulation modes: static facts and fluents, derived facts, event types with preconditions and effects, and time steps in which unaffected facts persist. |

Three build rulings were made on October 6, in the planning conversation that wrote this specification:

- **One definition, two runtimes.** The disembodied mode and the 3D engine share a world model, not a runtime. Python generates a world and writes the world's definition as data. A Python runtime runs the disembodied mode. The Rust engine will later load the same files and run them in 3D. The two modes share the format of the definition, the format of histories, and a set of conformance fixtures.
- **Conformance fixtures from phase (a).** The fixtures are written as the Python runtime is built, so the Rust runtime has its tests waiting when the 3D link is built.
- **A new package.** The Python runtime and the world generator live in `python/semantic_world/world/`. The taxonomy generator stays the generator of each entity's initial static facts.

### What it replaces

- In `TAXONOMY_GENERATOR.md`: CAN features, which become one-place event types and derived capacities, and the single `instances.csv` table, which becomes base and derived tables plus views.
- In `TAXONOMY_RELATIONS.md`: Part B moves from the taxonomy package to the world package, with "verb" renamed "event type". Exposure (`verbs.projections.expose_*`) is removed (REL.16).
- In `CORPUS_GENERATOR.md`: the sections named in "Phase (a): the corpus" and "Phase (b): the corpus".
- CG.4 ("events never change state") is superseded. Event-level propositions are still never negated.

The older specifications stay as the record of what was built before. Each stage that changes behavior described in an older specification adds a short note at the top of the affected section, pointing to this specification.

## Terms

- **World definition.** The data that defines a world: its entities' base facts, its feature rules, and its event types. Nothing else is part of the definition (REL.16).
- **Entity.** An individual in the world. In phases (a) and (b), every entity is a taxonomy instance.
- **Base fact.** A fact that is stored. **Derived fact.** A fact computed from other facts by a rule. A derived fact is never stored as an input, and never written by an event.
- **Static.** Fixed for the whole life of the world. **Fluent.** Able to change over time. Every fact is static or fluent, and base or derived, so there are four kinds: base static (a free IS feature), derived static (a determined IS feature), base fluent (a state that events change), and derived fluent (computed from fluents).
- **Event type.** A kind of event, with participant roles (`agent`, and for a two-place event type, `patient`), a requirement, a precondition, and effects.
- **Binding.** An assignment of entities to the roles of an event type. A binding never uses one entity twice (REL.5).
- **Requirement.** A rule over the static facts of a binding. When the requirement holds, the entities are **able** to take part in the event: the owl can eat the mouse. Capacities ("can") are requirements.
- **Precondition.** A rule over the fluent facts of a binding. The precondition says whether the event can happen now: the owl is awake.
- **Legal.** A binding is legal in a state when its requirement holds and its precondition holds in that state.
- **Effect.** A change that an event makes to a base fluent of one of its participants.
- **Event.** One occurrence of an event type, with a binding, in one time step of one episode.
- **State.** The values of every base fluent of every entity at one time point. Static facts are not part of the state, because they never change.
- **Time point and step.** An episode has time points `T.1`, `T.2`, and so on. Step k takes the state at `T.k` to the state at `T.k+1`. The events of step k happen between the two time points.
- **Episode.** A run of the world over a set of participants and a number of steps. The corpus calls an episode a scene, and keeps the label `SN.<n>`.
- **History.** The record of an episode: its participants, its initial state, and each step's events and changes.
- **Selection policy.** The procedure that chooses which legal events happen. A selection policy is never part of the world.
- **Runtime.** The code that computes derived facts, legal bindings, and next states from a world definition. The runtime has no randomness.
- **View.** A choice of facts that a model, or a file, shows. Views are experiment settings, never part of the world (REL.16).

## Architecture

### One definition, two runtimes

```
taxonomy generator ──► world generator ──► world definition (data files)
                                                 │
                       ┌─────────────────────────┴──────────────────────────┐
              Python runtime (disembodied mode)                  Rust runtime (3D mode, later)
                       │                                                    │
                  histories ─────────────► corpus generator ◄───────── histories (later)
```

### What lives where

| Part | Language | Package or crate |
| --- | --- | --- |
| Generating entities' initial static facts: the tree, the free features, the feature rules, the scalars, the instances | Python | `semantic_world.taxonomy` |
| Generating the rest of a world: fluents, event types, requirements, preconditions, effects; writing the definition | Python | `semantic_world.world` |
| Analyses of a rule set: the fixed-by-rule test (which `NEC` needs), capacities, relation proportions, thematic relatedness | Python | `semantic_world.world` (moved from the taxonomy package) |
| The runtime: derive, legal, apply | Python now; Rust later | `semantic_world.world.runtime`; a Rust crate when the 3D link is built |
| Episodes in the disembodied mode, and the selection policy | Python | `semantic_world.world.episodes` |
| Space, bodies, sensors, rendering, ticks, durative actions, detectors | Rust | the existing crates |
| The corpus and the language programs | Python | `semantic_world.corpus`, `semantic_world.wordforms` |

Everything in the Python column runs with Python alone, without building the Rust core: NumPy, polars, and PyYAML only (the decision of September 29).

### The runtime's operations

The runtime is a small interpreter of the definition. Its operations, in Python:

- `derive(definition, state, entities) -> DerivedFacts`: the derived static facts (cached once, by rule-set identity) and the derived fluent facts of the given entities in the given state.
- `able(definition, event_type, bindings) -> bool array`: whether each binding's requirement holds. Static, so the result can be cached.
- `legal(definition, state, event_type, bindings) -> bool array`: whether each binding is legal in the state.
- `legal_bindings(definition, state, event_type, entities) -> list of bindings`: every legal binding among the given entities, in a fixed order (by agent, then patient, in entity order).
- `apply(definition, state, events) -> State`: the state after one step. Every event must be legal in `state`, and the events must not interfere (see "Time, steps, and inertia"). Otherwise `apply` raises an error that names the event and the reason.

The runtime never draws a random number, and never chooses an event. The same definition, state, and events always give the same result.

### Conformance fixtures

A conformance fixture is one JSON file in `tests/fixtures/world/`. Each fixture holds:

- `definition`: a complete world definition, inline, in the JSON form of the definition files;
- `initial`: the base fluents that are true at `T.1`, for each entity;
- `steps`: a list of steps. Each step holds its `events` (event type and binding) and what must come out: `legal` (for each event type, the legal bindings in the state before the step), `derived` (the derived fluent facts that are true after the step), and `state` (the base fluents that are true after the step);
- `error`: null, or the kind of error that `apply` must raise at a given step (an illegal event, or interfering events).

Two kinds of fixture are written:

- **Hand-written fixtures**, at least eight in phase (a), each with its expected values worked out by hand: a fluent with no event; an effect that sets a fluent; an effect that clears one; an effect that sets a fluent already true (no change); a derived fluent that changes because a base fluent changed; a precondition that blocks an event; two events in one step that do not interfere; and two events in one step that do interfere (an error).
- **Generated fixtures**, made from the tiny world configuration. Their expected values are computed by a brute-force evaluator written separately from the runtime: the evaluator reads the rules' truth tables, never the matrices.

`python -m semantic_world.world check-fixtures` runs the Python runtime on every fixture. The Rust runtime will run the same files.

## The world definition

### Labels

| Object | Label | Example | Replaces |
| --- | --- | --- | --- |
| Base or derived fluent (Boolean) | `ST.<n>` | `ST.3` | new |
| One-place event type | `EV.<n>` | `EV.4` | `CAN.<n>` as a feature |
| Two-place event type, or a category of two-place event types | `E<i>.<j>...` | `E1.2` | `V<i>.<j>...` |
| Event-type feature | `EF.<n>` | `EF.5` | `VF.<n>` |
| Constraint of an event-type feature, or an event type's own constraint | `K.EF.<n>`, `K.<event type>` | `K.EF.5`, `K.E1.2.3` | `K.VF.<n>`, `K.<verb>` |
| Capacity: able to be the agent of an event type (derived) | `CAN.<event type>` | `CAN.EV.4`, `CAN.E1.2` | `CAN.<n>`, `CAN.<verb>` |
| Capacity: able to be the patient of a two-place event type (derived) | `CANBE.<event type>` | `CANBE.E1.2` | `CANBE.<verb>` |
| Time point within an episode | `T.<k>` | `T.3` | new |

Labels for categories, instances, IS, HAS, and scalar dimensions are unchanged. Inside requirements and preconditions, a literal names its role with a prefix, as now: `a.` for the agent and `p.` for the patient (`a.HAS.4`, `p.ST.2`).

### Files

A world run writes one folder, by default `runs/world/<name>_seed<seed>/`:

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved world configuration, including the resolved taxonomy configuration, all seeds, the git commit hash (flagged when the working tree had uncommitted changes), and the package version. |
| `taxonomy/` | The taxonomy run that supplies the entities, in the taxonomy's own output format (see "Taxonomy outputs" under phase (a)). |
| `definition.json` | The definition: every symbol, every rule in Boolean form and in matrix form, and every event type. The schema is below. |
| `entities.csv` | One row per entity: its label, its leaf, its base static facts (free IS and HAS features, scalars), and its initial base fluents. |
| `derived/` | Derived values, each file tagged with the rule-set identity (see "Rule-set identity"). |
| `world_stats.yaml` | The statistics in "World statistics". |

`definition.json` is the language-neutral part of the definition. It holds no Python expressions and no code. Its top-level keys:

- `version`: the definition format's version, starting at 1;
- `rule_set_id`: the rule-set identity;
- `symbols`: one entry per symbol: label, kind (`is`, `has`, `scalar`, `fluent`, `event_type`, `event_type_category`), base or derived, static or fluent, and arity;
- `literals`: the literal table: each literal's index, and what the literal reads (a feature of one role, a threshold on a scalar, or a comparison between two roles' scalars, each with its numbers);
- `rules`: one entry per derived feature, with its output, its inputs, its truth table, and its expression (the Boolean form);
- `layers`: the matrix form (see the next section);
- `event_types`: one entry per event type and event-type category: roles, requirement (the constraints and the layer that combines them), precondition, and effects.

Floating-point numbers are written in full (17 significant digits), so both runtimes read the same values. Comparisons are written with their operator spelled out (`>` or `>=`), never implied.

### Rules: Boolean form and matrices (REL.18)

The Boolean form is the readable definition: each rule's inputs and truth table, with its expression. The matrix form is what the runtime computes.

Each rule is turned into two threshold layers:

- **Literal vector.** For an entity or a binding, `ℓ` holds the value of every literal the layer reads, followed by every complement (1 minus the value).
- **Term layer.** Each row `j` of `W1` marks the literals of one term of the rule's disjunctive normal form, with threshold `θ_j` equal to the number of literals in the term. Term `j` is true when `W1_j · ℓ ≥ θ_j`.
- **Output layer.** Each row of `W2` marks the terms of one output, with threshold 1. The output is true when at least one of its terms is true.

A rule whose output is always true has one term with no literals and threshold 0. A rule whose output is always false has no terms. The terms are the rule's minimal DNF, which the taxonomy already computes for the complexity score. Any DNF equal to the truth table is valid: the agreement test is what guarantees correctness.

Rules form layers in dependency order, as the taxonomy's rule layers do now. A threshold literal (`SC.2 > 0.4127`) and a comparison (`a.SC.1 - p.SC.1 > 0.15`) are computed before the first layer, as literals. A two-place requirement is the conjunction of its constraints: each constraint is a rule with its own two layers, and one more layer ANDs the constraints' outputs.

In `definition.json`, each layer is written sparsely: for each term, the indices of its literals in the literal table and whether each is complemented; for each output, the indices of its terms.

**The agreement test.** For every rule, the matrix form and the truth table give the same output for every entity of the run. For every rule with at most 12 inputs, they also agree on every setting of the inputs. The test runs at the end of every world run, and the run fails when the test fails.

### Rule-set identity and derived values (REL.16)

The rule-set identity is the SHA-256 hash of the canonical JSON of `symbols`, `literals`, `rules`, and `event_types`, written in hexadecimal. The rule-set identity changes exactly when a rule, a literal's numbers, or an event type changes.

Derived values are written in `derived/`, and `derived/manifest.yaml` lists each file with the rule-set identity that produced it. A loader that finds a different identity refuses the file, with an error that names the file and both identities. Derived values are never read back as inputs to a world. Nothing in `derived/` is edited by hand.

| File in `derived/` | Contents |
| --- | --- |
| `static_features.csv` | One row per entity: every derived static feature (determined IS and HAS). |
| `capacities.csv` | One row per entity: every capacity, `CAN.<event type>` and `CANBE.<event type>`, intensional, with a flag column for approximate capacities, then the extensional capacities (as `projections.csv` holds now). |
| `capacity_roles.csv` | For each category and each one-place event type: whether the capacity is fixed at 1, fixed at 0, or free, and whether the exact or the local test decided. `NEC` statements use it. |
| `relation_proportions.csv`, `relation_pairs.csv`, `event_type_stats.csv`, `thematic.csv` | As in `TAXONOMY_RELATIONS.md`, with "verb" read as "event type". Requirements only: these statistics describe what entities are able to do, never what is legal in a state. |

A capacity is a requirement only. A sleeping owl can still eat a mouse, because eating is in the owl's capacities. Eating is not legal while the owl is asleep.

### Views (REL.16)

A view chooses columns from the base facts and the derived values, and writes one table for a model. Views replace the taxonomy's single `instances.csv` and its exposure settings.

```
python -m semantic_world.world view RUN_FOLDER --preset classic [--include IS,HAS,CAN.EV] [--out FILE]
```

Presets:

- `base`: the base vector only (free IS and HAS features, scalars);
- `static`: the base vector and the derived static features;
- `classic`: the columns of the old `instances.csv`: ISA, IS, HAS, the one-place capacities in the place of CAN, and the scalars.

`--include` takes label prefixes, and adds or narrows columns. A view records, in a sidecar YAML file, its preset, its columns, and the rule-set identity.

## Phase (a): the world

### Taxonomy outputs

The taxonomy generator keeps everything it does now for categories, instances, free features, determined IS and HAS features, roles, scalars, and statistics. Three things change:

- **CAN features leave the taxonomy.** Their rules become the requirements of one-place event types, generated by the world package. Every configuration key that configured CAN features moves to `event_types.unary` in the world configuration, with the same meaning.
- **Verbs leave the taxonomy.** The verb tree, verb features, constraints, projections, and relation statistics move to the world package, renamed as above. The `verbs` block of the taxonomy configuration moves to `event_types.binary` in the world configuration, with the same keys and meanings, except the exposure keys, which are removed.
- **Base and derived values are written apart.** `instances.csv` is replaced by `base.csv` (labels, leaf, free IS and HAS features, scalars) and `derived/static_features.csv` with its manifest. The category-vector files keep both kinds of feature, because they describe categories, not world inputs. The taxonomy also writes its rules in matrix form (`rule_matrices.json`, in the `layers` format of `definition.json`), and runs the agreement test.

The taxonomy still runs alone (`python -m semantic_world.taxonomy`), for studies that need only categories and features.

### Fluents

A world has `fluents.count` Boolean fluents, `ST.1` to `ST.n`. Free fluents are base fluents: events change them. A proportion `fluents.derived_proportion` of fluents is derived instead.

- **Initial values.** Each base fluent gets an initial rate, drawn uniformly from `fluents.initial_rates` (default `[0.0, 0.2, 0.8, 1.0]`). Each entity's initial value is drawn at that rate, once, when the world is generated. A rate of 0 or 1 gives a fluent that starts the same for every entity. The initial values are part of `entities.csv`.
- **No inheritance.** Fluents have no roles (defining, characteristic, undiagnostic) and do not follow the tree. Whether initial values should depend on category is a later option (see "Decisions to confirm").
- **Derived fluents.** A derived fluent's rule reads at least one fluent (base or derived), and may also read static features and threshold literals. A rule that reads no fluent would be static, so the generator never draws one. Derived fluents are layered like determined IS and HAS features, with their own `fluents.max_chain_depth` (default 1). Rule complexity comes from the same settings as other rules.
- **Static rules stay static.** Rules for determined IS and HAS features read static facts only. No rule for a static fact ever reads a fluent.

### Event types

**One-place event types** (`EV.1` to `EV.n`) replace CAN features. The rule that computed `CAN.k` becomes the requirement of `EV.k`, sampled in the same way. One-place event types are flat: they have no tree.

**Two-place event types** replace verbs. The event-type tree, event-type features, constraint families, base relations, and the own constraint are those of Part B of `TAXONOMY_RELATIONS.md`, renamed. A two-place event type's requirement is the conjunction of its constraints, and an event-type category's requirement is its base relation. A requirement may contain cross-role constraints and comparisons, so a requirement is a rule over the whole binding, not only over each role alone. REL.17's "a requirement rule for each role" is the special case of a requirement with agent conditions and patient conditions only.

Requirements read static facts only: base and derived static features, threshold literals, and comparisons. A requirement never reads a fluent.

### Preconditions

A precondition is a conjunction of fluent literals: `a.ST.2 AND NOT p.ST.5`. A literal reads a base or a derived fluent of one role, positive or negated. An empty precondition is always true.

- **One-place event types** get a number of precondition literals drawn from `event_types.preconditions.literals` (default weights `{0: 0.3, 1: 0.5, 2: 0.2}`), all on the agent.
- **Two-place event types** get precondition literals in two ways, mirroring constraints. Each event-type feature carries a precondition literal with probability `event_types.preconditions.feature_rate` (default 0.3), and an event type gets the literals of every event-type feature that is true for it. Each event type also gets its own literals, drawn from `event_types.preconditions.literals`. So event types in one branch of the tree share preconditions, as they share base relations. Each literal's role is drawn from `event_types.preconditions.roles` (default `{agent: 0.5, patient: 0.5}`).
- **Enabling.** A literal is drawn from the values that some event type's effect produces with probability `event_types.preconditions.enabled_share` (default 0.7), and from all values otherwise. Enabling literals make chains of events: one event makes another legal.
- **Achievable.** Every literal must be achievable: its value is the initial value of some entity, or some event type's effect produces it. The generator redraws a literal that is not.
- **Consistent.** A precondition never holds a literal and its negation for the same role. The generator redraws a literal that would.

### Effects

An effect sets one base fluent of one role, to true or to false: `p.ST.3 := 1`. Effects never write a derived fluent or a static fact. There are no conditional effects.

- **One-place event types** get a number of effects drawn from `event_types.effects.count` (default weights `{1: 0.7, 2: 0.3}`), all on the agent.
- **Two-place event types** get effects in the same two ways as preconditions: an event-type feature carries an effect with probability `event_types.effects.feature_rate` (default 0.5), and each event type gets its own effects, drawn from `event_types.effects.count`. Each effect's role is drawn from `event_types.effects.roles` (default `{agent: 0.4, patient: 0.6}`).
- **No contradictions.** An event type never has two effects that set the same fluent of the same role to different values. When inherited effects contradict, the effect of the lower-numbered event-type feature is kept, and the others are dropped and counted in `world_stats.yaml`. When an event type's own effect would contradict an inherited one, the own effect is redrawn.
- **No empty effects.** An effect must be able to change something. An effect that sets a fluent to the value that the event's own precondition already requires is redrawn.

### Explicit event types

An event file (`event_types.event_file`) can define event types by hand, as a rule file defines rules. An event file can give an event type's requirement, precondition, and effects explicitly, and can name the fluents it uses. Event types without an explicit entry are sampled. The file lets a study build a known chain, for example a search, find, chase, catch, eat sequence over abstract labels.

```yaml
fluents:
  ST.1: {initial_rate: 1.0}        # say, awake
  ST.2: {initial_rate: 0.0}        # say, caught
event_types:
  E1.1:
    precondition: "a.ST.1 AND NOT p.ST.2"
    effects: ["p.ST.2 := 1"]
  E1.2:
    precondition: "a.ST.1 AND p.ST.2"
    effects: ["p.ST.1 := 0"]
```

An explicit entry that breaks a rule above (an effect on a derived fluent, a requirement that reads a fluent, a contradiction) is a validation error that names the file and the entry. `data/world/events/chain_example.yaml` is a worked example.

### Time, steps, and inertia

An episode starts at time point `T.1`, with each participant's initial base fluents. Step k takes the state at `T.k` to the state at `T.k+1`:

1. **Legality.** Every event of the step must be legal in the state at `T.k`.
2. **Non-interference.** The events of one step must not interfere. Two events interfere when both write the same fluent of the same entity, or when one writes a base fluent that the other's precondition reads. A precondition reads a derived fluent's whole cone: every base fluent the derived fluent depends on, directly or through other derived fluents. The same event (same event type and binding) occurs at most once in a step.
3. **Effects.** Each event's effects are applied to the state at `T.k`.
4. **Inertia.** Every base fluent that no effect wrote keeps its value at `T.k+1`.
5. **Derivation.** Derived fluents are recomputed from the state at `T.k+1`. They are never carried over.

Because the events of a step do not interfere, applying them together gives the same state as applying them one at a time in any order. The Rust runtime may therefore apply them in sequence.

An effect that sets a fluent to the value it already has is still applied. The fluent does not change, and the history records no change.

### Episodes and selection

Episodes replace the corpus's scene generator, which moves from `semantic_world.corpus.scenes` to `semantic_world.world.episodes`. Participants are drawn exactly as scenes draw them now: a seed instance, then others weighted by thematic relatedness, taxonomic similarity, and a constant. The scene settings stay in the corpus configuration, under `scene`.

- **Initial state.** Each episode starts from the participants' initial values in `entities.csv`. Episodes are independent: nothing that happens in one episode carries into another. With `scene.initial: redraw`, each episode draws its participants' initial values afresh, at each fluent's initial rate, from the episode's own part of the stream.
- **Each step.** The number of events in the step is drawn from a Poisson distribution with mean `scene.events_per_step`, as now. Events are then drawn one at a time by the selection policy. Each draw chooses among the events that are legal in the state at `T.k` and do not interfere with the events already chosen for the step. When no such event remains, the step has fewer events. After the draws, the runtime applies the step.
- **The default selection policy**, `uniform_event_type`, keeps today's order of draws: the kind of event first (`scene.transitive_share`), then the event type among those with at least one available binding, weighted by `scene.event_type_weights` (the heir of `scene.verb_weights`), then a binding uniformly among the available ones. An event type that is legal for many bindings is therefore no more frequent than one legal for few.
- **Other policies** are registered by name in `semantic_world.world.policies`, and chosen with `scene.policy`. A policy receives the definition, the state, the participants, and its random generator, and returns the events of one step. A policy may never change the state itself.
- **Quiescence.** When no event is legal at a time point, the episode ends there, and its history records `quiescent: true`. Otherwise the episode ends after its drawn number of steps.

### Histories

A history is one JSON object per episode. The corpus writes them to `scenes.jsonl`, and `python -m semantic_world.world simulate` writes them to `episodes.jsonl`. The schema is the same, and the 3D engine will write the same schema when the link is built.

```json
{"label": "SN.8", "seed": "I1.3.2.5", "participants": ["I1.3.2.5", "I1.5.1.2", "I2.1.1.3"],
 "policy": "uniform_event_type", "rule_set_id": "4f1c...",
 "initial": {"I1.3.2.5": ["ST.1", "ST.4"], "I1.5.1.2": ["ST.1"], "I2.1.1.3": []},
 "steps": [
   {"step": 1, "events": [
     {"label": "SN.8.1", "type": "E1.2", "agent": "I1.3.2.5", "patient": "I1.5.1.2",
      "changes": [{"entity": "I1.5.1.2", "fluent": "ST.2", "to": true}]}]},
   {"step": 2, "events": []}],
 "final": "T.3", "quiescent": false}
```

- `initial` lists, for each participant, the base fluents that are true at `T.1`.
- `changes` lists only base fluents whose value changed. A derived fluent's change is never recorded: it is recomputed.
- Events are numbered within their episode in time order, and within a step in the order they were drawn, as now. Events have no aspect (CG.64).
- `final` is the last time point.

With `--legal` (off by default, because of size), `simulate` also records, for each step, the number of legal bindings of each event type. Legality and occurrence can then be studied apart.

### World statistics

`world_stats.yaml` reports:

- counts of fluents (base and derived), event types, precondition literals, and effects, by kind and by role;
- the effects dropped as contradictions;
- the enabling graph: an edge from event type A to event type B when an effect of A produces a value that a precondition literal of B requires; the graph's number of edges, and its longest chain;
- absorbing fluents: a base fluent that some effect sets to one value and no effect sets back;
- from 1,000 episodes of the default policy (`world:stats` stream): for each event type, the share of steps at which it had a legal binding among the participants, and the share at which it occurred; the mean number of changes per event; the share of quiescent episodes; and a warning for each event type that was never legal.

### Configuration

`data/world/default.yaml`:

```yaml
name: default
seed: 1
taxonomy: {config: data/taxonomy/default.yaml, seed: null}   # null: the world's seed

fluents:
  count: 8
  derived_proportion: 0.25
  max_chain_depth: 1
  initial_rates: [0.0, 0.2, 0.8, 1.0]        # each base fluent's initial rate is drawn from this list

event_types:
  unary: {}                                   # the old CAN settings, moved; the stage proposal lists each key
  binary: {}                                  # the old `verbs` block, moved, without the exposure keys
  preconditions:
    literals: {0: 0.3, 1: 0.5, 2: 0.2}        # weights over the number of an event type's own precondition literals
    feature_rate: 0.3                         # the probability that an event-type feature carries a precondition literal
    roles: {agent: 0.5, patient: 0.5}
    enabled_share: 0.7                        # the share of literals drawn from values that some effect produces
  effects:
    count: {1: 0.7, 2: 0.3}                   # weights over the number of an event type's own effects
    feature_rate: 0.5
    roles: {agent: 0.4, patient: 0.6}
  event_file: null                            # a file of explicit event types
```

`data/world/tiny.yaml` builds on the tiny relations taxonomy, with 4 fluents, and runs in seconds. With `fluents.count: 0`, a world has no fluents, no preconditions, and no effects: every event is legal whenever it is able, and nothing changes. Validation follows the base conventions: unknown keys are errors, and every error names the file and the field.

## Phase (a): the corpus

### Inputs

`world: {config: <file>, seed: N}` or `world: {run: <folder>}` replaces the `taxonomy` key. A run folder is regenerated in memory from its `config.yaml` and checked against its files, as taxonomy run folders are now. A configuration that still has `taxonomy` gets an error that names the new key.

### Lexicon

| Concept | Part of speech | Concept label | Configuration key |
| --- | --- | --- | --- |
| Every category | noun | `C1.3.2` | `category` |
| IS feature (free or derived) | adjective | `IS.12` | `is` |
| HAS feature | part noun | `HAS.4` | `has` |
| Fluent (base or derived) | state adjective | `ST.3` | `state` (new) |
| One-place event type | intransitive verb | `EV.4` | `event_unary` (was `can`) |
| Two-place event type | transitive verb | `E1.2.3` | `event` (was `verb`) |
| Category of two-place event types | transitive verb (more general) | `E1.2` | `event_category` (was `verb_category`) |
| Patient capacity | adjective | `CANBE.E1.1` | `patient_projection` |
| Scalar dimension | two adjectives | `SC.1.HIGH`, `SC.1.LOW` (until phase (b)) | `scalar` |
| The generic head noun | noun | `THING` | — |

- Which patient capacities get words is a language setting (REL.16). The default keeps today's proportion: `named_proportion.patient_projection: 0.25`.
- An event type whose requirement holds for every binding, or for none, never gets a word, as for verbs now.
- The function word `become` is added (see "States and changes"). Its tense is marked like any verb's.

### Quantifiers (CG.59 to CG.62)

**Operators.** A class-level proposition's quantifier is one of `NEC_ALL`, `ALL`, `MOST`, `SOME`, `NO`, and `NEC_NO`. `GEN` is removed (CG.60). Rendered: `NEC(ALL(restrictor, scope))`, `ALL(...)`, `MOST(...)`, `SOME(...)`, `NO(...)`, `NEC(NO(...))`.

**Truth.**

- `ALL`: every member of the subject set has the predicate. `NO`: none does. Both are extensional, and both are available for every predicate, including patient capacities and subjects with relative clauses. The bans of decisions 24 and 34 and CG.E61 are lifted.
- `NEC_ALL`: the rules guarantee the predicate for the subject: the fixed test, as the law-like reading tests `all` now. "Fixed by the rules" means fixed by the feature rules and by the requirements of event types (`capacity_roles.csv`). `NEC_NO` is the same with the predicate fixed at 0. `NEC_ALL` implies `ALL`, because a subject set is never empty.
- `NEC` applies to one-place predicates only: IS, HAS, one-place capacities, and membership. Relation facts (a two-place event type with a patient category) and patient capacities take the extensional quantifiers only, because no fixed test exists for them.
- `MOST`: more than half of the subject set (CG.62).
- `SOME`: at least one member.
- Membership ("penguins are birds") is `NEC_ALL` by construction. A rule statement is always `NEC_ALL` (CG.61).

**Strongest true quantifier.** A document still states a fact with its strongest true quantifier, in the order `NEC_ALL`, `ALL`, `MOST`, `SOME` (and `NEC_NO`, `NO`, `MOST ... NOT`, `SOME ... NOT` for negative facts), among the quantifiers the language can say and its usage rules allow. `quantifiers.weights` gains the keys `nec_all` and `nec_none`.

**The language's words.** These settings decide what the words mean, never what a proposition means:

- `quantifiers.universal_words: nec | extensional | either` (the heir of `quantifiers.all_grounding`). With `nec`, the default, "all" and "no" express `NEC_ALL` and `NEC_NO` only. With `extensional`, they express `ALL` and `NO` only. With `either`, they express both, and the sentence's logical form records the strongest true one. A fact whose strongest true quantifier has no word in the language falls to the next quantifier down.
- `quantifiers.most.usage_min: 0.7` (the heir of `quantifiers.most.min_proportion`): a speaker says "most" only when at least this share has the predicate. A `MOST` proposition stays true above one half.
- `quantifiers.some.exclude_all` is unchanged.
- `quantifiers.bare_plural.expresses: [most]` (the heir of `quantifiers.generic.means`): the quantifiers that a bare plural can express. A fact is stated with a bare plural, at `quantifiers.generic_rate`, when its stated quantifier is in the list. Membership and rule statements may always use the bare plural, for `NEC_ALL`.

The old keys (`all_grounding`, `most.min_proportion`, `generic.means`) get an error that names the new key.

**Readings.** A bare-plural sentence records its quantifier readings in `readings`: every quantifier in `bare_plural.expresses`, plus `nec_all`, because membership and rule statements use the bare plural for `NEC_ALL`. Readings are still worked out from the tree and the tokens alone.

**Law-like test sets.** A false class-level item whose quantifier is `NEC_ALL` or `NEC_NO`, and whose extensional twin (`ALL` or `NO`) is true, is law-like, and goes into the `_lawlike` twin of its set (CG.59). The ordinary sets hold the other items.

### Referring content (CG.63)

A sentence about instances has two parts: the **assertion**, which is the main clause, and the **descriptions** that identify its referents.

- In a definite mention, the noun, the modifiers, and a restrictive relative clause are descriptions.
- In an indefinite mention (a first mention, "a penguin"), the noun and the modifiers are asserted: the sentence introduces the referent and says what the referent is.
- A pronoun has no description.
- Class-level sentences have no descriptions: a restriction or a relative clause on a class-level subject is part of the quantifier's restrictor, and so part of what is asserted.

In the JSON logical form, each mention gains `"descriptive": true` or `false`. In the propositional rendering, descriptions come first, inside braces, followed by the assertion:

| Sentence | Propositional rendering |
| --- | --- |
| the furry dog has legs | `{C1.3.2(R.1) AND IS.12(R.1)} HAS.4(R.1)` |
| a penguin swam | `C1.3(R.1) AND EVENT(SN.8.5, PAST, SIMPLE, EV.3(R.1))` |
| the dog that chased the cat ran | `{C1.3.2(R.1) AND C1.4.1(R.2) AND EVENT(SN.3.2, PAST, SIMPLE, E1.2(R.1, R.2))} EVENT(SN.3.4, PAST, SIMPLE, EV.7(R.1))` |
| it has legs | `HAS.4(R.1)` |

- `renderings.propositional.descriptions: marked | omitted` (default `marked`). With `omitted`, the braces and their contents are left out, and the rendering holds the assertion only.
- `test_sets.seen.descriptions: true | false` (default `true`) decides whether a description counts as stated, for `seen`. The default keeps today's counting.
- The propositional rendering still parses back to the logical form.

### Aspect (CG.64)

Events no longer have an aspect. Each report of an event chooses its aspect, at `documents.progressive_rate` (the heir of `propositions.events.progressive_rate`), from the document's part of `corpus:mentions`. Both aspects are true of any event that occurred. With `documents.one_aspect_per_event: true` (the default), every report of one event in one document uses the aspect chosen at its first report, so documents keep today's agreement. A test item draws its aspect from its own part of `corpus:grammar`. `scenes.jsonl` no longer records aspect. CG.E56 and decision 39 of `CORPUS_GENERATOR.md` are superseded.

### Events

- An event-level proposition reports an event of a history. Its atom is `EV.3(R.1)` or `E1.2(R.1, R.2)`, named at a level of the event-type tree as verbs are now.
- Truth is unchanged: the event occurred.
- The grounding's `possible` field is replaced by two fields: `able` (the requirement holds for the binding) and `legal` (the binding was legal at some time point of the scene).
- Event-level propositions are never negated.

### States and changes

Two new kinds of proposition, both at the instance level and both inside a scene:

- **State.** `HOLDS(SN.8, T.2, PAST, ST.3(R.1))`: the fluent `ST.3` held of `R.1` at time point `T.2` of scene `SN.8`, as in "the mouse was asleep". A negative state is `HOLDS(SN.8, T.2, PAST, NOT ST.3(R.1))`: "the mouse was not asleep". Realized with the copula and the state adjective.
- **Change.** `BECOME(SN.8, T.2, ST.3(R.1))`: `ST.3` was false of `R.1` at `T.2` and true at `T.3`, as in "the mouse became asleep". `BECOME(SN.8, T.2, NOT ST.3(R.1))` is the opposite change. Realized with `become` and the state adjective. A change says nothing about its cause, so the proposition stays precise when a derived fluent changes. The grounding records the event whose effect wrote the base fluent, when there is one (`caused_by`).

States and changes take the scene's tense (`propositions.events.tense`) and no aspect. Their JSON form is the instance-level form, with `"level": "state"` or `"level": "change"`, the scene, the time point, and the tense. Their predicate is `{"kind": "state", "fluent": "ST.3"}`.

In narratives:

- **Initial states.** When a participant is first mentioned, a state sentence about the participant follows at `documents.initial_state_rate` (default 0.2). The sentence states one of the participant's fluents at the time point before the participant's first reported event. The polarity is drawn at `propositions.negation_rate.state` (default 0.1).
- **Results.** After an event sentence, a change sentence follows at `documents.result_rate` (default 0.5). The sentence states one change that the event's step brought to a participant of the event: a base fluent the event's own effects changed, or a derived fluent of a participant that changed in the same step. An event that changed nothing has no result sentence.
- **Modifiers stay static.** Mentions use static features only, for modifiers and for distinguishing referents. A fluent changes over a scene, so a fluent modifier would need a time point.

**Readings.** A verb phrase whose predicate word is a state adjective, or `become`, has the reading `state`. The lexeme tells a state adjective from a static one.

### Test sets

- **Event sets.** A false event-level item is one of three kinds, each in a test set of its own: `possible` (the binding was legal at some time point of the scene, and the event did not happen), `blocked` (the binding is able, and was never legal in the scene), and `impossible` (the binding is not able). The sets are `event_<change>_possible`, `event_<change>_blocked`, and `event_<change>_impossible`. The false-event rule is unchanged: no event with the item's event type (or any event type below it) and participants happened in the scene.
- **State sets** (new). A state item is a continuation of a situational narrative, and states a fluent of one of its participants at the scene's final time point: `HOLDS(SN.8, T.final, PAST, ST.3(R.1))`, with the scene's actual final label. A false item is made by a predicate swap (another fluent with a word, false of the referent at that time) or a subject swap (another participant, for which the fluent is false). Each item is marked `changed` when the fluent's value at the final time point differs from its value at `T.1`. The sets are `state_<change>_changed` and `state_<change>_unchanged`: an item that changed can only be answered by tracking what the events did.
- All the rules of "False propositions and test sets" in `CORPUS_GENERATOR.md` still hold: matched pairs, one format, `seen`, context, and the stream rules.

### Outputs

- `scenes.jsonl` holds histories, in the schema of "Histories".
- `documents.jsonl` gains the new levels, `descriptive` on mentions, and `able` and `legal` in event groundings.
- `stats.yaml` gains counts of state and change sentences, the share of events with a result sentence, and the share of state items marked `changed` in each state set.
- `config.yaml` records the world run's identity (its configuration hash, seed, and rule-set identity) in place of the taxonomy run's.

### Configuration changes

| Old key | New key |
| --- | --- |
| `taxonomy` | `world` |
| `lexicon.named_proportion.can`, `.verb`, `.verb_category` | `.event_unary`, `.event`, `.event_category`; and `.state` (new) |
| `quantifiers.all_grounding` | `quantifiers.universal_words` |
| `quantifiers.most.min_proportion` | `quantifiers.most.usage_min` |
| `quantifiers.generic.means` | `quantifiers.bare_plural.expresses` |
| `propositions.events.progressive_rate` | `documents.progressive_rate`, with `documents.one_aspect_per_event` |
| `scene.verb_weights` | `scene.event_type_weights`; and `scene.policy`, `scene.initial` (new) |
| `mention.verb_level_weights` | `mention.event_level_weights` |
| — | `documents.initial_state_rate`, `documents.result_rate`, `propositions.negation_rate.state`, `renderings.propositional.descriptions`, `test_sets.seen.descriptions` |

Every old key gets an error that names its new key.

## Phase (a): determinism

- Event types draw from the streams that draw CAN rules and verbs now, so a world with fluents off reproduces today's CAN rules and verbs for the same seed. New world streams: `world:fluents` (fluent rules and initial rates), `world:initial` (initial values), `world:preconditions`, `world:effects`, and `world:stats`. Stream seeds use the function in `semantic_world.taxonomy.streams`.
- Episodes draw from `corpus:scenes`, each scene from its own part, named by its label, as now. A redrawn initial state (`scene.initial: redraw`) draws from the scene's own part too.

Properties, each tested:

- the same world configuration and seed give a byte-identical world run;
- the `taxonomy/` folder of a world run is byte-identical to the output of the taxonomy run alone with the same configuration and seed;
- changing fluent, precondition, or effect settings never changes the taxonomy, the requirements, the capacities, or the relation statistics;
- the runtime is pure: `apply` and `legal` depend only on their arguments;
- with `fluents.count: 0`, every able binding is legal at every time point, every history records no change, and episodes have the same participants and events as a selection over a static pool;
- the corpus properties of `CORPUS_GENERATOR.md` still hold: grammar settings never change a logical form, rendering word forms changes only word-form fields, and test-set settings never change the documents.

## Phase (a): Python package

`python/semantic_world/world/`. Suggested modules:

| Module | Contents |
| --- | --- |
| `config.py`, `streams.py`, `labels.py`, `errors.py`, `io.py` | Configuration, streams, labels, errors, reading and writing |
| `matrices.py` | REL.18: rules to layers, evaluation, the agreement test. The taxonomy imports it for its own rules |
| `fluents.py` | Fluents, initial rates and values, derived fluent rules |
| `event_types.py` | One-place event types, and the event-type tree (moved from `taxonomy/verbs.py`) |
| `constraints.py` | Constraint families (moved from the taxonomy) |
| `preconditions.py`, `effects.py` | Generation, as above |
| `event_file.py` | Explicit event types |
| `definition.py` | The definition: building, `definition.json`, `entities.csv`, rule-set identity, loading with identity checks |
| `runtime.py` | `derive`, `able`, `legal`, `legal_bindings`, `apply` |
| `episodes.py`, `policies.py`, `history.py` | Episodes (moved from `corpus/scenes.py`), selection policies, the history schema |
| `capacities.py`, `relation_stats.py` | Capacities and relation statistics (moved from the taxonomy) |
| `stats.py`, `views.py` | `world_stats.yaml`; views |
| `fixtures.py` | Conformance fixtures: the brute-force evaluator, generating fixtures, checking them |
| `generate.py`, `__main__.py` | A whole world run, and the command line |

The command line:

```
python -m semantic_world.world define data/world/default.yaml [--seed N] [--out DIR]
python -m semantic_world.world simulate RUN_FOLDER --episodes N [--seed N] [--legal] [--out FILE]
python -m semantic_world.world view RUN_FOLDER --preset classic [--include ...] [--out FILE]
python -m semantic_world.world check-fixtures [FOLDER]
```

Add `data/world/default.yaml`, `data/world/tiny.yaml`, and `data/world/events/chain_example.yaml`. Tests live in `tests/world/`, and fixtures in `tests/fixtures/world/`. Shared modules (Boolean functions, expressions) may move to a common package if Claude Code finds that cleaner. The stage proposal records the move.

## Phase (a): build stages

Work on one branch per stage (`world-a1`, and so on). Each stage ends with its tests passing, the full check list in `CLAUDE.md` passing, and a commit. Stages a1 to a4 build the world package beside the existing programs, without changing what the taxonomy and the corpus produce. Stage a5 cuts over.

1. **a1. Matrices and derived values.** `matrices.py`, the taxonomy's `rule_matrices.json`, the agreement test, rule-set identity, and the derived-value manifest. *Accept:* matrices and truth tables agree for every rule of the default and tiny taxonomies, over every instance and, for rules with at most 12 inputs, over every input setting; constant rules give their constant; the rule-set identity is stable across runs and changes when any one rule changes.
2. **a2. The world generator.** Configuration, fluents, one-place and two-place event types (moved code, imported rather than removed for now), preconditions, effects, the event file, `definition.json`, `entities.csv`, capacities, views, world statistics, and `define`. *Accept:* every rule in "Preconditions" and "Effects" holds on the default world (no fluent read by a requirement, no effect on a derived fluent, no contradiction, every literal achievable); with `fluents.count: 0` and the old settings moved over, requirements, capacities, and relation statistics equal what the taxonomy computes today for the same seed (if the move makes equal draws impractical, the stage proposal says why, and the test instead re-evaluates every requirement by brute force from the truth tables); the chain example loads, and a broken event file fails with the right error; `view --preset classic` gives the columns of the old `instances.csv`.
3. **a3. The runtime and the fixtures.** `runtime.py`, the brute-force evaluator, eight or more hand-written fixtures, generated fixtures, and `check-fixtures`. *Accept:* every fixture passes; on the tiny world, `legal` and `derive` agree with the brute-force evaluator in 1,000 random states; interfering events raise the documented error; applying a step's events together equals applying them one at a time in every order, on 1,000 random non-interfering sets.
4. **a4. Episodes and histories.** `episodes.py`, the default policy, `history.py`, and `simulate`. *Accept:* every event in every history was legal in its step's starting state; no two events of a step interfere; replaying a history's events through `apply` from its initial state reproduces every recorded change and nothing else; with `fluents.count: 0`, every able binding is available at every step; on the chain example, the chain's events occur in the order the preconditions force.
5. **a5. The cut-over.** The taxonomy loses CAN features, verbs, and exposure, and writes `base.csv`. The corpus reads world runs, uses `world.episodes`, and takes the new labels, the quantifiers (CG.59 to CG.62), aspect (CG.64), and `able` and `legal` in event groundings. *Accept:* every corpus test passes with its expectations updated, and each changed expectation is listed in the stage proposal; every generated proposition passes an independent recomputation of its truth from the world run's files; the law-like sets hold only `NEC` items whose extensional twin is true; the old configuration keys fail with errors that name the new keys.
6. **a6. States, changes, and descriptions.** State adjectives, `become`, `HOLDS` and `BECOME`, initial-state and result sentences, readings, the state test sets, the three kinds of false event, CG.63's descriptions in the logical form and the rendering, and `seen`. *Accept:* every state and change sentence is true of its scene's history; every result sentence's change happened in its event's step; every `changed` mark agrees with the history; the propositional rendering parses back in both `marked` and `omitted` forms; `interpret(tree)` recovers every logical form.
7. **a7. Documentation and datasets.** A new guide, `docs/guides/WORLD.md`; updates to `TAXONOMY.md` and `CORPUS.md`; the notes at the top of the superseded sections of older specifications; `CLAUDE.md`'s reading list and commands; and the default and tiny runs regenerated end to end (world, corpus, word forms, render). *Accept:* the full chain runs on the tiny configuration from the guide's commands alone; the default world and corpus run in under 15 minutes on a laptop, with the time recorded in the guide.

## Phase (b): comparisons derived from scalars

Scalars stay static in phase (b). Comparisons are derived static facts: computed when needed, never stored, and never written by an event.

### The world

- **`GREATER(SC.i, x, y)`**: the value of `x` on `SC.i` exceeds the value of `y`. The comparison is strict, so two equal values give false in both directions. `x` and `y` are entities, or category means.
- **`ABOVE(SC.i, x, K, z)`**: the value of `x` is at least `z` standard deviations above the mean of comparison class `K`. **`BELOW(SC.i, x, K, z)`**: at least `z` standard deviations below. `K` is a category, or `ENTITY`, the world symbol for all entities, which replaces `THING` as the comparison class of a top-level category.
- **`MEAN(C)`**: the mean value of category `C`'s instances on the scalar, used as an argument (CG.65).
- **Comparison classes.** `derived/comparison_classes.csv` holds, for each category, `ENTITY`, and each scalar: the number of instances, the mean, and the standard deviation, computed as the corpus computes them now.
- **The runtime.** `derive` gains `greater`, `above`, and `below`, evaluated on demand for given entities or category means. Requirement literals with margins (`a.SC.1 - p.SC.1 > 0.15`) are unchanged: they are literals of the world's rules, not comparisons that the language states.
- **Fixtures.** Phase (b) adds hand-written fixtures for ties, for `ENTITY`, and for a category mean.

### The corpus

**Relative gradable adjectives (CG.65).** The proposition is the comparison, with its comparison class and threshold:

| Sentence | Propositional rendering |
| --- | --- |
| the big mouse is red | `{C1.5(R.1) AND ABOVE(SC.1, R.1, C1.5, 1.0)} IS.4(R.1)` |
| elephants are big | `ABOVE(SC.1, MEAN(C1.4), C1, 1.0)` |
| big penguins can swim | `MOST(C1.3(X.1) AND ABOVE(SC.1, X.1, C1.3, 1.0), ABLE(EV.3(X.1)))` |

- A class-level comparison is a statement about the category, not a quantification over its members. Its level is `class`, with no quantifier, and its predicate kind is `comparison`. The language realizes it with a bare plural, which here carries no quantifier reading.
- "Big" and "small" are the language's words for `ABOVE` and `BELOW` at the language's own z. `scalar_adjectives.z` stays as that language setting, and every proposition records the z it used.
- The comparison class follows CG.3: the noun's category at the instance level, the subject's parent at the class level, and `ENTITY` for a top-level category. The setting `scalar_adjectives.class_levels_up` (default 1) lets the planner choose an ancestor further up, at the class level.
- Concept labels `SC.1.HIGH` and `SC.1.LOW` remain the labels of the two adjectives' concepts in the lexicon.

**Comparatives (CG.66).**

| Sentence | Propositional rendering |
| --- | --- |
| the owl is bigger than the mouse | `{C1.2(R.1) AND C1.5(R.2)} GREATER(SC.1, R.1, R.2)` |
| the mouse is smaller than the owl | `{C1.5(R.2) AND C1.2(R.1)} GREATER(SC.1, R.1, R.2)` |
| owls are bigger than mice | `GREATER(SC.1, MEAN(C1.2), MEAN(C1.5))` |

- `GREATER` is the only comparative operator. "Smaller than" is the language's word for `GREATER` with its arguments in the other order. The language chooses the word, and the proposition stays one form.
- The grammar gains a comparative form of each pole adjective, with `grammar.morphology.comparative: {realization: affix | word, position: after | before}`, and the function word `than`.
- **Instance level.** In narratives, after an event sentence, a comparative description of the event's two participants follows at `documents.comparative_rate` (default 0.1), on a scalar where the two values differ.
- **Class level.** Category documents gain a content kind, a comparison fact: the topic compared with a sibling or a thematic partner, on one scalar. The kind has the same chance as the other kinds.
- **Test sets.** `instance_comparison_role` swaps the arguments, and `instance_comparison_predicate` swaps the scalar for one on which the comparison is false. The class-level comparison facts join the class-level sets, with the same two changes.

**Rule statements with thresholds.** Terms that read a threshold literal are still skipped in phase (b), unless Jon chooses otherwise (decision 23 below).

### Build stages

1. **b1. The world.** Comparisons in the runtime, comparison classes, `ENTITY`, and the fixtures. *Accept:* every comparison agrees with direct arithmetic on the scalar values; the comparison-class statistics agree with a recomputation from `base.csv`; the new fixtures pass.
2. **b2. The corpus.** CG.65 and CG.66: the propositions, the comparative grammar, the planner's new content, the test sets, and the guide. *Accept:* every comparison sentence is true by an independent recomputation; every false comparison item is false; `interpret(tree)` recovers every logical form; with the comparative realized as an affix and as a word, trees are correct for all six clause orders.

## Phase (c): stored fluent relations, their profiles, and locations (design)

This section sets the design. Its detailed specification, with configuration, outputs, and build stages, is written and reviewed before the phase's build starts.

**Relations are abstract.** A stored relation is `RL.<n>`, as a feature is `IS.<n>`. No relation has a built-in meaning. A relation behaves like nearness, or like location, because of its profile, never because of its name. What a language calls a relation is a setting of the language.

- **Stored relations.** Binary fluent relations, stored sparsely as sets of ordered pairs (REL.19). There are no static stored relations.
- **Profiles.** Each stored relation has a profile, drawn from a configured mix or given in an event file. The profiles of phase (c):
  - *symmetric*: the relation holds both ways or neither way. An effect on `(a, p)` writes `(p, a)` too;
  - *asymmetric*: never both ways;
  - *irreflexive*: never between an entity and itself (true of every relation, by REL.5);
  - *functional*: each entity has at most one partner;
  - *total functional*: each entity has exactly one partner. Adding a new pair replaces the entity's old pair.
  
  A profile is an integrity constraint that the runtime maintains. An event whose effects would break a profile is not legal, as if its precondition had failed.
- **Effects** gain `add RL.2(a, p)` and `delete RL.2(a, p)`, between the roles of the event's binding. **Preconditions** gain relation literals over the binding: `RL.2(a, p)`, `NOT RL.2(p, a)`.
- **Derived fluents** gain existential relation literals: `SOME.RL.2(x)`, true when `x` has at least one partner in `RL.2`, and `SOMEOF.RL.2(x)`, true when `x` is someone's partner. These literals are computed before the matrices, as threshold literals are.
- **Non-interference** extends to relations: two events of one step interfere when one writes a pair that the other writes or reads. A symmetric pair counts as written in both directions.
- **Locations.** Location is a total functional relation from entities to places: each entity is at exactly one place. An event type whose patient is a place, with an effect that adds a pair of the location relation, moves its agent. Until sorts arrive in phase (d), places are a fixed second kind of entity, labeled `PL.<n>`.
- **The 3D link.** A relation may carry a `spatial` mark that names a 3D detector and its parameters. Only the 3D runtime reads the mark: in 3D, the detector computes the relation from geometry, and the relation is not stored. The disembodied runtime ignores the mark. The mark is the one place where an abstract relation is tied to a concrete meaning, and the tie exists only in the 3D mode.
- **The corpus** gains relational states and changes, such as `HOLDS(SN.8, T.2, PAST, RL.2(R.1, R.2))` and `BECOME` over relations, words for relations and places, and location sentences. A language's word for a symmetric relation ("near", in an English gloss) is a lexicon entry like any other. The relation's concept label stays `RL.2` (CG.67, as reworded).

Questions to settle before the detailed specification:

1. Can one relation be constrained by another? For example, may a symmetric relation hold only between entities that share a location? If so, is that a kind of profile, and does a move then delete the mover's pairs of the constrained relation?
2. Do relations start empty in each episode, or are initial pairs drawn at a rate?
3. Are places generated with their own features, or are they bare labels?
4. Which profiles are in the default mix, and with what weights?
5. Which relation statements does the corpus make at the class level, if any?

## Phase (d): sorts, derived relations, and parts (design)

As for phase (c), this section sets the design, and the detailed specification follows before the build.

- **Sorts.** A small, coarse set of sorts (at first: entity, place, and part). Sorts are structural, like time points: they say what kinds of thing a symbol takes, not what the things are like. Every symbol gets a signature: the sorts of its arguments. Role requirements gain sort checks. Sorts replace the fixed second kind of phase (c).
- **More profiles, and derived relations.** Phase (d) adds the *acyclic* and *transitive* profiles. Derived relations come from profiles: the converse of a relation, and the transitive closure of an acyclic one. They are computed by stratified rules, with positive recursion only.
- **Parts.** Parts become entities of the part sort, with their own static features ("red arms", in an English gloss). Each part is linked to its whole by a stored relation whose profile is functional (a part has one whole) and acyclic. Parthood, like location, is a profile and a sort, not a built-in relation. Whole-level features can be derived from parts: `HAS.n(x)` when some part of `x` has kind `n`.

Questions to settle before the detailed specification:

1. Are part kinds generated by the taxonomy (a second tree, or a list), and does each whole category inherit its parts the way it inherits free features now?
2. Do today's HAS features all become derived from parts, or do some stay free features?
3. Is the part relation static at first, and fluent only when events attach and detach parts?
4. Do counts of parts ("four legs") come in this phase?

## The Rust runtime (later)

The Rust runtime is not built in this specification. It is written when the 3D link is built. The format and the fixtures above are what it must meet:

- A crate loads `definition.json` and `entities.csv`, and implements `derive`, `able`, `legal`, and `apply` with the semantics of phase (a), and later of phases (b) to (d).
- The crate passes every fixture in `tests/fixtures/world/`.
- The 3D engine maps its actions onto event types, with an actuator and a duration for each. A step of the world model is a decision point. Relations with a `spatial` mark are computed by their detectors.
- Milestone 1's tags, stocks, and states correspond to static facts, numeric fluents, and Boolean fluents. Numeric fluents are not part of phases (a) to (d), and are added with the link.
- The link adds to contracts 2, 5, and 7 in `docs/CONTRACTS.md`. Nothing in this specification changes the contracts.

## Questions for Jon

When a question comes up that this specification does not answer, and the answer would change the world's content, a file format, or what models are given, stop and ask. Write the question, the options, and a recommendation in `docs/proposals/`, one file per question. Small engineering choices inside a module need no question. Record them in the stage's proposal file.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. **Labels.** One-place event types are `EV.<n>`, two-place event types and their categories `E<i>.<j>...`, event-type features `EF.<n>`, fluents `ST.<n>`, and time points `T.<k>`. `CAN.` and `CANBE.` remain, as the labels of derived capacities.
2. **Fluents are Boolean, and a feature type of their own.** Numeric fluents (needs, amounts) are added with the 3D link, not in phases (a) to (d).
3. **Initial values.** Each base fluent's initial rate is drawn from `[0.0, 0.2, 0.8, 1.0]`. Each entity's initial values are fixed when the world is generated, and every episode starts from them, with an option to redraw per episode. Initial values do not depend on category.
4. **Static rules read static facts only.** A rule that reads a fluent makes a derived fluent.
5. **"Can" is the requirement only.** `ABLE` and capacities ignore state. Legality adds the precondition.
6. **Preconditions are conjunctions of fluent literals**, in the style of STRIPS. Requirements keep the full rule families.
7. **Effects set base fluents to true or false**, with no conditional effects.
8. **Preconditions and effects attach to event-type features** as well as to event types, so a branch of the event-type tree shares them, as it shares base relations (REL.10).
9. **A requirement is a rule over the whole binding**, so cross-role constraints and comparisons stay. REL.17's per-role wording is read as the special case.
10. **Parallel steps.** A step can hold several events, which must not interfere, and which are applied together. An episode ends early when nothing is legal.
11. **The default selection policy** keeps today's order of draws: the kind of event, then the event type, then a binding among the legal ones.
12. **Scenes move to the world package**, as episodes. The scene settings stay in the corpus configuration.
13. **Histories** extend `scenes.jsonl` with initial states and changes. The 3D engine writes the same schema once the link is built.
14. **The definition's format** is `definition.json` plus `entities.csv`, with rules in both Boolean and matrix form, and the matrices made from the minimal DNF.
15. **Taxonomy outputs** split the base vector (`base.csv`) from derived values (`derived/`, tagged with the rule-set identity). Views, with a `classic` preset, replace `instances.csv`.
16. **A quarter of patient capacities get words** by default, as before. This is now a language setting (REL.16).
17. **Default word meanings.** "All" and "no" express `NEC_ALL` and `NEC_NO` (as the old default, the fixed reading, did). The bare plural expresses `MOST`.
18. **Relation facts and patient capacities are never `NEC`**, because no fixed test exists for them.
19. **Descriptions (CG.63).** The noun, modifiers, and restrictive relative clause of a definite mention are descriptions, and an indefinite mention asserts its noun and modifiers. The rendering marks descriptions with braces. `seen` counts descriptions by default, as now.
20. **States in narratives.** Initial-state sentences (rate 0.2) and result sentences (rate 0.5). Modifiers stay static in phase (a).
21. **Test sets.** State items come from situational narratives, at the scene's final time point, split into `changed` and `unchanged`. False events split into `possible`, `blocked`, and `impossible`.
22. **Causal rule statements are deferred.** Class-level statements of effects and preconditions ("things that are caught become ...") need new grammar, and are left for after phase (a). In phase (a), narratives carry the evidence for effects.
23. **Threshold terms in rule statements.** Still skipped in phase (b). Two options for later: state a threshold by comparison with the category whose mean is nearest (the language approximates, and the proposition keeps the exact threshold); or add number words.
24. **Class-level comparatives** compare category means, and category documents gain comparison facts with a sibling or a thematic partner.
25. **Relations stay abstract.** Stored relations are `RL.<n>`, with no built-in meaning. Profiles move into phase (c), so a relation behaves like nearness or location because of its profile, not its name. Places are a fixed second kind in phase (c), ahead of the sorts of phase (d). The `spatial` mark is read only by the 3D runtime.

## References

- Barwise, J., and Cooper, R. (1981). Generalized quantifiers and natural language. *Linguistics and Philosophy*, 4, 159–219.
- Blum, A. L., and Furst, M. L. (1997). Fast planning through planning graph analysis. *Artificial Intelligence*, 90, 281–300. (Non-interfering actions in one step.)
- Fikes, R. E., and Nilsson, N. J. (1971). STRIPS: a new approach to the application of theorem proving to problem solving. *Artificial Intelligence*, 2, 189–208.
- Fox, M., and Long, D. (2003). PDDL2.1: an extension to PDDL for expressing temporal planning domains. *Journal of Artificial Intelligence Research*, 20, 61–124.
- Kennedy, C. (2007). Vagueness and grammar: the semantics of relative and absolute gradable adjectives. *Linguistics and Philosophy*, 30, 1–45.
- Kowalski, R., and Sergot, M. (1986). A logic-based calculus of events. *New Generation Computing*, 4, 67–95.
- Thiébaux, S., Hoffmann, J., and Nebel, B. (2005). In defense of PDDL axioms. *Artificial Intelligence*, 168, 38–69.
