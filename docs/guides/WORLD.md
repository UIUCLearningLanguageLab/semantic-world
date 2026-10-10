# The world package

The world package defines a world and runs it. A world is a set of entities with fixed features, a set of fluents that change over time, and a set of event types that say what can happen to the entities and what each event changes. The package generates a world from a taxonomy of the taxonomy generator, writes the world's definition as data files, runs episodes of the world, and writes their histories. The corpus generator reads worlds and writes documents about them. The 3D engine will later read the same definition files.

This guide covers running the package, what a world is, the commands, the output files, the configuration file, the default world's figures, the conformance fixtures, the two-place share, and the Python interface. The design is specified in `docs/specs/WORLD_AND_LANGUAGE.md`, and every design decision is listed in `docs/DECISIONS.md` (WM.1 to WM.34, and the engineering choices WM.E1 and following).

**Status.** Phase (a) of the world-and-language refactor is complete (stages a1 to a8): Boolean fluents, one-place and two-place event types with requirements, preconditions, and effects, time steps with inertia, the Python runtime, episodes with selection policies, histories, views, world statistics, and the conformance fixtures. Phase (b) has begun: stage b1 adds conditional effects (an effect that applies only when a condition on the binding holds). Numeric fluents and comparisons derived from scalars come in stage b2, and the corpus's conditional causal statements and comparatives in stage b3; stored relations and places in phase (c); sorts and parts in phase (d). The Rust runtime is not built yet: the fixtures are its tests, waiting.

## Quick start

From the root of the repository (see `README.md` in this folder for setup):

```
PYTHONPATH=python python -m semantic_world.world define data/world/tiny.yaml
```

The program prints one line, and a note for each event type whose preconditions had to be drawn again:

```
wrote runs/world/tiny_seed1: 12 entities, 4 fluents (1 derived), 4 one-place and 4 two-place event types, 12 constraints; rule set 6777fdc1b604
note: the preconditions of EVENTTYPE1.4 were drawn again 1 time(s)
note: the preconditions of EVENTTYPE2.1.2 were drawn again 2 time(s)
note: the preconditions of EVENTTYPE2.2.1 were drawn again 1 time(s)
```

The tiny world builds on the tiny relations taxonomy (6 categories, 12 instances, 8 PROPERTY and 8 PART features, 1 scalar dimension) and adds 4 fluents, 4 one-place event types, and 4 two-place event types in a tree of 2 categories. The 12 constraints are the rules of the requirements: one for each one-place event type, and one for each constraint of the two-place event types. One of its 10 effects is conditional (`when agent.BOOLFL.3 AND NOT agent.PROPERTY.1 then agent.BOOLFL.2 := 1`, on `EVENTTYPE1.1`). The run is small enough to read every file by eye. The whole run, with its 1,000 statistics episodes, takes about 3 seconds.

Two options change a run without editing the configuration:

- `--seed N` replaces the master seed. Different seeds give different worlds from the same settings. The embedded taxonomy takes the world's seed unless the configuration gives it one of its own.
- `--out DIR` writes the output folder somewhere else. The default is `runs/world/<name>_seed<seed>/`.

The example configurations in `data/world/` are:

| File | What it makes |
| --- | --- |
| `tiny.yaml` | The tiny world above. |
| `default.yaml` | The default world: the default relations taxonomy (4 superordinates, depth 3, 284 instances, 40 PROPERTY and 40 PART features, 2 scalar dimensions) with 8 fluents, 20 one-place event types, and 20 two-place event types in 5 categories. Every parameter appears with its value, so this file is also the reference for defaults; five settings of the event types depart from the code's defaults (see "The two-place share"). |
| `tiny_chain.yaml` | The tiny world with the event file `events/chain_example.yaml`, which defines two event types by hand so that they form a chain: catch, then eat. |

## Concepts

### What a world is

A world definition has three parts: the entities' base facts, the feature rules, and the event types. Nothing else is part of the definition. What a model sees, and what the language names, are settings of the experiment and of the language.

- **Entities** are the instances of the embedded taxonomy, `INSTANCE.1.3.2.5`. Each has a leaf category, free PROPERTY and PART features, and scalar values. These are its **base static facts**: stored, and fixed for the life of the world. Its **derived static facts**, the determined PROPERTY and PART features, are computed from them by the taxonomy's rules and never stored as inputs.
- **Fluents** are Boolean facts that change over time, `BOOLFL.1` to `BOOLFL.n`. A **base fluent** is stored, starts at an initial value for each entity, and changes when an event's effect writes it. A **derived fluent** is computed by a rule that reads at least one fluent, and may read static features too; it is recomputed at every time point and never written. So every fact is static or fluent, and base or derived.
- **Event types** are the kinds of thing that can happen. A **one-place event type** (`EVENTTYPE1.4`) has one role, the agent. A **two-place event type** (`EVENTTYPE2.1.2`) has an agent and a patient; two-place event types form a tree, and a category of the tree (`EVENTTYPE2.1`) is a more general kind of event, as "hunt" is to "chase". A **binding** assigns entities to the roles, never the same entity twice.
- **The requirement** of an event type is a rule over the static facts of a binding. When it holds, the entities are **able** to take part: the owl can eat the mouse. Capacities ("can") are requirements. A one-place requirement is a rule over the agent's features, sampled as the taxonomy's rules are. A two-place requirement is the conjunction of constraints over both roles, which can compare the two entities' scalars (the agent is bigger than the patient) or match their features (barbers cut hair).
- **The precondition** is a conjunction of fluent literals over the binding, `agent.BOOLFL.2 AND NOT patient.BOOLFL.5`. It says whether the event can happen now: the owl is awake. A binding is **legal** in a state when its requirement holds and its precondition holds in that state.
- **The effects** set base fluents of the participants, `patient.BOOLFL.3 := 1`. An event type never has two effects that set the same fluent of the same role to different values, and never an effect that sets a fluent to the value its own precondition requires. Since stage b1 an effect may carry a **condition**, `when patient.PROPERTY.4 AND patient.BOOLFL.2 then patient.BOOLFL.3 := 1`: a conjunction of one or two literals over the binding, each a static feature (PROPERTY or PART, free or determined) or a Boolean fluent (base or derived) of one role, positive or negated. The condition is judged in the state at the start of the step, before any effect of the step; when it is false, the effect does nothing. A condition never bears on whether the event is legal. A share of the effects (`event_types.effects.conditional_share`, 0.2 by default) gets a condition when the world is generated; with the share at 0 the world is the unconditional world of phase (a), byte for byte.
- **Time.** An episode has time points `TIME.1`, `TIME.2`, and so on. Step k takes the state at `TIME.k` to the state at `TIME.k+1`. Every event of a step must be legal at `TIME.k`, and the events of one step must not interfere: no two write the same fluent of the same entity (every effect counts as a write, whatever its condition), and none writes a fluent that another's precondition, or a condition of another's effect, reads (a condition on a derived fluent reads every base fluent the derived fluent depends on; a condition on a static feature reads nothing an event can write). The effects whose conditions hold are applied, every other base fluent keeps its value (inertia), and the derived fluents are recomputed. Applying a step's events together gives the same state as applying them one at a time in any order.
- **Episodes and histories.** An episode runs the world over a few participants for a few steps. Participants are drawn around a seed entity, weighted by thematic relatedness and taxonomic similarity. At each step a selection policy draws events among the legal ones; by default (`uniform_event`) it draws uniformly among all legal events, so an event type that is legal for many bindings happens more often, as in a world where some things simply happen more. An episode ends after its steps, or earlier when nothing is legal (quiescence). A **history** records the episode: its participants, its initial state, and each step's events with the changes they made. The corpus calls an episode a scene.
- **Views** choose which columns of the base facts and the derived values a model sees. A view is an experiment setting, never part of the world.

The event types attach their preconditions and effects to the tree as well: an event-type feature can carry a precondition literal or an effect, and every event type with that feature inherits it, so a branch of the tree shares what it needs and what it does, as it shares its constraints.

### Labels

Every label is a word a reader can read without a key, a period, and an index. The full table is in `docs/specs/WORLD_AND_LANGUAGE.md`, "Labels". The world's own labels:

| Object | Label | Example |
| --- | --- | --- |
| Boolean fluent | `BOOLFL.<n>` | `BOOLFL.3` |
| One-place event type | `EVENTTYPE1.<n>` | `EVENTTYPE1.4` |
| Two-place event type, or a category of them | `EVENTTYPE2.<path>` | `EVENTTYPE2.1.2`, `EVENTTYPE2.1` |
| Event-type feature | `EVENTFEAT.<n>` | `EVENTFEAT.5` |
| Constraint of an event-type feature, or an event type's own | `CONSTRAINT.<owner>` | `CONSTRAINT.EVENTFEAT.5` |
| Capacity, as agent or as patient (derived) | `CAN.<event type>`, `CANBE.<event type>` | `CAN.EVENTTYPE1.4`, `CANBE.EVENTTYPE2.1.2` |
| Role prefix in a rule | `agent.`, `patient.` | `patient.PROPERTY.7` |
| Episode (a scene), event, time point | `SCENE.<n>`, `SCENE.<n>.EVENTINSTANCE.<k>`, `TIME.<k>` | `SCENE.8.EVENTINSTANCE.5` |

### Where the world comes from

`define` runs the taxonomy generator first, in memory, and writes the run to `taxonomy/` inside the world's folder, byte-identical to the taxonomy run alone with the same configuration and seed. The taxonomy supplies the entities' static facts and the feature rules. The world package then draws, from streams of its own, the one-place requirements (`world:requirements`), the event-type tree and its constraints (`world:event_tree`, `world:constraints`, `world:pairs`), the fluents and their rules (`world:fluents`), the initial values (`world:initial`), the preconditions (`world:preconditions`), and the effects (`world:effects`), and runs the statistics episodes (`world:stats`). Separate streams mean that changing the fluent, precondition, or effect settings never changes the taxonomy, the requirements, the capacities, or the relation statistics. `TAXONOMY.md`, "Where CAN features and verbs went", describes the requirements and the relation statistics in detail.

### Rules as matrices, and the rule-set identity

Every rule of a world (a determined feature, a derived fluent, a constraint, a requirement) is kept in two forms: the Boolean form, a truth table with an expression, which a reader can check, and the matrix form, two threshold layers, which the runtime computes (`docs/specs/WORLD_AND_LANGUAGE.md`, "Rules: Boolean form and matrices"). At the end of every run, the agreement test evaluates both forms on every entity and every ordered pair, and on every input setting of every rule with at most 12 inputs; a disagreement fails the run, because it would be a bug in the generator.

The **rule-set identity** is the SHA-256 hash of the canonical JSON of the definition's symbols, literals, rules, and event types. It changes exactly when a rule, a literal's number, or an event type changes. Every derived file and every history carries it, and a loader refuses a derived file or a history whose identity differs from the definition in use. `tests/fixtures/world/README.md` gives the exact rules of the canonical JSON, for whoever builds the Rust runtime.

## The commands

Every command runs from the root of the repository. With the repository's virtual environment active, `PYTHONPATH=python` is not needed.

### `define`

```
PYTHONPATH=python python -m semantic_world.world define data/world/tiny.yaml [--seed N] [--out DIR]
```

Generates a world and writes its folder (see "Reading the outputs"). The run embeds the taxonomy run, draws the fluents and the event types, builds the definition, runs the agreement test and the checks of the dynamics (no requirement reads a fluent, no effect writes a derived fluent, no contradiction, every precondition literal achievable), and runs 1,000 statistics episodes. An event type that is never legal in the statistics episodes has its own precondition literals drawn again, up to a few times; the command prints a note for each redraw, and a warning for each event type that is still never legal. The default world takes about 8 seconds, of which the static side is about a second. Every stage records this time, measured with nothing else running, and flags a growth of more than half in its proposal file; there is no hard time limit, and no step is made cheaper by changing what it produces by default (Jon's ruling 3 on stage a8).

### `simulate`

```
PYTHONPATH=python python -m semantic_world.world simulate runs/world/tiny_seed1 --episodes 100 [--seed N] [--legal] [--out FILE] [--config data/corpus/default.yaml]
```

Runs episodes of a world run and writes them as histories, one JSON object per line, to `RUN/episodes.jsonl` or to `--out`. The command prints one line:

```
wrote runs/world/tiny_seed1/episodes.jsonl: 100 episodes, 762 events, 0 quiescent, policy uniform_event
```

The master seed is the run's seed unless `--seed` gives another; each episode draws from its own part of the stream `world:episodes`, named by its label, so episode 7 is the same whether 10 or 1,000 episodes are run. The scene settings (the number of participants, the number of steps, the events per step, the participant weights, the policy, the event-type weights, and whether the initial state is kept or redrawn) are the corpus's defaults, or the `scene` block of a corpus configuration file given with `--config`. With `--legal`, each step also records the number of legal bindings of each event type among the participants, so legality and occurrence can be studied apart; it is off by default because of size.

### `view`

```
PYTHONPATH=python python -m semantic_world.world view runs/world/tiny_seed1 --preset classic [--include PROPERTY,PART] [--out FILE]
```

Writes one table for a model, by default to `RUN/views/<preset>.csv`, with a sidecar YAML file beside it that records the preset, the columns, and the rule-set identity. The presets:

- `base`: the base vector only, `label`, `leaf`, the free PROPERTY and PART features, and the scalars;
- `static`: the base vector and the derived static features;
- `classic`: the columns of the taxonomy's old `instances.csv`: the ISA features, every PROPERTY and PART feature, the one-place capacities `CAN.EVENTTYPE1.<n>` in the place of the old CAN features, and the scalars.

`--include` takes label prefixes. A prefix that matches columns of the preset narrows the view to them; a prefix that matches none adds every available column with that prefix (ISA, base facts, derived static features, capacities, initial fluents). For example, `--preset base --include PROPERTY` gives the free PROPERTY features alone, and `--preset base --include BOOLFL` adds the initial fluents.

### `check-fixtures` and `make-fixtures`

```
PYTHONPATH=python python -m semantic_world.world check-fixtures [tests/fixtures/world]
```

Runs the Python runtime and then the brute-force evaluator on every conformance fixture of the folder, prints one line per fixture, and exits with status 1 at the first failure. On the committed fixtures it ends with `23 fixtures passed the runtime and the brute-force evaluator`. See "The conformance fixtures" below.

```
PYTHONPATH=python python -m semantic_world.world make-fixtures data/world/tiny.yaml [--out tests/fixtures/world] [--count N] [--steps N]
```

Regenerates the generated fixtures of a world configuration: `count` fixtures of random steps without an error (the default 4, each of 6 steps), then one that ends in an illegal event and one that ends in interfering events. Their expected values come from the brute-force evaluator, never from the runtime. Regenerating the tiny world's fixtures gives the committed files byte for byte, and a test requires it. The hand-written fixtures are made by `python tests/world/hand_world.py`, from the hand world and its hand-typed cases.

### The levers of the two-place share

```
python examples/two_place_levers.py [--seeds 1,2,3] [--only TEXT] [--out FILE]
```

Defines the default world with one setting changed at a time and reports, for each lever and seed, the share of two-place events among the events of the statistics episodes, the mean one-place capacity rate, and the mean two-place pair density. The whole table takes about half an hour; `--only` runs the levers whose names contain the text. See "The two-place share".

## Reading the outputs

A run writes one folder:

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved world configuration, with the resolved taxonomy configuration under `taxonomy.resolved`, every stream seed, the git commit (flagged when the working tree had uncommitted changes), and the package version. Running `define` on a run's `config.yaml` reproduces the run exactly. |
| `taxonomy/` | The embedded taxonomy run, in the taxonomy generator's own format (`TAXONOMY.md`, "Reading the outputs"). |
| `definition.json` | The definition: every symbol, every literal, every rule in Boolean form, the matrix form, and every event type. |
| `entities.csv` | One row per entity: `label`, `leaf`, the free PROPERTY and PART features, the scalars (full precision), and the initial value of every base fluent. |
| `derived/` | The derived values, each file tagged in `manifest.yaml` with the rule-set identity that produced it. |
| `world_stats.yaml` | The world's statistics (below). |
| `views/` | The views written by `view`, each with its sidecar. |
| `episodes.jsonl` | The histories written by `simulate`, when it has run. |

### `definition.json`

The language-neutral part of the definition, with no code in it. Its top-level keys:

- `version`: the format's version, 1;
- `rule_set_id`: the rule-set identity;
- `symbols`: one entry per feature, scalar, fluent, event type, and event-type category: `label`, `kind` (`property`, `part`, `scalar`, `fluent`, `event_type`, `event_type_category`), `derived`, `fluent`, `arity`, and, for a base fluent, `initial_rate`;
- `literals`: everything the rules read, each with its `index`, its `key`, its `kind` (`feature`, `threshold`, `fluent`, `comparison`, `constraint`), its `role` (null for the entity itself, `agent`, `patient`, or `binding`), and the fields of its kind;
- `rules`: every rule's `output`, `scope` (`entity` or `binding`), `family`, `inputs` (literal indices), `truth_table`, and `expression`. A derived feature and a derived fluent are entity rules; a constraint and a requirement are binding rules;
- `layers`: the matrix form, written sparsely: for each layer its `terms` (the literals of one term of the rule's minimal DNF, whether each is complemented, and the threshold) and its `outputs`;
- `event_types`: one entry per event type and category: `label`, `kind`, `arity`, `roles`, `parent`, `level`, `features`, `explicit`, the `requirement` (its constraints, the rule that is the requirement, and the expression), the `precondition` (its literals, each a `role`, `fluent`, and `value`, and the expression), and the `effects` (each a `role`, `fluent`, `value`, and expression, and, for a conditional effect, a `condition` with its literals, each a `role`, `kind` (`feature` or `fluent`), `symbol`, and `value`, and its expression; an unconditional effect has no `condition` key).

Floating-point numbers are written with 17 significant digits, so both runtimes read the same values. `tests/fixtures/world/README.md` describes every field, and how the layers are evaluated.

### `derived/`

| File | Contents |
| --- | --- |
| `static_features.csv` | One row per entity: every derived static feature. |
| `capacities.csv` | One row per entity: `CAN.EVENTTYPE1.<n>` for every one-place event type, then `CAN.` and `CANBE.` for every two-place event type (whether some possible partner would make the requirement hold), an `approximate` flag, and the `ACTUAL_` columns (whether the entity has a partner among the world's entities). |
| `capacity_roles.csv` | For each category and one-place event type: whether the capacity is `fixed_1`, `fixed_0`, or `free` for the category's members, and whether the `exact` or the `local` test decided. The corpus's `NEC` statements rest on it. |
| `relation_proportions.csv`, `relation_pairs.csv`, `event_type_stats.csv`, `thematic.csv` | The relation statistics of the two-place event types: for each event type and pair of categories at the same level, the share of pairs related; sampled true and false pairs; one row of statistics per event type (the proportion of pairs related, the agents and patients with a partner, the constraints and their families, the leaf-pair density); and, for every pair of leaves, the thematic relatedness score beside the taxonomic similarity. `TAXONOMY.md` has an example row. |

A derived file is read through the package, which refuses a file whose identity differs from the rule set in use:

```python
from semantic_world.world import load_derived_csv

frame = load_derived_csv("runs/world/tiny_seed1/derived", "capacities.csv", rule_set_id)
```

Nothing in `derived/` is an input to anything, and nothing there is edited by hand. The statistics describe what entities are able to do, never what is legal in a state: a sleeping owl can still eat a mouse.

### `world_stats.yaml`

- `fluents`: the counts of base and derived fluents.
- `event_types`: the counts of one-place event types, two-place event types, and categories.
- `precondition_literals` and `effects`: the totals, by kind of event type, by role, and how many come from event-type features; under `effects`, the `conditional` effects by kind of event type and the `condition_literals` by symbol kind (`feature` or `fluent`), by role, and from event-type features (stage b1).
- `fix_ups`: the effects, literals, and conditions redrawn or dropped while the dynamics were drawn (a contradiction, a duplicate, an effect that changes nothing, a literal that nothing can make true, a condition that breaks a rule of "Conditional effects": `conditions_redrawn` and `conditions_dropped`).
- `enabling_graph`: the edges from an event type whose effect produces a value to an event type whose precondition requires it, and the longest chain.
- `absorbing_fluents`: the base fluents that some effect sets to one value and no effect sets back.
- `relations`: the event-type features, categories, event types, and constraints by family, and the density report.
- `episodes`: the 1,000 statistics episodes of the default policy with the corpus's default scene settings, on the stream `world:stats`: for each event type the share of steps at which it had a legal binding (`legal_share`) and at which it occurred (`occurrence_share`), the mean number of changes per event, the share of quiescent episodes, the two-place share (`two_place_share`: the share of two-place events among all events), the precondition redraws, and the event types that were never legal.

### Histories

A history is one JSON object per episode: its `label`, its `seed` entity, its `participants`, the `policy`, the `rule_set_id`, the `initial` base fluents of each participant, the `steps` (each with its `events`, and each event with its `label`, `type`, `agent`, `patient`, and `changes`), the `final` time point, and `quiescent`. From the tiny world:

```json
{"label": "SCENE.1", "seed": "INSTANCE.1.1.2", "participants": ["INSTANCE.1.1.2", "INSTANCE.1.1.3", "INSTANCE.1.2.1"],
 "policy": "uniform_event", "rule_set_id": "bd02c67e7286...",
 "initial": {"INSTANCE.1.1.2": ["BOOLFL.3"], "INSTANCE.1.1.3": ["BOOLFL.1", "BOOLFL.3"], "INSTANCE.1.2.1": ["BOOLFL.3"]},
 "steps": [{"step": 1, "events": [
   {"label": "SCENE.1.EVENTINSTANCE.1", "type": "EVENTTYPE2.2.1", "agent": "INSTANCE.1.1.3", "patient": "INSTANCE.1.2.1",
    "changes": [{"entity": "INSTANCE.1.1.3", "fluent": "BOOLFL.2", "to": true}]},
   {"label": "SCENE.1.EVENTINSTANCE.2", "type": "EVENTTYPE1.2", "agent": "INSTANCE.1.2.1",
    "changes": [{"entity": "INSTANCE.1.2.1", "fluent": "BOOLFL.1", "to": true}]}]}, ...],
 "final": "TIME.5", "quiescent": false}
```

`changes` lists only the base fluents whose value changed: an effect that sets a fluent to the value it already has records nothing, an effect whose condition failed records nothing, and a derived fluent's change is never recorded, because it is recomputed. The corpus's `scenes.jsonl` holds the same schema, and the 3D engine will write it too. `tests/fixtures/world/README.md`, "Histories", gives every field.

## The configuration file

A configuration file is YAML. Any parameter left out takes its default, unknown keys are errors, and every error names the file and the field. `data/world/default.yaml` lists every parameter.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `name` | `default` | Names the output folder. |
| `seed` | 1 | The master seed. |
| `taxonomy` | `{config: data/taxonomy/relations.yaml, seed: null}` | The taxonomy configuration file, and its seed (null: the world's seed). |
| `fluents.count` | 8 | The number of Boolean fluents. 0 gives a world with no fluents, no preconditions, and no effects, in which every able binding is legal and nothing changes. |
| `fluents.derived_proportion` | 0.25 | The share of fluents that are derived (rounded half up). |
| `fluents.max_chain_depth` | 1 | The layers of derived fluents. |
| `fluents.initial_rates` | `[0.0, 0.2, 0.8, 1.0]` | Each base fluent's initial rate is drawn from this list; each entity's initial value is drawn at that rate, once, when the world is generated. |
| `event_types.unary.count` | 20 | The number of one-place event types. |
| `event_types.unary.rules` | `{}` | Overrides of the taxonomy's `rules` settings (arity, operator mix, negation probability, families, nesting depth) for the one-place requirements. |
| `event_types.binary` | `{}` | The two-place event types, with the keys below; null turns them off. |
| `event_types.binary.features` | `{count: 12, expected_true: 3}` | The event-type features: how many, and how many are true of an event type on average. Each true feature adds one constraint to the event type's requirement. |
| `event_types.binary.taxonomy` | `{superordinates: 3, depth: 2, branching: [2, 3]}` | The shape of the event-type tree. |
| `event_types.binary.inheritance` | `{proportion_defining: 0.4, proportion_characteristic: 0.4, characteristic_probability: 0.9}` | How event-type features are inherited down the tree. |
| `event_types.binary.own_constraint` | true | Whether each event type has a constraint of its own beside those of its features. |
| `event_types.binary.constraint_families` | every family 1 | Weights over the families `agent`, `patient`, `cross`, `key_lock`, and `comparison`. |
| `event_types.binary.key_lock_pairs` | `{1: 0.5, 2: 0.3, 3: 0.2}` | Weights over the number of matching pairs in a key-lock constraint. |
| `event_types.binary.comparison` | `{window_probability: 0.3, cross_dimension_probability: 0.2, margin_quantiles: [0.1, 0.9]}` | How comparisons are drawn: a window with two margins, a comparison across two dimensions, and the quantiles the margins come from. |
| `event_types.binary.rules` | `{}` | Overrides of the taxonomy's `rules` settings for the constraints. |
| `event_types.binary.pairs` | `{sampled_true: 1000, sampled_false: 1000, max_exact_pairs: 50000000}` | The pairs sampled for `relation_pairs.csv`, and above how many pairs the proportions are estimated. |
| `event_types.binary.density` | `{min: 0.0, max: 1.0, max_tries: 200}` | The leaf-pair density range of every event type. The default redraws an event type only when its requirement holds for no leaf pair or for every leaf pair; `{min: 0.01, max: 0.3}` restores the old check. |
| `event_types.binary.constraint_min_density` | null | The smallest leaf-pair density a single constraint may have; null turns the check off. |
| `event_types.preconditions.literals` | `{0: 0.3, 1: 0.5, 2: 0.2}` | Weights over the number of an event type's own precondition literals. |
| `event_types.preconditions.feature_rate` | 0.3 | The probability that an event-type feature carries a precondition literal, which every event type with the feature inherits. |
| `event_types.preconditions.roles` | `{agent: 0.5, patient: 0.5}` | Weights over the role of a two-place literal. |
| `event_types.preconditions.enabled_share` | 0.7 | The share of literals drawn from the values that some effect produces, which makes chains of events. |
| `event_types.effects.count` | `{1: 0.7, 2: 0.3}` | Weights over the number of an event type's own effects. |
| `event_types.effects.feature_rate` | 0.5 | The probability that an event-type feature carries an effect. |
| `event_types.effects.roles` | `{agent: 0.4, patient: 0.6}` | Weights over the role of a two-place effect. |
| `event_types.effects.conditional_share` | 0.2 | The share of effects that get a condition (stage b1): a conjunction of one or two literals over the binding, judged in the step's starting state, under which alone the effect applies. An inherited effect brings its condition to every event type with the feature. 0 gives the unconditional world of phase (a). |
| `event_types.effects.condition_literals` | `{1: 0.7, 2: 0.3}` | Weights over the number of literals of a condition. |
| `event_types.event_file` | null | A file of explicit event types, read relative to the configuration file's folder. |

Every precondition literal must be achievable (its value is some entity's initial value, or some effect produces it) and consistent (a precondition never holds a literal and its negation for one role); the generator redraws or drops a literal that is not, and counts it under `fix_ups`. Every condition must be satisfiable with its event type's requirement and precondition: some able binding satisfies its static literals (for every event type that inherits the effect), each fluent literal is achievable, names no fluent of a role that the precondition names, and never names the fluent the effect sets; the conditions are drawn from a stream of their own (`world:conditions`), so changing their settings changes nothing else, and a condition that breaks a rule is drawn again (`fix_ups.conditions_redrawn`).

**The event file.** An event file defines fluents and event types by hand, over the world's abstract labels, as a rule file defines rules: a base fluent's initial rate, and an event type's requirement, precondition, and effects. An entry replaces the sampled dynamics of the event type it names; event types without an entry are sampled. An effect may carry a condition, written `when agent.PROPERTY.1 AND NOT patient.BOOLFL.2 then patient.BOOLFL.2 := 1`. `data/world/events/chain_example.yaml` builds a chain of two event types, catch and eat, whose second needs the first's effect:

```yaml
fluents:
  BOOLFL.1: {initial_rate: 1.0}        # say, awake
  BOOLFL.2: {initial_rate: 0.0}        # say, caught
event_types:
  EVENTTYPE2.1.1:
    requirement: "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0"
    precondition: "agent.BOOLFL.1 AND NOT patient.BOOLFL.2"
    effects: ["patient.BOOLFL.2 := 1"]
  EVENTTYPE2.1.2:
    requirement: "agent.SCALARDIM.1 - patient.SCALARDIM.1 > 0.0"
    precondition: "agent.BOOLFL.1 AND patient.BOOLFL.2"
    effects: ["patient.BOOLFL.1 := 0"]
```

An entry that breaks a rule of the world (an effect on a derived fluent, a requirement that reads a fluent, a contradiction, an unachievable literal, a condition that no able binding satisfies or that reads the effect's own fluent or a fluent the precondition fixes) is an error that names the file and the entry. `data/world/tiny_chain.yaml` uses the file with the tiny world.

## The default world

The figures of `data/world/default.yaml` with seed 1, from a run of October 9, 2026 (stage b1); `define` took 8 seconds (7.6 s; 7.8 s in stage a8).

| What | Figure |
| --- | --- |
| Entities | 284, in 4 superordinates of depth 3 (the default relations taxonomy, with 40 PROPERTY and 40 PART features and 2 scalar dimensions) |
| Fluents | 8: 6 base, 2 derived |
| Event types | 20 one-place; 20 two-place in 5 categories of 4, with 12 event-type features and 12 constraints (4 patient, 3 agent, 2 cross, 2 comparison, 1 key-lock) |
| Precondition literals | 62: 18 on one-place and 44 on two-place event types; 35 on the agent, 27 on the patient; 5 inherited from event-type features |
| Effects | 81: 22 on one-place and 59 on two-place event types; 52 on the agent, 29 on the patient; 5 inherited from event-type features |
| Conditional effects | 25 of the 81 (2 on one-place and 23 on two-place event types): 6 own effects, and 2 effects of event-type features that 14 and 5 event types inherit (`when NOT patient.PROPERTY.6 then agent.BOOLFL.4 := 1`, `when agent.PROPERTY.37 then patient.BOOLFL.3 := 1`); 26 condition literals, 24 on static features and 2 on fluents, 11 on the agent and 15 on the patient |
| Fix-ups | 11 effects redrawn and 6 dropped as changing nothing; 4 literals redrawn and 17 dropped as unachievable; 8 conditions redrawn, none dropped |
| Enabling graph | 323 edges; longest chain 1 |
| Absorbing fluents | `BOOLFL.3` |
| One-place requirements | hold for 58% of the entities on average (13% to 98% by event type) |
| Two-place requirements | hold for 17.1% of the ordered pairs on average (0.4% to 96.5% by event type) |
| Statistics episodes | 1,000 episodes, 5,490 steps, 8,337 events; 0.50 changes per event (0.56 before the conditions: an effect whose condition fails changes nothing); no quiescent episode; every event type legal at some step; no precondition redraw |
| Two-place share | 0.313 of the events of the statistics episodes (0.318 before the conditions) |

Every category of event types keeps a word in the corpus: a category whose base relation holds for every pair, or for none, would get none, because it says nothing. The world's rule set is `32356b8c6be9...` (`123a4a6eb9f3...` with `conditional_share: 0`, the world of stage a8: the conditions are the only difference).

## The two-place share

Two-place events are the events with a patient. Their share among all events is a frequency of the world, and follows from the world's constraints ("A principle for defaults" in the specification): a scene of a few participants has many able agents and few able pairs, so one-place events dominate unless the requirements make room. Stage a7a retuned five settings of `data/world/default.yaml` so that the share is about a third on the default world (Jon's ruling): the event-type tree has 5 superordinates with 4 event types each (as many two-place event types as one-place ones); a two-place event type has no constraint of its own beside its features' (`own_constraint: false`), so a requirement is a conjunction of about three constraints rather than four; the key-lock family, which holds for the fewest pairs, has half the weight of the others; the constraints weigh OR four times as much as AND and XOR, so each holds for more pairs; and the one-place requirements weigh AND three times as much as OR and XOR, so they are stricter. Without own constraints, two event types that share their true event-type features have the same requirement, and one whose true features are a subset of another's has the weaker one; the tree keeps every pair of event types apart on some feature.

**The target is the default world's share**, 0.318 on seed 1 before the conditional effects of stage b1 and 0.313 with them (Jon's ruling, October 9, 2026). Other seeds draw other constraints, and the share varies with the densities they draw (the table gives the densities and rates of the static side, which conditions do not touch, and the shares of stage a8 and of stage b1):

| Seed | Two-place pair density, mean over event types (min, max) | One-place capacity rate, mean | Two-place share, stage a8 | Two-place share, stage b1 | Categories of event types with a word | Never legal | Precondition redraws (stage b1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (the default world) | 0.171 (0.004, 0.965) | 0.580 | 0.318 | 0.313 | 5 of 5 | none | none |
| 2 | 0.212 (0.004, 0.757) | 0.473 | 0.546 | 0.519 | 5 | `EVENTTYPE1.12` (never able) | none |
| 3 | 0.067 (0.000, 0.569) | 0.440 | 0.374 | 0.379 | 5 | none | `EVENTTYPE2.1.4`, `EVENTTYPE2.3.3`, `EVENTTYPE2.5.1` once each, `EVENTTYPE2.2.4` twice |
| 4 | 0.060 (0.000, 0.629) | 0.463 | 0.118 | 0.119 | 4 | `EVENTTYPE2.1.2` to `.1.4`, `EVENTTYPE2.2.1` to `.2.3` (never able) | `EVENTTYPE2.3.1`, once |
| 5 | 0.129 (0.001, 0.940) | 0.542 | 0.538 | 0.509 | 3 | `EVENTTYPE1.17`, `EVENTTYPE2.5.1`, `.5.2`, `.5.4` (never able) | 4 event types (1, 5, 5, 5) |
| Mean | 0.128 | 0.500 | 0.379 | 0.368 | | | |

Before the retune the share was 0.090, 0.193, 0.264, 0.074, and 0.083 on the same seeds. The spread (0.12 to 0.52) follows the spread of the pair densities: seeds 3 and 4 draw sparse constraints. The one-place strictness moves the share little; the two-place densities set it. The conditional effects of stage b1 move the shares by a few hundredths at most (an effect whose condition fails leaves the state as it was, so other events are legal later), and change nothing on the static side.

**The levers.** `examples/two_place_levers.py` runs the default world with one setting changed at a time, on seeds 1 to 3, and reports the share, the one-place capacity rate, and the two-place pair density (the baseline row is the retuned default; the scene settings and the event-type weights are those of the statistics episodes, which belong to the corpus configuration). The table is from stage a7a, before the conditional effects of stage b1, which move the baseline on seed 1 from 0.32 to 0.31 and leave the rates and densities as they are:

| Lever | Two-place share (seeds 1, 2, 3) | Mean | One-place capacity rate | Two-place pair density |
| --- | --- | --- | --- | --- |
| baseline (the default world) | 0.32, 0.55, 0.37 | 0.41 | 0.50 | 0.150 |
| unary.count: 10 | 0.29, 0.70, 0.37 | 0.45 | 0.51 | 0.150 |
| unary.count: 40 | 0.32, 0.38, 0.25 | 0.32 | 0.42 | 0.150 |
| binary.taxonomy: 3 superordinates, 2 to 3 each (the code's default, 7 event types) | 0.19, 0.03, 0.19 | 0.14 | 0.50 | 0.128 |
| binary.taxonomy: 5 superordinates, 8 each (40 event types) | 0.24, 0.46, 0.37 | 0.36 | 0.50 | 0.124 |
| binary.features.expected_true: 1 | 0.62, 0.67, 0.60 | 0.63 | 0.50 | 0.381 |
| binary.features.expected_true: 6 | 0.02, 0.01, 0.12 | 0.05 | 0.50 | 0.032 |
| binary.features.count: 6, expected_true: 3 | 0.38, 0.02, 0.44 | 0.28 | 0.50 | 0.109 |
| binary.own_constraint: true (the code's default) | 0.21, 0.27, 0.33 | 0.27 | 0.50 | 0.077 |
| binary.constraint_families: equal (the code's default) | 0.16, 0.31, 0.42 | 0.30 | 0.50 | 0.083 |
| binary.constraint_families: no key_lock | 0.34, 0.55, 0.39 | 0.43 | 0.50 | 0.176 |
| binary.constraint_families: comparison only | 0.31, 0.40, 0.59 | 0.43 | 0.50 | 0.175 |
| binary.rules.operator_mix: equal (the taxonomy's default) | 0.20, 0.54, 0.31 | 0.35 | 0.50 | 0.110 |
| binary.rules.arity: {1: 0.5, 2: 0.5} | 0.12, 0.37, 0.47 | 0.32 | 0.50 | 0.130 |
| unary.rules.operator_mix: equal (the taxonomy's default) | 0.29, 0.50, 0.32 | 0.37 | 0.62 | 0.150 |
| unary.rules.operator_mix: AND only | 0.42, 0.60, 0.44 | 0.49 | 0.34 | 0.150 |
| unary.rules.arity: {2: 0.2, 3: 0.4, 4: 0.4} | 0.49, 0.62, 0.43 | 0.51 | 0.33 | 0.150 |
| preconditions.literals: {0: 1} and feature_rate: 0 (no preconditions) | 0.47, 0.63, 0.40 | 0.50 | 0.50 | 0.150 |
| preconditions.feature_rate: 0 | 0.52, 0.62, 0.25 | 0.46 | 0.50 | 0.150 |
| preconditions.feature_rate: 0.6 | 0.34, 0.07, 0.03 | 0.15 | 0.50 | 0.150 |
| preconditions.literals: {0: 0.1, 1: 0.4, 2: 0.5} | 0.27, 0.44, 0.36 | 0.36 | 0.50 | 0.150 |
| scene.size: [1, 3] | 0.19, 0.39, 0.23 | 0.27 | 0.50 | 0.150 |
| scene.size: [6, 10] | 0.49, 0.71, 0.58 | 0.59 | 0.50 | 0.150 |
| scene.events_per_step: 0.5 | 0.33, 0.54, 0.42 | 0.43 | 0.50 | 0.150 |
| scene.events_per_step: 4 | 0.28, 0.53, 0.33 | 0.38 | 0.50 | 0.150 |
| scene.event_type_weights: two-place event types weighed 3 | 0.53, 0.76, 0.61 | 0.64 | 0.50 | 0.150 |
| scene.event_type_weights: one-place event types weighed 3 | 0.15, 0.31, 0.18 | 0.21 | 0.50 | 0.150 |

What each lever does, against the baseline's share of 0.41 over seeds 1 to 3:

- **The number of event types of each kind** moves the share as a count: forty one-place event types lower it to 0.32 and ten raise it to 0.45; seven two-place event types lower it to 0.14 and forty raise it to 0.36 against the baseline's twenty at 0.41, because every event type's own feature draw changes the densities too.
- **The number of constraints per two-place event type** is the strongest lever: one true event-type feature per event type raises the density to 0.38 and the share to 0.63, six true features lower them to 0.03 and 0.05, and the own constraint of the code's default gives 0.08 and 0.27. Each constraint removes roughly half the pairs, so the density falls geometrically with their number.
- **The constraint-family weights**: the key-lock family holds for the fewest pairs, so equal weights give a density of 0.08 and a share of 0.30, while no key-lock, or comparisons alone, give 0.18 and 0.43.
- **The complexity of the constraints** matters less: the taxonomy's equal operator mix gives 0.11 and 0.35 against the OR-weighted mix's 0.15 and 0.41, and arities of 1 and 2 alone give 0.13 and 0.32. A constraint with more OR holds for more pairs.
- **The complexity of the one-place requirements** moves the one-place capacity rate (0.62 with the equal mix, 0.50 with AND weighed 3, 0.34 with AND alone, 0.33 with arities of 2 to 4) and the share by a few hundredths to a tenth (0.37, 0.41, 0.49, 0.51): fewer able agents leave more room for the two-place events.
- **The precondition settings**: with no preconditions at all the share is 0.50, with no feature literals 0.46, with more own literals 0.36, and with a feature rate of 0.6 only 0.15. Two-place event types carry the literals of their features beside their own, so preconditions bear on them more, and feature literals most of all: a literal on a feature blocks every event type of the branch.
- **The scene size** moves the share as the number of ordered pairs against the number of participants: small scenes of 2 to 4 participants give 0.27, large ones of 7 to 11 give 0.59. **`events_per_step`** barely moves it (0.43 at 0.5 and 0.38 at 4): the policy draws one event at a time among what is legal, and more draws mostly add one-place events once the few able pairs are taken.
- **`event_type_weights`** moves the share directly under `uniform_event`: two-place event types weighed 3 give 0.64, one-place event types weighed 3 give 0.21.

## The conformance fixtures

The Rust runtime, when it is built, must give the same results as the Python runtime. The conformance fixtures in `tests/fixtures/world/` are its tests, written now. Each fixture is one JSON file with a complete definition inline, the entities' static facts, the initial state, and a list of steps; each step gives its events and what must come out: the legal bindings of every event type before the step, and the state and the derived fluents after it. A fixture can end in an error that `apply` must raise, an illegal event or interfering events.

Two kinds are committed: seventeen hand-written fixtures over a hand world of three entities, with every expected value worked out by hand (a fluent with no event; an effect that sets, clears, or leaves a fluent; a derived fluent that changes; a precondition that blocks; two events that do not interfere; the kinds of interference; and, since stage b1, a static condition that holds and one that fails, a condition on a derived fluent that holds and one that fails, and interference through a condition), and six generated from the tiny world, with expected values from a brute-force evaluator that reads the truth tables and never the matrices. `check-fixtures` runs every fixture through the runtime and then through the evaluator, so the two implementations check each other. `tests/fixtures/world/README.md` is written for whoever builds the Rust runtime: the fixture format, the definition record, the canonical JSON behind the rule-set identity, how literals and layers are evaluated, and the semantics of each operation.

## Using the package from Python

```python
from semantic_world.world.config import load_config
from semantic_world.world.generate import define

config = load_config("data/world/tiny.yaml", seed=7)
result = define(config)
result.write("runs/world/tiny_seed7")
```

`result` holds everything the run produced: `result.taxonomy` (the embedded taxonomy run), `result.statics` (the static side: the one-place requirements, the event-type tree, the relations, the capacities, and the relation statistics), `result.fluents`, `result.event_types` (every event type with its requirement, precondition, and effects), `result.definition` (the definition, with `result.rule_set_id`), `result.stats` (the contents of `world_stats.yaml`), and `result.warnings`. `config_from_mapping` in the same module builds a configuration from a dictionary, for sweeping a setting in code.

The runtime works on a loaded definition:

```python
from semantic_world.world.definition import load_definition
from semantic_world.world.runtime import apply, derive, initial_state, legal_bindings

definition = load_definition("runs/world/tiny_seed1")
state = initial_state(definition)
legal = legal_bindings(definition, state, "EVENTTYPE2.1.2")   # every legal binding, as pairs of entity indices
```

`derive(definition, state, entities)` gives the derived facts, `able(definition, event_type, bindings)` and `legal(definition, state, event_type, bindings)` judge bindings, `legal_bindings` lists every legal binding among given entities in a fixed order, and `apply(definition, state, events)` gives the state after one step, or raises an `IllegalEventError` or an `InterferenceError` that names the event and the reason. The runtime draws no random number and chooses no event. Episodes are run by `semantic_world.world.episodes.simulate(folder, episodes, seed, settings)`, or by an `EpisodeGenerator` over a definition in memory; `semantic_world.world.history.replay(definition, history)` checks a history against the runtime and returns its states.

## Recipes

- **Several worlds with the same settings.** Run the same configuration with `--seed 1`, `--seed 2`, and so on. The two-place share and the densities vary with the seed, as the table above shows.
- **A world without conditions.** `event_types: {effects: {conditional_share: 0}}` gives the unconditional world of phase (a), with the same definition and identity as before stage b1; `conditional_share: 1` makes every effect conditional.
- **A world without change.** `fluents: {count: 0}`. Every able binding is legal at every time point, histories record no change, and `scene.policy: uniform_event_type` in the corpus then gives the scenes of the corpus's old scene generator.
- **A known chain of events.** Write an event file (above), and name it under `event_types.event_file`. `simulate` on the world shows the chain's events in the order the preconditions force.
- **More two-place events.** Lower `event_types.binary.features.expected_true`, or weigh the two-place event types in `scene.event_type_weights` of the corpus configuration; see "The two-place share" for every lever.
- **A model's input.** `view --preset classic` for the columns of the old `instances.csv`; `view --preset base --include ISA` for the base vector with the categories. Join `derived/capacities.csv` on `label` for the two-place capacities.
- **Legality apart from occurrence.** `simulate --legal` records, at every step, how many bindings of each event type were legal, beside the events that happened.

## Troubleshooting

- **"the preconditions of ... were drawn again".** A note, not an error: the event type was never legal in the statistics episodes, so its own precondition literals were redrawn. A warning names an event type that is still never legal after the redraws; it still exists, and the corpus never reports an event of it.
- **"never able".** An event type whose requirement holds for no entity or no pair gets no events and no word. The world keeps it; another seed, or looser `event_types.unary.rules`, gives another requirement.
- **A derived file is refused.** The file's rule-set identity differs from the definition's. The run folder holds files of two runs; regenerate it.
- **The agreement test fails.** The error names the rule and the first disagreement. The two forms of one rule set must agree, so this is a bug in the generator, not in the configuration. Keep the configuration and seed and report it.
- **An event file entry is refused.** The error names the file and the entry: an effect on a derived fluent, a requirement that reads a fluent, a contradiction, a literal that nothing can make true, or a condition that breaks a rule (no able binding satisfies it, or it reads the effect's own fluent or a fluent the precondition fixes).
- **`taxonomy.config` is not found.** Paths in a configuration are read from the folder the command runs in. Run from the root of the repository, as the examples do.

## Reference

- `docs/specs/WORLD_AND_LANGUAGE.md`: the design, with "Terms", "Architecture", "The world definition", "Phase (a): the world", "Phase (a): determinism", and the build stages.
- `tests/fixtures/world/README.md`: the fixture format, the definition record, the canonical JSON, and the semantics of the runtime, for the Rust runtime.
- `docs/proposals/2026-10-07-world-stage-a1-decisions.md` to `2026-10-09-world-stage-a8-decisions.md`: the engineering choices of each stage, and Jon's rulings on them.
- `docs/DECISIONS.md`, "World model": WM.1 to WM.34 and WM.E1 and following.
- `docs/guides/TAXONOMY.md`: the taxonomy inside the world, and "Where CAN features and verbs went". `docs/guides/CORPUS.md`: the documents written about a world.
