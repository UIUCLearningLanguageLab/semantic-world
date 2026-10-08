# Conformance fixtures for the world runtime

This folder holds the conformance fixtures of `docs/specs/WORLD_AND_LANGUAGE.md` ("Conformance fixtures"). Each fixture is one JSON file. The Python runtime passes every fixture (`python -m semantic_world.world check-fixtures`), and the Rust runtime must pass the same files. This note is written for whoever builds the Rust runtime: the fixture format, the exact rules of the canonical JSON behind the rule-set identity, and the semantics of each operation. The Python reference is `python/semantic_world/world/runtime.py`; the brute-force reference, which reads truth tables and never the matrices, is `BruteForce` in `python/semantic_world/world/fixtures.py`.

Two kinds of fixture are here. The `hand_*` files are hand-written over one small world of three entities, with every expected value worked out by hand; `tests/world/hand_world.py` holds that world and the cases, and assembles the files. The `tiny_*` files are generated from `data/world/tiny.yaml` (`python -m semantic_world.world make-fixtures data/world/tiny.yaml`), with expected values from the brute-force evaluator; regenerating them gives the committed files byte for byte. `check-fixtures` runs every file through the runtime and then through the brute-force evaluator, so the two implementations check each other on every fixture.

## The fixture file

| Key | Contents |
| --- | --- |
| `version` | The fixture format's version: 1. |
| `name` | The file's stem. |
| `description` | What the fixture shows. |
| `rule_set_id` | The rule-set identity of the inline definition. A checker recomputes it from `definition` and refuses the fixture when they differ. |
| `definition` | A complete world definition, in the form of `definition.json`. Its own `rule_set_id` is the same identity. |
| `entities` | The entities' static facts: one object per entity with `label`, `leaf`, every free feature (0 or 1), and every scalar (a float). These are the columns of `entities.csv` without the fluent columns. |
| `initial` | The state at `TIME.1`: for every entity label, the list of base fluents that are true. A fluent not listed is false. Every entity is listed. |
| `steps` | The steps, in order. Step k (counted from 1) takes the state at `TIME.k` to the state at `TIME.k+1`. |
| `error` | `null`, or `{"step": k, "kind": "illegal" | "interference"}`: `apply` must raise that kind of error at step k. The error step is always the last step. |

Each step holds:

| Key | Contents |
| --- | --- |
| `events` | The events of the step: `{"event_type": label, "binding": {role: entity label}}`, in the order they are given to `apply`. |
| `legal` | For every event type of kind `event_type` (never a category), the legal bindings in the state before the step, as `{role: entity label}` objects, in the fixed order below. Every event type is listed, with `[]` when none is legal. |
| `derived` | After the step: for every entity label, the derived fluents that are true, in symbol order. Absent on the error step. |
| `state` | After the step: for every entity label, the base fluents that are true, in symbol order. Absent on the error step. |

A checker runs, for each step: `legal_bindings` for every event type and compares with `legal`, as ordered lists; then `apply` with the step's events. On the error step `apply` must raise the named kind of error. Otherwise the new state must equal `state`, and `derive` on the new state must equal `derived`.

## The definition record

`definition` has the keys `version` (1), `rule_set_id`, `symbols`, `literals`, `rules`, `layers`, and `event_types`. The four tables `symbols`, `literals`, `rules`, and `event_types` are hashed; `layers` and `version` are not.

- **Symbols.** `label`, `kind` (`property`, `part`, `scalar`, `fluent`, `event_type`, `event_type_category`), `derived`, `fluent`, `arity`. A static feature is a `property` or `part` symbol; it is free when `derived` is false. A fluent symbol is base when `derived` is false.
- **Literals.** `index` (the position in the table), `key` (a readable name, not used by the runtime), `kind`, `role`, and the fields of the kind. `feature`: `role` (`null`, `agent`, or `patient`) and `feature`. `threshold`: `role`, `scalar`, `operator` (`>`), `threshold` (float). `fluent`: `role` `null` and `fluent`. `comparison`: `role` `binding`, `agent_scalar`, `patient_scalar`, `operator` (`>`), `low` (float), `high` (float or `null`). `constraint`: `role` `binding` and `constraint` (the output of a binding rule). Literals with `role` `null` belong to the entity scope; the others to the binding scope.
- **Rules.** `output`, `scope` (`entity` or `binding`), `family`, `inputs` (literal indices, in truth-table order), `truth_table` (a bit string of length 2 to the number of inputs), `expression`. Bit i of the table is the output for the input setting whose binary representation is i, with the first input as the most significant bit. An entity rule computes a derived static feature or a derived fluent; a binding rule computes a constraint or a requirement.
- **Layers.** One entry per layer: `scope`, `layer` (1 and up), `terms`, `outputs`. A term is `literals` (indices into the literal table), `complemented` (one flag per literal), `threshold` (the number of literals). An output is `output`, `terms` (indices into the layer's `terms`), `threshold` (always 1).
- **Event types.** `label`, `kind`, `arity`, `roles` (`["agent"]` or `["agent", "patient"]`), `parent`, `level`, `features`, `explicit`, `requirement` (`constraints`, `output`: the binding rule that is the requirement, `expression`), `precondition` (`literals`: each `role`, `fluent`, `value`; and `expression`), `effects` (each `role`, `fluent`, `value`, `expression`). Effects write base fluents only. A category (`event_type_category`) has no events.

## Canonical JSON and the rule-set identity

The rule-set identity is the SHA-256 hash, as 64 lowercase hexadecimal digits, of the UTF-8 bytes of the canonical JSON of this document:

```
{"event_types": <the event_types table>, "literals": <the literals table>, "rules": <the rules table>, "symbols": <the symbols table>}
```

The tables are copied from the definition exactly, every entry and every field. The canonical JSON is:

- **Objects.** Keys sorted by their UTF-16 code units (for ASCII keys, byte order). No whitespace anywhere: `{"a":1,"b":[1,2]}`.
- **Strings.** Every character outside printable ASCII is escaped: `"` as `\"`, `\` as `\\`, newline, carriage return, tab, backspace, and form feed as `\n`, `\r`, `\t`, `\b`, `\f`, every other control character and every non-ASCII character as `\uXXXX` with lowercase hex digits (characters above the basic plane as a surrogate pair). Every label in a definition is ASCII, so in practice no escape occurs.
- **Integers** in decimal with no exponent. **Booleans** as `true` and `false`. **Null** as `null`.
- **Floats** with 17 significant digits, as C's `printf("%.17g")` writes them: rounded to 17 significant digits, trailing zeros removed, and the decimal point removed when nothing follows it; fixed notation when the decimal exponent is between -4 and 16 inclusive, otherwise scientific notation with one digit before the point, `e`, a sign, and at least two exponent digits. Then `.0` is appended when the text has neither a point nor an `e`, so a float is never written like an integer. Examples: `0.5`, `-0.58220000000000005`, `1.0000000000000001e-05`, `1e+17`, `123456789.0`, `-0.0`. NaN and infinities never occur.
- **Which numbers are floats.** In the hashed tables: `threshold` of a threshold literal, and `low` and `high` of a comparison. Every other number is an integer (`index`, `arity`, `level`, the term thresholds of the unhashed layers). A reader must keep the two apart: `1` and `1.0` hash differently.

The definition files on disk (`definition.json`, and `definition` inside a fixture) are the same values in an indented form that keeps the key order as written. Only the canonical form is hashed.

## Literal values

A literal's value for an entity (entity scope) or a binding (binding scope) is 0 or 1:

- `feature`: the feature's value for the entity of the role (the entity itself when `role` is `null`). A derived feature's value comes from its rule.
- `threshold`: 1 when `scalar > threshold` for the entity of the role, strictly.
- `fluent`: the fluent's value for the entity itself, in the state (a base fluent) or from its rule (a derived fluent).
- `comparison`: 1 when `agent_scalar of the agent - patient_scalar of the patient > low`, and, when `high` is not `null`, also `< high`, both strictly.
- `constraint`: the output of the named binding rule for the binding.

A one-place event type's requirement reads literals of the agent only. The runtime may leave patient and comparison literals at 0 for a one-place binding.

## Evaluating the layers (REL.18)

For a scope, build the literal vector of every entity or binding: the values of the base literals, with 0 for every literal that a rule of the scope computes. Then take the scope's layers in increasing `layer` order. In a layer, a term is true when the number of its literals that hold (the literal's value, or 1 minus it when complemented) is at least its threshold, so a term with no literals is true. An output is true when at least one of its terms is true, so an output with no terms is false. After a layer, write each output's value into the literal column of the same key, when the literal table has one, so later layers read it. The entity scope is evaluated before the binding scope, because binding literals read derived static features.

The truth table and the matrix form of every rule agree on every entity, every ordered pair of entities, and every input setting of rules with at most 12 inputs; the checker verifies this on the fixture's inline definition before it runs the steps.

## The operations

- **`derive(definition, state, entities)`**: the derived static features and the derived fluents of the given entities in the state. Static features depend on the entities' free features and scalars only, so they can be computed once per definition and cached. Derived fluents are recomputed from the state at every time point and never carried over.
- **`able(definition, event_type, bindings)`**: for each binding, the value of the event type's requirement (the binding rule named by `requirement.output`). A binding is `(agent)` or `(agent, patient)`; a binding never uses one entity twice.
- **`legal(definition, state, event_type, bindings)`**: `able` and, for every precondition literal, the fluent's value for the entity of the literal's role equals the literal's `value`. An empty precondition holds.
- **`legal_bindings(definition, state, event_type, entities)`**: every legal binding among the given entities, ordered by agent, then patient, in the order of the entities table. For a two-place event type the candidates are every ordered pair of distinct entities. A category of event types has no bindings.
- **`apply(definition, state, events)`**: the state after one step. (1) Every event must be legal in the state; otherwise the error kind is `illegal`. An event whose event type is unknown or a category, whose binding has the wrong number of entities, names an unknown entity, or names one entity twice, is illegal too. (2) The events must not interfere; otherwise the error kind is `interference`. Two events interfere when both write the same fluent of the same entity (whatever the values), or when one writes a base fluent that the other's precondition reads. A precondition that reads a derived fluent reads every base fluent of its cone: every base fluent the derived fluent's rule reads, directly or through other derived fluents. The same event (same event type and binding) twice in a step is interference. (3) Every effect sets its fluent of the entity of its role to its value, even when the value is unchanged. (4) Every base fluent that no effect wrote keeps its value. Applying the events together equals applying them one at a time in any order. Legality is checked for every event before interference is checked for any pair; the Python runtime's error message names the event or events and the reason.

The runtime draws no random number and chooses no event. The same definition, state, and events always give the same result.

## Histories

A history is the record of one episode: one JSON object, written as one line of a JSON lines file (`episodes.jsonl` from `python -m semantic_world.world simulate`, `scenes.jsonl` from the corpus). The 3D engine writes the same schema. The Python reference is `python/semantic_world/world/history.py`, and `replay` there checks a history against the runtime.

| Key | Contents |
| --- | --- |
| `label` | `SCENE.<n>`, the episode's label. |
| `seed` | The seed instance: the first participant. |
| `participants` | The entity labels, the seed first, then the others in the order they were drawn. |
| `policy` | The name of the selection policy that chose the events. |
| `rule_set_id` | The rule-set identity of the definition the episode ran on. A history replays only on that definition. |
| `initial` | For each participant, the base fluents true at `TIME.1`, in symbol order. Entities that are not participants keep the definition's initial values. |
| `steps` | One object per step, in order: `step` (counted from 1: step k takes `TIME.k` to `TIME.k+1`), `events`, and, when the run recorded legality, `legal`. |
| `final` | The last time point, `TIME.<number of steps + 1>`. |
| `quiescent` | `true` when the episode ended because no event was legal at its last time point. |

Each event of a step is `label` (`SCENE.<n>.EVENTINSTANCE.<k>`, numbered within the episode in time order and within a step in the order the events were drawn), `type` (the event type), `agent`, `patient` (two-place event types only), and `changes`: the base fluents the event's effects changed, each `{"entity", "fluent", "to"}` with `to` the new value. An effect that sets a fluent to the value it already has records no change. A derived fluent's change is never recorded. Numeric fluents (phase (b)) will add their changes, with their causes, to the same list.

`legal`, when present, maps every event type that has events to the number of its legal bindings among the participants in the state before the step.

A history replays when, from its initial state, every step's events are legal and free of interference under `apply`, every event's recorded changes are exactly the base fluents its effects changed, and `final` names the time point after the last step. Events of one step are applied together; because they do not interfere, applying them one at a time in any order gives the same state.
