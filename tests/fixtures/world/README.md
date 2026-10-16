# Conformance fixtures for the world runtime

This folder holds the conformance fixtures of `docs/specs/WORLD_AND_LANGUAGE.md` ("Conformance fixtures"). Each fixture is one JSON file. The Python runtime passes every fixture (`python -m semantic_world.world check-fixtures`), and the Rust runtime must pass the same files. This note is written for whoever builds the Rust runtime: the fixture format, the exact rules of the canonical JSON behind the rule-set identity, and the semantics of each operation. The Python reference is `python/semantic_world/world/runtime.py`; the brute-force reference, which reads truth tables and never the matrices, is `BruteForce` in `python/semantic_world/world/fixtures.py`.

Two kinds of fixture are here. The `hand_*` files are hand-written over one small world of three entities, with every expected value worked out by hand; `tests/world/hand_world.py` holds that world and the cases, and assembles the files. Since stage b1 the hand world has two event types with conditional effects, and `hand_13` to `hand_17` show a static condition that holds and one that fails, a fluent condition (on a derived fluent) that holds and one that fails, and interference through a condition. Since stage b2 the hand world has two numeric fluents (one of which drifts) and three more event types (a numeric effect, a numeric precondition literal, a numeric condition): `hand_18` to `hand_20` are the comparison fixtures (ties, `ENTITY` as the comparison class, a category mean as an argument), and `hand_21` to `hand_27` the numeric fixtures (an effect, the drift, clamping at both ends, an effect and the drift in one step, a precondition at its threshold, a numeric condition, and interference through a numeric fluent), and `hand_28` shows the `RANGE` comparisons at the cut itself (a value exactly at the fraction `q` is not `above` and is `below`). The `tiny_*` files are generated from `data/world/tiny.yaml` (`python -m semantic_world.world make-fixtures data/world/tiny.yaml`), with expected values from the brute-force evaluator, numeric fluents included; regenerating them gives the committed files byte for byte. `check-fixtures` runs every file through the runtime and then through the brute-force evaluator, so the two implementations check each other on every fixture.

## The fixture file

| Key | Contents |
| --- | --- |
| `version` | The fixture format's version: 1. |
| `name` | The file's stem. |
| `description` | What the fixture shows. |
| `rule_set_id` | The rule-set identity of the inline definition. A checker recomputes it from `definition` and refuses the fixture when they differ. |
| `definition` | A complete world definition, in the form of `definition.json`. Its own `rule_set_id` is the same identity. |
| `entities` | The entities' static facts: one object per entity with `label`, `leaf`, every free feature (0 or 1), and every scalar (a float). These are the columns of `entities.csv` without the fluent columns, Boolean and numeric. |
| `initial` | The state at `TIME.1`: for every entity label, the list of base Boolean fluents that are true. A fluent not listed is false. Every entity is listed. |
| `initial_numeric` | Since stage b2, and only in a world with numeric fluents: for every entity label, an object with every numeric fluent's value at `TIME.1` (a float). |
| `steps` | The steps, in order. Step k (counted from 1) takes the state at `TIME.k` to the state at `TIME.k+1`. |
| `error` | `null`, or `{"step": k, "kind": "illegal" | "interference"}`: `apply` must raise that kind of error at step k. The error step is always the last step. |
| `comparisons` | Since stage b2, optional: a list of comparisons with their expected values, judged after the steps (below). |

Each step holds:

| Key | Contents |
| --- | --- |
| `events` | The events of the step: `{"event_type": label, "binding": {role: entity label}}`, in the order they are given to `apply`. |
| `legal` | For every event type of kind `event_type` (never a category), the legal bindings in the state before the step, as `{role: entity label}` objects, in the fixed order below. Every event type is listed, with `[]` when none is legal. |
| `derived` | After the step: for every entity label, the derived fluents that are true, in symbol order. Absent on the error step. |
| `state` | After the step: for every entity label, the base Boolean fluents that are true, in symbol order. Absent on the error step. |
| `numeric` | After the step, in a world with numeric fluents: for every entity label, every numeric fluent's value, as `{"NUMFL.1": 0.75, ...}`. Absent on the error step. |

A checker runs, for each step: `legal_bindings` for every event type and compares with `legal`, as ordered lists; then `apply` with the step's events. On the error step `apply` must raise the named kind of error. Otherwise the new state must equal `state`, every numeric value must equal `numeric` exactly (the same float64, by the arithmetic under "The order within a step" below), and `derive` on the new state must equal `derived`.

Each entry of `comparisons` is `{"kind": ..., "dimension": ..., "x": ..., "expected": true | false}` with the fields of its kind. `kind` is `greater` (with `y`), `above` or `below` (with `class` and `z`), or `increase` or `decrease` (with `time`). `dimension` is a scalar (`SCALARDIM.1`) or a numeric fluent (`NUMFL.1`). An argument `x` or `y` is an entity label, or `{"mean": category}` for the mean of a category's members (`{"mean": "ENTITY"}` for all entities). A comparison of a numeric fluent carries `time`: the time point `TIME.<time>` whose state it is judged in (1 is the initial state, k+1 the state after step k); `increase` and `decrease` compare the values at `TIME.<time>` and `TIME.<time + 1>`. See "Comparisons" below for the semantics.

## The definition record

`definition` has the keys `version` (1), `rule_set_id`, `symbols`, `literals`, `rules`, `layers`, and `event_types`. The four tables `symbols`, `literals`, `rules`, and `event_types` are hashed; `layers` and `version` are not.

- **Symbols.** `label`, `kind` (`property`, `part`, `scalar`, `fluent`, `numeric_fluent`, `event_type`, `event_type_category`), `derived`, `fluent`, `arity`. A static feature is a `property` or `part` symbol; it is free when `derived` is false. A fluent symbol is base when `derived` is false, and carries `initial_rate`: the rate at which an entity's initial value of a base fluent is true (a float between 0 and 1, part of the hashed tables), or `null` for a derived fluent. The runtime draws at these rates when an episode redraws its initial state; the fixtures give every initial state outright. A `numeric_fluent` symbol (stage b2) is a base fluent with a real value, and carries `range` (`[low, high]`, two floats with `low < high`), `initial` (the distribution of an entity's initial value: `{"kind": "uniform"}` over the range, `{"kind": "fixed", "value": v}`, or `{"kind": "beta", "mean": m, "concentration": c}`, with `v` and `m` floats inside the range and `c` a float above 0), and `drift` (a float: the change per step, 0.0 for none), all part of the hashed tables. Other symbols have none of these keys.
- **Literals.** `index` (the position in the table), `key` (a readable name, not used by the runtime), `kind`, `role`, and the fields of the kind. `feature`: `role` (`null`, `agent`, or `patient`) and `feature`. `threshold`: `role`, `scalar`, `operator` (`>`), `threshold` (float). `fluent`: `role` `null` and `fluent`. `numeric_threshold` (stage b2): `role` `null`, `fluent` (a numeric fluent), `operator` (`>`), `threshold` (float); read by the rules of derived fluents. `comparison`: `role` `binding`, `agent_scalar`, `patient_scalar`, `operator` (`>`), `low` (float), `high` (float or `null`). `constraint`: `role` `binding` and `constraint` (the output of a binding rule). Literals with `role` `null` belong to the entity scope; the others to the binding scope.
- **Rules.** `output`, `scope` (`entity` or `binding`), `family`, `inputs` (literal indices, in truth-table order), `truth_table` (a bit string of length 2 to the number of inputs), `expression`. Bit i of the table is the output for the input setting whose binary representation is i, with the first input as the most significant bit. An entity rule computes a derived static feature or a derived fluent; a binding rule computes a constraint or a requirement.
- **Layers.** One entry per layer: `scope`, `layer` (1 and up), `terms`, `outputs`. A term is `literals` (indices into the literal table), `complemented` (one flag per literal), `threshold` (the number of literals). An output is `output`, `terms` (indices into the layer's `terms`), `threshold` (always 1).
- **Event types.** `label`, `kind`, `arity`, `roles` (`["agent"]` or `["agent", "patient"]`), `parent`, `level`, `features`, `explicit`, `requirement` (`constraints`, `output`: the binding rule that is the requirement, `expression`), `precondition` (`literals`, each a Boolean literal `role`, `fluent`, `value`, or, since stage b2, a numeric threshold literal `role`, `fluent` (a numeric fluent), `operator` (`>` or `<=`), `threshold` (a float strictly inside the fluent's range); and `expression`), `effects` (each a Boolean effect `role`, `fluent`, `value`, `expression`, or, since stage b2, a numeric effect `role`, `fluent` (a numeric fluent), `operator` (`+=` or `-=`), `amount` (a float above 0), `expression`; and, since stage b1, `condition` when the effect has one). Effects write base fluents only. A category (`event_type_category`) has no events. A reader tells the kinds apart by their keys: a literal with a `threshold` is numeric, an effect with an `amount` is numeric.
- **Conditions (stage b1).** An effect's `condition` is `{"literals": [...], "expression": "..."}`: a conjunction of one or two literals, each `role` (a role of the event type), `kind` (`feature` for a static feature, `fluent` for a Boolean fluent, base or derived, or, since stage b2, `numeric` for a numeric threshold), `symbol` (the feature's or the fluent's label), and `value` (a `feature` or `fluent` literal) or `operator` (`>` or `<=`) and `threshold` (a `numeric` literal). The effect's `expression` is then `when <condition> then <assignment>`. An effect without a condition has no `condition` key and is written exactly as before stage b1, so the identity of a world without conditions is unchanged; a reader must treat a missing key as no condition. A condition never names a symbol twice, never names the Boolean fluent the effect sets (a numeric condition on the effect's own numeric fluent is allowed: "when low, add"), and never names a fluent of a role that the precondition names.

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
- **Which numbers are floats.** In the hashed tables: `threshold` of a threshold literal and of a numeric threshold literal, `low` and `high` of a comparison, `initial_rate` of a base fluent, and, since stage b2, the `range` ends, the `initial` distribution's `value`, `mean`, and `concentration`, and the `drift` of a numeric fluent symbol, the `threshold` of a numeric precondition or condition literal, and the `amount` of a numeric effect. Every other number is an integer (`index`, `arity`, `level`, the term thresholds of the unhashed layers). A reader must keep the two apart: `1` and `1.0` hash differently.

The definition files on disk (`definition.json`, and `definition` inside a fixture) are the same values in an indented form that keeps the key order as written. Only the canonical form is hashed.

## Literal values

A literal's value for an entity (entity scope) or a binding (binding scope) is 0 or 1:

- `feature`: the feature's value for the entity of the role (the entity itself when `role` is `null`). A derived feature's value comes from its rule.
- `threshold`: 1 when `scalar > threshold` for the entity of the role, strictly.
- `fluent`: the fluent's value for the entity itself, in the state (a base fluent) or from its rule (a derived fluent).
- `numeric_threshold`: 1 when the numeric fluent's value for the entity itself, in the state, is `> threshold`, strictly (stage b2).
- `comparison`: 1 when `agent_scalar of the agent - patient_scalar of the patient > low`, and, when `high` is not `null`, also `< high`, both strictly.
- `constraint`: the output of the named binding rule for the binding.

A one-place event type's requirement reads literals of the agent only. The runtime may leave patient and comparison literals at 0 for a one-place binding.

## Evaluating the layers (REL.18)

For a scope, build the literal vector of every entity or binding: the values of the base literals, with 0 for every literal that a rule of the scope computes. Then take the scope's layers in increasing `layer` order. In a layer, a term is true when the number of its literals that hold (the literal's value, or 1 minus it when complemented) is at least its threshold, so a term with no literals is true. An output is true when at least one of its terms is true, so an output with no terms is false. After a layer, write each output's value into the literal column of the same key, when the literal table has one, so later layers read it. The entity scope is evaluated before the binding scope, because binding literals read derived static features.

The truth table and the matrix form of every rule agree on every entity, every ordered pair of entities, and every input setting of rules with at most 12 inputs; the checker verifies this on the fixture's inline definition before it runs the steps.

## The operations

A state (stage b2) holds, beside every entity's base Boolean fluents, every entity's value of each numeric fluent: a float64 inside the fluent's range.

- **`derive(definition, state, entities)`**: the derived static features and the derived fluents of the given entities in the state. Static features depend on the entities' free features and scalars only, so they can be computed once per definition and cached. Derived fluents are recomputed from the state at every time point and never carried over; a rule that reads a numeric threshold literal reads the state's numeric values.
- **`able(definition, event_type, bindings)`**: for each binding, the value of the event type's requirement (the binding rule named by `requirement.output`). A binding is `(agent)` or `(agent, patient)`; a binding never uses one entity twice.
- **`legal(definition, state, event_type, bindings)`**: `able` and, for every precondition literal, the fluent's value for the entity of the literal's role equals the literal's `value` (a Boolean literal), or the numeric fluent's value is `> threshold` for the operator `>` and `<= threshold` for `<=`, with `>` strict (a numeric literal, stage b2). An empty precondition holds.
- **`legal_bindings(definition, state, event_type, entities)`**: every legal binding among the given entities, ordered by agent, then patient, in the order of the entities table. For a two-place event type the candidates are every ordered pair of distinct entities. A category of event types has no bindings.
- **`apply(definition, state, events)`**: the state after one step. (1) Every event must be legal in the state; otherwise the error kind is `illegal`. An event whose event type is unknown or a category, whose binding has the wrong number of entities, names an unknown entity, or names one entity twice, is illegal too. A condition never bears on legality. (2) The events must not interfere; otherwise the error kind is `interference`. Two events interfere when both write the same fluent of the same entity (whatever the values, and whatever the effects' conditions: every effect counts as a write of its fluent, whether or not its condition holds in the state; a numeric effect writes its numeric fluent), or when one writes a base fluent, Boolean or numeric, that the other's precondition, or a condition of one of the other's effects, reads. A precondition or a condition that reads a derived fluent reads every base fluent of its cone: every base fluent the derived fluent's rule reads, directly or through other derived fluents, numeric fluents read through numeric threshold literals included; a numeric literal reads its numeric fluent. A condition literal on a static feature reads nothing that an event can write. The same event (same event type and binding) twice in a step is interference. (3) Every effect whose condition holds in the state (the state at `TIME.k`, before any effect of the step; an effect without a condition always holds) sets its Boolean fluent of the entity of its role to its value, even when the value is unchanged, or adds its constant to its numeric fluent of the entity of its role (`+=`) or subtracts it (`-=`), the sum clamped to the fluent's range. An effect whose condition fails does nothing. A condition's literal holds when the symbol has the literal's value for the entity of the literal's role: a static feature from the static facts, a base fluent from the state, a derived fluent from its rule in the state, a numeric threshold from the state's numeric value as a precondition literal is judged. (4) Every base fluent that no fired effect wrote keeps its value; then every numeric fluent of every entity moves by its drift, clamped to the range (see "The order within a step"). Applying the events together equals applying them one at a time in any order, the drift added once, after all of them. Legality is checked for every event before interference is checked for any pair; the Python runtime's error message names the event or events and the reason.

The runtime draws no random number and chooses no event. The same definition, state, and events always give the same result.

## The order within a step (stage b2)

For each entity and each numeric fluent, with `v` the value at `TIME.k`, `e` the constant of the one effect that wrote the fluent in the step (signed: positive for `+=`, negative for `-=`; non-interference allows at most one, and an effect whose condition failed wrote nothing), and `d` the fluent's drift, the value at `TIME.k+1` is

```
clamp(clamp(v + e) + d)
```

where `clamp(x)` is `min(max(x, low), high)` over the fluent's range, every operation in IEEE 754 float64 (binary64) with round to nearest, ties to even, in exactly this order: the addition of the effect, the first clamp, the addition of the drift, the second clamp. Without an effect the value is `clamp(v + d)`. The histories and the fixtures record the results of exactly this arithmetic, and a runtime must reproduce them bit for bit: `v + e` and `v + d` are single additions of two float64 values, with no fused operation, no extended precision, and no reordering. The Python reference is `step_value` in `python/semantic_world/world/numeric.py`.

**The float format.** Every float of a fixture (the `numeric` values after a step, `initial_numeric`, the `threshold`, `amount`, `range`, `initial`, and `drift` fields of the definition) is written with 17 significant digits, as "Canonical JSON" says, which reads back to the same float64. The histories (`episodes.jsonl`, `scenes.jsonl`) write their floats in Python's shortest round-trip form (`repr`), which also reads back exactly; a reader that parses JSON floats correctly rounded gets the same values either way.

## Comparisons (stage b2)

The comparisons of "Phase (b): the world" are judged on demand by direct arithmetic on the values, in float64, and never stored. A fixture's `comparisons` list gives their expected values. The Python reference is `greater`, `above`, `below`, `increase`, and `decrease` in `runtime.py`; the brute-force reference is `BruteForce.compare`.

- **Members of a category.** The members of a category are the entities whose `leaf` is the category or below it, which the labels give: `CATEGORY.1.3.2` is below `CATEGORY.1.3` and `CATEGORY.1`, because a category's label is a prefix of its descendants' labels followed by a period. `ENTITY` names every entity.
- **An argument's value** on a dimension is the entity's value (a scalar from the entities table, a numeric fluent from the state at the time point), or, for `{"mean": category}`, the arithmetic mean of the members' values (the sum in entity order, divided by the count, in float64).
- **`greater(dimension, x, y)`**: the value of `x` is strictly greater than the value of `y`. Two equal values give false both ways, so no entity is greater than itself, and a category mean equal to a member's value is not greater than it.
- **`above(dimension, x, class, z)`** on a scalar: the value of `x` is at least `mean + z * sd`, where `mean` and `sd` are the mean and the population standard deviation (the square root of the mean of the squared deviations, not the sample estimate) of the scalar over the members of `class`, a category or `ENTITY`; a class with no spread (`sd` 0) has nothing above or below it. `derived/comparison_classes.csv` of a world run holds the same `instances`, `mean`, and `sd` for every category and `ENTITY`, to six decimals. **`below`**: at most `mean - z * sd`.
- **`above(fluent, x, "RANGE", q)`** on a numeric fluent, at a time point: the value is strictly greater than `low + q * (high - low)`, the point of the fluent's range at the fraction `q`; **`below`**: at most that point (the value is `<= point`). The two are a numeric threshold literal and its negation: `above` at `q` is `> point` and `below` at `q` is `<= point`, so a value exactly at the point is not above and is below. (Until October 15, 2026, `above` was inclusive, `>=`; Jon's ruling WM.35 made it strict.) No comparison class is involved.
- **`increase(fluent, x, time)`**: the entity's value at `TIME.<time + 1>` is strictly greater than at `TIME.<time>`; **`decrease`**: strictly less. The drift alone can make either true.

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

Each event of a step is `label` (`SCENE.<n>.EVENTINSTANCE.<k>`, numbered within the episode in time order and within a step in the order the events were drawn), `type` (the event type), `agent`, `patient` (two-place event types only), and `changes`: the base fluents the event's effects changed, each a Boolean change `{"entity", "fluent", "to"}` with `to` the new value, or, since stage b2, a numeric change `{"entity", "fluent", "from", "to", "cause"}` with `from` and `to` the values before and after the effect (after the effect and before the drift: `from` is the value at `TIME.k`, `to` is `clamp(v + e)`) and `cause` the event's own label. An effect that sets a Boolean fluent to the value it already has records no change, a numeric effect that leaves the value where it was (clamped at the end of the range) records none, and an effect whose condition failed in the step's starting state records none either (stage b1). A derived fluent's change is never recorded.

Since stage b2 a world with numeric fluents also writes `initial_numeric` after `initial` (for each participant, every numeric fluent's value at `TIME.1`, as `{"NUMFL.1": 0.4, ...}`), and each step whose drift changed some participant's value writes `drift` after `events`: the numeric changes the drift made to the participants, each `{"entity", "fluent", "from", "to", "cause": "drift"}` with `from` the value after the effects (`clamp(v + e)`, or `v` when no effect wrote it) and `to` the value at `TIME.k+1`. A step without such changes has no `drift` key, and a world without numeric fluents writes neither key, so its histories are as before. Every numeric change of a step is therefore recorded with its cause, an event's label or `drift`, and a reader can rebuild every participant's numeric values at every time point from `initial_numeric` and the changes alone, without the drift settings.

`legal`, when present, maps every event type that has events to the number of its legal bindings among the participants in the state before the step.

A history replays when, from its initial state (`initial` and `initial_numeric` for the participants, the entities table for the rest), every step's events are legal and free of interference under `apply`, every event's recorded Boolean changes are exactly the base Boolean fluents its effects changed, every event's recorded numeric changes are exactly its numeric effects' changes with the values of "The order within a step" reproduced bit for bit, every step's `drift` is exactly the drift's changes to the participants, and `final` names the time point after the last step. Events of one step are applied together; because they do not interfere, applying them one at a time in any order gives the same state, the drift being added once after all of them.
