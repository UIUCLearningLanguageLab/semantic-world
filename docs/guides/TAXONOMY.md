# The taxonomy generator

The taxonomy generator builds an artificial world of categories and objects. The categories form a tree. The objects, called instances, sit at the tree's leaves. Every category and every instance has a vector of binary features, and optionally some continuous scalar dimensions. The features follow controlled rules of inheritance down the tree, and controlled logical rules between features. Every rule and every inheritance decision is written to the output, so we always know the ground truth a model is trying to learn.

This guide covers running the generator, the ideas behind it, the configuration file, the output files, and the Python interface. The design is specified in `docs/specs/TAXONOMY_GENERATOR.md` and `docs/specs/TAXONOMY_RELATIONS.md`, with the changes of the world-and-language refactor (`docs/specs/WORLD_AND_LANGUAGE.md`, "Labels" and "Taxonomy outputs").

**Status.** Complete: the base generator and scalar dimensions. Since stage a5b of the world-and-language refactor, the generator writes the labels of that specification (`CATEGORY.1.2`, `INSTANCE.1.2.3`, `PROPERTY.4`, `PART.5`, `SCALARDIM.1`), writes the base vector and the derived values apart (`base.csv` and `derived/`), and makes categories, instances, features, and scalars only. The actions (the old CAN features) and the verbs with their relations are made by the world package; see "Where CAN features and verbs went" below.

## Quick start

From the root of the repository (see `README.md` in this folder for setup):

```
PYTHONPATH=python python -m semantic_world.taxonomy data/taxonomy/tiny.yaml
```

The program prints one line:

```
wrote runs/taxonomy/tiny_seed1: 6 categories, 4 leaves, 12 instances, 4 rules
```

The tiny configuration makes 2 superordinate categories with 2 subcategories each, 3 instances per leaf, 8 PROPERTY features, and 8 PART features. The run is small enough to read every file by eye, which is the best way to learn the outputs.

Two options change a run without editing the configuration:

- `--seed N` replaces the master seed. Different seeds give different worlds from the same settings.
- `--out DIR` writes the output folder somewhere else. The default is `runs/taxonomy/<name>_seed<seed>/`.

The example configurations in `data/taxonomy/` are:

| File | What it makes |
| --- | --- |
| `tiny.yaml` | The tiny world above. |
| `default.yaml` | A mid-sized world: 4 superordinates, depth 3, 40 PROPERTY and 40 PART features. Every parameter appears with its default value, so this file is also the reference for defaults. |
| `rule_file.yaml` | The default world with rules drawn from a rule file (`rules/example.yaml`) instead of the automatic settings. |
| `tiny_relations.yaml` | The tiny world with 1 scalar dimension. The tiny world of the world package (`data/world/tiny.yaml`) builds on it. |
| `relations.yaml` | The default world with 2 scalar dimensions. The default world of the world package (`data/world/default.yaml`) builds on it. |

## Concepts

### Labels

Every label is a word a reader can read without a key, a period, and an index. Indices start at 1, and periods separate indices.

| Object | Label | Example |
| --- | --- | --- |
| Superordinate category | `CATEGORY.<i>` | `CATEGORY.1` |
| Subcategory | `CATEGORY.<i>.<j>...` | `CATEGORY.1.2` is subcategory 2 of superordinate 1 |
| Instance | `INSTANCE.<category indices>.<k>` | `INSTANCE.1.2.3` is instance 3 of leaf `CATEGORY.1.2` |
| Property feature | `PROPERTY.<n>` | `PROPERTY.4` |
| Part feature | `PART.<n>` | `PART.12` |
| Membership feature | `ISA.<category>` | `ISA.CATEGORY.1.2` |
| Scalar dimension | `SCALARDIM.<n>` | `SCALARDIM.1` |

The old labels (`C1.2`, `I1.2.3`, `IS.4`, `HAS.12`, `SC.1`) appear in the two taxonomy specifications, which keep them as the record of what was built. The full table of labels, including the world's event types and the corpus's documents, is in `docs/specs/WORLD_AND_LANGUAGE.md`, "Labels".

### Feature types

- **ISA** features record membership. An instance has `ISA.CATEGORY.1 = 1` and `ISA.CATEGORY.1.2 = 1` when it belongs to leaf `CATEGORY.1.2`. ISA features come from the tree alone. No rule reads them.
- **PROPERTY** features are properties, and **PART** features are parts. The type names follow the labels since stage a6 of the world-and-language refactor: `features.property` configures the PROPERTY features and `features.part` the PART features, and the `type` column of `features.csv` says `property` or `part`. The old keys and type names (`is`, `has`) fail with an error that names the new ones.
- **Scalar** dimensions (`SCALARDIM.<n>`) are continuous values, such as size. Scalars are off unless the configuration turns them on.

PROPERTY and PART features are either **free** or **determined**. A free feature's value comes from sampling and inheritance. A determined feature's value is computed by a rule from other features. A configured proportion of each type is determined.

Actions (the old CAN features, every one computed by a rule) are not features of the taxonomy any more. They are the one-place event types of the world package, whose requirements are rules of the same kind over the taxonomy's features; see "Where CAN features and verbs went".

### Rules

A rule is a Boolean function of a few input features. Rules range from simple (`PROPERTY.7 = PART.2`) to complex. Four settings control complexity: the number of inputs (arity), the mix of operators (AND, OR, XOR), how deeply operators nest, and how often inputs are negated.

- With 1 input, the rule copies or negates its input.
- With 2 inputs, the rule is one of the 10 functions that depend on both inputs.
- With 3 inputs, the rule is usually one of the six category structures of Shepard, Hovland, and Jenkins (1961), types I to VI. The `shj_type` field in `rules.yaml` names the type.
- With 4 or more inputs, the rule is a random formula that uses each input once.

Rules are organized in layers so they never form a cycle. Free features are layer 0. With `max_chain_depth: 1`, every rule reads free features only. With a larger chain depth, some determined features are computed from other determined features, which gives chains of inference.

Every rule records a complexity score: the number of literals in its shortest disjunctive normal form (`min_dnf_literals`).

Some rules produce features that barely vary, or that carry no information about categories. Those features are kept on purpose. One research question is whether models learn to attend to the informative features. `feature_stats.csv` reports how informative every feature is.

### The tree and inheritance

The tree has `depth` levels. Superordinates are level 1, and leaves are at level `depth`. Each superordinate's free features are drawn from base rates. Its determined features are computed from its free features.

At every category, each free feature gets one of three roles. The roles decide how the category's children inherit the feature:

- **defining:** every child copies the value. A defining feature stays defining all the way down, so every member of the category has the same value;
- **characteristic:** each child copies the value with a high probability (`characteristic_probability`), and flips it otherwise;
- **undiagnostic:** each child draws a new value from the base rate.

Roles are assigned separately at every category, so a feature can be defining in one branch and undiagnostic in another. A defining value can be 0: a category can be defined by lacking something.

Instances inherit from their leaf in the same way, using the roles assigned at the leaf. `instances.characteristic_probability` sets the copy probability for instances separately from the tree.

A determined feature can also end up fixed for a whole category, when its rule's output no longer depends on anything that varies below that category. The generator detects these cases and reports them as `fixed_by_rule`.

**Base rates.** By default, each free feature has a base rate of its own, drawn from a Beta distribution whose mean is `expected_true_free` divided by the number of free features of the type and whose concentration is `features.base_rate_heterogeneity` (2 by default), so some features are common and others rare, as in the real world ("A principle for defaults" in `docs/specs/WORLD_AND_LANGUAGE.md`). `base_rate_heterogeneity: null` gives every free feature of a type the same base rate. The tiny configurations set it to null: with 6 free features of a type, a heterogeneous draw can leave a superordinate with no true feature, which the similarity bound cannot place.

### Scalar dimensions

A superordinate's scalar values are drawn from a standard normal distribution. Each child's value is its parent's value plus normal noise with standard deviation `drift`. Each instance adds noise with standard deviation `instance_drift`. Related categories therefore have similar values.

Rules can read scalars through threshold literals, such as `SCALARDIM.1 > -0.5822`. Inside a rule, a threshold literal acts like a binary input. Thresholds come from the configuration alone, never from the realized data. With `scalars.count: 0`, the world is purely binary.

### Three vectors for every category

Every category, leaves included, gets three vectors:

1. **Generative:** the vector the category's children were generated from.
2. **Defining:** the value of every feature fixed for all members of the category, and `NaN` for every feature that varies.
3. **Mean:** the mean of every feature over all instances below the category.

## Reading the outputs

Each run writes one folder. The files below appear in every run. Examples come from the tiny configuration with seed 1.

### `tree.csv`

One row per category.

```
label,parent,level,children,instances
CATEGORY.1,,1,2,6
CATEGORY.1.1,CATEGORY.1,2,0,3
CATEGORY.1.2,CATEGORY.1,2,0,3
```

### `features.csv`

One row per feature: `label`, `type` (`isa`, `property`, `part`, or `scalar`), `kind` (`free` or `determined`), `layer`, and `base_rate` (free binary features only).

### `rules.yaml`

One entry per determined feature. For example:

```yaml
- output: PROPERTY.7
  layer: 1
  family: compositional
  shj_type: null
  arity: 4
  nesting_depth: 1
  inputs: [PROPERTY.1, PROPERTY.4, PART.2, PART.3]
  relevant_inputs: [PROPERTY.1, PROPERTY.4, PART.2, PART.3]
  expression: PROPERTY.1 OR PROPERTY.4 OR NOT PART.2 OR PART.3
  truth_table: '1101111111111111'
  min_dnf_literals: 4
```

`PROPERTY.7` is true unless `PART.2` is the only one of its four inputs that is true, so every input is relevant. The truth table gives the output for input settings 0000 to 1111, with the first input as the most significant bit. A rule with three inputs is usually an SHJ type, named in `shj_type`; a type that depends on only two of its inputs lists two `relevant_inputs`. Rules that read scalars also list their `thresholds`.

### `base.csv`

The base vector: one row per instance, with its label, its leaf, every free PROPERTY and PART feature, and every scalar. These are the inputs of the world; nothing in the file is computed by a rule.

```
label,leaf,PROPERTY.1,PROPERTY.2,PROPERTY.3,PROPERTY.4,PROPERTY.5,PROPERTY.6,PART.1,PART.2,...
INSTANCE.1.1.1,CATEGORY.1.1,0,1,0,0,0,1,0,0,...
```

`base.csv` replaced the old `instances.csv`, which held the ISA columns, the determined features, and the CAN features as well. The determined features are in `derived/static_features.csv`. A table with the old columns comes from the world package: `python -m semantic_world.world view RUN --preset classic` writes the ISA, PROPERTY, PART, one-place capacity, and scalar columns of a world run (`docs/specs/WORLD_AND_LANGUAGE.md`, "Views").

### `categories_generative.csv`, `categories_defining.csv`, `categories_mean.csv`

The three vectors for every category, with the ISA columns, every PROPERTY and PART feature (free and determined), and the scalars. The category files keep both kinds of feature because they describe categories, not world inputs. In the defining file, `NaN` marks features that vary among the category's members. The word-form pipeline reads `categories_generative.csv` as its meanings.

### `roles.csv`

One row per category and feature: the feature's role at that category. The roles are `defining_new` (made defining at this category), `defining_inherited` (defining at an ancestor), `characteristic`, `undiagnostic`, `determined` (computed by a rule), and `fixed_by_rule`. For `fixed_by_rule`, the `fixed_test` column says whether the exact or the approximate test found it.

### `similarity.csv`

One row per category, using the configured similarity metric:

- `within`: the mean similarity among the category's children (for a leaf, among its instances);
- `between`: the mean similarity between the category's children and the children of its sibling categories;
- `instances`: the mean similarity among all instances below the category.

A category whose `within` is well above its `between` is a tight, distinct category.

### `feature_stats.csv`

One row per feature: the proportion of instances with the feature, its entropy, how many categories give it each role, and for each level L the mutual information between the feature and the category at level L (`mi_level_L`). Scalar rows give the mean, the standard deviation, and the proportion of variance explained by the category at each level (`eta2_level_L`). These columns show which features are informative about categories, and at which level.

### `summary.yaml` and `config.yaml`

`summary.yaml` gives counts, the mean number of true features per instance, constant features, duplicate leaves and instances, and any warnings. `config.yaml` records the complete resolved configuration and every random seed. Running the generator on an output folder's `config.yaml` reproduces the run exactly.

### `rule_matrices.json`

The same rules as `rules.yaml`, in the form the world runtime computes: two-layer threshold matrices (`docs/specs/WORLD_AND_LANGUAGE.md`, "Rules: Boolean form and matrices"). The file has six keys.

- `version`: the definition format's version, 1.
- `rule_set_id`: the rule-set identity, the SHA-256 hash of the canonical JSON of `symbols`, `literals`, `rules`, and (for a taxonomy, an empty) `event_types`. The same configuration and seed always give the same identity. Changing any one rule, or any threshold, changes it.
- `symbols`: one entry per PROPERTY and PART feature and per scalar: its label, kind (`property`, `part`, or `scalar`), whether it is derived (computed by a rule) or base, whether it is a fluent (never, in a taxonomy), and its arity.
- `literals`: the literal table: everything the rules read. Each literal has an `index`, its `key` (the variable name in expressions, such as `PROPERTY.3` or `SCALARDIM.2>0.4127`), its `kind` (`feature` or `threshold`), and what it reads.
- `rules`: every rule's `output`, `inputs` (indices into the literal table), `truth_table`, and `expression`.
- `layers`: the matrices, written sparsely. Rules are grouped in dependency layers: a rule that reads only free features and threshold literals is in layer 1, a rule that reads a layer-1 output is in layer 2, and so on. Each layer lists its `terms` (the rows of the first matrix: the literal indices of one term of the rule's minimal DNF, whether each is complemented, and the threshold, which is the number of literals) and its `outputs` (the rows of the second matrix: the output and the indices of its terms within the layer, with threshold 1). A rule that is always true has one term with no literals and threshold 0; a rule that is always false has no terms.

```json
{"literals": [0, 1], "complemented": [false, true], "threshold": 2}
{"output": "PROPERTY.7", "terms": [0, 1, 2, 3], "threshold": 1}
```

The first line is a term that is true when `PROPERTY.1` is true and `PROPERTY.2` is false. The second says `PROPERTY.7` is true when any of its first four terms is true.

**The agreement test.** At the end of every run, the generator evaluates the matrices on every instance and compares them with the truth tables, and checks every rule with at most 12 inputs on every setting of its inputs. If any rule disagrees anywhere, the run fails: nothing is written, and the command line prints an error that names the rule and the first disagreement and exits with status 1. A failure means a bug in the generator, not in the configuration; report it.

### `derived/`

Derived values: columns computed from the base vector by the rules, written apart from the inputs (REL.16). A taxonomy run holds one table.

- `static_features.csv`: one row per instance, with its label and every determined PROPERTY and PART feature.
- `manifest.yaml`: for each file in the folder, the rule-set identity that produced it.

```yaml
version: 1
files:
  static_features.csv:
    rule_set_id: a2e8f084f05f969836fdfde01e00823cb4fa26b99d9122a64a9b5dd6574e86e6
```

A derived file is read through the world package, which refuses a file whose identity differs from the rule set in use, with an error that names the file and both identities:

```python
from semantic_world.world import load_derived_csv

frame = load_derived_csv("runs/taxonomy/tiny_seed1/derived", "static_features.csv", result.rule_set_id)
```

Nothing in `derived/` is an input to anything, and nothing there is edited by hand. A world run's `derived/` folder holds more tables: the capacities, the capacity roles, and the relation statistics (see "Where CAN features and verbs went").

## The configuration file

A configuration file is YAML. Any parameter left out takes its default. An unknown key is an error, and every error names the field. `data/taxonomy/default.yaml` lists every base parameter with its default value.

### Main parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `name` | `default` | Names the output folder. |
| `seed` | 1 | The master seed. |
| `features.property.count`, `features.part.count` | 40, 40 | Number of PROPERTY and PART features. |
| `features.<type>.proportion_determined` | 0.25 | Proportion of PROPERTY or PART features computed by rules. |
| `features.<type>.expected_true_free` | 6 | Expected number of true free features per object. Base rate = this number ÷ the number of free features. |
| `features.base_rate_override` | null | One base rate for every free feature. |
| `features.base_rate_heterogeneity` | 2 | Each free feature gets its own base rate from a Beta distribution with this concentration; null gives every free feature of a type the same base rate. |
| `rules.max_chain_depth` | 1 | Number of layers of determined features. |
| `rules.arity` | `{1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}` | Weights over the number of rule inputs. |
| `rules.operator_mix` | equal AND, OR, XOR | Weights over operators. |
| `rules.negation_probability` | 0.2 | Probability that a rule input is negated. |
| `rules.arity_3_families` | equal | Weights over SHJ types I–VI and `compositional` for 3-input rules. |
| `rules.nesting_depth` | `{1: 0.5, 2: 0.5}` | Weights over nesting depth, for rules with 4 or more inputs. |
| `rules.input_type_weights` | `{property: 1, part: 1, scalar: 1}` | Weights over input types. The scalar weight matters only when scalars are on. |
| `rules.overrides` | `{}` | Different rule settings for PROPERTY (`property`) or PART (`part`) outputs, for example `{part: {arity: {2: 1, 3: 1}}}`. |
| `rules.allow_duplicate_rules` | false | Whether two determined features can have identical rules. |
| `rules.variance_bound` | null | `[low, high]`: resample rules whose expected proportion of true outputs falls outside the range. |
| `rules.source`, `rules.file` | automatic, null | Set `source: file` and name a rule file to use rule templates and explicit rules (see below). |
| `taxonomy.superordinates` | 4 | Number of superordinates. |
| `taxonomy.depth` | 3 | Number of category levels. |
| `taxonomy.branching` | `[2, 4]` | Children per category: a number, a range `[min, max]`, or a per-level list (below). |
| `superordinates.similarity_bound` | phi at most 0.3, on free features | Limits the similarity between superordinates. Set to null to turn off. |
| `inheritance.proportion_defining` | 0.1 | Proportion of still-varying free features that become defining at each category. |
| `inheritance.proportion_characteristic` | 0.5 | Proportion that are characteristic. The rest are undiagnostic. |
| `inheritance.characteristic_probability` | 0.9 | Probability that a child copies a characteristic feature. |
| `inheritance.require_distinct_leaves` | true | Whether any two leaves must differ in some feature. |
| `instances.per_leaf` | `[5, 10]` | Instances per leaf: a number or a range. |
| `instances.characteristic_probability` | 0.9 | Probability that an instance copies a characteristic feature from its leaf. |
| `analysis.similarity_metric` | cosine | `phi`, `cosine`, or `jaccard`, for `similarity.csv`. |
| `scalars.count` | 0 | Number of scalar dimensions. |
| `scalars.drift` | 0.5 | Standard deviation of a child's change from its parent. |
| `scalars.instance_drift` | 0.2 | Standard deviation of an instance's change from its leaf. |
| `scalars.threshold_quantiles` | `[0.2, 0.8]` | Range of quantiles for rule thresholds on scalars. |
| `scalars.thermometer_bins` | 0 | A number above 0 writes binary thermometer codes for the scalars. |

The keys that configured CAN features and verbs are no longer taxonomy keys. `features.can` and `rules.overrides.can` fail with an error that names `event_types.unary` of the world configuration, and `verbs` with one that names `event_types.binary`.

### Parameters that change with depth

`inheritance.proportion_defining`, `inheritance.proportion_characteristic`, `inheritance.characteristic_probability`, and `scalars.drift` can change with level. Each takes one of four forms:

```yaml
proportion_defining: 0.1                                                  # the same at every level
proportion_defining: {schedule: linear, start: 0.3, end: 0.05}            # level 1 to the last level
proportion_defining: {schedule: exponential, start: 0.3, asymptote: 0.05, rate: 1.0}
proportion_defining: {schedule: list, values: [0.3, 0.1, 0.05]}           # one value per level
```

`taxonomy.branching` takes a number, a range, or a per-level list in the schedule form: `{schedule: list, values: [2, [3, 5]]}`. A bare two-number list such as `[2, 4]` is always a range. `config.yaml` in every output folder shows the per-level values a run actually used.

### Rule files

A rule file lists rule templates with weights, and optionally explicit rules for named features. `data/taxonomy/rules/example.yaml` shows the format:

```yaml
templates:
  - {family: literal, weight: 0.1}
  - {family: shj, type: IV, weight: 0.2}
  - {family: shj, type: VI, weight: 0.05, applies_to: has}
  - {family: compositional, arity: 4, operators: {AND: 1, OR: 1}, nesting_depth: 2, weight: 0.3}
explicit:
  - {output: PROPERTY.35, expression: "(PART.2 AND NOT PROPERTY.5) OR PROPERTY.7"}
```

The rule file's path is read relative to the configuration file's folder. A template applies to PROPERTY outputs (`property`), PART outputs (`part`), or both. The requirement of a one-place event type is given by hand in the world's event file, not in a rule file.

## Where CAN features and verbs went

The taxonomy once made actions (CAN features) and verbs with their relations. Both are made by the world package since stage a5b, from the taxonomy's features, with the same machinery under new names: CAN features are the **one-place event types** `EVENTTYPE1.<n>`, and verbs are the **two-place event types** `EVENTTYPE2.<path>`. A world configuration (`data/world/default.yaml`) names a taxonomy configuration and adds the event types:

```yaml
taxonomy: {config: data/taxonomy/relations.yaml, seed: null}   # null: the world's seed
event_types:
  unary: {count: 20, rules: {}}      # the old features.can.count and rules.overrides.can
  binary: {}                         # the old verbs block, every key at its default; null turns it off
```

`python -m semantic_world.world define data/world/tiny.yaml` runs the taxonomy (the run is written to `taxonomy/` in the world's folder, byte-identical to the taxonomy run alone), then draws the event types from the world's own streams (`world:requirements`, `world:event_tree`, `world:constraints`, `world:pairs`), the fluents, the preconditions, and the effects. The corpus generator reads worlds, not taxonomy runs.

**One-place event types.** The requirement of `EVENTTYPE1.<k>` is a rule over the agent's PROPERTY and PART features of any layer and threshold literals, sampled with the taxonomy's `rules` settings and the overrides in `event_types.unary.rules`, exactly as the old CAN rules were. `derived/capacities.csv` of a world run holds the column `CAN.EVENTTYPE1.<k>`: whether each entity meets the requirement. `derived/capacity_roles.csv` says, for each category and one-place event type, whether the capacity is fixed at 1, fixed at 0, or free for the category's members, and which test decided.

**Two-place event types.** An event type holds for a pair of entities (an agent and a patient) when every one of its constraints holds. A constraint is a rule over the static facts of the agent (`agent.`), the patient (`patient.`), or both, from the same five families as before:

| Family | Example | Models |
| --- | --- | --- |
| `agent` | `agent.PROPERTY.24 OR agent.PART.38` | what the agent must be like |
| `patient` | `patient.PROPERTY.12 AND NOT patient.PART.4` | what the patient must be like |
| `cross` | `agent.PART.5 OR NOT patient.PART.3` | any rule mixing both roles |
| `key_lock` | `(agent.PROPERTY.6 AND NOT patient.PROPERTY.5) OR (NOT agent.PART.1 AND NOT patient.PART.6)` | matching pairs: barbers cut hair, lumberjacks cut wood |
| `comparison` | `-0.6950 < agent.SCALARDIM.1 - patient.SCALARDIM.1 < -0.0460` | scalar comparisons: owls eat things smaller than themselves, but not much smaller |

Comparisons need scalar dimensions. An entity is never related to itself. Two entities of the same leaf can be related.

The event-type tree (`EVENTTYPE2.1`, `EVENTTYPE2.1.2`) is generated by the same machinery as the category tree, with binary event-type features (`EVENTFEAT.<n>`) in defining, characteristic, and undiagnostic roles. Every event-type feature carries one constraint (`CONSTRAINT.EVENTFEAT.<n>`), and every event type (a leaf of the tree) has one of its own (`CONSTRAINT.EVENTTYPE2.<path>`). An event type's requirement is the conjunction of the constraints of its true features and its own constraint. Every category of event types has a base relation, the constraints of its defining features, which every event type below it entails. `definition.json` of a world run lists every constraint and every event type with its requirement, preconditions, and effects.

**Density.** By default, the generator redraws an event type only when its requirement holds for no pair of leaf categories or for every pair, and a constraint only when it holds for no leaf pair or for every leaf pair, so the share of pairs an event type relates follows from its constraints ("A principle for defaults"). The old checks stay as settings of `event_types.binary`: `density: {min: 0.01, max: 0.3}` keeps every event type between 1% and 30% of leaf pairs, and `constraint_min_density: 0.1` keeps every constraint at 10% or more. An event type that cannot be brought into range is kept as close as possible and named in the warnings.

**Capacities.** Each two-place event type gives every entity two derived one-place facts, in `derived/capacities.csv`: `CAN.<event type>`, the entity could be the agent with some possible patient, and `CANBE.<event type>`, it could be the patient with some possible agent. "Possible" means any combination of features an entity could have, not only the entities in the world. The `ACTUAL_` columns say whether the entity has a partner among the world's entities. Exposure is gone: a view (`python -m semantic_world.world view`) chooses which columns a model sees.

**Relation statistics.** A world run's `derived/` folder holds the relation files of the old taxonomy, with `event_type` for `verb`: `relation_proportions.csv` (category-level facts: for each event type, agent category, and patient category at the same level, the share of pairs related), `relation_pairs.csv` (sampled true and false pairs), `event_type_stats.csv` (one row per event type: the proportion of pairs related, the numbers of agents and patients with a partner, the constraints and their families, the symmetric and within-leaf shares, and with the density check its leaf-pair density and tries), and `thematic.csv` (for every pair of leaves, a `thematic` score, the sum over event types and both directions of the leaf-pair proportions, and the leaves' taxonomic `similarity`). For example, this row of the tiny world says that `EVENTTYPE2.1.1` holds for 16 of the 30 pairs with a `CATEGORY.1` agent and a `CATEGORY.1` patient:

```
event_type,level,agent,patient,true_pairs,total_pairs,proportion,estimated
EVENTTYPE2.1.1,1,CATEGORY.1,CATEGORY.1,16,30,0.533333,false
```

**Event-type parameters.** `event_types.binary` takes the keys of the old `verbs` block, without `projections`: `features` (`{count: 12, expected_true: 3}`), `taxonomy` (3 superordinates, depth 2, branching `[2, 3]`), `inheritance` (0.4 defining, 0.4 characteristic, 0.9 copy probability), `own_constraint` (true), `constraint_families` (equal weights), `key_lock_pairs` (`{1: 0.5, 2: 0.3, 3: 0.2}`), `comparison` (windows 0.3, cross-dimension 0.2), `rules` (the taxonomy's rule settings), `pairs` (1000 true, 1000 false, `max_exact_pairs`), `density` (`{min: 0.0, max: 1.0, max_tries: 200}`: the degenerate check), and `constraint_min_density` (null). `event_types.unary` takes `count` (20) and `rules` (overrides of the taxonomy's rule settings for the requirements). `config.yaml` of a world run shows every value.

**The default world's retuned settings.** `data/world/default.yaml` departs from the code's defaults in five settings, so that the default world's events are not almost all one-place: the event-type tree has 5 superordinates with 4 event types each (20 two-place event types, as many as the one-place ones; stage a6); a two-place event type has no constraint of its own beside those of its event-type features (`own_constraint: false`), so that a requirement is a conjunction of about three constraints rather than four; the key-lock family, which holds for the fewest pairs, has half the weight of the other families (`constraint_families: {..., key_lock: 0.5, ...}`); the constraints weigh OR four times as much as AND and XOR (`binary.rules.operator_mix: {AND: 1, OR: 4, XOR: 1}`), so that each holds for more pairs; and the one-place requirements weigh AND three times as much as OR and XOR (`unary.rules.operator_mix: {AND: 3, OR: 1, XOR: 1}`), so that they are stricter (stage a7a, Jon's ruling on the requirement densities). In the default world of seed 1, two-place requirements then hold for 17% of the ordered pairs of entities on average (0.4% to 97% by event type), one-place requirements for 58% of the entities (13% to 98%), every category of event types keeps a word, and 32% of the events of the 1,000 statistics episodes are two-place (`episodes.two_place_share` of `world_stats.yaml`; 0.12 to 0.55 on seeds 1 to 5, mean 0.38, following the spread of the densities). Without own constraints, two event types that share their true event-type features have the same requirement, and one whose true features are a subset of another's has the weaker one; the tree keeps every pair of event types apart on some feature. `examples/two_place_levers.py` runs the default world with one setting changed at a time and reports the two-place share, the one-place capacity rate, and the two-place pair density; the table is in `docs/proposals/2026-10-09-world-stage-a7a-decisions.md`.

## Using the generator from Python

```python
from semantic_world.taxonomy import load_config, generate

config = load_config("data/taxonomy/tiny.yaml", seed=7)
result = generate(config)
result.write("runs/taxonomy/tiny_seed7")
```

`result` holds everything the run produced:

- `result.tree`: the categories, with `result.tree.leaves` and `result.tree.superordinates`;
- `result.instances`: the instances, with `result.instances.values` (the PROPERTY and PART features as a NumPy array, without the ISA columns) and `result.instances.labels`;
- `result.rules`: the rules, with `result.rules.rule_for("PROPERTY.7")` for one rule;
- `result.matrices`: the rules in matrix form (`result.matrices.matrices.evaluate(...)`), and `result.rule_set_id`, the rule-set identity;
- `result.features`: the feature layout;
- `result.frames()`: every output table as a polars data frame, keyed by file name;
- `result.summary` and `result.warnings`.

The event types, the capacities, and the relations of a world are in `semantic_world.world`: `define(config)` returns a result whose `statics` holds the one-place requirements (`statics.unary`), the event-type tree (`statics.event_tree`), the relations (`statics.relations`, with `holds(event_type, agents, patients)` over aligned arrays of entity indices and `matrix(event_type)` over every ordered pair), the capacities (`statics.projections`), and the relation statistics (`statics.relation_stats`).

To change settings in code, build the configuration from a dictionary with `config_from_mapping`. Any key left out takes its default:

```python
from semantic_world.taxonomy import config_from_mapping, generate

for seed in range(1, 11):
    config = config_from_mapping({"name": "sweep", "seed": seed, "taxonomy": {"depth": 4}})
    generate(config).write(f"runs/taxonomy/sweep_seed{seed}")
```

## Recipes

**Several worlds with the same settings.** Run the same configuration with `--seed 1`, `--seed 2`, and so on. Each seed gives a different world with the same statistical structure.

**Loading outputs for analysis.**

```python
import polars as pl

base = pl.read_csv("runs/taxonomy/tiny_seed1/base.csv")
features = base.select(pl.exclude("label", "leaf"))
```

`base.csv` holds the free features and the scalars. For the determined features too, join `derived/static_features.csv` on `label`; for the ISA columns and the one-place capacities as well, write a view of a world run with `python -m semantic_world.world view RUN --preset classic`. Drop the ISA columns when a model should learn categories from features alone.

**A flatter or deeper world.** `taxonomy.depth` and `taxonomy.branching` set the shape. With `depth: 1`, the superordinates are the leaves.

**Tighter or looser categories.** Raise `inheritance.proportion_defining` and `characteristic_probability` for tight categories with sharp boundaries. Lower them for loose categories with graded, family-resemblance structure. `similarity.csv` shows the result.

**Taxonomic versus thematic structure.** Compare the `similarity` and `thematic` columns of a world run's `derived/thematic.csv`. A model that learns from features alone should track similarity. A model that learns from who does what to whom should track thematic relatedness.

**Simpler or harder rules.** Put all of `rules.arity`'s weight on 1 and 2 for simple rules. Weight SHJ type VI for the hardest 3-input rules. Raise `rules.max_chain_depth` for chains of inference.

## Troubleshooting

- **The similarity bound cannot be met.** With few features, or a tight bound, no set of superordinates satisfies `superordinates.similarity_bound`. The error reports the closest similarity reached. Loosen `max`, add features, set `features.base_rate_heterogeneity: null` in a small configuration, or set the bound to null.
- **Leaves cannot be made distinct.** With few features and many leaves, two leaves may be forced to match. Add features, reduce branching, or set `require_distinct_leaves: false`.
- **The largest arity does not fit.** A rule cannot have more inputs than the features available to it. Lower the largest arity with nonzero weight, or add free features.
- **A key is unknown.** Check the spelling against `data/taxonomy/default.yaml`. `features.can`, `rules.overrides.can`, and `verbs` moved to the world configuration, and the error says where.
- **The agreement test fails.** The error begins "the matrix form of the rule for ... disagrees with its truth table". The matrices and the truth tables are two forms of one rule set, so a disagreement is a bug in the generator, not a problem with the configuration. Keep the configuration and seed and report it.

## Reference

- `docs/specs/TAXONOMY_GENERATOR.md`: the full design of the base generator (with the old labels).
- `docs/specs/TAXONOMY_RELATIONS.md`: scalars, verbs, and relations (with the old labels; the verbs are the world's two-place event types now).
- `docs/specs/WORLD_AND_LANGUAGE.md`: the world model; "Labels", "Taxonomy outputs", "Event types", "Rules: Boolean form and matrices", and "Rule-set identity and derived values".
- `docs/proposals/`: decisions made during the build, where the build differs from the first draft of a specification.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13, Whole No. 517).
