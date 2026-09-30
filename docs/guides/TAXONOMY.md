# The taxonomy generator

The taxonomy generator builds an artificial world of categories and objects. The categories form a tree. The objects, called instances, sit at the tree's leaves. Every category and every instance has a vector of binary features, and optionally some continuous scalar dimensions. The features follow controlled rules of inheritance down the tree, and controlled logical rules between features. Every rule and every inheritance decision is written to the output, so we always know the ground truth a model is trying to learn.

This guide covers running the generator, the ideas behind it, the configuration file, the output files, and the Python interface. The design is specified in `docs/specs/TAXONOMY_GENERATOR.md` and `docs/specs/TAXONOMY_RELATIONS.md`.

**Status.** Complete: the base generator, scalar dimensions, and verbs and relations.

## Quick start

From the root of the repository (see `README.md` in this folder for setup):

```
PYTHONPATH=python python -m semantic_world.taxonomy data/taxonomy/tiny.yaml
```

The program prints one line:

```
wrote runs/taxonomy/tiny_seed1: 6 categories, 4 leaves, 12 instances, 8 rules
```

The tiny configuration makes 2 superordinate categories with 2 subcategories each, 3 instances per leaf, 8 IS features, 8 HAS features, and 4 CAN features. The run is small enough to read every file by eye, which is the best way to learn the outputs.

Two options change a run without editing the configuration:

- `--seed N` replaces the master seed. Different seeds give different worlds from the same settings.
- `--out DIR` writes the output folder somewhere else. The default is `runs/taxonomy/<name>_seed<seed>/`.

The example configurations in `data/taxonomy/` are:

| File | What it makes |
| --- | --- |
| `tiny.yaml` | The tiny world above. |
| `default.yaml` | A mid-sized world: 4 superordinates, depth 3, 40 IS, 40 HAS, and 20 CAN features. Every parameter appears with its default value, so this file is also the reference for defaults. |
| `rule_file.yaml` | The default world with rules drawn from a rule file (`rules/example.yaml`) instead of the automatic settings. |
| `tiny_relations.yaml` | The tiny world with 1 scalar dimension and a small verb tree. |
| `relations.yaml` | The default world with 2 scalar dimensions and the default verbs. |

## Concepts

### Labels

All labels are formal. Indices start at 1, and periods separate indices.

| Object | Label | Example |
| --- | --- | --- |
| Superordinate category | `C<i>` | `C1` |
| Subcategory | `C<i>.<j>...` | `C1.2` is subcategory 2 of superordinate 1 |
| Instance | `I<category indices>.<k>` | `I1.2.3` is instance 3 of leaf `C1.2` |
| Features | `IS.<n>`, `HAS.<n>`, `CAN.<n>` | `IS.4` |
| Membership features | `ISA.<category>` | `ISA.C1.2` |
| Scalar dimensions | `SC.<n>` | `SC.1` |

### Feature types

- **ISA** features record membership. An instance has `ISA.C1 = 1` and `ISA.C1.2 = 1` when it belongs to leaf `C1.2`. ISA features come from the tree alone. No rule reads them.
- **IS** features are properties, and **HAS** features are parts.
- **CAN** features are actions. Every CAN feature is computed by a rule from IS and HAS features.
- **Scalar** dimensions (`SC.<n>`) are continuous values, such as size. Scalars are off unless the configuration turns them on.

IS and HAS features are either **free** or **determined**. A free feature's value comes from sampling and inheritance. A determined feature's value is computed by a rule from other features. A configured proportion of IS and HAS features is determined.

### Rules

A rule is a Boolean function of a few input features. Rules range from simple (`CAN.3 = HAS.2`) to complex. Four settings control complexity: the number of inputs (arity), the mix of operators (AND, OR, XOR), how deeply operators nest, and how often inputs are negated.

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

### Scalar dimensions

A superordinate's scalar values are drawn from a standard normal distribution. Each child's value is its parent's value plus normal noise with standard deviation `drift`. Each instance adds noise with standard deviation `instance_drift`. Related categories therefore have similar values.

Rules can read scalars through threshold literals, such as `SC.1 > -0.5822`. Inside a rule, a threshold literal acts like a binary input. Thresholds come from the configuration alone, never from the realized data. With `scalars.count: 0`, the world is purely binary.

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
C1,,1,2,6
C1.1,C1,2,0,3
C1.2,C1,2,0,3
```

### `features.csv`

One row per feature: `label`, `type` (`isa`, `is`, `has`, `can`, or `scalar`), `kind` (`free` or `determined`), `layer`, and `base_rate` (free binary features only).

### `rules.yaml`

One entry per determined feature. For example:

```yaml
- output: IS.8
  layer: 1
  family: shj
  shj_type: II
  arity: 3
  nesting_depth: null
  inputs: [HAS.4, HAS.5, HAS.6]
  relevant_inputs: [HAS.4, HAS.5]
  expression: (NOT HAS.4 AND NOT HAS.5) OR (HAS.4 AND HAS.5)
  truth_table: '11000011'
  min_dnf_literals: 4
```

`IS.8` is true when `HAS.4` and `HAS.5` agree. The rule is SHJ type II, which depends on only two of its three inputs, so `relevant_inputs` lists two. The truth table gives the output for input settings 000 to 111, with the first input as the most significant bit. Rules that read scalars also list their `thresholds`.

### `instances.csv`

The main feature matrix: one row per instance, with its label, its leaf, and every feature. Columns are ordered ISA, IS, HAS, CAN, then scalars.

```
label,leaf,ISA.C1,ISA.C1.1,ISA.C1.2,ISA.C2,ISA.C2.1,ISA.C2.2,IS.1,IS.2,...
I1.1.1,C1.1,1,1,0,0,0,0,0,1,...
```

### `categories_generative.csv`, `categories_defining.csv`, `categories_mean.csv`

The three vectors for every category, with the same columns as `instances.csv` (without `leaf`). In the defining file, `NaN` marks features that vary among the category's members.

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

## The configuration file

A configuration file is YAML. Any parameter left out takes its default. An unknown key is an error, and every error names the field. `data/taxonomy/default.yaml` lists every base parameter with its default value.

### Main parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `name` | `default` | Names the output folder. |
| `seed` | 1 | The master seed. |
| `features.is.count`, `features.has.count` | 40, 40 | Number of IS and HAS features. |
| `features.<type>.proportion_determined` | 0.25 | Proportion of IS or HAS features computed by rules. |
| `features.<type>.expected_true_free` | 6 | Expected number of true free features per object. Base rate = this number ÷ the number of free features. |
| `features.can.count` | 20 | Number of CAN features. |
| `features.base_rate_override` | null | One base rate for every free feature. |
| `features.base_rate_heterogeneity` | null | A number gives each free feature its own base rate, from a Beta distribution with this concentration. |
| `rules.max_chain_depth` | 1 | Number of layers of determined features. |
| `rules.arity` | `{1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}` | Weights over the number of rule inputs. |
| `rules.operator_mix` | equal AND, OR, XOR | Weights over operators. |
| `rules.negation_probability` | 0.2 | Probability that a rule input is negated. |
| `rules.arity_3_families` | equal | Weights over SHJ types I–VI and `compositional` for 3-input rules. |
| `rules.nesting_depth` | `{1: 0.5, 2: 0.5}` | Weights over nesting depth, for rules with 4 or more inputs. |
| `rules.input_type_weights` | `{is: 1, has: 1, scalar: 1}` | Weights over input types. The scalar weight matters only when scalars are on. |
| `rules.overrides` | `{}` | Different rule settings for IS, HAS, or CAN outputs, for example `{can: {arity: {2: 1, 3: 1}}}`. |
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
  - {family: compositional, arity: 4, operators: {AND: 1, OR: 1}, nesting_depth: 2, weight: 0.3}
explicit:
  - {output: CAN.3, expression: "(HAS.2 AND NOT IS.5) OR IS.7"}
```

The rule file's path is read relative to the configuration file's folder.

## Verbs and relations

Verbs are two-place relations between instances: an agent and a patient. "Penguins chase fish" is a verb relating penguin instances as agents to fish instances as patients. Verbs are off unless the configuration has a `verbs` block. `data/taxonomy/tiny_relations.yaml` and `relations.yaml` turn them on. The design is in `docs/specs/TAXONOMY_RELATIONS.md`.

### How verbs work

**Constraints.** A verb holds for a pair of instances when every one of its constraints holds. A constraint is a rule over the features of the agent (`a.`), the patient (`p.`), or both. There are five families:

| Family | Example | Models |
| --- | --- | --- |
| `agent` | `a.IS.24 OR a.HAS.38` | what the agent must be like |
| `patient` | `p.IS.12 AND NOT p.HAS.4` | what the patient must be like |
| `cross` | `a.HAS.5 OR NOT p.HAS.3` | any rule mixing both arguments |
| `key_lock` | `(a.IS.6 AND NOT p.IS.5) OR (NOT a.HAS.1 AND NOT p.HAS.6)` | matching pairs: barbers cut hair, lumberjacks cut wood |
| `comparison` | `-0.6950 < a.SC.1 - p.SC.1 < -0.0460` | scalar comparisons: owls eat things smaller than themselves, but not much smaller |

Comparisons need scalar dimensions. With `scalars.count: 0`, the comparison family is never drawn.

An instance is never related to itself. Two instances of the same leaf can be related: a penguin can chase another penguin.

**The verb tree.** Verbs are generated by the same machinery as the noun tree. Verb categories (`V1`, `V1.2`) have binary verb features (`VF.<n>`), with defining, characteristic, and undiagnostic roles, exactly as nouns do. Every verb feature carries one constraint (`K.VF.<n>`). The verbs are the leaves of the verb tree. A verb's relation is the conjunction of:

- the constraints of every verb feature that is true for the verb;
- the verb's own constraint (`K.<verb>`), which keeps two verbs from being the same relation.

Every verb category also has a **base relation**: the constraints of its defining verb features. Defining features stay defining down the tree, so every verb entails the base relation of every category above it. If `V1` is predation, then chasing (`V1.1`) and eating (`V1.2`) are both kinds of predation.

**Density.** Each verb must relate between 1% and 30% of pairs of leaf categories (`verbs.density`), and no single constraint may relate fewer than 10% (`verbs.constraint_min_density`). The generator resamples constraints and verb features until each verb fits. A verb that cannot fit is kept as close as possible and named in the warnings. Density is measured on the leaves, not the instances, so changing the instance count never changes the verbs. Without the density check, most verbs were empty or nearly empty.

**Projections.** Each verb gives every instance two derived one-place features:

- `CAN.<verb>`: the instance could be the verb's agent with some possible patient;
- `CANBE.<verb>`: the instance could be the verb's patient with some possible agent.

"Possible" means any combination of features an object could have, not only the instances in the run. The `ACTUAL_` versions report whether the instance actually has a partner among the run's instances. By default, all `CAN` projections and a quarter of the `CANBE` projections appear as columns in `instances.csv`. `CANBE` features model words like "edible", which languages lexicalize only for some verbs.

### Relation outputs

| File | Contents |
| --- | --- |
| `verb_tree.csv` | One row per verb category: label, parent, level, number of children. |
| `verb_features.csv` | One row per verb feature: label, base rate, and the label of its constraint. |
| `verb_roles.csv`, `verbs_generative.csv`, `verbs_defining.csv` | The verb tree's roles and vectors, in the same formats as the noun files. |
| `constraints.yaml` | One entry per constraint: family, literals, comparisons, expression, truth table, and leaf-pair density. |
| `relations.yaml` | One entry per verb category: its constraints and the relation's full expression. `base: true` marks base relations of internal categories. |
| `verb_stats.csv` | One row per verb (below). |
| `relation_proportions.csv` | Category-level facts (below). |
| `relation_pairs.csv` | Sampled instance pairs: `verb`, `agent`, `patient`, and `holds` (1 or 0). By default, 1000 true pairs and 1000 false pairs per verb. |
| `projections.csv` | Every projection for every instance, with the `ACTUAL_` columns and a flag for any projection computed approximately. |
| `thematic.csv` | Thematic relatedness and taxonomic similarity for every pair of leaves (below). |

`verb_stats.csv` columns: `proportion_true` (the proportion of ordered instance pairs related), `agents` and `patients` (the number of instances with an actual partner), `constraints` and `families`, `symmetric_proportion` (the share of true pairs whose reverse is also true), `within_leaf_proportion` (the share of true pairs within one leaf), `leaf_pair_density`, and `tries`.

`relation_proportions.csv` holds the category-level facts that generic sentences describe. One row per verb, agent category, and patient category at the same level, with `true_pairs`, `total_pairs`, and `proportion`. Rows with no true pairs are left out. For example, this row from the tiny relations run says that verb `V1.1` holds for 6 of the 36 pairs with a `C1` agent and a `C2` patient:

```
verb,level,agent,patient,true_pairs,total_pairs,proportion,estimated
V1.1,1,C1,C2,6,36,0.166667,false
```

A proportion near 1 at the leaf level is a generic truth ("penguins chase fish"). `relation_pairs.csv` holds the episodic facts ("this penguin chased that fish").

`thematic.csv` gives, for every pair of leaves, a `thematic` score (the sum over verbs and both directions of the leaf-pair proportions) and the leaves' taxonomic `similarity`. Across seeds of `relations.yaml`, the two measures are nearly uncorrelated. Leaves can be thematic partners without being similar, as lions and deer are.

### Verb parameters

The `verbs` block takes these keys. `verbs: {}` turns verbs on with every default. `config.yaml` in any relations run shows them all.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `features` | `{count: 12, expected_true: 3}` | Number of verb features, and the expected number true per verb. |
| `taxonomy` | 3 superordinates, depth 2, branching `[2, 3]` | The shape of the verb tree, in the same form as the noun `taxonomy` block. |
| `inheritance` | 0.4 defining, 0.4 characteristic, 0.9 copy probability | Verb-tree inheritance, in the same form as the noun `inheritance` block. |
| `own_constraint` | true | Whether every verb gets a constraint of its own. |
| `constraint_families` | equal weights | Weights over `agent`, `patient`, `cross`, `key_lock`, and `comparison`. |
| `key_lock_pairs` | `{1: 0.5, 2: 0.3, 3: 0.2}` | Weights over the number of matching pairs in a key-lock constraint. |
| `comparison` | windows 0.3, cross-dimension 0.2 | How often a comparison is a window, and how often it compares two different scalars. |
| `rules` | the top-level rule settings | Complexity settings for constraints, with the same keys as the top-level `rules` block. |
| `projections` | `{expose_agent: 1.0, expose_patient: 0.25}` | Proportions of projections shown in `instances.csv`. |
| `pairs` | 1000 true, 1000 false | Sampled pairs per verb, and `max_exact_pairs`, above which proportions are estimated from a sample. |
| `density` | `{min: 0.01, max: 0.3, max_tries: 200}` | The allowed leaf-pair density of every verb. Null turns the check off. |
| `constraint_min_density` | 0.1 | The smallest density allowed for a single constraint. Null turns the check off. |

## Using the generator from Python

```python
from semantic_world.taxonomy import load_config, generate

config = load_config("data/taxonomy/tiny.yaml", seed=7)
result = generate(config)
result.write("runs/taxonomy/tiny_seed7")
```

`result` holds everything the run produced:

- `result.tree`: the categories, with `result.tree.leaves` and `result.tree.superordinates`;
- `result.instances`: the instances, with `result.instances.values` (the IS, HAS, and CAN features as a NumPy array, without the ISA columns) and `result.instances.labels`;
- `result.rules`: the rules, with `result.rules.rule_for("CAN.3")` for one rule;
- `result.features`: the feature layout;
- `result.frames()`: every output table as a polars data frame, keyed by file name;
- `result.summary` and `result.warnings`;
- with verbs on, `result.verbs` (the verb tree), `result.relations`, and `result.projections`.

`result.relations.holds(verb, agents, patients)` says whether a verb holds for pairs of instances, given as aligned arrays of instance indices (rows of `instances.csv`). `result.relations.matrix(verb)` gives the verb over every ordered pair of instances, as a square Boolean array.

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

instances = pl.read_csv("runs/taxonomy/tiny_seed1/instances.csv")
features = instances.select(pl.exclude("label", "leaf"))
```

The ISA columns give category labels at every level. Drop them when a model should learn categories from features alone.

**A flatter or deeper world.** `taxonomy.depth` and `taxonomy.branching` set the shape. With `depth: 1`, the superordinates are the leaves.

**Tighter or looser categories.** Raise `inheritance.proportion_defining` and `characteristic_probability` for tight categories with sharp boundaries. Lower them for loose categories with graded, family-resemblance structure. `similarity.csv` shows the result.

**Taxonomic versus thematic structure.** Compare the `similarity` and `thematic` columns of `thematic.csv`. A model that learns from features alone should track similarity. A model that learns from who does what to whom should track thematic relatedness.

**Simpler or harder rules.** Put all of `rules.arity`'s weight on 1 and 2 for simple rules. Weight SHJ type VI for the hardest 3-input rules. Raise `rules.max_chain_depth` for chains of inference.

## Troubleshooting

- **The similarity bound cannot be met.** With few features, or a tight bound, no set of superordinates satisfies `superordinates.similarity_bound`. The error reports the closest similarity reached. Loosen `max`, add features, or set the bound to null.
- **Leaves cannot be made distinct.** With few features and many leaves, two leaves may be forced to match. Add features, reduce branching, or set `require_distinct_leaves: false`.
- **The largest arity does not fit.** A rule cannot have more inputs than the features available to it. Lower the largest arity with nonzero weight, or add free features.
- **A key is unknown.** Check the spelling against `data/taxonomy/default.yaml`.

## Reference

- `docs/specs/TAXONOMY_GENERATOR.md`: the full design of the base generator.
- `docs/specs/TAXONOMY_RELATIONS.md`: scalars, verbs, and relations.
- `docs/proposals/`: decisions made during the build, where the build differs from the first draft of a specification.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13, Whole No. 517).
