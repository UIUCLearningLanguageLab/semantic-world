# Taxonomy feature generator

Draft, September 29, 2026. Built in stages 1 to 6; the user guide is `docs/guides/TAXONOMY.md`. This document is the build specification for a Python program that generates an artificial dataset of objects in a taxonomic hierarchy, with their features. The specification is written for handoff to Claude Code, or to any developer, and should be complete without access to any other planning material.

## Goal

The generator builds a tree of categories and a set of instances at the leaves of the tree. Every category and instance has a binary feature vector. The features follow controlled rules of inheritance and controlled logical rules between features. Every rule and every inheritance decision is recorded, so we always know the ground truth that a model is trying to learn.

The generator is the content generator for the generative (blocks) world. The same datasets also serve as shared comparison data for cognitive models. The generator is a standalone Python program. The generator does not touch the Rust engine, and does not change anything in `docs/CONTRACTS.md`.

Out of scope for now:

- names in a language (a lexicon will be generated separately from the formal labels below);
- IS features of parts ("red arms"), and part-whole structure between HAS features;
- rendering objects as 3D shapes;
- connecting the generator to the simulation engine.

## Terms

- **Category.** A node in the tree. Superordinates are level 1. Leaves are at level `depth`.
- **Instance.** An individual object. Instances belong to leaves only.
- **Feature types.** ISA, IS (properties), HAS (parts), and CAN (actions). All features are binary.
- **Free feature.** An IS or HAS feature whose value is set by sampling and inheritance.
- **Determined feature.** An IS, HAS, or CAN feature whose value is computed by a rule from other features. Every CAN feature is determined. A parametric proportion of IS and HAS features are determined.
- **Rule.** A Boolean function that computes one determined feature from a list of input features.
- **Role.** At each category, each free feature has one of three roles for that category's members: defining, characteristic, or undiagnostic.

## Labels

All labels are formal. Indices start at 1. Periods separate every index.

| Object | Label | Example |
| --- | --- | --- |
| Superordinate category | `C<i>` | `C1` |
| Subcategory | `C<i>.<j>...` | `C1.3.2` is subcategory 2 of subcategory 3 of superordinate 1 |
| Instance | `I<category indices>.<k>` | `I1.3.2.5` is instance 5 of category `C1.3.2` |
| IS, HAS, CAN features | `IS.<n>`, `HAS.<n>`, `CAN.<n>` | `IS.4`, `HAS.12`, `CAN.3` |
| ISA features | `ISA.<category label>` | `ISA.C1.3` |

Instance labels are unambiguous because the prefix differs (`I` rather than `C`) and all leaves are at the same depth.

Within each type, free features are numbered first, then determined features in layer order (see "Rules"). Categories are ordered depth-first by their index tuples (`C1`, `C1.1`, `C1.1.1`, `C1.1.2`, `C1.2`, ...). Instances are ordered by leaf, then by index.

## Feature types

### ISA

ISA features are determined by the tree alone. There is one ISA feature per category, including leaves. An instance has `ISA.<c> = 1` for its leaf and every ancestor of its leaf, and 0 for every other category. A category's own vectors follow the same rule, with the category itself in place of the leaf. No rule may read an ISA feature.

### IS and HAS

The number of IS features and the number of HAS features are parameters. For each type, a parametric proportion is determined, and the rest are free.

Free features have base rates. The default parameter is the expected number of true free features per object, for each type. The base rate is that target divided by the number of free features of the type. The base rate therefore falls automatically as the feature count grows. An explicit base rate can override the target. An optional heterogeneity setting draws a separate base rate for each free feature from a Beta distribution with the configured mean and a configured concentration. By default, heterogeneity is off, and every free feature of a type has the same base rate.

Determined IS and HAS features have no base rate. Their frequencies emerge from their rules. The summary output reports the realized frequencies.

### CAN

The number of CAN features is a parameter. Every CAN feature is determined by a rule. Rules for CAN features read IS and HAS features only. No rule may read a CAN feature.

## Rules

The rule set is global: one rule set holds for the whole world, at every category and every instance.

### Layers and the dependency graph

Rules form an acyclic graph. Free IS and HAS features are layer 0. Determined IS and HAS features are split as evenly as possible across layers 1 to `max_chain_depth`, with the assignment drawn at random. A rule for a layer-k feature takes inputs from layers below k, and at least one input must come from layer k−1. That constraint makes the chain depth real: with `max_chain_depth: 2`, some features are computed from features that are themselves computed. With `max_chain_depth: 1`, every rule reads free features only.

CAN rules may take inputs from any IS or HAS layer. The input pool can be weighted toward IS or HAS inputs with a parameter.

Inputs to one rule are distinct features. Evaluation order is layer order, so every input is known before the rule runs.

### Rule families

The complexity of a rule is controlled by four things: the arity (number of inputs), the operator mix, the nesting depth, and the negation probability. The arity is drawn from a weighted distribution. How a function is built depends on the arity.

**Arity 1.** The output copies the input, or negates the input with the negation probability.

**Arity 2.** There are 10 Boolean functions of two inputs that depend on both inputs. The 10 functions are the three operators with every pattern of negated inputs: AND with negations (AND, A AND NOT B, NOT A AND B, NOR), OR with negations (OR, A OR NOT B, NOT A OR B, NAND), and XOR with negations (XOR, XNOR). The operator is drawn from the operator mix. Each input is negated with the negation probability.

**Arity 3.** The family is drawn from weights over the six Shepard, Hovland, and Jenkins (1961) types and a "compositional" family. For an SHJ type, the generator takes the canonical function of that type, applies a random permutation of the three inputs, and negates each input with the negation probability. The compositional family builds the function as described for arity 4 and above. The compositional family covers unbalanced functions such as a three-input AND.

The canonical SHJ functions are given by the input combinations that make the output true. Each combination lists the three inputs in order. The numbering follows Nosofsky, Gluck, Palmeri, McKinley, and Glauthier (1994).

| Type | Output true for | Minimal DNF |
| --- | --- | --- |
| I | 000, 001, 010, 011 | NOT A |
| II | 000, 001, 110, 111 | (A AND B) OR (NOT A AND NOT B) |
| III | 000, 001, 010, 101 | (C AND NOT B) OR (NOT A AND NOT C) |
| IV | 000, 001, 010, 100 | (NOT A AND NOT B) OR (NOT A AND NOT C) OR (NOT B AND NOT C) |
| V | 000, 001, 010, 111 | (A AND B AND C) OR (NOT A AND NOT B) OR (NOT A AND NOT C) |
| VI | 000, 011, 101, 110 | three-input even parity |

These canonical forms were checked when this specification was written. Under permutation and negation of inputs, the six forms generate six classes of sizes 6, 6, 24, 8, 24, and 2. Together the six classes cover all 70 three-input functions that are true for exactly four of the eight combinations. Each class is closed under negating the output, so negating the output never changes the type.

SHJ types I and II ignore some of their three inputs. Type I depends on one input, and type II depends on two. The rule still lists all three inputs, because the three-input structure is what defines the type. The rules file marks which inputs are relevant.

**Arity 4 and above.** The function is a read-once formula: a random expression tree whose leaves are the inputs, each used exactly once. The nesting depth is drawn from a weighted distribution. Depth 1 is a single operator over all inputs. Depth d allows up to d levels of operators, and the generator splits the inputs randomly among the children at each level. Each operator is drawn from the operator mix. Each leaf is negated with the negation probability. A read-once formula over AND, OR, and XOR always depends on every input, so no input is wasted.

### Complexity scores

Every rule records its family, its SHJ type when there is one, its arity, its nesting depth, and a complexity score. The complexity score is the literal count of a minimal disjunctive normal form (DNF). The minimal DNF literal count is a proxy for Feldman's (2000) Boolean complexity, which counts literals in the minimal formula of any form. The proxy is exact to compute and comparable across arities. The proxy does not reproduce the human difficulty ordering of the SHJ types exactly (for example, type III scores 4 and type IV scores 6), so the SHJ type label is kept alongside the score.

### Duplicate rules and constant outputs

By default, two determined features may not have the same function of the same inputs, because two such features would be identical columns. A parameter allows duplicates.

Features that carry no information are wanted. A central question is whether models learn to attend to the informative features and ignore the others. An optional variance bound, off by default, resamples a rule whose output is true for a proportion of instances outside a configured range. The feature statistics output reports how informative every feature is, so we can check informativeness by observation instead.

### Automatic rules and rule files

Rules come from one of two sources, and both compile to the same internal representation.

- **Automatic.** Rules are sampled from the complexity parameters in the configuration file.
- **Rule file.** A YAML file lists rule templates, each with a sampling weight. A template can name an SHJ type, a fixed operator at a given arity, a single-input literal, or a compositional family with its own arity, operator mix, and nesting depth. A template can apply to all determined features, or only to IS, HAS, or CAN outputs. The file can also list explicit rules that fix a named output feature to a named expression. Determined features without an explicit rule are sampled from the templates. An explicit rule that breaks the layer constraints is a validation error.

Example rule file:

```yaml
templates:
  - {family: literal, weight: 0.1}
  - {family: fixed, arity: 2, operator: XOR, weight: 0.1}
  - {family: shj, type: IV, weight: 0.2}
  - {family: shj, type: VI, weight: 0.05, applies_to: can}
  - {family: compositional, arity: 4, operators: {AND: 1, OR: 1}, nesting_depth: 2, weight: 0.3}
explicit:
  - {output: CAN.3, expression: "(HAS.2 AND NOT IS.5) OR IS.7"}
```

### Expressions

Rules are written and printed as expressions. Literals are feature labels. The operators are `NOT`, `AND`, `OR`, and `XOR`, with parentheses. Precedence, from tightest to loosest, is `NOT`, `AND`, `XOR`, `OR`. Printed expressions put parentheses around every nested operator, so no reader has to rely on precedence. SHJ rules are printed as their minimal DNF over the actual input labels.

Internally, every rule is stored as its input list and a truth table. Evaluation is a table lookup: the index is the input values read as a binary number, with the first input as the most significant bit. The lookup is vectorized over all objects at once with NumPy.

## The tree

### Shape

- `superordinates`: the number of level-1 categories.
- `depth`: the number of category levels. Leaves are at level `depth`. With `depth: 1`, the superordinates are the leaves.
- `branching`: the number of children for a category at level L, for L from 1 to depth − 1. Each value is either a fixed number or a range `[min, max]` drawn uniformly for each category. The minimum is at least 1, so every leaf is at level `depth`. Branching is either one value for all levels or a list with one value per level.

### Superordinates

Each superordinate's free features are drawn from their base rates. Its determined features are then computed from the rules.

An optional bound limits the similarity between superordinates. The bound has three parameters:

- `metric`: `phi` (the Pearson correlation of binary vectors), `cosine`, or `jaccard`;
- `scope`: `free` (free features only), `is_has` (all IS and HAS features), or `all` (IS, HAS, and CAN features);
- `min` and `max`: the allowed range for every pair of superordinates; either can be null.

Superordinates are generated in index order. A candidate is accepted when its similarity to every accepted superordinate lies in the range. The generator tries rejection sampling first. If rejection sampling reaches `max_tries` without success, and `local_search` is on, the generator runs a search that flips free features of the candidate one at a time to reduce the largest violation. Computing a full vector is cheap, because determined features depend only on the same object's free features, so every candidate is checked on the configured scope. If no candidate satisfies the bound, the generator stops with an error that reports the closest similarity it reached and suggests which parameters to loosen. Phi is undefined for a constant vector; a candidate with an undefined similarity is rejected.

### Roles

Roles are assigned at every category, including leaves. The roles at a category govern how that category's children are generated. For a leaf, the children are its instances.

A feature that is defining at a category is defining at every descendant of that category, and is fixed for every instance below the category. Roles are assigned independently at every category. So a feature can be defining in one branch and undiagnostic in another.

At a category N at level L, let F be the free features that are not already defining at N's parent. For a superordinate, F is all free features. The generator draws, without replacement from F:

- `proportion_defining(L) × |F|` features that become newly defining at N;
- `proportion_characteristic(L) × |F|` features that are characteristic at N;
- the rest of F, which are undiagnostic at N.

The proportions are therefore proportions of the features still free to vary. Counts are rounded stochastically, so the expected count equals the proportion times |F|. The two proportions must sum to at most 1 at every level.

### Generating children

A child of category N gets its free features from N's generative vector, according to N's roles:

- **defining** at N (inherited or new): the child copies N's value;
- **characteristic** at N: the child copies N's value with probability `characteristic_probability(L)`, and flips it otherwise;
- **undiagnostic** at N: the child draws a new value from the feature's base rate.

The child's determined features are then computed from the rules. A defining value can be 0: a category can be defined by lacking something.

### Distinct leaves

With `require_distinct_leaves` on, no two leaves may have the same generative vector on their IS, HAS, and CAN features. When a new leaf duplicates an earlier leaf, the generator redraws the new leaf's non-defining free features from its parent, up to `distinct_max_tries` times, and then stops with an error. Leaves are checked in category order. Whether or not the option is on, the similarity output lets us inspect distinctness by observation.

## Instances

Each leaf gets a number of instances, set by its own parameter: a fixed number, or a range `[min, max]` drawn for each leaf. The instance count is independent of the branching parameters.

Instances are not a new level of the tree. Instances receive no roles of their own. Each instance follows the roles assigned at its leaf:

- **defining** at the leaf: the instance copies the leaf's value;
- **characteristic** at the leaf: the instance copies the leaf's value with probability `instances.characteristic_probability`, and flips it otherwise;
- **undiagnostic** at the leaf: the instance draws a new value from the feature's base rate.

The instance's determined IS, HAS, and CAN features are then computed from the rules. The instance characteristic probability is separate from the tree schedules, so within-category variability can be tuned apart from the tree.

## Depth schedules

Four parameters can vary with level: `proportion_defining`, `proportion_characteristic`, `characteristic_probability`, and `branching`. Each of the first three accepts one of these forms:

- a bare number: constant at every level;
- `{schedule: linear, start: a, end: b}`: a at level 1, b at level `depth`, linear between;
- `{schedule: exponential, start: a, asymptote: c, rate: r}`: the value at level L is c + (a − c) × exp(−r × (L − 1));
- `{schedule: list, values: [...]}`: one value per level, with exactly `depth` values.

Branching accepts a bare value (a number or a range) or a list with one value per level from 1 to depth − 1. The resolved per-level values of every schedule are written to the resolved configuration, so every run records the exact numbers it used.

## Node vectors

Every category, including every leaf, gets three vectors over all features (ISA, IS, HAS, CAN):

1. **Generative vector.** The vector the category's children were generated from.
2. **Defining vector.** The value of every feature that is fixed for all members of the category, and NaN for every other feature. The fixed features are:
   - free features defining at the category (inherited or new);
   - determined features fixed by rule (see below);
   - ISA features: 1 for the category and its ancestors, 0 for every category that is neither an ancestor nor a descendant, and NaN for strict descendants.
3. **Mean vector.** The mean of every feature over all instances below the category.

### Fixed by rule

A determined feature is fixed by rule at category N when its output is the same for every setting of the free features that are not defining at N. For example, an AND rule with a defining input of 0 is fixed at 0. The test uses each determined feature's cone: the free features it depends on, directly or through other determined features. When the number of non-defining features in the cone is at most 20, the generator enumerates all their settings, which is exact. Above 20, the generator uses a local test that treats each non-fixed input of the rule as independent. The local test never marks a feature fixed when it is not, but it can miss a fixed feature. The roles output marks which test was used.

## Configuration

A run is defined by one YAML file. Unknown keys are errors. Every error names the file and the field. The example below shows every parameter with its default value.

```yaml
name: default
seed: 1                          # master seed, a 64-bit unsigned integer

features:
  is:  {count: 40, proportion_determined: 0.25, expected_true_free: 6}
  has: {count: 40, proportion_determined: 0.25, expected_true_free: 6}
  can: {count: 20}
  base_rate_override: null       # one probability for all free features; replaces expected_true_free
  base_rate_heterogeneity: null  # null: equal base rates within a type; a number: Beta concentration

rules:
  source: automatic              # automatic or file
  file: null                     # path to a rule file when source is file
  max_chain_depth: 1
  arity: {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}      # weights
  operator_mix: {AND: 1, OR: 1, XOR: 1}          # weights
  negation_probability: 0.2
  arity_3_families: {I: 1, II: 1, III: 1, IV: 1, V: 1, VI: 1, compositional: 1}
  nesting_depth: {1: 0.5, 2: 0.5}                # compositional rules only
  input_type_weights: {is: 1, has: 1}
  overrides: {}                  # optional per-output-type settings, e.g. {can: {arity: {2: 1, 3: 1}}}
  allow_duplicate_rules: false
  variance_bound: null           # [low, high] proportion of instances true; null is off

taxonomy:
  superordinates: 4
  depth: 3
  branching: [2, 4]              # a number, a range, or a list with one value per level 1..depth-1

superordinates:
  similarity_bound: {metric: phi, scope: free, min: null, max: 0.3, max_tries: 10000, local_search: true}

inheritance:
  proportion_defining: 0.1
  proportion_characteristic: 0.5
  characteristic_probability: 0.9
  require_distinct_leaves: true
  distinct_max_tries: 1000

instances:
  per_leaf: [5, 10]              # a number or a range
  characteristic_probability: 0.9

analysis:
  similarity_metric: cosine      # phi, cosine, or jaccard
  similarity_features: non_isa   # non_isa or all
  max_pairs: 200000              # pairs above this number are sampled
```

Validation checks, at least: counts are non-negative; the two role proportions sum to at most 1 at every level; the branching minimum is at least 1; `max_chain_depth` is at least 1 when any IS or HAS feature is determined; the largest arity with nonzero weight fits the smallest eligible input pool; every schedule list has the right length; explicit rules obey the layer constraints.

## Determinism

Follow the determinism section of `docs/specs/MILESTONE_1.md`, implemented in Python. A stream's seed is SHA-256 of the master seed (8 bytes, little-endian) followed by the stream name (UTF-8). The digest, read as an integer, seeds `numpy.random.default_rng`. The streams are:

- `taxonomy:base_rates`
- `taxonomy:rules`
- `taxonomy:superordinates`
- `taxonomy:tree`
- `taxonomy:instances`
- `taxonomy:analysis` (pair sampling only)

Separate streams mean that changing the instance count never changes the rules or the tree, and changing the tree never changes the rules. All loops run in category order and feature order. No result may depend on set or dictionary iteration order.

## Outputs

A run writes one folder, by default `runs/taxonomy/<name>_seed<seed>/`. The `runs/` folder is ignored by git. CSV files have a header row and one ID column first. Binary values are written as 0 and 1. Missing values are written as `NaN`. Means are written with 6 decimal places. Feature columns are ordered ISA (in category order), then IS, HAS, and CAN (in index order).

| File | Contents |
| --- | --- |
| `config.yaml` | The fully resolved configuration: every default filled in, every schedule resolved to per-level values, the seed, the git commit hash (with a flag if the working tree had uncommitted changes), and the package version. |
| `features.csv` | One row per feature: label, type, free or determined, layer, base rate (free features only). |
| `rules.yaml` | One entry per determined feature: output, family, SHJ type, arity, nesting depth, inputs, relevant inputs, expression, truth table as a bit string, minimal DNF literal count, layer. |
| `tree.csv` | One row per category: label, parent, level, number of children, number of instances below. |
| `roles.csv` | One row per category and feature: the role of the feature at the category (`defining_inherited`, `defining_new`, `characteristic`, `undiagnostic`, `determined`, or `fixed_by_rule`), and for `fixed_by_rule` whether the exact or local test was used. |
| `categories_generative.csv` | One row per category: the generative vector. |
| `categories_defining.csv` | One row per category: the defining vector. |
| `categories_mean.csv` | One row per category: the mean vector over instances. |
| `instances.csv` | The feature matrix: one row per instance, with its label, its leaf, and every feature. |
| `similarity.csv` | One row per category: the three similarity statistics below. |
| `feature_stats.csv` | One row per feature: the statistics below. |
| `summary.yaml` | Run-level statistics below, and any warnings. |

Example rule entry:

```yaml
- output: CAN.3
  layer: 1
  family: shj
  shj_type: IV
  arity: 3
  nesting_depth: null
  inputs: [HAS.2, IS.5, IS.7]
  relevant_inputs: [HAS.2, IS.5, IS.7]
  expression: "(NOT HAS.2 AND NOT IS.5) OR (NOT HAS.2 AND NOT IS.7) OR (NOT IS.5 AND NOT IS.7)"
  truth_table: "11101000"
  min_dnf_literals: 6
```

The truth table lists outputs for input settings 000, 001, ..., 111, with the first input as the most significant bit.

### Similarity statistics

For each category N, computed with the configured metric and features:

- `within`: the mean pairwise similarity among N's children's generative vectors (for a leaf, among its instances);
- `between`: the mean similarity between N's children and the children of N's siblings (for a superordinate, the siblings are the other superordinates);
- `instances`: the mean pairwise similarity among all instances below N.

A statistic with no pairs, or with only undefined similarities, is written as NaN. When a statistic has more pairs than `max_pairs`, the pairs are sampled from the `taxonomy:analysis` stream.

### Feature statistics

For each feature: type, free or determined, layer, proportion of instances true, entropy in bits, the number of categories where the feature is defining (inherited and new separately), characteristic, undiagnostic, and fixed by rule, and, for each level L, the mutual information in bits between the feature and the category at level L, computed over instances.

### Summary

The summary reports: the counts of categories, leaves, and instances; the mean number of true features per instance for each type; the number of features that are constant across all instances; the number of duplicate leaves and duplicate instances; the range of superordinate similarities achieved, and the number of tries used; and every warning raised during the run.

## Python package

Put the generator in `python/semantic_world/taxonomy/`. Suggested modules:

```
python/semantic_world/taxonomy/
  __init__.py      # load_config, generate, TaxonomyResult
  __main__.py      # command line
  config.py        # configuration types, defaults, validation, schedules
  streams.py       # seeded random streams
  boolean.py       # truth tables, arity-2 enumeration, SHJ types, read-once formulas, minimal DNF, expressions
  rules.py         # layering, rule sampling, rule files
  tree.py          # superordinates, roles, children, distinct leaves
  instances.py
  fixed.py         # defining vectors and the fixed-by-rule test
  analysis.py      # similarity and feature statistics
  io.py            # writing the output folder
```

Module names are recommendations. `generate(config)` returns a `TaxonomyResult` holding NumPy arrays and polars data frames. `TaxonomyResult.write(path)` writes the output folder. The command line is:

```
python -m semantic_world.taxonomy data/taxonomy/default.yaml [--seed N] [--out DIR]
```

Put example configurations in `data/taxonomy/`: `default.yaml` with the values above, `tiny.yaml` small enough to read by eye (2 superordinates, depth 2, 8 IS, 8 HAS, 4 CAN, 3 instances per leaf), and one example rule file.

Minimal DNF can come from `sympy.logic.SOPform` (add `sympy`, pinned to a minor version) or from a small Quine–McCluskey implementation. Either is fine.

## Build stages

Work on a branch for each stage (for example, `taxonomy-stage-1`). Each stage ends with its tests passing and a commit. Run the full check list in `CLAUDE.md` before each commit.

1. **Configuration, streams, and schedules.** Configuration types, defaults, validation, and the resolved configuration. The stream seeds. The four schedule forms. *Accept:* the example configurations load; a set of deliberately broken configurations each fail with an error naming the field; schedules give the right per-level values.
2. **Boolean functions.** Truth tables, arity-2 enumeration, the SHJ types, read-once formulas, the expression parser and printer, and minimal DNF. *Accept:* the arity-2 enumeration gives exactly 10 functions, each depending on both inputs; the six canonical SHJ forms generate classes of sizes 6, 6, 24, 8, 24, and 2 under permutation and negation of inputs, and cover all 70 balanced three-input functions; every read-once formula depends on every input; every printed expression parses back to the same truth table.
3. **Rules.** Layering, rule sampling, duplicate checks, the variance bound, and rule files. *Accept:* the rule graph is acyclic; no rule reads an ISA or CAN feature; every layer-k rule has an input from layer k−1; explicit rules that break the layer constraints are rejected; with duplicates disallowed, no two rules share inputs and truth table.
4. **The tree.** Superordinates with the similarity bound, roles, children, and distinct leaves. *Accept:* all superordinate pairs lie within the bound; an infeasible bound fails with the closest value reached; every defining feature at a category has the same value in all descendant categories; on a large tree, the empirical copy rate of characteristic features is within tolerance of the configured probability; with distinct leaves on, no two leaves share a vector.
5. **Instances, node vectors, and the fixed-by-rule test.** *Accept:* every non-NaN entry of every defining vector matches every instance below the category; on the tiny configuration, the fixed-by-rule result matches brute-force enumeration for every category and determined feature; the mean vectors match means computed directly from `instances.csv`; recomputing every determined feature from the instances' free features reproduces `instances.csv` exactly.
6. **Outputs, statistics, and the command line.** *Accept:* every CSV loads in polars with the expected columns; labels follow the label table; the same configuration and seed give byte-identical output folders; different seeds give different outputs; changing only the instance count leaves `rules.yaml`, `tree.csv`, and `categories_generative.csv` unchanged; the default configuration runs in under a minute on a laptop.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. Rule files may include explicit rules for named output features, in addition to weighted templates.
2. Exact duplicate rules are disallowed by default.
3. Instance labels use periods throughout: `I1.3.2.5`.
4. Free features are numbered before determined features within each type.
5. A Beta-distributed base-rate heterogeneity option exists, off by default.
6. The complexity score is the minimal DNF literal count, reported alongside the SHJ type.

## References

- Feldman, J. (2000). Minimization of Boolean complexity in human concept learning. *Nature*, 407, 630–633.
- Nosofsky, R. M., Gluck, M. A., Palmeri, T. J., McKinley, S. C., & Glauthier, P. (1994). Comparing models of rule-based classification learning: A replication and extension of Shepard, Hovland, and Jenkins (1961). *Memory & Cognition*, 22, 352–369.
- Shepard, R. N., Hovland, C. I., & Jenkins, H. M. (1961). Learning and memorization of classifications. *Psychological Monographs*, 75(13, Whole No. 517).
