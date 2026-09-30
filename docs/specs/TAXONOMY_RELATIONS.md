# Taxonomy relations: scalar dimensions and transitive verbs

Draft, September 29, 2026. This document is the build specification for two extensions to the taxonomy generator: scalar dimensions, and two-argument relations (transitive verbs) with a verb taxonomy. The specification is written for handoff to Claude Code, or to any developer, and should be complete together with `docs/specs/TAXONOMY_GENERATOR.md`.

## How this specification relates to the base specification

`docs/specs/TAXONOMY_GENERATOR.md` (the base specification) is built, in `python/semantic_world/taxonomy/`. This specification extends the base generator and does not replace it. Everything in the base specification still holds, including the two decided proposals in `docs/proposals/`. Where this specification changes base behavior, the section "Changes to the base generator" says so explicitly.

One rule protects the existing work: **with scalar dimensions set to 0 and relations turned off, the generator produces exactly the output it produces today.** Every output file except `config.yaml` must be byte-identical to the current output for the same configuration and seed, and `config.yaml` may differ only by added keys. A regression test enforces the rule (see "Build stages").

## Goal

The base generator gives objects properties (IS), parts (HAS), and one-place actions (CAN). Natural language also has relations between objects: "penguins chase fish", "barbers cut hair", "owls eat snakes". This specification adds:

1. **Scalar dimensions**, such as size and strength. Scalars inherit down the tree by drift, and rules can test them against thresholds ("CAN A if SIZE > x").
2. **Transitive verbs.** A verb is a relation between an agent and a patient. Whether a verb holds for a pair of objects is computed from the two objects' features.
3. **A verb taxonomy.** Verbs are generated from a tree of verb categories with verb features, using the base generator's own machinery. Verbs in the same branch share argument requirements, so the verb system has structure like the noun system.
4. **Derived one-place features.** Each verb gives every object two derived features: whether the object can be the verb's agent (CAN), and whether the object can be the verb's patient (CAN BE).

These extensions let us control two kinds of semantic structure independently: taxonomic similarity (lions and tigers share features) and thematic relatedness (lions and deer take part in the same events).

Out of scope: verbs with three or more arguments, event schemas with changing states, and the other items listed in "Future additions".

## Labels

Labels follow the base convention: formal labels, indices starting at 1, and periods between indices.

| Object | Label | Example |
| --- | --- | --- |
| Scalar dimension | `SC.<n>` | `SC.2` |
| Verb category (a node of the verb tree) | `V<i>.<j>...` | `V1`, `V1.2` |
| Verb (a leaf of the verb tree) | `V<i>.<j>...` | `V1.2.3` |
| Verb feature | `VF.<n>` | `VF.4` |
| Constraint attached to a verb feature | `K.VF.<n>` | `K.VF.4` |
| A verb's own constraint | `K.<verb label>` | `K.V1.2.3` |
| Agent projection of a verb | `CAN.<verb label>` | `CAN.V1.2.3` |
| Patient projection of a verb | `CANBE.<verb label>` | `CANBE.V1.2.3` |

Inside constraint expressions, a literal names its argument with a prefix: `a.` for the agent and `p.` for the patient, as in `a.HAS.4`, `p.IS.7`, and `a.SC.1`.

## Part A: scalar dimensions

### Values and inheritance

The number of scalar dimensions is a parameter, `scalars.count`. With 0, the world is purely binary, as before.

- A superordinate's value on each scalar is drawn from the standard normal distribution.
- A child category's value is its parent's value plus normal noise with standard deviation `scalars.drift(L)`, where L is the parent's level. `drift` accepts every schedule form that the base specification defines for level-dependent parameters.
- An instance's value is its leaf's value plus normal noise with standard deviation `scalars.instance_drift`.

Scalars are always free. No rule determines a scalar. Scalars have no roles (defining, characteristic, undiagnostic): drift plays the role that inheritance roles play for binary features. A drift of 0 at a level fixes the scalar for that level's children.

Scalar values are drawn from their own streams (see "Determinism"), so adding scalars never changes the tree, the roles, or the free binary features of any category or instance. Determined features can change, because their rules may read threshold literals (see "Threshold literals in rules"). With `rules.input_type_weights.scalar` at 0, no rule reads a scalar, and every binary feature is unchanged.

### Threshold literals in rules

A rule's input can be a threshold literal on a scalar: `SC.2 > θ`. Inside the rule, a threshold literal acts exactly like a binary input: it is true or false for each object. Negating the literal gives `SC.2 <= θ`, so the negation probability also decides the direction of the comparison.

- **Thresholds.** θ is drawn at rule-sampling time as a quantile of the scalar's model distribution: the normal distribution with mean 0 and variance 1 plus the sum of the squared drifts down to the instances. The quantile is drawn uniformly from `scalars.threshold_quantiles`. Thresholds are rounded to 4 decimal places when drawn, so printed expressions parse back exactly. The model distribution depends only on the configuration, so thresholds never depend on the tree or the instances.
- **The input pool.** Threshold literals join the input pool of every rule (determined IS, determined HAS, and CAN rules), weighted by `rules.input_type_weights.scalar`. A threshold literal is a layer-0 input. One rule uses at most one literal per scalar.
- **Expressions.** A threshold literal is written `SC.2>0.4127` or `SC.2 > 0.4127`. The expression parser and printer, explicit rules in rule files, and `rules.yaml` all accept threshold literals. Each `rules.yaml` entry lists its threshold literals with their thresholds.
- **Expected proportions.** The expected-proportion calculation for the variance bound treats a threshold literal as an independent input that is true with probability 1 − (its quantile).
- **Fixed by rule.** A threshold literal counts as fixed at a category when every drift below the category's level, including the instance drift, is 0. Otherwise the literal counts as free to vary.

### Outputs for scalars

- `features.csv` gets one row per scalar, with type `scalar`.
- `instances.csv` and the three category-vector files get scalar columns after the CAN columns, written with 6 decimal places. In `categories_defining.csv`, a scalar column holds the category's value when the scalar is fixed for the category's members (all drift below is 0), and NaN otherwise.
- `feature_stats.csv` gets one row per scalar: mean, standard deviation, and for each level L the proportion of variance explained by the category at level L (η²), in place of mutual information. Binary-only columns are NaN.
- Similarity statistics and the superordinate similarity bound use binary features only. Scalars are excluded.
- With `scalars.thermometer_bins: k` above 0, the generator also writes `instances_scalar_codes.csv`: for each scalar, k binary columns `SC.<n>>q<j>`, one per quantile of the realized instance values. The file gives binary-only models a binary view of the scalars.

## Part B: transitive verbs

### Relations

A verb v is a relation R_v(a, p) between an agent instance a and a patient instance p. R_v is a conjunction of constraints:

R_v(a, p) = C_1(a, p) AND C_2(a, p) AND ...

Each constraint is a rule over the features of one or both arguments. The inputs of a constraint are binary IS and HAS features (free or determined) and scalar threshold literals, from either argument. No constraint reads an ISA feature, a CAN feature, or a projection (see "Projections").

Two limits on arguments:

- An instance is never related to itself: R_v(a, a) is false for every verb.
- Two different instances of the same leaf can be related: a penguin can chase another penguin.

### Constraint families

Five families of constraints are built now. The weights in `verbs.constraint_families` choose among them.

1. **Agent condition.** A rule over the agent's features only. Built exactly like a CAN rule, with the same complexity settings.
2. **Patient condition.** A rule over the patient's features only.
3. **Cross-role rule.** A Boolean rule whose inputs come from both arguments, with at least one input from each. Built with the same complexity settings as other rules (arity, operator mix, SHJ types at arity 3, read-once formulas, negation), over the combined pool of agent and patient literals. Every cross-role rule must depend on at least one agent input and at least one patient input. Resample otherwise.
4. **Key-lock rule.** A named special case of the cross-role rule: an OR of k pairs, each pair an AND of one agent literal and one patient literal, as in `(a.HAS.4 AND p.IS.7) OR (a.HAS.9 AND p.IS.2)`. k is drawn from `verbs.key_lock_pairs`. Literals are negated with the negation probability. Key-lock rules model instruments and materials: barbers cut hair and lumberjacks cut wood.
5. **Scalar comparison** (only when `scalars.count` is above 0). A comparison between the arguments' scalars:
   - order: `a.SC.i - p.SC.j > m`;
   - window: `m1 < a.SC.i - p.SC.j < m2`.
   
   The scalars i and j are the same by default. With probability `comparison.cross_dimension_probability` they differ, as in "a's strength exceeds p's weight". A comparison is a window with probability `comparison.window_probability`, and an order otherwise. Margins are drawn as quantiles, from `comparison.margin_quantiles`, of the model distribution of the difference between two independent instances, and are rounded to 4 decimal places. Comparisons model food chains and dominance: owls eat things smaller than themselves, but not much smaller.

Constraint expressions use the base syntax plus role prefixes, threshold literals, and comparisons, for example `(a.HAS.4 AND p.IS.7) OR (a.HAS.9 AND p.IS.2)`, `p.SC.1 <= 0.2031`, `0.1500 < a.SC.1 - p.SC.1 < 1.2200`. Every constraint is stored as its Boolean skeleton (a truth table over its literals) plus the definitions of its literals, and printed expressions parse back to the same constraint.

### The verb taxonomy

Verbs are generated by the base generator's own machinery, applied to a second tree.

- **Verb features.** `verbs.features.count` binary verb features `VF.<n>`, all free, with base rate `verbs.features.expected_true / count`. Verb features have no rules among themselves.
- **The verb tree.** Built exactly like the noun tree: superordinates, depth, branching, per-category roles (defining, characteristic, undiagnostic), inheritance schedules, and the option to require distinct leaves. The settings live under `verbs.taxonomy` and `verbs.inheritance`, with the same keys and forms as the base configuration. Verb superordinates have an optional similarity bound, off by default.
- **Verbs.** The leaves of the verb tree are the verbs. Verbs have no instances.
- **Constraints of verb features.** Every verb feature gets one constraint, `K.VF.<n>`, drawn from the constraint families.
- **A verb's relation.** The conjunction of the constraints of every verb feature that is true for the verb, plus, when `verbs.own_constraint` is on, one constraint of the verb's own, `K.<verb label>`. The own constraint keeps two verbs with the same verb features from being the same relation.

**Base relations.** Every verb category, not only every verb, has a base relation: the conjunction of the constraints of the verb features that are defining with value 1 at that category. Defining features stay defining all the way down, so every verb entails the base relation of every ancestor. For example, if V1 is a predation category, then R_{V1.2}(a, p) implies R_{V1}(a, p) for every verb V1.2 below V1. Shared base relations are therefore not a separate mechanism. They are the defining structure of the verb tree.

With `verbs.taxonomy.depth: 1` and no defining features, verbs are independent of each other. That is the flat case.

### Projections

Every verb gives every object two derived one-place features:

- **Agent projection** `CAN.<verb>`: true when some possible patient would make R_v true with this object as agent.
- **Patient projection** `CANBE.<verb>`: true when some possible agent would make R_v true with this object as patient.

"Possible" means any combination of features that an object could have, not only the objects that exist in the run. So "barbers can cut" is true because of the barber's features, not because some hair happens to be in the data.

**Computing projections exactly.** For the agent projection, the patient-side literals of the verb's constraints are treated as unknowns. The generator enumerates the settings of the free features in the cones of those literals (the same cone machinery as the fixed-by-rule test). For each setting, the generator checks, scalar by scalar, that the intervals implied by the patient's threshold literals and the comparisons with this agent's scalars intersect. The projection is true when some setting satisfies every constraint. The patient projection is computed the same way with the roles swapped. When the unknown side has more than 20 free features in its cones, the generator uses the local test (each literal treated as independent) and marks the projection as approximate.

**Extensional projections.** The generator also reports, for each instance and verb, whether the instance actually has a partner among the run's instances, as agent and as patient. An actual agent is always a possible agent, and a test checks the inclusion.

**Exposure.** Projections are ground truth. Whether models see them is a parameter. `verbs.projections.expose_agent` and `verbs.projections.expose_patient` give the proportions of agent and patient projections that appear as columns in `instances.csv`, chosen at random. The defaults are 1.0 for agent projections, because they parallel the base CAN features, and 0.25 for patient projections, because natural languages lexicalize some patient capacities ("edible", "breakable") and not others. All projections, exposed or not, are written to `projections.csv`. Projections never feed any rule or constraint.

### Evaluating relations

A relation is a pure function of two instances' features, so the generator never needs to store every pair. `result.relations.holds(verb, agents, patients)` evaluates any pairs on demand.

The generator computes and writes three things:

- **Category proportions.** For every verb, and for every pair of categories at the same level (leaves, and every level above), the number of instance pairs where the relation holds, the total number of pairs, and the proportion. Rows with a proportion of 0 are left out. Category proportions are what generic statements describe: "penguins chase fish" is a proportion near 1 for the leaf pair, and "birds eat worms" is a proportion at a higher level.
- **Sampled pairs.** For every verb, `verbs.pairs.sampled_true` pairs drawn uniformly from the true pairs and `verbs.pairs.sampled_false` pairs drawn uniformly from the false pairs, from the `taxonomy:pairs` stream. Sampled pairs are what episodic statements describe: "this penguin chased that fish". A verb with fewer true pairs than requested gets all of them, and the summary says so.
- **Verb statistics.** For every verb: the proportion of all pairs that are true, the number of actual agents and patients, the number of constraints and their families, the proportion of true pairs whose reverse is also true, and the proportion of true pairs within one leaf.

Exact evaluation covers every ordered pair of distinct instances, computed in chunks. Each literal is evaluated once per instance, and the chunks combine the agent and patient literal matrices. When the number of pairs exceeds `verbs.pairs.max_exact_pairs`, the generator estimates category proportions from a uniform sample of pairs of that size, and marks the proportions as estimates.

### Thematic relatedness

For every pair of leaves, the generator reports a thematic relatedness score: the sum, over verbs and over both directions, of the category proportions for that pair. The score sits beside the taxonomic similarity statistics already in the output, so the two kinds of structure can be compared directly.

## Changes to the base generator

- `FeatureSet` gains scalar dimensions. The base matrix of binary features is unchanged, and scalar values are held in a separate float array for categories and instances.
- `RuleSet.compute` takes scalar values as well as free binary values.
- Rule sampling adds threshold literals to the input pool when `scalars.count` is above 0.
- The expression parser and printer accept threshold literals, and for constraints, role prefixes and comparisons.
- The tree builder and the instance generator draw scalar drift from new streams.
- The fixed-by-rule test, the expected-proportion calculation, node vectors, feature statistics, and the output writer handle scalars as described in Part A.
- `rules.input_type_weights` accepts a `scalar` key, with default weight 1. The weight has no effect when `scalars.count` is 0.

None of these changes may alter any output when `scalars.count` is 0 and `verbs` is null. In particular, no existing stream may draw a different sequence of numbers when scalars are off.

## Configuration

New top-level keys, with defaults. The defaults keep scalars and verbs off, so existing configurations behave as before.

```yaml
scalars:
  count: 0
  drift: 0.5                     # a schedule over levels 1..depth-1: the standard deviation of a child's change from its parent
  instance_drift: 0.2
  threshold_quantiles: [0.2, 0.8]
  thermometer_bins: 0            # 0 writes no thermometer file

rules:
  input_type_weights: {is: 1, has: 1, scalar: 1}

verbs: null                      # null: no verbs. The example below shows every key with its default.
```

Example of a `verbs` block, with every key at its default value:

```yaml
verbs:
  features: {count: 12, expected_true: 3}
  taxonomy: {superordinates: 3, depth: 2, branching: [2, 3]}
  superordinates: {similarity_bound: null}
  inheritance:
    proportion_defining: 0.4
    proportion_characteristic: 0.4
    characteristic_probability: 0.9
    require_distinct_leaves: true
    distinct_max_tries: 1000
  own_constraint: true
  constraint_families: {agent: 1, patient: 1, cross: 1, key_lock: 1, comparison: 1}
  key_lock_pairs: {1: 0.5, 2: 0.3, 3: 0.2}
  comparison: {window_probability: 0.3, cross_dimension_probability: 0.2, margin_quantiles: [0.1, 0.9]}
  rules: {}                      # overrides of the rule-complexity settings for constraints; default: the top-level rules settings
  projections: {expose_agent: 1.0, expose_patient: 0.25}
  pairs: {sampled_true: 1000, sampled_false: 1000, max_exact_pairs: 50000000}
```

The `comparison` family has no effect when `scalars.count` is 0, and its weight is ignored. Validation adds, at least: every schedule has the right length; quantile ranges lie inside (0, 1) with low below high; at least one constraint family has a positive weight that can be used with the current settings.

Add example configurations to `data/taxonomy/`: `relations.yaml` (the default taxonomy with 2 scalars and the default verbs) and `tiny_relations.yaml` (the tiny taxonomy with 1 scalar, 4 verb features, and a verb tree of 2 superordinates with 2 verbs each), small enough to check by brute force.

## Determinism

New streams, added to the stream list: `taxonomy:scalars` (superordinate values and drift in the tree), `taxonomy:scalar_instances` (instance drift), `taxonomy:verb_tree` (verb features, the verb tree, and its roles), `taxonomy:constraints` (constraint sampling, and the choice of exposed projections), and `taxonomy:pairs` (pair sampling and pair estimates). Thresholds in noun rules come from `taxonomy:rules`.

The separate streams give these properties, and tests check each one:

- changing the scalar settings never changes the tree, the roles, or any free binary feature of any category or instance, and never changes any binary feature when `rules.input_type_weights.scalar` is 0;
- changing the instance count never changes the rules, the thresholds, the verb tree, or the constraints;
- changing the verb settings never changes any noun output;
- the same configuration and seed give byte-identical output folders.

## Outputs

New files, written only when the feature they describe is on:

| File | Contents |
| --- | --- |
| `instances_scalar_codes.csv` | Thermometer codes for the scalars (only with `thermometer_bins` above 0). |
| `verb_features.csv` | One row per verb feature: label, base rate, constraint label. |
| `verb_tree.csv` | One row per verb category: label, parent, level, number of children. |
| `verb_roles.csv` | One row per verb category and verb feature: the role, as in `roles.csv`. |
| `verbs_generative.csv`, `verbs_defining.csv` | One row per verb category: the generative and defining vectors over the verb features. |
| `constraints.yaml` | One entry per constraint: label, family, agent literals, patient literals (with thresholds), comparisons (with margins), expression, truth table of the Boolean skeleton, minimal DNF literal count. |
| `relations.yaml` | One entry per verb category: the constraints in its relation (base relation for internal categories, full relation for verbs) and the relation's expression. |
| `projections.csv` | One row per instance: every agent and patient projection, intensional, with a flag column for approximate projections, followed by the extensional projections. |
| `relation_proportions.csv` | One row per verb, agent category, and patient category at the same level, where the proportion is above 0: level, true pairs, total pairs, proportion, and whether the proportion is estimated. |
| `relation_pairs.csv` | The sampled pairs: verb, agent, patient, whether the relation holds. |
| `verb_stats.csv` | The verb statistics described above. |
| `thematic.csv` | One row per pair of leaves with a score above 0: the thematic relatedness score, and the taxonomic similarity of the two leaves' generative vectors under the configured metric. |

Exposed projections also appear as columns in `instances.csv`, after the scalar columns. The summary gains verb counts, constraint-family counts, and the proportion of approximate projections.

## Build stages

Continue the base numbering: stages 7 to 12. Work on one branch per stage (`taxonomy-stage-7`, and so on), each branched from the previous stage's branch unless Jon has merged it into `main`. Each stage ends with its tests passing, the full check list in `CLAUDE.md` passing, and a commit.

7. **Scalars in the tree and the instances.** Configuration, scalar values, drift, node vectors, outputs, statistics, and thermometer codes. *Accept:* the regression test (every existing configuration, with `scalars.count: 0`, gives byte-identical output files except `config.yaml`, which differs only by added keys); on a large tree, the standard deviation of child-minus-parent differences at each level is within tolerance of the configured drift; binary features are unchanged when only scalar settings change.
8. **Threshold literals in rules.** Sampling, expressions, rule files, expected proportions, and fixed-by-rule. *Accept:* the regression test still passes; printed expressions with threshold literals parse back to the same rule; with every drift at 0, a threshold literal is fixed at every category, and the fixed-by-rule result matches brute force on the tiny configuration with scalars.
9. **The verb tree.** Verb features, the verb tree, and verb roles, reusing the tree builder. *Accept:* the tree tests of stage 4, repeated for the verb tree; the flat case (depth 1) gives independent verbs; noun outputs are unchanged when only verb settings change.
10. **Constraints and relations.** The five families, constraint expressions, relation assembly, base relations, and pair evaluation. *Accept:* on the tiny relations configuration, every relation matches brute-force evaluation over all instance pairs; every cross-role and key-lock constraint depends on both roles; no instance is related to itself; every verb's relation implies the base relation of every ancestor, checked over all pairs; comparisons agree with direct arithmetic on the scalar values.
11. **Projections.** Intensional and extensional projections, and exposure. *Accept:* on the tiny relations configuration, intensional projections match brute-force enumeration over feature settings; every extensional projection implies the matching intensional projection; exposed columns in `instances.csv` equal the matching columns in `projections.csv`.
12. **Outputs, statistics, and the command line.** Category proportions, pair sampling, verb statistics, thematic relatedness, the new files, and `result.relations.holds`. *Accept:* category proportions match counts from brute-force evaluation on the tiny configuration; every sampled pair's `holds` value is correct; all the determinism properties above; every new CSV loads in polars with the expected columns; the relations example configuration runs in under five minutes on a laptop.

## Future additions

These were discussed and deliberately left out. They are documented here so the design leaves room for them.

**More constraint families.**

- **Similarity.** A weighted overlap between the two arguments' features, on a subset of features, compared with a threshold. Similarity models mating and competition between conspecifics as the effect of feature overlap. Heavy weights on a few features would capture biology, and weights on others could capture culture. Similarity relations are symmetric. A dissimilarity variant would require difference.
- **Relations derived from relations.** Converses (flee(p, a) whenever chase(a, p)), symmetric closures, and compositions or shared targets (two animals compete when they eat the same things).
- **Context and state.** Relations that depend on hunger, location, or time. These belong with event schemas.

**Modifiers for any family.**

- **A symmetry constraint**, forcing R(a, p) = R(p, a).
- **Stochastic strength**, so a relation holds with a probability when its constraints are met. Owls usually eat mice, but not always.

**Relations based on tree position.** No rule or constraint reads ISA features, and no constraint compares the arguments' positions in the tree. This design is deliberately anti-nominalist. Membership matters only through features. For example, competing with conspecifics comes from feature overlap, not from membership itself. A future parameter could allow relations that depend on tree position, to test what nominal structure adds.

**Event schemas.** Scripts like search, find, chase, attack, kill, and eat are ordered, and not arbitrary. A later specification could add:

- mutable states, a new feature type distinct from permanent IS and HAS features (alive, caught, hungry);
- preconditions and effects for verbs (kill requires alive(p) and makes p dead; eat requires p to be caught);
- schemas as ordered chains of verbs with transition probabilities, whose order follows from the preconditions.

Semantic World's rule engine already models actions with preconditions and effects, in a layer compatible with PDDL (the Planning Domain Definition Language). So event schemas are where the taxonomy generator will meet the world simulation. Event schemas are deliberately left out now, to avoid building the world simulation inside the generator.

**Other extensions.** Verbs with three or more arguments. Scalars determined by rules. Rules among verb features. Explicit constraints in rule files.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. Scalars and verbs are off by default, so existing configurations are unchanged. The new example configurations turn them on.
2. Scalars drift by a normal random walk, starting from a standard normal at the superordinates, and have no inheritance roles.
3. Thresholds and margins are quantiles of the model distribution implied by the configuration, not of the realized data, so they never depend on the tree or the instances.
4. A verb's relation is the conjunction of its constraints. Shared base relations are the defining features of the verb tree.
5. The key-lock rule is a named family, separate from the general cross-role rule, so its weight can be set on its own.
6. Default exposure: all agent projections, and a quarter of patient projections.
7. Category proportions are computed for categories at the same level only, not for mixed levels ("birds eat mice").

## References

- Estes, Z., Golonka, S., & Jones, L. L. (2011). Thematic thinking: The apprehension and consequences of thematic relations. *Psychology of Learning and Motivation*, 54, 249–294.
- Schank, R. C., & Abelson, R. P. (1977). *Scripts, plans, goals, and understanding*. Lawrence Erlbaum.
