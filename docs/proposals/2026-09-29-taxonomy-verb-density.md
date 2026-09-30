# Proposal: a density range for verbs

September 29, 2026. Raised while reviewing stage 11 of `docs/specs/TAXONOMY_RELATIONS.md` on real runs. Status: decided. Jon chose a target range with resampling on September 29, 2026. To be built as stage 12a, after stage 12.

## The question

A verb's relation is the conjunction of the constraints of its true verb features plus its own constraint. Every constraint removes pairs, and the removals multiply. Across 10 seeds of `data/taxonomy/relations.yaml` (78 verbs):

- 27% of verbs hold for no pair of instances at all;
- 65% of verbs hold for fewer than 0.5% of ordered instance pairs, and the median verb holds for 0.07%;
- only 15% of verbs hold for more than 5% of pairs.

The median proportion falls with the number of constraints: 5% with 2 constraints, 2.7% with 3, 0.09% with 4, and 0 with 6 or more. The median verb has 4 or 5 constraints. The specification has no check on how often a verb holds, so dead and nearly dead verbs are common.

## Options

1. **A target range with resampling.** Each verb must hold for a proportion of leaf pairs inside a configured range. The generator resamples until each verb fits, and warns when it cannot.
2. **Tune the defaults only.** Fewer true verb features per verb and looser constraints, with no check.
3. **Leave the design, and report the sparsity.**

## Decision

Option 1.

## Design

### The measure

A verb's **leaf-pair density** is the proportion of ordered pairs of leaves (L1, L2), including pairs with L1 = L2, for which the verb's relation holds between the leaves' generative vectors, with each leaf's own scalar values. The measure uses the noun tree only, never the instances, so:

- changing the instance count never changes the constraints or the verb tree;
- changing the verb settings never changes any noun output.

Constraints now depend on the noun tree, which is fixed before any verb is generated. That is a new dependency. The determinism section of the relations specification should say so.

### New parameters

```yaml
verbs:
  density: {min: 0.01, max: 0.3, max_tries: 200}   # null turns the check off
  constraint_min_density: 0.1                      # null turns the check off
```

- `density`: the allowed range of leaf-pair density for every verb.
- `constraint_min_density`: the smallest leaf-pair density allowed for any single constraint, measured with that constraint alone. The floor keeps any one constraint from being so restrictive that no verb using it can reach `density.min`.

### The procedure

1. **Constraints of verb features.** When each verb-feature constraint is sampled, resample it until its leaf-pair density is at least `constraint_min_density`, up to `density.max_tries` tries.
2. **The verb tree, then each verb in category order.** Compute the density of the verb's relation without its own constraint.
   - If that density is at or above `density.min`, sample the verb's own constraint until the full relation lies inside the range, up to `max_tries` tries. An own constraint can only remove pairs, so it can bring a verb that is too dense down into the range.
   - If that density is below `density.min`, redraw the verb's non-defining verb features from its parent, as the distinct-leaves check does, then retry. The verb's defining features stay fixed.
3. **When the range cannot be reached**, for example because the defining features of the verb's ancestors already make the base relation too sparse, keep the closest result and add a warning naming the verb and its density.

The order of the build changes: verb-feature constraints are sampled before the verb tree, because the verb tree's leaf checks need them. Every draw uses the same streams as before (`taxonomy:constraints` for constraints, `taxonomy:verb_tree` for verb features).

### Outputs

- `verb_stats.csv` gains `leaf_pair_density` and `tries`.
- `constraints.yaml` gains `leaf_pair_density` for every constraint.
- `summary.yaml` lists the number of verbs outside the range. Each such verb also appears in the warnings.

### Acceptance tests

- On the tiny relations configuration, every leaf-pair density matches brute-force evaluation over all leaf pairs.
- Across 10 seeds of `relations.yaml`, every verb lies inside the range, or appears in the warnings. Report the share of verbs inside the range, and the distribution of instance-pair proportions, beside the numbers above.
- Changing only the instance count leaves `constraints.yaml`, `relations.yaml`, and every verb output unchanged.
- With both checks set to null, the output equals the output before this change.
