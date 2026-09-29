# Proposal: when the variance bound is checked

September 29, 2026. Raised while building stage 3 of `docs/specs/TAXONOMY_GENERATOR.md`. Status: waiting for Jon.

## The question

The specification says that the optional variance bound "resamples a rule whose output is true for a proportion of instances outside a configured range". Rules are sampled in stage 3, before the tree and the instances exist. The proportion of instances with a true output is known only after stage 5. The specification also says that separate streams mean "changing the instance count never changes the rules". A rule resampled because of an instance statistic would depend on the instances. The two statements cannot both hold if the bound is read literally.

## Options

1. **Check an expected proportion at rule-sampling time.** For each candidate rule, compute the probability that the output is true when every free feature is drawn independently at its base rate. The computation enumerates the rule's cone (the free features it depends on, directly or through other determined features) when the cone has at most 20 features, and samples it from the `taxonomy:rules` stream otherwise. Rules stay independent of the tree and the instances. The estimate ignores the correlation that inheritance creates among instances, so the realized proportion, which `feature_stats.csv` reports, can differ from the expected one.
2. **Check the realized proportion after instances exist, and regenerate.** Sample rules, build the tree and the instances, compute the proportions, resample the offending rules, and recompute the determined features (the free features are unchanged). With the similarity bound on the `free` scope, the tree does not need to change. Rules then depend on the tree and the instances, so changing the instance count can change the rules whenever the bound is on, and the run is slower.
3. **Drop the variance bound.** Report informativeness in `feature_stats.csv` only, as the specification already suggests, and let us filter or rerun by hand.

## Recommendation

Option 1. It answers the question the bound is for, whether a rule is close to constant given the base rates, and it keeps every stated determinism property. The realized proportion remains available in `feature_stats.csv` for checking by observation.

Stage 3 implements option 1 provisionally. The check lives in `expected_true_proportion` in `python/semantic_world/taxonomy/rules.py`, with the cone limit shared with the fixed-by-rule test of stage 5.
