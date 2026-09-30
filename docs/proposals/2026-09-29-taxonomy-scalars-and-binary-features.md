# Proposal: which binary features scalar settings leave unchanged

September 29, 2026. Raised while building stage 8 of `docs/specs/TAXONOMY_RELATIONS.md`. Status: decided. Jon chose option 1 on September 29, 2026.

## The question

The relations specification says, in "Values and inheritance" and again in "Determinism", that adding scalars or changing the scalar settings "never changes any binary feature of any category or instance". Stage 7 satisfied that statement, and its acceptance test checked it. Stage 8 adds threshold literals to the input pool of every rule. A determined feature whose rule reads `SC.2 > 0.4127` now depends on a scalar value. Turning scalars on changes the input pool, so different rules are drawn, and changing the drift changes the model distribution, so the thresholds move. Either way, determined binary features change. The statement as written cannot hold once rules read scalars.

## Options

1. **Narrow the statement to what the streams guarantee.** Changing the scalar settings never changes the tree, the roles, or any free binary feature of any category or instance. Determined features can change, because their rules may read threshold literals. When `rules.input_type_weights.scalar` is 0, no rule reads a scalar, and every binary feature is unchanged, as before.
2. **Keep every binary feature independent of the scalars** by evaluating threshold literals against a scalar value that does not depend on the scalar settings. There is no such value: the literal is about the object's scalar.
3. **Keep the strong statement only for turning scalars on**, by drawing threshold literals from a stream that is consumed even when scalars are off. This would change the rules of existing configurations and break the regression rule.

## Recommendation

Option 1. It is what the separate streams actually give, it keeps the regression rule, and it matches the purpose of the statement: adding scalars does not disturb the inheritance structure. Stage 8 implements option 1 provisionally. The stage 7 acceptance test now checks the free features, the tree, and the roles, and a second test checks every binary feature when the scalar weight is 0. The thresholds themselves come from the `taxonomy:rules` stream, so they never depend on the tree or the instances.
