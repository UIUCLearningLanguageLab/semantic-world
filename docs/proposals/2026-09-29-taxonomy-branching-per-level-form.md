# Proposal: the per-level form of `taxonomy.branching`

September 29, 2026. Raised while building stage 1 of `docs/specs/TAXONOMY_GENERATOR.md`. Status: decided. Jon chose option 1 on September 29, 2026.

## The question

The specification says that `branching` accepts "a bare value (a number or a range) or a list with one value per level from 1 to depth − 1". A range is itself a list, `[min, max]`. So the default, `branching: [2, 4]` with `depth: 3`, has two readings: a range of 2 to 4 at both levels, or a fixed 2 at level 1 and a fixed 4 at level 2. The two readings give different trees. We need one rule that tells them apart.

## Options

1. **A bare list is always a range. The per-level form uses the schedule mapping.** A list of two integers is a range. A per-level list is written `{schedule: list, values: [...]}`, with exactly `depth − 1` values, each a number or a range. Example: `branching: {schedule: list, values: [2, [3, 5]]}`.
2. **Disambiguate by length.** A list is a range when it has exactly two integers and a per-level list otherwise. Then a per-level list of two fixed numbers cannot be written when `depth` is 3.
3. **Nested lists for the per-level form.** A per-level list must nest each level: `[[2], [3, 5]]`. Unusual to read and write.

## Recommendation

Option 1. The other three level-dependent parameters already use `{schedule: ...}` mappings, so the per-level form of `branching` reads the same way. The resolved configuration writes `branching` in this form, with one value per level, which is what "every schedule resolved to per-level values" asks for. The default `[2, 4]` keeps the meaning the specification's "Shape" section gives it: a range drawn for each category.

Stage 1 implements option 1 provisionally. If Jon prefers another option, the change is confined to `resolve_branching` in `python/semantic_world/taxonomy/config.py`, its tests, and the comment in `data/taxonomy/default.yaml`.
