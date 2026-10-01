# Proposal: decisions for stage 2 of the corpus generator

October 1, 2026. Raised at the end of stage 1 of `docs/specs/CORPUS_GENERATOR.md`, and answered by Jon before stage 2 (propositions and truth). Status: decided. The specification was updated to match, as decisions 23 to 26.

## Jon's decisions

### 23. Where are the words for verb categories used?

The lexicon gives a word to every verb category ("hunt" above "chase"), but the specification never said where the word is used.

**Decision.** In two places. The first is class-level and instance-level capacity sentences ("owls hunt mice"). The second is naming events: each event's verb is named at a level drawn by weight, the same way nouns choose between "penguin" and "bird". A new setting, `mention.verb_level_weights`, gives the weights, heaviest at the leaf by default. Stage 2 builds the capacity sentences. The event half belongs to stages 3 and 5.

### 24. Do patient projections appear at the class level?

The specification listed exposed patient projections ("is edible") as instance-level predicates only.

**Decision.** They also appear at the class level ("most mice are edible"). Truth for `most`, `some`, and the generic comes from the share of instances in the subject set that have the projection. The share uses the same instance values as the instance-level sentences. `all` and `no` are allowed only under the observed reading (`quantifiers.all_grounding: observed`).

### 25. Can a rule term pass the limits on modifiers?

A rule term can have four literals, and the limits allow three adjectives and two with-phrases.

**Decision.** Rule statements are exempt from the limits on adjectives, with-phrases, and content words. Rules of varied complexity are a core feature of the world, so long terms must still be stated. An optional cap, `propositions.rule_statements.max_literals`, is null (no cap) by default. When a cap is set, terms above it are skipped and counted.

### 26. The morphology switch

**Decision.** The key `on` becomes `enabled` in the configuration, the specification, and the data files. YAML reads a bare `on` as the boolean true, and the loader no longer needs a workaround. A configuration that still says `on` gets an error that names the new key.

## Choices made by Claude

These are engineering choices made while building stage 2. Each one is the working design unless Jon changes it.

1. **`no` and polarity.** In the logical form, a negative polarity denies the predicate ("most penguins can not fly"). The quantifier `no` replaces sentence negation, so `no` always has a positive polarity and counts as a negative proposition. `all` never has a negative polarity.
2. **The strongest true quantifier.** For a subject and a predicate, a document states `all` before `most` before `some`. For a negative fact, it states `no` before "most ... not" before "some ... not". The bare generic can replace any of them when the generic is true.
3. **`some` and the implicature.** `some` is true when the proportion is above 0. `quantifiers.some.exclude_all` limits what documents state, and never makes a `some` sentence false. So a false test item is never a `some` sentence that is merely odd.
4. **Negative membership.** "No penguins are fish" and "penguins are not fish" are true exactly when the two categories share no instance. The generic of a membership sentence always means all, whatever `quantifiers.generic.means` says. A sentence about a category names a category at its own level or above, never one below.
5. **Instance-level membership.** The predicate can be the instance's leaf or any category above it. The planner must keep the noun and the predicate apart ("the bird is a penguin", never "the penguin is a penguin").
6. **Scalar poles.** The comparison uses the population standard deviation. A comparison class whose values do not vary has no poles. A class-level pole uses the mean over the subject set. A pole in a restriction is relative to the subject's category ("big penguins" are big for a penguin), is never negated, and holds no binary feature fixed in the fixed test.
7. **The comparison class of an instance-level pole.** The logical form records the comparison class (`"class": "C1.3"`), because the truth of "the mouse is big" depends on the noun. The planner must name that category when it mentions the subject.
8. **The fixed test with a restriction.** The free features that are defining at the category, and the restriction's literals on free features, are held. The other free features in the cone are enumerated. A restriction's literal on a determined feature keeps only the settings that satisfy it. Above 2^20 settings, the taxonomy generator's local test is used on a copy of the category in which the held features are defining. The local test never calls a feature fixed when it is not.
9. **Projections and the generic that means all.** With `quantifiers.generic.means: all`, the generic of a patient projection is true when every instance of the subject set has the projection.
10. **Verbs.** A verb sentence needs at least one pair of distinct instances. The patient category can take a restriction, grounded like the subject's.
11. **Rule statements.** A rule statement takes `all`. Its bare generic is true too, and the planner can use it. A term that no instance satisfies is skipped as vacuous (decision 19), and counted. In the default world, 59 of 103 terms are stated: 13 read a scalar threshold, 27 need more than one relative clause, and 4 have no instance.
12. **False items.** A predicate swap keeps the predicate's kind, its patient, and its comparison class. A subject swap keeps the restriction, and takes a category of the same level. A quantifier swap keeps the polarity. A role swap exchanges the subject and the patient, with their restrictions. A false item is never vacuous.
13. **The grounding record.** `proportion` is always the share that has the predicate, whatever the polarity. `test` says how the truth was decided: `exact`, `local`, `observed`, `tree`, `mean`, or `value`.
14. **Negation's position.** Unchanged from stage 1: `after_auxiliary` or `before_auxiliary`.

## Still open

- **Stage 5: the rule statement in the rendering examples.** The specification renders "things with wings and with feathers can fly" as `ALL(...)`, but the sentence is a bare generic, which would be `GEN(...)`. Is a rule statement "all things with wings ..." (`ALL`), or the bare generic (`GEN`)? Stage 2 builds both forms.
- **Stage 6: `all` in the test sets.** Under the law-like reading, "all penguins swim" is false when every penguin in the world swims but nothing fixes it. In the default world, 1,158 of the 2,278 pairs of a category and a feature on which every instance agrees are not fixed, 1,059 of them at the leaf level. A quantifier swap can therefore make a false `all` or `no` item that no instance contradicts. Should the test sets keep such items, leave them out, or mark them?
- **Stage 6:** what makes an event-level test item false, and which instance does an instance-level test item refer to outside a document? (From the first proposal.)
