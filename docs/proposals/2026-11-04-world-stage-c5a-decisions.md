# Proposal: decisions for stage c5a of the world-and-language refactor

November 4, 2026. Raised in the orientation for stage c5a of `docs/specs/WORLD_PHASE_C.md` ("c5a. The corpus at the class level: grammar and statements"), and while building the stage. Status: working design. Every choice below is Claude Code's unless Jon changes it. The choices are logged as WM.E353 and following in `docs/DECISIONS.md`; Jon's decision for the stage is WM.74.

Stage c5a builds, in this order: WM.74's rule for comparisons; the words `them` and `SOMEPLACE`; the grammar of the bound pronoun, of relative clauses with a prepositional predicate with `grammar.word_order.stranding`, and of quantified objects of prepositions; the causal statements with pair literals; the rule for conditional effects with pair conditions; the profile statements; and the regenerated runs with the word forms remade.

## Jon's decisions recorded by this stage

- **WM.74 (November 4, 2026, on stage c4c's open questions).** (1) A comparison of two places, or of two participants, may use the compared scalar's poles as modifiers only when the two mentions are measured against different comparison classes, because then the sentence does not follow from its own words ("the cold desert is hotter than the hot meadow" is stated when true). Within one comparison class the poles stay out. When nothing else tells the two mentions apart, the comparison is not made. `stats.yaml` counts the comparisons that use the compared poles across comparison classes and the comparisons the rule skips. (2) A blocked sentence may introduce a place, or a participant, not yet mentioned, with "a", as built. (3) `documents.place_description_rate` stays at 0.1. Proposed by Claude Code in the c4c proposal and by Claude (chat), decided by Jon.

## Orientation

### What was read

This session is a fresh one. It read `CLAUDE.md` in full; `docs/specs/WORLD_PHASE_C.md` in full; `docs/proposals/2026-10-29-world-stage-c4c-decisions.md` in full, and "What the corpus skips or filters" of the c4a proposal; the corpus package: `facts.py`, `grammar.py`, `realize.py`, `interpret.py`, `logical.py`, `mentions.py`, and `readings.py` in full, and of `propositions.py` (the terms, the predicates, `condition_of`, `effect_matches`, `Truth.has_entry`, `_evaluate_causal`, `observed`, `_predicate_problem`), `renderings.py` (the formula classes, `_causal`, `formula`, the parser, `_causal_proposition`, `proposition_of`), `planner.py` (the feature documents, the comparatives of participants and of places), `lexicon.py` (the function words, the concepts, the lexemes), `request.py`, `world.py` (the pair fluents and the place event types), `testsets.py` (the causal candidates and `_causal_group`), `stats.py` and `figures.py` (the `comparisons` block); `tests/corpus/truth_oracle.py` (`entries`, `causal`, `_effect_matches`, `observed`); the word-form configurations of the two corpora; and the definitions of the tiny and default worlds of the reference run.

### How the pieces stand today

Main holds stage c4c (b448557). A causal statement is a class-level proposition whose subject is an `EventTerm` and whose predicate is an `effect` or a `precondition` of a base fluent, or a numeric effect. `condition_of` gives no condition for an effect under a pair literal, `effect_matches` refuses a pair effect, and `Truth.has_entry` reads no pair literal as an entry. The oracle's `entries` and `causal` skip the same records. `interpret` refuses a prepositional predicate with a category as its subject. The causal statements run over the one-place event types and the two-place event types and categories with a word; the place event types are kept apart.

The reference run's worlds show what the stage can state. A pair fluent has a preposition when some precondition or condition literal reads it as a pair (stage c4b). In the tiny world that is the location fluent `PAIRFL.3` alone, with `SAME.PAIRFL.3`; `PAIRFL.1` and `PAIRFL.2` are read through `SOME` and written by a delete, and have no preposition. In the default world `PAIRFL.1` (symmetric), `PAIRFL.4` (general), and the location fluent `PAIRFL.5` have prepositions, with `SAME.PAIRFL.5`; `PAIRFL.2` and `PAIRFL.3` (both symmetric) have none.

### Plan for the stage

In the order of the scope:

1. **WM.74's rule 1.** One helper of `Mentions` makes the two mentions of a comparison: with the compared scalar's poles avoided; then, when the two nouns name different categories, as ordinary mentions; and it says whether something tells the two apart. The planner skips a comparison whose mentions nothing tells apart, and counts the skips and the comparisons that use the compared poles across classes.
2. **The words.** `them` at the end of `FUNCTION_WORDS`. `SOMEPLACE` as the last content concept of a world with places, so that every earlier lexeme keeps its label.
3. **The grammar.** A bound pronoun is a class-level noun phrase without a noun. A causal statement's predication may be a prepositional predicate, with `them` or a quantified generic noun as its object. A profile statement's subject is the generic noun with an object relative whose predicate is the copula and the preposition, the gap its object. `grammar.word_order.stranding` places the preposition of that clause.
4. **Causal statements with pair literals.** The predicate of an effect or a precondition may be a pair fluent, a `SAME` literal, or a `SOME` literal. The subject's role is the first argument of the pair. The place event types join the event types of the causal statements, with `SOMEPLACE` as the other participant.
5. **Conditional statements.** `condition_of` keeps giving no condition for a pair condition, so no statement is made, and the unconditional form stays false.
6. **Profile statements.** A new subject term, a new predicate kind `profile`, the truth test from the pair fluent's symmetry, the rendering, the parser, and the reading.
7. **The test sets.** The causal sets keep to the statements of stage c4c, by one filter.
8. **The runs,** the snapshot, the guides, and the documents.

### Reading of the scope

- **"Told apart."** Two mentions are told apart when the noun and the modifiers of one of them do not fit the other's referent. A pronoun is told apart by the discourse rule that allowed it.
- **"Different comparison classes."** The two mentions have nouns that name different categories. A pole in a mention is relative to its noun's category.
- **"Place event types join."** Every causal statement about a place event type with a verb is enumerated: its Boolean and numeric effects and preconditions on the agent, and its location literals. A document about a fluent then states the place event types that set it or need it. The statements about the location fluent appear in no document until stage c5b, because no document of this stage is about a place event type or a pair fluent.
- **"Stated with a quantified object."** `SOME.PAIRFL.k(x)` is "near some things", and its negation "near no things". The verb phrase is never negated: the determiner carries the value.
- **A symmetric pair fluent, and `SAME`.** A pair effect or a pair literal on a symmetric pair fluent holds in both directions, so the statement about either role is true by the definition, and both are stated. The same holds for `SAME`.

### Questions

No question of the orientation changes the design or a file format beyond what `WORLD_PHASE_C.md` and the stage prompt say. One finding is recorded as an open question at the end: a pair fluent that only `SOME` literals read, or only effects write, has no preposition, so its statements cannot be said. The lexicon stays as stage c4b built it. The stage proceeds.

## Engineering choices made while building

Each choice is logged in `docs/DECISIONS.md` as the WM.E row named beside it.

1. **The two mentions of a comparison (WM.E353, WM.74).** `Mentions.comparison_phrases` makes both mentions with the compared scalar's poles left out. When the two nouns name different categories, each definite mention is told apart again with the compared poles allowed, keeping its noun. Two mentions are told apart when the noun and the modifiers of one of them do not fit the other's referent. A pronoun is told apart by the rule that allowed it. The planner drops a comparison whose mentions nothing tells apart, and draws no other comparison in its place. A comparison of participants did not leave the compared poles out before this stage; it does now, within one comparison class. A comparative on a numeric fluent has no pole among the modifiers, so only the told-apart rule applies to it.
2. **The two words (WM.E354).** `them` is the last of `FUNCTION_WORDS`. `SOMEPLACE` is a concept of the type `generic`, made only in a world with places, and the last content concept, so every earlier content lexeme keeps its label. The lexeme labels of the function words move up by one in a world with places.
3. **Pair literals in the proposition layer (WM.E355).** The predicate of an effect or a precondition statement names its pair literal as its label: `PAIRFL.k`, `SAME.PAIRFL.k`, or `SOME.PAIRFL.k`, with the value. The subject's role is the literal's first argument, and the other participant its second. The `causal` record gains `second` for a pair and for `SAME`. A profile statement's subject is a new term, `PairTerm`, and its predicate the new kind `profile` with the new field `form`. The truth tests refuse a pair literal that its event type cannot have.
4. **Which entries are statements (WM.E356).** A pair effect or a `pair` literal is the entry of the statement about the role of its first argument. On a symmetric pair fluent it is the entry of the statement about either role, because the runtime writes and reads both directions. A `SAME` literal is the entry of the statement about either role. A `SOME` literal is the entry of the statement about its role. A `SOMEOF` literal is the entry of no statement, also on a symmetric pair fluent, where it equals `SOME`. A conditional pair effect is stated when its condition holds no pair literal. The place event types with a verb join the event types of the causal statements, about their agent alone, with their Boolean and numeric entries and their location literals. The new entries of a role are enumerated after the Boolean and numeric ones, so the statements of the earlier stages keep their relative order.
5. **The grammar of the bound pronoun, the quantified object, and the prepositional predicate (WM.E357).** The bound pronoun is a class-level noun phrase without a noun, whose referent is the generic noun of the participant it refers back to. A causal statement about a pair literal is a predication of the kind `effect` or `precondition` whose label is the preposition's concept and whose object is the bound pronoun, or, for `SOME`, the generic noun with `some` or `no`. The prepositional phrase stands where an object stands. `before` stays at the edge of the verb phrase, after the prepositional phrase when the phrase is inside the verb phrase. `interpret` reads a preposition with a category as its subject as an effect, a precondition, or a profile statement.
6. **Relative clauses with a prepositional predicate, and `stranding` (WM.E358, WM.58).** The clause is an object relative with one predication of the kind `pair_state`, admitted in the subject of a profile statement alone. With `strand` the preposition stands alone in a `PP` where the object would stand. With `pied_pipe` the first daughter of the clause is a `PP` of the preposition and the relative word, ordered by `word_order.adposition`.
7. **The truth of a profile statement (WM.E359).** A profile statement is true exactly when the pair fluent's symmetry in the definition is the statement's form. The statement is valid whether or not a pair of the fluent can ever hold. The grounding records `can_hold`.
8. **The rendering and the readings (WM.E360).** A pair or a `SAME` literal is an atom over the two participants, the statement's role first. `SOME` is an atom of the role alone. A profile statement is `NEC(ALL(PAIRFL.k(VAR.2, VAR.1), [NOT] PAIRFL.k(VAR.1, VAR.2)))`. `profile` is a seventh reading, after `causal`.
9. **Where the statements appear, the test sets, and the statistics (WM.E361).** A document about an event type between entities states its pair literals. A document about a fluent states the place event types that set it or need it. The statements about the location fluent, and the profile statements, enter no document. The causal sets leave out the new statements by one filter. `stats.yaml` gains the counts listed under "File formats added or changed".
10. **The lexicon of prepositions stays as stage c4b built it (WM.E362).** A pair fluent that only `SOME` literals read, or only effects write, has no preposition. Its statements are not made, and are counted. Open question 1 asks Jon about it.
11. **The tests (WM.E363).** Listed under "Test results" below, with the earlier tests that were changed.

Two fixes came out of the slow tier. `content_words` counted the bound pronoun as a content word; a pronoun of either kind is now a function word. `stats.yaml`'s `mentions` block counts the bound pronouns apart (`bound_pronouns`), because `pronouns` counts the mentions of instances.

## WM.74's counts

The comparisons that use the compared scalar's poles across comparison classes, and the comparisons the rule skips, from `stats.yaml` (`comparisons.compared_poles_across_classes` and `comparisons.skipped_not_told_apart`):

| Corpus | Comparisons of participants | With the compared poles across classes | Skipped | Comparisons of places | With the compared poles across classes | Skipped |
| --- | --- | --- | --- | --- | --- | --- |
| tiny (20 documents) | 1 | 0 | 0 | 0 | 0 | 0 |
| default (10,000 documents) | 1,216 | 27 | 0 | 116 | 24 | 0 |
| tiny, 400 documents, for illustration | 22 | 3 | 0 | 4 | 4 | 4 |
| tiny, 400 documents, `comparative_rate: 0.9`, for illustration | 184 | 14 | 1 | 34 | 34 | 20 |

The tiny corpus of 20 documents holds one comparison, and the rule changes nothing in it. The two runs of 400 tiny documents show the rule at work, because the tiny world's two places of a category differ by their one place scalar alone: every comparison of two places of one category is skipped, and every comparison that is made is between places of different categories, with the compared poles. The default world's places and entities have enough features, so no comparison of the default corpus is skipped. Every comparison with the compared poles is true by the oracle, and each of its two mentions fits its own referent by the oracle (`test_comparisons_use_the_compared_poles_across_comparison_classes_alone`).

## The filters of stage c4a: lifted, and what remains

The c4a proposal's items 4 to 6 and 8, on causal statements:

- **Item 4** (`condition_of` and `effect_matches`). Lifted in part. `effect_matches` reads a pair effect as an entry. `condition_of` still gives no condition for an effect under a pair literal, by WM.59, so such an effect is the entry of no statement.
- **Item 5** (`Truth.has_entry`). Lifted. A pair literal of a precondition is an entry: `pair`, `pair_same`, and `pair_some`. `pair_someof` stays out, by WM.55.
- **Item 6** (`Facts._conditional_report`). The reason `pair_effect` is gone: a conditional pair effect is stated like any other. `pair_condition` stays, by WM.59. The report covers the place event types with dynamics too.
- **Item 8** (the oracle's `entries` and `causal`). Lifted. The oracle reads the pair effects and the pair literals, and `profile` and `profile_held` are new.

`interpret`'s refusal of a prepositional predicate with a category as its subject is replaced by the causal and profile readings.

What remains filtered, and why:

- **The causal test sets** take none of this stage's statements (scope item 7). The one filter is in `TestSetBuilder._causal_group`: a statement with a pair literal, or about a place event type, is left out of the pool. Stage c5b lifts it, with the role swap that skips symmetric pair fluents.
- **`SOMEOF`** makes no statement (WM.55).
- **A conditional effect under a pair condition** makes no statement, in either form (WM.59).
- **A pair fluent without a preposition** makes no statement (choice 10).
- **The profile statements** enter no document (stage c5b).
- **The statements about the location fluent** enter no document, because no document of this stage is about a place event type or a pair fluent (stage c5b).

## The pair entries of the two worlds

From `stats.yaml`, `causal.pair_entries`: the pair effects and pair precondition literals of the event types with events, compiled literals included.

| World | Pair effects | Pair precondition literals | Stated | `SOMEOF` | Pair condition | No word |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 3 | 6 | 6 | 0 | 0 | 3 |
| default | 8 | 22 | 26 | 2 | 0 | 2 |

The language can make 24 causal statements about the tiny world (12 before this stage) and 156 about the default world (109 before). Of the new ones, the tiny world gives 4 with a pair, 6 with `SAME`, and none with `SOME`, and 6 about place event types; the default world gives 15 with a pair, 24 with `SAME`, 2 with `SOME`, and 13 about place event types. The default world gives one profile statement, about the symmetric `PAIRFL.1`, and the tiny world none.

The entries without a word: in the tiny world `NOT SOME.PAIRFL.1(agent)`, `NOT SOME.PAIRFL.2(agent)`, and the delete on `PAIRFL.2`; in the default world the two deletes on `PAIRFL.2`. The conditional effects with a pair condition are 0 in the tiny world and 8 of 12 in the default world, as in stage c4a.

## File formats added or changed

- `documents.jsonl`: an effect or a precondition predicate whose `label` is `PAIRFL.k`, `SAME.PAIRFL.k`, or `SOME.PAIRFL.k`; a `causal` record with `second`; an event term whose `event` is a place event type; a referent `SOMEPLACE` with the noun `SOMEPLACE`, and a referent `THING` or `SOMEPLACE` with no noun (the bound pronoun). The reading `profile` can appear in `readings`. The predicate kind `profile` (with `form`) and the pair term (`{"head": "THING", "pair": "PAIRFL.2", "role": "second"}`) are read and written by `Proposition`, and no document holds one in this stage.
- `stats.yaml`: `causal.statements`, `pair_literal_statements`, `place_event_statements`, `pair_literal_sentences`, `place_event_sentences`, `pair_entries`, and `profile_statements`; `comparisons.compared_poles_across_classes` and `skipped_not_told_apart`; `mentions.bound_pronouns`; `ambiguity.readings` may hold `profile`. The key `causal.conditional_effects.unstated.pair_effect` is gone.
- The corpus configuration: `grammar.word_order.stranding` (`strand` or `pied_pipe`, default `strand`), listed in `data/corpus/default.yaml`.
- `wordform_request.yaml`: one more lexeme (`SOMEPLACE`, a noun) and one more function word (`them`).
- No format of a definition, a history, a world run, or a test set changed.

## Rule-set identities

| Configuration | Before (main, b448557) | After stage c5a |
| --- | --- | --- |
| `tiny.yaml` | `2f80537bb72b` | `2f80537bb72b` |
| `default.yaml` | `baa625dfd0fb` | `baa625dfd0fb` |

The stage changes no world and no world code. The regenerated tiny and default worlds are byte-identical to the reference run's, apart from the git state that the three `config.yaml` files of each run record.

## Specification edits

Every edit of `docs/specs/WORLD_PHASE_C.md` is a parenthetical "(As built, stage c5a: ...)" note, or the c5a entry of "Build stages". No decided text changed.

- "Lexicon": `them`, `SOMEPLACE`, and the prepositions that stay as stage c4b built them.
- "Grammar": the bound pronoun; relative clauses with a prepositional predicate and the two settings of `stranding`; the quantified object; the limits kept, and where `before` stands.
- "Instance level", "Comparative descriptions": WM.74's rule.
- "Class level": the causal statements with pair literals (the roles, the symmetric pair fluents, the place event types, `SOMEOF`, the `causal` record); the conditional statements with pair conditions; the profile statements (the pair term, `can_hold`).
- "Readings": the reading `profile`.
- "Test sets", "Causal sets": the filter of this stage.
- "Outputs": `documents.jsonl` and `stats.yaml`.
- "Build stages", the c5a entry: Jon's rulings WM.74 and the as-built summary.

## Open questions

1. **Prepositions for every pair fluent that the dynamics use.** A pair fluent has a preposition when some precondition or condition literal reads it as a pair (stage c4b). A pair fluent that only `SOME` literals read, or only effects write, has no word. The corpus then cannot state its `SOME` literals, its adds and deletes, or its profile. In the default world that is `PAIRFL.2`, a symmetric pair fluent with initial pairs that two event types delete; `PAIRFL.3` is neither read nor written. In the tiny world it is `PAIRFL.1` and `PAIRFL.2`, so the tiny world has no profile statement at all. The options: (a) keep the rule; (b) give a preposition to every pair fluent that some literal of any kind reads or some effect writes; (c) give one to every pair fluent. Options (b) and (c) add content lexemes. To keep every earlier lexeme's label, the new prepositions would follow `SOMEPLACE`. Recommendation: (b), built in stage c5b, which regenerates the runs and the word forms anyway.
2. **The profile statement of a pair fluent that can never hold.** The default world's one profile statement is about `PAIRFL.1`, which has no initial pairs and no add anywhere in the world. The statement is true by the definition, and true of no pair. The stage keeps it valid and records `can_hold: false` in its grounding. Should stage c5b's documents and profile sets leave such a statement out? Recommendation: leave it out of the documents and the sets, as a rule statement is left out when no instance satisfies its term. With option (b) of question 1 the default world would then state the profile of `PAIRFL.2`, which holds.
3. **"Told apart".** The stage reads WM.74's "tells the two mentions apart" as: the noun and the modifiers of one mention do not fit the other's referent. So "the dry place is hotter than the place" is made when the other place is not dry, although "the place" alone fits both. A stricter reading would require each mention to exclude the other's referent. Is the reading right? And a skipped comparison is not replaced by a comparison on another scalar: should the planner try the other scalars first?
4. **Both roles of a symmetric pair fluent, and of `SAME`.** The stage states "things that approach things become near them" and "things that things approach become near them", and both `SAME` statements, because both are true by the definition. In the default world 24 of the 41 statements with a pair literal are `SAME` statements, half of them the mirror image of the other half. Should the mirror statements be left out, or stated at a lower weight?
5. **Place event types in the documents about fluents.** A document about a fluent now states the place event types that set it or need it ("things that go to places are not asleep before"), with the generic noun for places, although no document is about a place event type until stage c5b. The default corpus has 41 such sentences. This follows the scope's "place event types join". Jon may prefer to hold them back until stage c5b.

## Notes for stage c5b

- **The causal sets.** Lift the filter in `TestSetBuilder._causal_group`. `_causal_candidates` needs three changes: the predicate swap draws among the base fluents, and a pair statement needs the other prepositions of its kind; the event swap reads `world.event_types`, which has no place event type (use `Truth.roles_of`); and the role swap of a statement about a symmetric pair fluent, or of a `SAME` statement, gives a true statement, which `false_items` drops, so the swap skips them by itself, and a test should say so.
- **`Truth.observed` and the oracle's `observed`** read a Boolean fluent's value, or a numeric change. A statement with a pair literal needs the replayed pairs: `Truth.pair_holds_at` judges a pair and `SAME`, and `SOME` needs the row of the participant. The oracle has `pairs_at`.
- **The profile sets.** `oracle.profile_held(form, scenes)` says whether a form held of every pair at every time point of the scenes, which is the `observed` mark of a false profile item; the mark also needs at least one scene with a pair of the fluent. `Truth` has no such method yet. `can_hold` in the grounding says whether a pair can ever hold.
- **Documents about a pair fluent.** `Facts.profile_statements(fluent)` gives the profile statement. `Facts.causal_statements(fluent=...)` matches the predicate's label exactly, so the statements about `SAME.PAIRFL.k` and `SOME.PAIRFL.k` need `pair_fluent_of_label`. `Planner._section` has no section for a profile statement, and `_feature_topics` holds no pair fluent and no place event type.
- **Documents about a place event type.** `Facts.causal_statements(event_type=label)` gives its statements, the location ones included.
- **Class-level facts about categories of places** still need `Truth._term_problem` to accept a category of places (note of stage c4c).
- **`clause_propositions`** (`mentions.py`) reads the relative clauses of instance-level plans. A test item that is a profile statement must not go through it: the relative clause of a profile statement's subject is no capacity.
- **The word forms.** The tiny word-form budget is 60 content words for 58 content lexemes. A new function word or content lexeme changes the requests, so the word forms are remade.
- **The reading `profile`** is counted by `reading_counts`, and `stats.yaml`'s `ambiguity.readings` will hold it once documents state profile statements.

## Earlier tests changed

Each earlier test that expected a filter this stage lifts, or a word list or a count this stage changes, was changed to the stage's property. None was loosened.

- `test_corpus_pairs_places.py::test_no_causal_statement_states_a_pair_literal` became `test_causal_statements_state_pair_literals_and_no_pair_condition`, its opposite: some statement's predicate is a pair fluent, `SAME`, or (default world) `SOME`, and some statement is about a place event type; no condition holds a pair literal, and the effects under a pair condition are counted, over every event type with events.
- `test_corpus_pairs_places.py::test_a_conditional_effect_with_a_pair_condition_is_stated_in_neither_form` (WM.59) is kept and widened: it now covers the pair effects and the place event types too.
- `test_corpus_pairs_places.py::test_the_lexicon_names_the_pair_fluents_and_the_places`: the lexicon has `them`, and `SOMEPLACE` is its last content lexeme; no word carries `SOME.`, `SOMEOF.`, or `EVERYPLACE`.
- `test_corpus_causal.py`: `test_every_true_causal_statement_of_the_world_is_enumerated` keeps its subject, the Boolean and numeric entries of the event types between entities, and checks every statement, the new ones included, by the oracle; `test_causal_statements_are_true_by_a_rereading_of_the_definition` checks the form and the record entry of the Boolean and numeric statements, and the oracle's verdict on all; `test_the_causal_sets_pair_every_statement_with_every_valid_false_item` and `test_capped_causal_sets_keep_every_statement` apply the stage's filter; `test_the_function_word_before` reads the last four base words.
- `test_corpus_numeric.py::test_conditional_statements_are_true_and_never_without_their_condition`: `stats.yaml` writes the report of the pair entries under its own key.
- `test_corpus_outputs.py`: the schema test admits the referent and the noun `SOMEPLACE`; the statistics test tells the pronouns that mention instances from the bound pronouns, and checks both counts.
- `test_corpus_documents.py::test_noun_levels_apply_to_instances_only`: a class-level noun phrase is named by its own noun, or is the bound pronoun.
- `test_corpus_realization.py::test_the_generated_sentences_cover_the_grammar`: `them` needs a causal statement about a pair literal, as `before` needs a causal statement.
- `test_corpus_lexicon.py`: `test_function_words` (the list gains `them`), `test_a_world_without_two_place_event_types_or_scalars`, `test_concepts_of_the_default_world`, `test_concept_order_follows_the_specification_table`, `test_each_type_has_its_own_proportion`, and `test_lexeme_labels` (two generic nouns, `SOMEPLACE` the last content lexeme, `them` the last base function word).
- `test_corpus_config.py::test_default_values`: the word order has `stranding: strand`.
- `test_corpus_renderings.py::test_formal_rendering`: the function words' lexeme labels move up by one in the tiny world (`the` is `LEXEME.60`).

`test_corpus_documents.py::test_content_words_count_nouns_adjectives_and_verbs` failed in the first slow-tier run and was not changed: it found that `content_words` counted the bound pronoun, which was fixed in the code.

## Changed expectations

The snapshot moved by 81 figures, all of the two corpora (`tiny.corpus`, `default.corpus`) and the one baseline that is a corpus hash. No world figure and no baseline identity moved.

- **The lexicon.** `lexemes.content` 57 to 58 (tiny) and 236 to 237 (default), for `SOMEPLACE`; `lexemes.function` 18 to 19 in both, for `them`; `lexemes.total` 75 to 77 and 254 to 256.
- **The sizes.** The tiny corpus has 196 sentences and 968 tokens (195 and 957), and the default corpus 100,660 sentences and 503,243 tokens (100,643 and 501,668), with `by_level.class` 123 to 124 and 49,720 to 49,737, and the `wrote` lines and the quick-start block that quote them. A document about a fluent now has the place event types' statements to draw from, so fewer such documents run out before their drawn length.
- **The causal sentences.** `causal.causal_sentences` 7 to 8 (tiny) and 2,305 to 2,605 (default); `precondition_sentences` 2 to 3 and 846 to 1,149; in the default corpus `effect_sentences` 1,459 to 1,456, `lowers_sentences` 149 to 165, and `conditional_sentences` 110 to 109. Most pair literals of the two worlds are precondition literals (`SAME`, the compiled literals), so the preconditions gained most.
- **The new counts of the `causal` block.** `statements` (24 and 156), `pair_literal_statements` (by `pair`, `same`, `some`: 4, 6, 0 and 15, 24, 2), `place_event_statements` (6 and 13), `pair_literal_sentences` (0 and 370), `place_event_sentences` (1 and 41), `pair_entries` with its five reasons, and `profile_statements` (none, and 1 symmetric). `conditional_effects.unstated.pair_effect` is gone from both.
- **The comparisons.** The new `compared_poles_across_classes` and `skipped_not_told_apart`, each for participants and for places; `pole_modifiers` 25 to 24 (tiny) and 3,739 to 3,730 (default), because a comparison of participants now leaves the compared poles out within one comparison class; `pole_restrictions` 208 to 204 in the default corpus, because the feature documents draw other sentences.
- **Smaller shifts of the default corpus.** `ambiguity.share` 0.06068 to 0.06067 (and the tiny one 0.046154 to 0.045918), because the corpora have more sentences; the co-occurrence of the feature documents (`thematic` 0.318 to 0.258, `taxonomic` 0.184 to 0.195), because the new causal sentences hold generic nouns and prepositions and no category noun; and the share of seen true items of four sets (`class_predicate` 0.642 to 0.638, `class_role` 0.36 to 0.356, `class_subject` 0.622 to 0.624, `instance_subject` 0.49 to 0.492), because the documents changed.
- **The baseline `conditional_effects_off`** keeps the hash of the tiny corpus without conditional effects, which moved because the corpus states the new sentences.

The guides' figures moved by 34, written by `python -m semantic_world.figures guides`.

## The word forms remade

The requests of both corpora changed as expected: one more content lexeme (`SOMEPLACE`, a noun, the last entry of `lexemes`) and one more function word (`them`). Every earlier `lexemes` entry is byte-identical to the reference run's, and `takes`, `affixes`, and `inflect` are unchanged. No new inflection came with the stage. The function words are listed by frequency: `them` is the last in both requests, and in the tiny request `not` and `can` changed places.

Both word-form runs were remade with `python -m semantic_world.wordforms all`, and both corpora rendered.

- **Every earlier content lexeme kept its word.** The `label`, `word`, and `spelling` columns of the rendered `lexicon.csv` equal those of the stage c4c run, and those of the reference run, for all 57 earlier content lexemes of the tiny corpus and all 236 of the default corpus. `SOMEPLACE` took `WORD.56` in the tiny run and `WORD.218` in the default run, words that no lexeme had.
- **The function words.** The default corpus's 18 earlier function words kept their forms, and `them` is `FUNCWORD.19`. In the tiny corpus 16 of the 18 kept their forms against the reference run: `not` and `can` exchanged forms, because the pipeline makes the function words' forms in the order of the request, by frequency, and the two changed places there. Stage c4b recorded the same behavior (Jon's ruling 1 of WM.72 keeps the order by frequency).
- **The times.** The tiny run took 9.9 seconds. The default run took 34.0 minutes (2,039 s): the audio cache held every clip but those of the two new words, and the front ends and the five embeddings were computed again over the 47,790 tokens.

## Run times

Measured on November 4, 2026, with nothing else running. The machine ran slower than on October 29: the reference run of stage c4c's code, made the same day, is the fair comparison, and the table gives both.

| Step | Stage c4c (October 29, 2026) | Reference run of main (November 4, 2026) | Stage c5a | Growth against the reference |
| --- | --- | --- | --- | --- |
| `define data/world/default.yaml` | 8.4 s | 9.0 s | 8.9 s | none (the world is unchanged) |
| `define data/world/tiny.yaml` | 1.8 s | 2.1 s | 1.9 s | none |
| `generate data/corpus/tiny.yaml` | 2.3 s | 2.5 s | 2.5 s | none |
| `generate data/corpus/default.yaml` | 163 s | 184 s | 181 s | none |
| `wordforms all`, default | 28.8 min (1,727 s) | not run (`forms`: 3.8 s) | 34.0 min (2,039 s) | +18% against stage c4c's run, on a slower day |
| `wordforms all`, tiny | about 9 s | not run (`forms`: 2.2 s) | 9.9 s | none |
| `render` tiny / default | 0.4 s / 2.8 s | 0.3 s / 2.8 s | 0.3 s / 2.8 s | none |
| the fast tier, `pytest -n auto` | 2 min 18 s (138 s, 2,060 tests) | 2 min 58 s (178 s, 2,060 tests) | 3 min 27 s (207 s, 2,078 tests) | +17% against main the same hour: the stage's 18 tests, their corpus runs of the hand worlds, and the larger enumeration of causal statements in every corpus run |

No step grew by more than half against the same day's measurement of main. Against the times recorded on October 29, the fast tier is exactly half again as long (138 s to 207 s), and `generate` 11% longer. The cause is the machine that day, not the stage: main's own fast tier took 178 s when timed in a worktree right after the stage's, and the reference `generate` took 184 s. The guides record the times as measured, with the reference beside them.

## Test results

- **The fast tier**, `pytest -n auto`: passed on November 4, 2026, with nothing else running, in 3 minutes 27 seconds (207 s, 2,078 tests). Main's fast tier, timed in a worktree right after it, took 2 minutes 58 seconds (178 s, 2,060 tests). The first fast-tier run of the build failed 16 tests: the two snapshot comparisons, which the regenerated runs settled, and 14 earlier tests that expected a filter, a word list, or a count this stage changes, each changed to the stage's property (listed above).
- **The slow tier** ran three times: once alone while building (`pytest --slow -n auto`, 8 minutes 7 seconds, 11 failures), once over the changed files, and once in the full check list. The first run's failures were the default snapshot, nine earlier default-world tests changed to the stage's property (listed above), and one finding, `test_content_words_count_nouns_adjectives_and_verbs`: `content_words` counted the bound pronoun as a content word, which was fixed in the code.
- **The full check list** (`cargo fmt --all --check`, `cargo clippy ... -D warnings`, `cargo test --workspace --all-features`, `ruff check python tests`, `ruff format --check python tests`, `pytest --all -n auto`): every step passed on November 4, 2026. `pytest --all -n auto` passed 2,405 tests in 10 minutes 59 seconds (659 s; 9 minutes 28 seconds in stage c4c, on a faster day).
- **The acceptance checks of the earlier stages.** Every acceptance check of phases (a) and (b) and of stages c1 to c4c that the test suite holds still passes. Twenty-one earlier tests were changed, as listed above; each change replaces an expectation that this stage's scope overturns (a filter lifted, a word added), and none removes a property of an earlier stage.
- Nothing is skipped and nothing is marked `xfail`.

The stage's own tests are in `tests/corpus/test_corpus_class_pairs.py` (18 in the fast tier and 4 more in the slow tier, on the default world):

- the two words, and a world without places;
- the statements with pair literals against an enumeration of the pair entries of `definition.json` made apart from the corpus, on the tiny world, the two worlds of the hand-written event files, and the default world; each statement true by the truth test and by the oracle, and false with the other value;
- the profile statements against the symbols' symmetries, and, on corpora of the two hand worlds, against the replayed pairs of every scene at every time point;
- every statement in the six clause orders, with stranding and pied-piping, the preposition before and after its noun phrase, and `before` at either end of the verb phrase (48 word orders): the tree's shape, `interpret(tree)` giving back the plan and the proposition, the propositional rendering parsing back, and the reading;
- the specification's examples in glosses;
- the plans the grammar refuses and the forms the truth tests and the rendering refuse;
- the documents that state the new statements, and the causal sets that hold none;
- WM.74, on the comparisons of the corpora and on `Mentions.comparison_phrases` over every ordered pair of the tiny world's places and entities.

## Documents changed

- `docs/DECISIONS.md`: WM.74 after WM.73; WM.E353 to WM.E363 after WM.E352.
- `docs/specs/WORLD_PHASE_C.md`: the as-built notes and the c5a entry listed under "Specification edits".
- `docs/guides/CORPUS.md`: the status line; the lexicon table (the generic noun for places) and the function words (`them`); "Causal statements" (the pair literals, the profile statements, and where the statements appear); "Comparisons of scalars" (WM.74); "Grammar" (the bound pronoun, the quantified object, and `stranding`); the readings; the `stats.yaml` row; the test sets' note on the causal sets; the parameter table (`grammar.word_order`); the reference to the specification; the run times; and the figures by the script.
- `docs/guides/README.md`: the status row of the corpus, the run times, and the chain paragraph. `docs/guides/WORLD.md`: the default world's run time. `docs/guides/WORDFORMS.md`: two figures by the script; the pipeline did not change. `README.md`: the chain's status line. `CLAUDE.md`: items 9 and 10 (stage c5a built, c5b next) and the fast-tier time.
- `data/corpus/default.yaml`: `grammar.word_order.stranding`. `data/wordforms/corpus_tiny.yaml`: the comment's count of content lexemes.
