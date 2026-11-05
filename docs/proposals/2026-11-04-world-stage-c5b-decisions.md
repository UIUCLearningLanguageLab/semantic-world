# Proposal: decisions for stage c5b of the world-and-language refactor

November 4, 2026. Raised in the orientation for stage c5b of `docs/specs/WORLD_PHASE_C.md` ("c5b. Documents, test sets, and the end of the phase"), and while building the stage. Status: working design. Every choice below is Claude Code's unless Jon changes it. The choices are logged as WM.E364 and following in `docs/DECISIONS.md`; Jon's decision for the stage is WM.75.

Stage c5b is the last stage of phase (c). It builds, in this order: Jon's rulings WM.75 (prepositions for every used pair fluent, no vacuous profile statement, comparisons that try another scalar, no mirror statements); documents about a pair fluent; places in category and feature documents; the causal, profile, and class test sets; the statistics; and the regenerated runs with the word forms remade and the full chain run from the guides' commands.

## Jon's decisions recorded by this stage

- **WM.75 (November 4, 2026, on stage c5a's open questions).** (1) Every pair fluent the dynamics use gets a preposition: a pair fluent that some precondition or condition literal reads in any form (a pair, `SAME`, `SOME`, or `SOMEOF`, compiled literals included) or that some effect writes. A pair fluent that nothing reads or writes gets none. The new prepositions are the last content concepts, after `SOMEPLACE`, so that every earlier content lexeme keeps its label and its word. (2) No profile statement about a pair fluent that can never hold: the statement is left out of the enumeration, the documents, and the test sets. (3) A comparison that WM.74 would skip tries the other scalars first, and is skipped only when no scalar allows it; "told apart" stays as stage c5a built it; `stats.yaml` counts the comparisons made on another scalar and those still skipped. (4) No mirror statements: on a symmetric pair fluent and for `SAME`, a causal statement is made in one direction only, with the agent as the subject; the mirror stays true by the truth test and by the oracle. (5) Place event types stay in fluent documents, as stage c5a built them. Proposed by Claude Code in the c5a proposal and by Claude (chat), decided by Jon.

## Orientation

### What was read

This session is a fresh one. It read `CLAUDE.md` in full; `docs/specs/WORLD_PHASE_C.md` in full; `docs/proposals/2026-11-04-world-stage-c5a-decisions.md` in full, and "Notes for stage c5a" of the c4c proposal; of `docs/specs/WORLD_AND_LANGUAGE.md`, "Phase (a): the corpus" (the lexicon, the quantifiers and the fixed test, causal statements, the test sets, the outputs), the corpus section of phase (b), and the phase (c) and phase (d) sections; `docs/guides/CORPUS.md` from its start to "Documents"; the corpus package: `facts.py` and `lexicon.py` in full, of `planner.py` the documents (the class-level facts, the category and feature documents, `draw_class_fact`) and the comparatives of participants and of places, of `propositions.py` the truth tests of the class level, of places, of causal and profile statements, `observed`, and the pair readers, of `world.py` the places and the fixed test, of `testsets.py` the candidates, the causal candidates, `_group`, `_causal_group`, `_pair`, and `_input`, of `mentions.py` `plan_for` and `comparison_phrases`, of `renderings.py` the terms, the predications, and the parser's kinds, and `stats.py` (the documents, the causal block, the comparisons); `figures.py` (`corpus_figures`); and `tests/corpus/truth_oracle.py` (the fixed test, the class level, the places, `entries`, `causal`, `profile`, `profile_held`, `observed`).

### How the pieces stand today

Main holds stage c5a (fca2229). The causal statements with pair literals and about place event types are enumerated, and a document about an event type between entities or about a fluent states them. The profile statements are enumerated and enter no document. The causal sets leave the new statements out by one filter. A category of places is a noun, and no class-level sentence is about one: `Truth._term_problem` refuses it. The place patient capacities have words and no sentence. A pair fluent has a preposition only when some literal reads it as a pair.

The reference run's worlds: in the tiny world `PAIRFL.1` and `PAIRFL.2` are used by the dynamics and have no preposition; in the default world `PAIRFL.2` is, and `PAIRFL.3` is neither read nor written.

### Plan for the stage

In the order of the scope:

1. **WM.75.** `World.pair_fluents_used` gives the pair fluents the dynamics use; the lexicon appends the prepositions of those that are not read as a pair after `SOMEPLACE`. `Facts.profile_statements` leaves out a pair fluent that can never hold. The planner tries the other dimensions of a comparison whose mentions nothing tells apart. `Facts._build_causal_statements` states a symmetric pair literal and a `SAME` literal about the agent alone.
2. **Documents about a pair fluent.** Every pair fluent with a preposition is a topic of feature documents. Its document holds its profile statement, in a new section `profile`, then its causal statements, one per sentence.
3. **Places in documents.** A category of places is a category term the truth tests accept, with place features, place poles, place patient capacities, and membership as its predicates, and the fixed test of the place taxonomy for `NEC`. A relation fact about a place event type has an entity category as its subject and a category of places as its patient. The categories of places are topics of category documents, and the place features and the place event types topics of feature documents.
4. **The test sets.** The causal sets take every statement. The profile sets are two new enumerated sets and a `_lawlike` twin. The class sets draw topics among the categories of places too.
5. **The statistics** and `figures.py`.
6. **The runs,** the word forms, the snapshot, the guides, the full tiny chain, and the documents.

### Reading of the scope

- **"Drawn uniformly with the categories of entities."** A category document draws its topic's level by `documents.topic_level_weights`, then a category of that level uniformly. The categories of places of a level join the categories of entities of that level.
- **"Relation facts" about a place topic.** A relation fact of a document about a category of places has the category of places as its patient, an entity category as its agent, and a place event type with a verb. A document about a category of entities keeps its relation facts as they were (open question 1).
- **"Place features are topics of feature documents."** A document about a place property or part states which categories of places have it and which lack it. The rules of the determined place features are not stated as rule statements (open question 2).
- **"The same form, about another pair fluent that lacks the profile."** The false item of `profile_predicate` is the statement's form about another pair fluent between entities with a preposition whose symmetry is not that form.
- **"Observed" for a profile item.** The form held of every pair at every time point of every scene of the corpus, and at least one time point of one scene held a pair of the fluent.

### Questions

No question of the orientation changes the design or a file format beyond what `WORLD_PHASE_C.md` and the stage prompt say. The open questions at the end record the readings above. The stage proceeds.

## The reference run

Made from main (fca2229) before any change, into `/Users/jon/Documents/Projects/semantic-world-reference/c5b/`, with `run.sh`, `run.log`, and `commit.txt`: `define` 2.1 s (tiny) and 9.3 s (default); `generate` 2.6 s (tiny) and 192.5 s (default); `wordforms forms` 2.3 s and 4.1 s; `render` 0.3 s and about 3 s. The worlds have the identities `2f80537bb72b` (tiny) and `baa625dfd0fb` (default).

## Engineering choices made while building

Each choice is logged in `docs/DECISIONS.md` as the WM.E row named beside it.

1. **The prepositions of every used pair fluent (WM.E364, ruling 1 of WM.75).** `World.pair_fluents_used` lists the pair fluents that some precondition or condition literal reads in any form, compiled literals included, or that some effect writes. The lexicon keeps the prepositions of stage c4b where they stand, and appends the others after `SOMEPLACE`, in label order, as concepts of the type `pair`. `Facts.pair_prepositions` follows the concept order, so the statements of the earlier stages keep their relative order.
2. **No profile statement about a pair fluent that can never hold (WM.E365, ruling 2).** `Facts.profile_statements` leaves out a pair fluent whose `Truth.can_hold` is false. The truth test still judges such a statement true, with `can_hold: false` in its grounding, so a profile item about such a pair fluent is never a false item either.
3. **A comparison tries the other dimensions (WM.E366, ruling 3).** `Planner._told_apart_comparison` draws one option as before. When nothing tells its two mentions apart, the planner tries the other dimensions of the same two arguments, in a random order from the comparatives stream, and makes the comparison on the first that tells them apart. The order is drawn only after a failure, so a document without one keeps its draws. For participants the dimensions are the scalars and the numeric fluents; for places, the place scalars of the same two places. `stats.yaml` counts both outcomes under `comparisons.made_on_another_scalar` and `comparisons.skipped_not_told_apart`.
4. **No mirror statements (WM.E367, ruling 4).** `Facts.mirror` says whether the statement about a role's pair literal is a mirror: a pair on a symmetric pair fluent, or `SAME`, about the patient. The enumeration skips it, in whichever direction the definition writes the entry. The truth test and the oracle still judge the mirror true. The report of pair entries counts an entry as stated when the statement about the agent is made.
5. **Documents about a pair fluent (WM.E368).** Every pair fluent with a preposition is a topic of feature documents, the location fluent included. The document holds the profile statement first, in the new section `profile`, which stands before `causal` in the template, and then causal statements alone, one per sentence: those whose literal is a pair of the fluent, `SAME` on it, or `SOME` of it (`Facts.causal_statements(pair_fluent=...)`). The document ends at its drawn length or when its statements run out, as a document about a fluent does.
6. **A category of places as a category term (WM.E369).** `Truth.is_place_term` tells the terms apart by the category's tree. A category of places has the places below it as its members, takes place properties, place parts, and place poles as restrictions, and takes no relative clause. Its predicates are the place properties and parts, the place patient capacities, the place poles against its parent in the place tree or `EVERYPLACE`, a comparative on a place scalar with another category of places, and membership in a category of places. `Truth._class_sort_problem` finds every crossing of the two kinds invalid, and the grammar refuses it.
7. **`NEC` by the fixed test of the place taxonomy (WM.E370).** `World.place_fixed` runs the fixed test over the place taxonomy's own features, rules, tree, and scalar settings. The test code is the entities', read through a small record of one taxonomy's side. The oracle reads the place run in `places/` of the world folder for its own fixed test.
8. **Relation facts about place event types (WM.E371).** The predicate kind `place_event_type` is used at the class level, with a category of places as its patient: "`CATEGORY.1.3` can `PLACEEVENTTYPE.2` `PLACECATEGORY.1`". The fact counts the bindings of an entity and a place, takes the extensional quantifiers alone, and has no role swap. A document about a category of places states the facts with the topic as the patient, and a document about a place event type states them over every pair of categories.
9. **The topics (WM.E372).** A category of places joins the categories of entities of its level in the draw of a category document's topic. The topics of feature documents gain the place properties and parts, the place event types with a verb, and the pair fluents with a preposition, after the earlier topics. A category of places is compared with its siblings whatever `documents.comparison_partners` says, because thematic relatedness is a relation between entity leaves.
10. **The test sets (WM.E373).** The causal sets take every causal statement. A predicate swap of a pair literal keeps its kind (another pair fluent, another `SAME`, `SOME` of another pair fluent). An event swap of a statement about a place event type brings in another place event type. The role swap gives no candidate for a place event type, a symmetric pair fluent, or `SAME`. `Truth.observed` reads the replayed pairs through `pair_literal_held`. The profile sets are the builder's level `profile`, enumerated as the causal sets are, with the changes `predicate` and `form`; `form` is a new value of `test_sets.changes`. `profile_form` has no `_lawlike` twin, because the other form is never observed. A run writes 56 sets. The class sets draw their topics among the categories of places too, and their swaps stay within the kind of thing.
11. **The statistics (WM.E374).** `stats.yaml` gains `documents.by_topic`, the block `places`, `causal.profile_sentences`, and `comparisons.made_on_another_scalar`. `figures.py` reads them under `class_places`.
12. **The tests (WM.E375).** Listed under "Test results" below, with the earlier tests that were changed.

## The filters: lifted, and what remains

Lifted in this stage:

- **The causal test sets.** The filter of stage c5a in `TestSetBuilder._causal_group`, which left out a statement with a pair literal or about a place event type, is gone. The sets take every causal statement.
- **The profile statements** enter documents, and have test sets.
- **The statements about the location fluent** enter documents: those about the location fluent itself, and those about the place event types.
- **A pair fluent without a preposition.** Every pair fluent the dynamics use has a word (ruling 1), so its statements are made.
- **A category of places as a term.** `Truth._term_problem` accepts it.

What remains filtered, and why:

- **`SOMEOF`** makes no statement (WM.55): it needs the pair fluent read backwards.
- **A conditional effect under a pair condition** makes no statement, in either form (WM.59).
- **A mirror statement** is not enumerated or stated (ruling 4 of WM.75). The role swap of the causal sets gives no item for a symmetric pair fluent or `SAME`, because the swapped statement is true.
- **A profile statement about a pair fluent that can never hold** is not made (ruling 2 of WM.75).
- **A pair fluent that nothing reads or writes** has no preposition (ruling 1 of WM.75), so no sentence names it.
- **A statement about the place of a place event type** is not made: a place has no fluent and stands first in no pair, so every causal statement about a place event type is about its agent, and such a statement takes no role swap.
- **A numeric threshold** is stated only at the language's cut (decision 23 and WM.35), as before.
- **A place** is never a pronoun's referent, never takes a relative clause, and is the subject of a sentence about its own static features alone (stage c4c); **a category of places** takes no relative clause and is the agent of nothing (this stage).
- **The rules of the determined place features** are not stated as rule statements (open question 2).
- **A relation fact with a place** appears in the documents about a category of places and about a place event type, and not in the documents about a category of entities (open question 1).

## Earlier tests changed

Each earlier test that expected a filter this stage lifts, a statement this stage no longer makes, or a word list this stage extends was changed to the stage's property. None was loosened.

- `test_corpus_class_pairs.py::test_them_and_someplace_are_the_new_words`: `SOMEPLACE` is no longer the last content lexeme. The test checks that only the prepositions of WM.75 follow it, in label order, and that `SOMEPLACE` keeps the label its position gives it.
- `test_corpus_class_pairs.py::test_statements_with_pair_literals_are_the_definitions_entries` (through `_record_pair_entries`): the enumeration made apart from the corpus lists a symmetric pair entry and a `SAME` literal once, about the agent (ruling 4).
- `test_corpus_class_pairs.py::test_the_words_of_the_statements_in_the_english_like_order`: the test asserted that the mirror statement is enumerated. It now asserts that the mirror is true by the truth test, that the grammar can still say it, and that no statement of the world is the mirror. Its `SOME` example is chosen among the negative `SOME` statements of two-place event types, because the hand world's `PAIRFL.3` has a preposition now and a one-place event type's positive `SOME` statement comes first.
- `test_corpus_class_pairs.py::test_profile_statements_are_the_symmetries_of_the_definition`: a pair fluent that can never hold has no profile statement (ruling 2); the test computes which pair fluents can hold from `definition.json`, and checks that the statement left out is still true with `can_hold: false`.
- `test_corpus_class_pairs.py::test_profile_statements_hold_in_the_replayed_scenes`: a document may state a profile statement, in the section `profile`.
- `test_corpus_class_pairs.py::test_documents_state_pair_literals_where_they_call_for_causal_statements`: a profile statement stands in the document about its pair fluent; a document about a pair fluent states the statements whose literal reads it; and the causal sets hold the statements with pair literals and about place event types, where the test asserted that they hold none.
- `test_corpus_causal.py::test_causal_statements_are_true_by_a_rereading_of_the_definition`: the test read the causal sentences of a small run, and its sanity counts depended on which topics the run drew. It now checks every causal statement of the world as well, with the form the documents would write, so the counts are over the world's statements.
- `test_corpus_causal.py::test_false_causal_items_are_false_under_nec_and_observed_marks_agree`: an item's fluent may be a pair literal with a preposition, and its record carries `second`; the test gains the properties of the role swap (never a place event type, a symmetric pair fluent, or `SAME`), of the event swap among place event types, and of the predicate swap that keeps the kind of pair literal.
- `test_corpus_causal.py::test_the_causal_sets_pair_every_statement_with_every_valid_false_item`: the expected pairs are over every causal statement, without the filter.
- `test_corpus_item_sets.py`: `ALL_SETS` gains the three profile sets, and `PROFILE_SETS` stands beside `CAUSAL_SETS` where a set holds what the world gives (`test_the_layout_of_the_test_sets`, `test_a_corpus_without_narratives_has_class_level_sets_only`, `test_an_item_without_a_noun_for_its_referent_is_left_out`, `test_instance_and_event_items_continue_a_document`); `test_true_items_are_true_and_false_items_differ_by_one_change` checks the two profile changes and their `observed` marks against the oracle, and reads a category's level in its own tree.
- `test_corpus_comparisons.py::test_every_comparison_sentence_is_true_by_the_oracle`: a class-level pole of a category of places compares with its parent in the place tree or `EVERYPLACE`. The test counted the class-level poles a run drew; it now judges every class-level pole fact of the world by the oracle, of the categories of entities and of places, because whether a run draws one depends on the draws.
- `test_corpus_documents.py::test_a_world_without_verbs_and_scalars`: the world's places keep their scalars, so a document about a category of places may state a place pole; the test asserts that no pole of an entity scalar and no two-place event type between entities is stated.
- `test_corpus_pairs_places.py::test_the_lexicon_names_the_pair_fluents_and_the_places`: the prepositions are those of every used pair fluent, the ones read as a pair first, the others after `SOMEPLACE`.
- `test_corpus_outputs.py`: the schema test admits a category of places as a referent and the reading `profile`; the statistics test counts the reading `profile`; and the profile sets need no document.
- `test_corpus_config.py::test_default_values`: `test_sets.changes` has `form`.
- `test_corpus_renderings.py::test_formal_rendering`: the function words' lexeme labels move up by two in the tiny world (`the` is `LEXEME.62`).

## WM.75's counts

**Ruling 1, the prepositions added.** The tiny world gains two, for `PAIRFL.1` and `PAIRFL.2` (both read through `NOT SOME` of the agent, `PAIRFL.2` also written by a delete): `LEXEME.59` and `LEXEME.60`. The default world gains one, for `PAIRFL.2` (written by two deletes): `LEXEME.238`. The default world's `PAIRFL.3` is neither read nor written, and has no word. The tiny corpus has 60 content lexemes (58 before) and the default corpus 238 (237 before). With the new words every pair entry of the two worlds but the `SOMEOF` literals makes a statement:

| World | Pair effects | Pair precondition literals | Stated, stage c5a | Stated, stage c5b | `SOMEOF` | No word |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 3 | 6 | 6 | 9 | 0 | 0 (3 before) |
| default | 8 | 22 | 26 | 28 | 2 | 0 (2 before) |

The new prepositions are used at the instance level too, because a narrative draws its pair states among the pair fluents with a preposition: the default corpus has 977 pair-state sentences (880 before).

**Ruling 2, the profile statements left out.** The default world's statement about the symmetric `PAIRFL.1`, which has an initial rate of 0 and no add anywhere, is left out. Its place is taken by the statement about the symmetric `PAIRFL.2` (initial rate 0.2), which has a word since ruling 1, so the default world still has one profile statement. In the tiny world the statement about `PAIRFL.1` (never holds) is left out and the statement about `PAIRFL.2` is made: one profile statement, where stage c5a had none. The default corpus states its profile statement in 17 documents, and the tiny corpus in 1.

**Ruling 3, the comparisons.** From `stats.yaml`, `comparisons.made_on_another_scalar` and `comparisons.skipped_not_told_apart`:

| Corpus | Comparisons of participants | Made on another dimension | Still skipped | Comparisons of places | Made on another dimension | Still skipped |
| --- | --- | --- | --- | --- | --- | --- |
| tiny (20 documents) | 1 | 0 | 0 | 0 | 0 | 0 |
| default (10,000 documents) | 1,219 | 0 | 0 | 114 | 0 | 0 |
| tiny, 400 documents, for illustration | 22 | 0 | 0 | 4 | 0 | 4 |
| tiny, 400 documents, `comparative_rate: 0.9`, for illustration | 185 | 1 | 0 | 33 | 0 | 19 |

Neither corpus has a comparison that nothing tells apart, as in stage c5a, so the rule changes nothing in them. The illustration runs show the rule at work: the one comparison of participants that stage c5a skipped is now made on another dimension, and the comparisons of two tiny places of one category are still skipped, because the tiny world's places have one place scalar and no other dimension to try.

**Ruling 4, the causal statements before and after the mirrors went.**

| World | Statements, stage c5a | Mirrors removed | New statements, from ruling 1 | Statements, stage c5b |
| --- | --- | --- | --- | --- |
| tiny | 24 | 3 (`SAME.PAIRFL.3` about the patient) | 3 (the delete on `PAIRFL.2`, `NOT SOME.PAIRFL.1`, `NOT SOME.PAIRFL.2`) | 24 |
| default | 156 | 14 (12 `SAME.PAIRFL.5` and 2 `PAIRFL.1` about the patient) | 2 (the two deletes on `PAIRFL.2`) | 144 |

By kind of pair literal, the tiny world has 5 statements with a pair, 3 with `SAME`, and 2 with `SOME` (4, 6, and 0 before), and the default world 15, 12, and 2 (15, 24, and 2 before). The statements about place event types are unchanged: 6 and 13.

## The test sets

A run writes 56 sets (53 before). The causal sets grew with the lifted filter, and the three profile sets are new:

| Set | tiny, before | tiny, after | default, before | default, after |
| --- | --- | --- | --- | --- |
| `causal_effect_predicate` / `_lawlike` | 12 / 1 | 12 / 2 | 235 / 49 | 242 / 53 |
| `causal_effect_polarity` | 8 | 11 | 66 | 75 |
| `causal_effect_event` / `_lawlike` | 20 / 0 | 20 / 0 | 500 / 65 | 500 / 73 |
| `causal_effect_role` / `_lawlike` | 4 / 0 | 4 / 0 | 33 / 9 | 36 / 9 |
| `causal_precondition_predicate` / `_lawlike` | 5 / 3 | 10 / 4 | 198 / 15 | 227 / 17 |
| `causal_precondition_polarity` | 4 | 13 | 43 | 69 |
| `causal_precondition_event` / `_lawlike` | 10 / 0 | 19 / 4 | 500 / 19 | 500 / 83 |
| `causal_precondition_role` / `_lawlike` | 2 / 1 | 3 / 1 | 30 / 2 | 31 / 2 |
| `profile_predicate` / `_lawlike` | none | 0 / 0 | none | 1 / 0 |
| `profile_form` | none | 1 | none | 1 |

The effect statements in the sets are 75 in the default world (66 before) and the precondition statements 69 (43). The profile sets are small because the two worlds have one profile statement each: the default world's `profile_predicate` pair says the symmetric form of the general `PAIRFL.4`, and its one `profile_form` pair the asymmetric form of `PAIRFL.2`; the tiny world has no other pair fluent that lacks the profile and can be named. The hand world `tiny_pairs`, with a symmetric, two asymmetric, and a general pair fluent, fills both sets in the tests.

## File formats added or changed

- `documents.jsonl`: a class-level subject, or the patient of a class-level relation, whose `category` is a category of places; a class-level predicate of the kind `place_event_type` with a category term as its `patient`; the predicate kind `profile` and the pair term as a subject, which stage c5a defined, now in documents; the section `profile`; a class-level referent that is a category of places.
- `stats.yaml`: `documents.by_topic`; the block `places`, after `causal`; `causal.profile_sentences`; `comparisons.made_on_another_scalar`; three more entries under `test_sets`.
- `tests/`: `profile_predicate.jsonl`, `profile_predicate_lawlike.jsonl`, and `profile_form.jsonl`, whose items carry `observed` and `statement` in `meta`.
- The corpus configuration: `form` is a value of `test_sets.changes`, listed in `data/corpus/default.yaml`.
- `wordform_request.yaml`: the new prepositions, as the last entries of `lexemes`.
- No format of a definition, a history, a world run, or a fixture changed.

## Rule-set identities

| Configuration | Before (main, fca2229) | After stage c5b |
| --- | --- | --- |
| `tiny.yaml` | `2f80537bb72b` | `2f80537bb72b` |
| `default.yaml` | `baa625dfd0fb` | `baa625dfd0fb` |

The stage changes no world and no world code. The regenerated tiny and default worlds are byte-identical to the reference run's, file for file, apart from `git_dirty` in the three `config.yaml` files of each run.

## Changed expectations

The snapshot moved by 267 figures, all of the two corpora (`tiny.corpus`, `default.corpus`) and the one baseline that is a corpus hash. No world figure and no baseline identity moved.

- **The lexicon.** `lexemes.content` 58 to 60 (tiny) and 237 to 238 (default); `lexemes.total` 77 to 79 and 256 to 257; `pairs_places.prepositions` 2 to 4 and 4 to 5.
- **The sizes.** The tiny corpus has 197 sentences and 1,044 tokens (196 and 968), and the default corpus 100,106 sentences and 499,870 tokens (100,660 and 503,243), with `by_level.class` 49,737 to 49,183, and the `wrote` lines and the quick-start block that quote them. The documents about pair fluents and about place features run out of statements before their drawn length more often than the documents they displaced.
- **The causal block.** `statements` 156 to 144 (default; 24 and 24 in the tiny world), `pair_literal_statements.same` 24 to 12 and 6 to 3, `pair_entries.stated` and `unstated.no_word` as in the table above, `causal_sentences` 2,605 to 2,669, `pair_literal_sentences` 370 to 585, `place_event_sentences` 41 to 223, `precondition_sentences` 1,149 to 1,237, `effect_sentences` 1,456 to 1,432, `raises_sentences`, `lowers_sentences`, and `conditional_sentences` (109 to 89), and the new `profile_sentences` (17 and 1). The documents about the location fluent and about the place event types state the statements that had no document before; the documents about event types between entities are fewer, because the topics are more.
- **The test sets.** `test_sets.count` 53 to 56, `pairs` 18,912 to 19,017 (default) and 597 to 634 (tiny), every causal set as in the table above, `effect_statements` 66 to 75 and `precondition_statements` 43 to 69, the three profile sets, and the shares of seen true items of the class sets, which moved because the documents changed.
- **The new figures.** `class_places` (25 figures in each corpus): the documents by the kind of their topic, the class-level sentences about places, and the pairs of the profile sets. In the default corpus 758 documents are about places (560 about a category of places, 152 about a place property, 46 about a place event type) and 55 about a pair fluent; 5,134 class-level sentences have a category of places as their subject, 240 of them a place patient capacity and 1,117 a law-like universal, and 1,955 are capacity facts about a place event type. And `comparisons.made_on_another_scalar`.
- **The comparisons.** `comparison_sentences.class` 94 to 282 and `comparative_sentences.class` 5,139 to 5,330, with the place poles and the comparison facts of the categories of places; `pole_restrictions` 204 to 329, with the place poles that restrict categories of places; small shifts of the counts of the narratives, whose draws moved with the lexicon.
- **The sentences about pairs and places.** `pair_state_sentences` 880 to 977 and the state items about pairs 1,012 to 1,229, because `PAIRFL.2` has a preposition; smaller shifts of the others.
- **Smaller shifts.** `ambiguity.share` 0.06067 to 0.061005; the co-occurrence of the encyclopedic documents (`encyclopedic_feature.thematic` 0.258 to 0.197, `encyclopedic_category.thematic` 0.112 to 0.103), because more of them are about places and pair fluents, whose words name no entity leaf; the shares of changed state items.
- **The baseline `conditional_effects_off`** keeps the hash of the tiny corpus without conditional effects, which moved because the corpus changed.

The guides' figures moved by 61, written by `python -m semantic_world.figures guides` (58, then 3 for the profile sets' sizes).

## The word forms remade

The requests of both corpora changed as expected. The `lexemes` of the reference run's requests are a prefix of the new ones, entry for entry, and the new prepositions are the last entries. `takes`, `affixes`, and `inflect` are unchanged. The function words are listed by frequency in the documents, and their order moved: in the default request `a` and `is` changed places, and `has`, `that`, and `become` rotated; in the tiny request 12 of the 19 moved.

- **The tiny budget.** `data/wordforms/corpus_tiny.yaml` makes 60 content words, and the tiny corpus now has 60 content lexemes: no word is spare. A larger count was checked first, as the prompt asks: with `count: 64` the pipeline gives another word to 57 of the 58 earlier content lexemes, because the set of words it makes depends on the count. With `count: 60` every earlier lexeme keeps its word. The count stays at 60, the lowest the lexemes allow, and the configuration's comment records it.
- **Every earlier content lexeme kept its label, word, and spelling.** The `label`, `concept`, `word`, and `spelling` columns of the rendered `lexicon.csv` equal those of the reference run for all 58 earlier content lexemes of the tiny corpus and all 237 of the default corpus. The new prepositions took `WORD.27` ("teenafs") and `WORD.33` ("dassin") in the tiny run and `WORD.133` ("snorshy") in the default run, words that no lexeme had.
- **The function words.** 14 of the default corpus's 19 function words kept their forms, and 5 changed (`a`, `is`, `has`, `that`, `become`); 7 of the tiny corpus's 19 kept theirs. The pipeline makes the function words' forms in the order of the request, by frequency, so a function word that changes its rank takes another form. Stages c4b and c5a recorded the same behavior (Jon's ruling 1 of WM.72 keeps the order by frequency). Open question 4 asks about it again, because the content lexemes are now stable across stages and the function words are not.
- **The times.** The tiny run with `all` took 6.5 seconds. The default run with `all` took 6.5 minutes (392 s): all 47,790 clips were read from the audio cache, because the 500 words of the default run are the same words as before, and the front ends and the five embeddings were computed again.

## The full chain on the tiny configuration

Run on November 4, 2026, from the repository root with `.venv` active, with the four commands of `docs/guides/README.md`, "The full chain", and `tiny` in place of `default` (the commands write into `runs/`):

| Command | Output line | Result |
| --- | --- | --- |
| `python -m semantic_world.world define data/world/tiny.yaml` | `wrote runs/world/tiny_seed1: 12 entities, 4 fluents (1 derived), 2 numeric fluents, 4 one-place and 4 two-place event types, 12 constraints, 4 places and 2 place event types, 3 pair fluents (2 between entities, 1 location fluent); rule set 2f80537bb72b` | exit 0, 2.0 s |
| `python -m semantic_world.corpus generate data/corpus/tiny.yaml` | `wrote runs/corpus/tiny_seed1: 20 documents, 197 sentences, 1044 tokens, 17 scenes, 56 test sets with 634 pairs` | exit 0, 2.6 s |
| `python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml` | `wrote runs/wordforms/corpus_tiny_seed1: 60 words (0 real), 1 minimal pairs, 122 candidates rejected`; `lexemes: 60 lexemes have forms: 6 by the mode, 54 at random, 0 sharing a form; 0 words have no lexeme` | exit 0, 6.4 s |
| `python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1` | `rendered runs/corpus/tiny_seed1: 197 sentences of 20 documents and 1268 test items, with the word forms of corpus_tiny (seed 1)` | exit 0, 0.3 s |

The chain worked from the guides' commands alone, in about 11 seconds. `runs/corpus/tiny_seed1/corpus.txt` holds the spelled rendering afterwards.

## Run times

Measured on November 4, 2026, with nothing else running. The reference run of main, made the same day, is the fair comparison.

| Step | Stage c5a (as recorded) | Reference run of main (today) | Stage c5b | Growth against the reference |
| --- | --- | --- | --- | --- |
| `define data/world/default.yaml` | 8.9 s | 9.3 s | 8.9 s | none (the world is unchanged) |
| `define data/world/tiny.yaml` | 1.9 s | 2.1 s | 1.9 s | none |
| `generate data/corpus/tiny.yaml` | 2.5 s | 2.6 s | 2.5 s | none |
| `generate data/corpus/default.yaml` | 181 s | 193 s | 175 s | none (9% less) |
| `wordforms all`, default | 34.0 min (2,039 s) | not run (`forms`: 4.1 s) | 6.5 min (392 s) | less: every clip came from the audio cache |
| `wordforms all`, tiny | 9.9 s | not run (`forms`: 2.3 s) | 6.5 s | none |
| `render` tiny / default | 0.3 s / 2.8 s | 0.3 s / about 3 s | 0.3 s / 2.8 s | none |
| the fast tier, `pytest -n auto` | 3 min 27 s (207 s, 2,078 tests) | 2 min 47 s (167 s, 2,078 tests) | 2 min 52 s (172 s, 2,093 tests) | +3% against main the same hour, for 15 more tests |
| `pytest --all -n auto` | 10 min 59 s (659 s, 2,405 tests) | not run | 10 min 16 s (616 s, 2,427 tests) | less than stage c5a's run, for 22 more tests, on a faster day |

No step grew by more than half, against the reference or against the times stage c5a recorded. Main's fast tier was timed in a worktree of fca2229 right after the stage's, as the c5a proposal's "Run times" describes. The machine ran faster than when stage c5a was measured: main's own fast tier took 167 s today against 178 s then.

## Test results

- **The fast tier**, `pytest -n auto`: passed on November 4, 2026, with nothing else running, in 2 minutes 52 seconds (171.67 s, 2,093 tests, exit status 0; pytest's summary line: `2093 passed, 2 warnings in 171.67s (0:02:51)`). Main's fast tier, timed in a worktree right after it, took 2 minutes 47 seconds (166.83 s, 2,078 tests). The first run of the corpus tests during the build had 42 failures. After the oracle and the rendering's parse-back of a relation fact were extended to places, 28 remained, all in earlier tests that expected a filter, a word list, or a statement this stage changes; each was changed to the stage's property (listed above).
- **The slow tier** ran while building: once alone (`pytest --slow -n auto`, 17 failures), twice over the corpus tests after the fixes, and once more in the full check list. The first run's failures were the default snapshot; 13 earlier default-world tests that needed the stage's property (those of `test_corpus_documents.py`, `test_corpus_causal.py`, `test_corpus_lexicon.py`, `test_corpus_outputs.py`, and `test_corpus_item_sets.py` listed above, with the slow-tier ones below); two of this stage's own tests, whose expectations were wrong (a contrast names a sibling of the topic, and a proportion is written at 6 decimals); and two findings in the code, both in `readings.py`: a membership sentence about a category of places and a place pole of a category of places were read by the entities' label prefixes alone, so a bare plural about places got the wrong quantifier readings. Both were fixed; no tiny sentence's readings changed.
- **The slow-tier tests changed** (beside the fast-tier ones listed under "Earlier tests changed"): `test_corpus_documents.py::test_category_documents` and `::test_feature_documents` keep their subject, the documents about categories of entities and about the entities' features, event types, and fluents, and leave the documents about places and pair fluents to `test_corpus_class_places.py`; `::test_with_shuffle_0_encyclopedic_documents_follow_the_template` admits a place event type in the section `relation`, and expects every section of the template but `profile`, which only a document about a pair fluent with a profile holds; `::test_sibling_contrasts` reads the siblings in the topic's own tree; `test_corpus_causal.py::test_feature_documents_about_event_types_and_fluents` and `::test_the_causal_statement_rate_is_a_setting` treat a document about a pair fluent as a document of statements alone, and a place event type as an event type; `::test_capped_causal_sets_keep_every_statement` expects every statement, without the filter; `::test_an_observed_false_item_held_of_every_event_in_the_scenes` judges a pair literal from the replayed pairs; `test_corpus_lexicon.py::test_concepts_of_the_default_world`, `::test_concept_order_follows_the_specification_table`, and `::test_lexeme_labels` expect the prepositions of WM.75 after `SOMEPLACE`; `test_corpus_outputs.py::test_the_statistics_count_what_the_documents_hold` lists the block `places`; and `test_corpus_item_sets.py::test_the_layout_of_the_test_sets` adds `state_object_changed` to the sets a run of 200 default documents may leave short (it held 24 of 30 pairs), with the sets of stage c4c that are listed for the same reason.
- **The full check list** (`cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace --all-features`, `ruff check python tests`, `ruff format --check python tests`, `pytest --all -n auto`): every step passed on November 4, 2026. `cargo test` passed 112 tests. `pytest --all -n auto` passed 2,427 tests in 10 minutes 16 seconds (616.29 s, exit status 0; pytest's summary line: `2427 passed, 2 warnings in 616.29s (0:10:16)`; 10 minutes 59 seconds for 2,405 tests in stage c5a).
- **The acceptance checks of the earlier stages.** Every acceptance check of phases (a) and (b) and of stages c1 to c5a that the test suite holds still passes. The earlier tests that were changed are listed above; each change replaces an expectation that this stage's scope or WM.75 overturns (a filter lifted, a mirror statement gone, a word added, a new kind of topic), or replaces a count of what a small run drew by an enumeration of the world's statements, and none removes a property of an earlier stage.
- Nothing is skipped and nothing is marked `xfail`.

The stage's own tests are in `tests/corpus/test_corpus_class_places.py` (15 in the fast tier and 7 more in the slow tier, on the default world):

- every class-level proposition about a category of places, plain or restricted, under every quantifier and both polarities, against the oracle, with the subject sets and the two fixed tests compared, and the comparison facts between sibling categories of places;
- the relation facts about place event types the same way, with their counts over the bindings and no role swap;
- the crossings of entities and places, each invalid by the truth test and never true by the oracle;
- the documents about categories of places, place features, place event types, and pair fluents: each sentence true by the oracle, read back from its tree, and parsed back from its rendering; the profile statement first; the statistics;
- the class sets about places, and their swaps within the kind of thing;
- the profile sets on the tiny world, the two hand worlds, and the default world: the pairs against an enumeration, and every `observed` mark against the oracle's replayed pairs;
- the causal items of the two hand worlds against the oracle, with their `observed` marks, and the mirrors: true, and no statement;
- the prepositions of WM.75 against a re-reading of `definition.json`, on four worlds;
- the comparison that tries the other dimensions, on a stub of the mentions, and the counts of the corpus.

## Acceptance

- **Every false causal, profile, and class item is false**: by the truth test and by the oracle (`test_true_items_are_true_and_false_items_differ_by_one_change`, `test_false_causal_items_are_false_under_nec_and_observed_marks_agree`, `test_the_profile_sets_and_their_observed_marks`, `test_the_class_sets_hold_facts_about_categories_of_places`).
- **Every `observed` mark agrees with the corpus's scenes**: the causal marks by the oracle's replay of the pairs, the profile marks by `profile_observed` (the same tests, and `test_causal_items_with_pair_literals_on_the_hand_worlds`).
- **The role swap never produces an item for a symmetric pair fluent or `SAME`**: checked on the tiny, default, and hand worlds.
- **No profile statement is about a pair fluent that can never hold, and no mirror statement is made**: `test_profile_statements_are_the_symmetries_of_the_definition`, `test_statements_with_pair_literals_are_the_definitions_entries`.
- **Every pair fluent the dynamics use has a preposition, and every earlier content lexeme keeps its label and its word**: `test_every_used_pair_fluent_has_a_preposition`, and the comparison of the rendered lexicons with the reference run.
- **Every causal, profile, and class statement in a document is true by a re-reading of `definition.json`, the place facts by the place taxonomy's fixed test**: `test_documents_about_places_and_pair_fluents`, `test_class_facts_about_places_agree_with_the_oracle`, and the earlier tests of the documents.
- **Every other sentence stays true by the oracle, `interpret(tree)` recovers every logical form, and the propositional rendering parses back**: the earlier tests of the documents, on the tiny and default worlds.
- **The full chain runs on the tiny configuration from the guides' commands alone**: above.
- **The tiny and default worlds keep their identities**, byte-identical to the reference run apart from the git state: above.
- **The full check list passes**: above.

## Specification edits

Every edit of `docs/specs/WORLD_PHASE_C.md` is a parenthetical "(As built, stage c5b: ...)" note, the status line, or the c5b entry of "Build stages". No decided text changed.

- The status line at the top: phase (c) is built, with the dates.
- "Lexicon": the prepositions of every used pair fluent (ruling 1).
- "Instance level", "Comparative descriptions": the other dimensions tried first (ruling 3).
- "Class level": the causal statements without mirrors (ruling 4); the profile statements without the vacuous ones, and where they stand (ruling 2); the documents about a pair fluent; the places in documents (the category term, its facts, the capacity facts, the topics, the fixed test).
- "Test sets": the causal sets, the profile sets, and the class sets.
- "Outputs": `documents.jsonl` and `stats.yaml`.
- "Build stages", the c5b entry: Jon's rulings WM.75 and the as-built summary.

`docs/specs/WORLD_AND_LANGUAGE.md`: the status of phase (c), in "What this specification does" and in the section "Phase (c)".

## Documents changed

- `docs/DECISIONS.md`: WM.75 after WM.74; WM.E364 to WM.E375 after WM.E363.
- `docs/specs/WORLD_PHASE_C.md` and `docs/specs/WORLD_AND_LANGUAGE.md`: as listed under "Specification edits".
- `docs/guides/CORPUS.md`: the status line; the lexicon table and its paragraph (the prepositions of every used pair fluent); "Propositions and truth" (class-level statements about categories of places); "Causal statements" (no mirrors, the profile statements, where the statements appear); "Comparisons of scalars" (ruling 3, with the illustration run); "Documents" (the table, and the paragraph on places and pair fluents in documents); the `stats.yaml` row and the logical form; the test sets (the class row, the causal row, the profile row, the change `form`, the paragraph on pair literals, places, and profiles, and the sets' sizes); the parameter table (`test_sets.changes`); the word forms (the tiny budget, the times); the run times; the reference to this file; and the figures by the script.
- `docs/guides/README.md`: the status row of the corpus, the run times, and the chain paragraph. `docs/guides/WORLD.md`: the status line (phase (c) complete) and the default world's run time. `docs/guides/WORDFORMS.md`: the tiny budget and time, and one figure by the script; the pipeline did not change. `README.md`: the chain's status line. `CLAUDE.md`: items 9 and 10 (stage c5b built, phase (c) complete) and the fast-tier time; no command changed.
- `data/corpus/default.yaml`: `form` in `test_sets.changes`. `data/wordforms/corpus_tiny.yaml`: the comment on the budget.

## Open questions

1. **Capacity facts about places in the documents about categories of entities.** "Owls can go to deserts" is stated in the document about deserts and in the document about the place event type, and not in the document about owls, whose relation facts are those of the two-place event types between entities, as before. The specification names the category of places as the patient of such a fact in a document about a category of places, and says nothing of the agent's document. Should a document about a category of entities draw such facts too, with the topic as the agent? Recommendation: yes, at the same chance as its other relation facts, as a change-track item; the stage left the entity documents as they were so that the earlier stages' documents moved as little as possible.
2. **Rule statements about the rules of place features.** A quarter of the default place properties are determined by rules, and the corpus states no rule statement about them ("dry places with sand are hot"), while it states the rules of the entities' features. Such a statement needs `SOMEPLACE` as the head of a restricted term, with its own truth test, grammar, and rendering. Recommendation: build it with phase (d)'s sorts, which would give the generic nouns one treatment.
3. **A negative pair state about a pair fluent that can never hold.** Ruling 2 leaves out the profile statement of such a pair fluent. At the instance level the corpus still says "the owl was not near the mouse" of a pair fluent that never holds (the default world's `PAIRFL.1`, since stage c4c; the tiny world's `PAIRFL.1`, since ruling 1 gave it a word), and the state sets hold such items, whose false item is the positive state. The sentence is true, and always true. Should initial states and state items leave such a pair fluent out? Recommendation: yes, by the same rule as the profile statement; the stage did not do it, because the prompt's rulings are about the class level.
4. **The function words' forms move when their frequencies move.** Every earlier content lexeme kept its word in this stage, and 5 of the default corpus's 19 function words took another form, because the request lists them by frequency and the pipeline makes their forms in that order (ruling 1 of WM.72 keeps the order). A model trained on one stage's rendered corpus cannot read the next stage's function words. The options: (a) keep the behavior; (b) list the function words in the order of `FUNCTION_WORDS`, so that a function word keeps its form whatever its frequency, and keep the frequency as a separate column for the pipeline's length rule. Recommendation: (b), as change-track work on the request and the pipeline's reading of it; it is outside this stage, which changes nothing in the pipeline.
5. **The guide's worked examples of the tiny corpus.** `docs/guides/CORPUS.md` shows sentences of the tiny corpus typed by hand ("One sentence, four renderings", the documents of "Documents", a tree in "Reading the outputs"). They show the right forms, and their labels and words date from before stage c4a: `DOC.19.SENT.3` of today's tiny corpus is another sentence, and `the` is `LEXEME.62`. Should the examples be retyped from the current run at the end of each stage, or written by the figures script? Recommendation: let `figures guides` write them between markers, as change-track work.
6. **The topics are drawn uniformly.** With the new topics, 55 of the default corpus's 10,000 documents are about a pair fluent, 46 about a place event type, and 560 about a category of places, and the documents about each earlier topic are fewer. No setting weights the kinds of topic. Is the uniform draw the wanted default now that the topics are of six more kinds?

## Notes for phase (d)

What the designer of phase (d) should know from building phase (c), beside the specification's own "Notes for phase (d)":

- **Two kinds of thing are built in by hand, in many places.** Phase (c) has entities and places, and every layer tells them apart by its own means: the world by separate index ranges and scopes (`place`, `place_binding`), the runtime by the shape of a location fluent's matrix, the corpus's truth tests by `is_place`, `is_place_term`, `_about_a_place`, and `_class_sort_problem`, the grammar by `NounPhrase.place` and `place_category` and by label prefixes, the renderings by prefixes, and the oracle by its own tables. Sorts in phase (d) should replace these checks with one signature per symbol, read by every layer. A third kind of thing (parts) built the same way would repeat every one of these checks.
- **The fixed test is per taxonomy.** `NEC` about places runs the fixed test over the place taxonomy's own features, rules, tree, and scalar settings (`World.place_fixed`). The two taxonomies share no feature and no rule. A rule that reads across sorts (a part's feature that fixes a whole's feature, or an adjacency computed from place features) has no fixed test today, and `NEC` about a derived pair fluent would need one.
- **Class-level statements about pairs state only what the definition guarantees.** Phase (c) states effects, preconditions, and profiles, and no fact about which categories stand in a pair fluent, because pairs change (WM.42). A static or rarely changing pair fluent (parthood) would give true class-level facts ("dogs have legs" as a pair fact), which need a truth test over pairs at the class level and a decision on REL.19.
- **`SOMEOF` and conditions with pair literals are still unsaid.** Both need a pair fluent read backwards or a second relative clause (WM.55, WM.59). A converse as a derived pair fluent in phase (d) would give `SOMEOF` a word of its own without new grammar: `SOMEOF.PAIRFL.k(x)` is `SOME` of the converse.
- **`SAME` is symmetric by meaning, and the corpus treats it so in three places.** The mirror rule (WM.75), the role swap of the causal sets, and the role swap of the state sets each know that `SAME` and a symmetric pair fluent hold both ways. A derived pair fluent with a declared profile would let one rule read the profile.
- **Vacuous truths need a rule each.** Phase (c) met three: an effect that can never change the state (WM.70), a pair fluent that can never hold (ruling 2 of WM.75), and a profile true of no pair. `Truth.can_hold` reads the initial rates and the adds of the whole world. Derived pair fluents will need the same question answered through their rules.
- **A comparison class has a root per kind of thing.** `ENTITY` and `EVERYPLACE` are separate roots, and a comparison never mixes an entity scalar and a place scalar. Parts as a third taxonomy would add a third root, or sorts would give one rule.
- **Places are static, and the corpus relies on it.** A place has no fluent, so a place is told apart by static features, a category of places is the agent of nothing, and every causal statement about a place event type is about its agent. A place fluent (a door that is open) would touch each of these.
- **Habitats are modest, and no sentence states them.** The corpus says which categories are able to act at which categories of places (capacity facts), and where an instance is in a scene, and never where a kind lives (WM.42 (e)). Strong kind-level habitats were deferred in stage c3b (WM.69).
- **The lexicon's order is append-only by practice.** Each stage of phase (c) appended its concepts after the earlier ones so that the earlier lexemes kept their labels and words, and the word-form pipeline gives the same words only for the same count of words (a larger tiny budget reshuffles nearly every word). A phase that adds words should append them, or accept new words for everything, and should decide the budget before the first run.
- **The topics of documents are drawn uniformly, so new topics dilute old ones.** Stage c5b added the categories of places, the place features, the place event types, and the pair fluents as topics, and the share of documents about each earlier topic fell. A study that needs a fixed amount of text about a kind of topic needs weights on the kinds of topic, which do not exist.
- **Tests that count what a small run drew are fragile.** Several earlier tests asserted that a run of a few dozen documents held some kind of sentence, and failed when the topics changed. They now check the world's statements by enumeration. New tests should enumerate where they can.
