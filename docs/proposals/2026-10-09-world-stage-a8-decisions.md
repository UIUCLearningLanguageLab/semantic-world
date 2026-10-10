# Proposal: decisions for stage a8 of the world-and-language refactor

October 9, 2026. Raised in the orientation for stage a8 of `docs/specs/WORLD_AND_LANGUAGE.md` ("a8. Documentation and datasets", the last stage of phase (a)), and while building the stage. Status: working design. Every choice below is Claude Code's unless Jon changes it. The choices are logged as WM.E165 and following in `docs/DECISIONS.md`.

Stage a8 builds three follow-ups of stage a7b (Jon's rulings 1 and 3 on that stage, and the quantifier-weight margin), the world guide, the updates of the other guides and the specifications, the regenerated datasets, and `CLAUDE.md`.

## Orientation

### What was read

- `CLAUDE.md`; the stage a7a and a7b proposal files in full, with their engineering choices, their "Notes for stage a7b" and "Notes for stage a8" (every place where the code departs from or adds to the specification), Jon's rulings on a7a, and the open questions of a7b; the proposal files of stages a1 to a6, skimmed for what the world guide must cover (the matrix form and the rule-set identity, a1; the world generator, the definition record, `entities.csv`, `derived/`, views, and the streams, a2; the runtime, the fixtures, and the brute-force evaluator, a3; episodes, policies, histories, `simulate`, and the statistics episodes, a4; the cut-over, a5a and a5b; the labels and the two-place share, a6).
- `docs/DECISIONS.md`, "World model": WM.1 to WM.34, and WM.E1 to WM.E164.
- `docs/specs/WORLD_AND_LANGUAGE.md` in full, with attention to "What it replaces", "Phase (a): Python package", stage a8 in "Phase (a): build stages", and the sections that the "Notes for stage a8" name: "States and changes", "'Can': what is held fixed", "Causal statements", "Referring content (CG.63)", "Test sets", "Outputs", "Configuration changes", and "Lexicon".
- The older specifications that the refactor supersedes in part: `docs/specs/TAXONOMY_GENERATOR.md`, `docs/specs/TAXONOMY_RELATIONS.md`, and `docs/specs/CORPUS_GENERATOR.md`. None of their sections carries a note pointing to `WORLD_AND_LANGUAGE.md` yet: a search for the specification's name, "refactor", and "world package" finds nothing in the three files. Stage a6 added such notes to `WORDFORM_PIPELINE.md` and `CONNECTED_SPEECH.md` ("Labels") only.
- `docs/guides/README.md`, `TAXONOMY.md`, `CORPUS.md`, and `WORDFORMS.md`.
- `python/semantic_world/world/` (the command line, `config.py` for every key and default, `generate.py`, `stats.py`, `episodes.py`, `views.py`, `fixtures.py`), `python/semantic_world/corpus/` (`planner.py`, the feature documents; `testsets.py`, the builder, `falsify`, `set_names`, `_kinds`; `facts.py`, `causal_statements`; `propositions.py`, `Truth.observed`; `stats.py`), `examples/two_place_levers.py`, and `tests/fixtures/world/README.md`.
- `tests/corpus/test_corpus_causal.py` and `tests/corpus/test_corpus_item_sets.py`, which encode the a7b layout of the causal sets (one pair per true statement, the two empty polarity law-like sets under `EMPTY_SETS`, 43 sets, and a true item used once per level and change), `tests/corpus/test_corpus_documents.py::test_quantifier_weights_rebalance_the_choice_of_facts` (the margin of 1.2), and `tests/corpus/test_corpus_outputs.py` (the message "43 test sets").
- The existing runs in `runs/`: `world/default_seed1` and `tiny_seed1` (stage a7a), `corpus/default_seed1` and `tiny_seed1` (stage a7b, October 9), `wordforms/corpus_default_seed1` (October 1, the old labels, 173 lexemes, 15 function words), `wordforms/corpus_tiny_seed1`, and the stage-4 and default word-form runs.

### How the pieces stand today

Main holds stage a7b (commit 29573dc, merged as PR #23). The world package defines worlds, runs episodes, and writes histories; the corpus runs on the world package; every program writes the labels of "Labels". Phase (a)'s features are all built. The corpus's feature documents draw their topics uniformly among the PROPERTY and PART features, the event types and categories with a word, and the base fluents with a word; a document about an event type or a fluent draws a causal statement at `propositions.causal_statement_rate` for each sentence, and a document about a fluent says nothing when the draw fails, so a fluent document's achieved length is binomial in its drawn length. The causal test sets hold one pair per true causal statement, because a true item is used once per level and change, and the two polarity law-like sets are empty by the world's own laws; the default corpus has 43 sets. The quantifier-weight test of `nec_none` asserts a rise of 1.2 with 300 documents.

Before the first change, a reference run of the tiny chain was made from `main` at 657af39 (`define data/world/tiny.yaml`, then `generate data/corpus/tiny.yaml`) into `/Users/jon/Documents/Projects/semantic-world-reference/a8/` (outside the repository), with the commit and the logs beside it: 12 entities, 4 fluents (1 derived), 4 one-place and 4 two-place event types, rule set `bd02c67e7286`; 20 documents, 203 sentences, 1,105 tokens, 17 scenes, 43 test sets with 518 pairs. The tiny and default worlds must not change in this stage; the tiny corpus changes only by the follow-ups below, and the differences are listed under "The tiny corpus against the reference".

### Jon's rulings on stage a7b

Recorded at the end of the stage a7b proposal file, and applied here: (1) fluent documents hold causal statements alone, one per sentence, with the rate applying to documents about event types and categories only; (2) the observed mark stays non-vacuous; (3) each true causal statement is paired with every valid false item, capped at `test_sets.size`, and the polarity law-like sets are dropped; (4) `test_sets.changes` keeps `polarity` and `event`; (5) a pronoun's `descriptive` mark stays false. WM.E146 to WM.E164 are decided.

### Plan for the stage

1. **Follow-up 1, fluent documents.** In `Planner._feature_document`, a document about a fluent takes a causal statement for every sentence, without the rate draw, until it reaches its drawn length or has stated every statement about its fluent. The rate keeps its meaning for documents about event types and categories.
2. **Follow-up 2, the causal sets.** A new path in `TestSetBuilder` for the causal levels: enumerate the true statements of the set's kind in the enumeration order of `Facts.causal_statements`, pair each with every valid false item of the set's change (false under `NEC`, expressible, statable), sort each pair into the ordinary set or the `_lawlike` twin by the `observed` mark, and cap each set at `test_sets.size` by a uniform draw without replacement from the set's stream. Each item records the true statement it tests. `_kinds` gives the polarity change one set, so 41 sets are written. The tests of a7b that encode the old layout change (listed under "Changed expectations").
3. **Follow-up 3, the quantifier-weight margin.** Measure the rise of the `nec_none` share with more documents and more seeds; restore 1.25 if the rise clears it with ample samples, or keep 1.2 and report the measurement.
4. **The world guide**, `docs/guides/WORLD.md`, and `docs/guides/README.md`.
5. **The other guides**: `TAXONOMY.md` (what moved, with a pointer), `CORPUS.md` (the follow-ups and every figure of the regenerated default corpus), `WORDFORMS.md` (the figures the regenerated runs change).
6. **The specification**: the edits that the "Notes for stage a8" and the a7a departures name, each marked "(as built, stage a7a)", "(as built, stage a7b)", or "(as built, stage a8)", and listed below by section.
7. **The older specifications**: a note at the top of every section whose behavior the refactor changed, listed below.
8. **The datasets**: the tiny and default chains regenerated end to end in `runs/`, each step timed.
9. **`CLAUDE.md`**: the reading list and the commands.
10. **Tests and documents**: the tests a change touches while building, the full check list at the end; this file, `docs/DECISIONS.md`.

### Reading of the scope

- "**Until it reaches its length or runs out of statements**" (ruling 1) is read as: a fluent document's achieved length is the smaller of its drawn length and the number of causal statements about its fluent, so a fluent with few statements gives a short document, and a fluent document is never shorter than its drawn length for any other reason. The feature documents' other draws (the topic, the length, the order) are unchanged, so every feature document about a feature or an event type is the same as before.
- "**Every valid false item of the set's change**" (ruling 3) is read as every candidate of `candidates(facts, true, change)` that the language can express and state and that the truth tests judge false, the same conditions `falsify` applies to the one it draws. "Valid" excludes nothing more: a false item that is observed goes into the `_lawlike` twin, as before.
- "**Up to `test_sets.size`**": each set (the ordinary set and the law-like twin apart) holds at most `test_sets.size` pairs; the pairs are enumerated first and then drawn uniformly without replacement when there are more, from the set's own stream (`tests:<level>_<change>`), and kept in enumeration order, so that the pairs of one true statement stay together.
- "**Which true statement it tests**": both items of a pair record the true statement, so that an analysis can group the pairs of a set by statement and can find a false item's statement without its pair.
- "**The two polarity law-like sets are no longer written**": the sets `causal_effect_polarity_lawlike` and `causal_precondition_polarity_lawlike` leave the file order and `set_names`; the polarity change keeps its ordinary set at each causal level; nothing else about the law-like twins changes.
- "**Give the test more samples**" (follow-up 3): more documents in the two runs that the test compares, or several seeds, with the margin of 1.25 restored if the measured rise clears it with the spread reported here.
- The world guide "**covers what a world is in plain terms**", so it describes the definition, the entities, the fluents, the event types, the requirements, the preconditions, the effects, the views, and the histories as the other guides describe their programs: for a reader who runs the programs, with the specification for the detail. Its figures come from a fresh run of this stage, which is the regenerated run of item 8.
- The spec edits "**change only what the notes and the follow-ups name**": each is a sentence or a clause added or changed in place, marked with its stage, and the design is untouched. An item of the notes that the specification already states correctly gets no edit.
- The notes in the older specifications go at the top of each changed section, after its heading, as one short paragraph that names the section of `WORLD_AND_LANGUAGE.md` that replaces it; a section whose behavior did not change gets none. The three files are not otherwise changed.

### Questions

No question of the orientation changes the design or a file format beyond what the stage prompt lists. The name of the record of which true statement an item tests is an engineering choice (choice 2 below). The stage proceeds.

## Engineering choices made while building

Numbered for `docs/DECISIONS.md` (WM.E165 and following). Each is Claude Code's choice inside the stage's scope.

1. **Fluent documents** (follow-up 1, Jon's ruling 1 on stage a7b). In `Planner._feature_document`, a document about a fluent takes a causal statement for every sentence, drawn uniformly among the statements about its fluent that it has not stated, without the rate draw, and ends when it reaches its drawn length or has stated every statement. `propositions.causal_statement_rate` applies to documents about event types and categories only, as before. Every other feature document is unchanged, because each document draws from its own parts of the four streams, so the follow-up changes the fluent documents of an existing corpus and nothing else.
2. **The causal sets are enumerated** (follow-up 2, Jon's ruling 3 on stage a7b). `TestSetBuilder._causal_group` replaces the drawing loop for the two causal levels: every true causal statement of the set's kind, in the enumeration order of `Facts.causal_statements`, is paired with every valid false item of the set's change, given by the new function `testsets.false_items` (the candidates of `candidates()` that are expressible, that the language can state, and that the truth tests judge false, the conditions `falsify` applies to the one it draws, now shared in `_valid_false`); each pair goes into the ordinary set or the `_lawlike` twin by the `observed` mark of its false item; a set with more than `test_sets.size` pairs keeps a uniform draw of them without replacement, in enumeration order, from the set's part of `corpus:tests` (`tests:<level>_<change>`), so that the pairs of one statement stay together; the true item of each kept pair takes a bare plural at the generic rate from the same part. `_kinds` gives the polarity change one set at each causal level, because a polarity swap is never observed (the true effect or literal guarantees the opposite value), so a run writes 41 sets; a polarity swap that the scenes did observe would be a `CorpusError`, since the world's laws rule it out. The item count of a set is therefore the smaller of `test_sets.size` and the pairs the world gives.
3. **The statement record.** Both items of a causal pair carry `statement` in their metadata: the propositional rendering of the true item, `NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))`, which is one string per statement, the same whether the item says "all" or a bare plural, and parses back to the statement. An analysis groups a set's pairs by it, and finds a false item's statement without its pair. `ItemSet.stats()` reports each causal set's distinct statements as `true_statements`, which `stats.yaml` carries.
4. **The observed mark's index.** `Truth` indexes the known events by event type on first use (`_events_of_type`), and `observed` scans the events of the candidate's event type, or of the event types below its category, in scene order, instead of every event of every scene; `add_scene` drops the index. The causal sets ask about thousands of candidates in the default corpus, and each needs the events of a few event types. The mark is a pure function of the scenes, so nothing in the output changes (the tiny and default corpora regenerate byte for byte with and without the index). The index did not shorten the run: the default corpus generates in 158 seconds with it, against 156 without, so the scan of the scenes was not the cost of the enumerated sets (the cost is in judging and realizing about 2,300 causal pairs in place of about 1,000). The index stays, because it does less work for the same answer, but it is not a speed-up.
5. **The quantifier-weight margin** (follow-up 3). `test_quantifier_weights_rebalance_the_choice_of_facts` compares two runs of 1,000 documents for the `nec_none` check (the other checks keep 300) and asserts a rise of 1.25 again. About 3% of the class-level encyclopedic sentences are `nec_no`, so 300 documents give about 50 of them and the rise swings from seed to seed; at 1,000 documents it settles above the margin. Measured on the default world, `quantifiers.weights.nec_none: 4` against the default: 1.47, 1.38, 1.31, 1.44, 1.44, 1.43 on seeds 1 to 6 at 1,000 documents (the test's seed 1 gives 1.47); 1.52, 1.32, 1.18 on seeds 1 to 3 at 600; 1.23, 1.44, 1.10 at 300. The test's two extra runs add about 35 seconds.
6. **The notes in the older specifications** are one line each, directly under the heading of a section whose behavior the refactor changed: "**Note (world-and-language refactor).**", the stage of the change, what changed, and the section of `WORLD_AND_LANGUAGE.md` that replaces it. Nothing else in the three files is changed. A section whose behavior did not change (the scalars of `TAXONOMY_RELATIONS.md`, the grammar's morphology table, the word-form request) gets no note. The notes are listed below.
7. **The specification's as-built edits** are parentheses added to the sentence that the "Notes for stage a8" or the a7a departures name, marked "(As built, stage a7a)", "(as built, stage a7b)", or "(as built, stages a7b and a8)", with Jon's ruling named where one applies; the three renderings whose examples lacked the tense (`ABLE_NOW` and the two `BECOME` forms) are rewritten with it. A note item that the specification already stated correctly got no edit. The 18 edits are listed below by section.
8. **The regenerated runs** go to the default folders of the commands, and the old folders at those paths are moved aside with the suffix `_pre_a8` rather than deleted or overwritten: a run is written in place (`mkdir` without clearing), so an old file that the new run does not write would stay in the folder (the old `runs/world/tiny_seed1/taxonomy/` still held the verb files of stage a5a). The superseded folders are listed below for Jon to delete.
9. **The full chain's first command is `define`**, optional because `generate` defines the world in memory from the same file; the guides' README, `CLAUDE.md`, the world guide, and the corpus guide give the four commands and their times at default scale.

## Changed expectations

Every expectation that changed, by file, with the reason. The worlds do not change (the world package and `data/world/` are untouched; the tiny and default worlds regenerate byte for byte, apart from the provenance block of `config.yaml`). The corpora change by the three follow-ups alone.

- `tests/corpus/test_corpus_item_sets.py`: `ALL_SETS` lists 41 sets (the two polarity law-like sets are gone) and `EMPTY_SETS` is removed; the layout test expects every set to have pairs, accepts a causal set below the size (the world gives fewer pairs), and checks that a true item is used once for the non-causal levels only, because a true causal statement is now paired with every valid false item; the no-narratives test and the two coverage checks drop their `EMPTY_SETS` exceptions.
- `tests/corpus/test_corpus_causal.py`: 41 sets and 14 causal sets in the tiny corpus; `test_feature_documents_about_event_types_and_fluents` requires a fluent document's length to be the smaller of its drawn length and the number of statements about its fluent; `test_the_causal_statement_rate_is_a_setting` expects fluent documents to hold the same sentences at every rate, including 0 (before: no fluent sentences at rate 0); two new tests, `test_the_causal_sets_pair_every_statement_with_every_valid_false_item` (every pair of a true statement and a valid false item is in a set when the set is not capped, a capped set holds a subset in enumeration order, no polarity law-like set, the `statement` record on both items and the `true_statements` count, and the same pairs on a second build) and `test_the_causal_sets_are_capped_at_the_size`.
- `tests/corpus/test_corpus_outputs.py`: the generate command prints "41 test sets".
- `tests/corpus/test_corpus_documents.py::test_quantifier_weights_rebalance_the_choice_of_facts`: the `nec_none` check runs on 1,000 documents with the margin 1.25 (follow-up 3).

Numbers of the default corpus (10,000 documents) in `docs/guides/CORPUS.md`, old to new: 99,448 sentences and 470,380 tokens to 99,448 sentences and 470,381 tokens (the fluent documents say the same number of sentences, in another order, with other bare plurals); 43 sets with 13,726 pairs to 41 sets with 15,509 pairs; the causal sets from 32 to 87 ordinary and 4 to 24 law-like pairs to the figures of the guide (the two event-swap sets full at 500, the others 32 to 321 and 4 to 299); 2,657 causal sentences (1,629 effects, 1,028 preconditions). Every other figure of the guide (the scenes, the lexicon, the rule terms, the narratives ending early, the co-occurrence table to two decimals, the seen shares, the state and `able_now` sets) is unchanged, because the follow-ups change the fluent documents and the causal sets only.

## The tiny corpus against the reference

The tiny world is byte-identical to the reference (`define data/world/tiny.yaml` gives the same files, apart from the provenance block of `config.yaml`; rule set `bd02c67e7286`), and so is the default world against the run of stage a7a (rule set `123a4a6eb9f3`; the regenerated folders differ from the previous ones in `config.yaml` and `taxonomy/config.yaml` alone, by the git commit and the dirty flag). The tiny corpus (`generate data/corpus/tiny.yaml`, seed 1) against the reference:

| What | Reference | This stage |
| --- | --- | --- |
| Documents, scenes | 20, 17 | 20, 17 (the scenes are identical, event for event) |
| Sentences, tokens | 203, 1,105 | 203, 1,105 |
| Documents whose sentences changed | | 2 of 20, the two documents about a fluent (`DOC.4` about `BOOLFL.3`, 3 sentences; `DOC.15` about `BOOLFL.1`, 5 sentences): each states the same statements as before, in another order and with other bare plurals, because the rate draws are gone. The 18 other documents are the same sentences in the same order |
| Test sets, pairs | 43, 518 | 41, 550: the 27 non-causal sets keep their pairs; the 14 causal sets hold 16, 4, 10, 20, 1, 5, 1 (effect: predicate, its law-like twin, polarity, event, its twin, role, its twin) and 12, 0, 7, 13, 0, 4, 1 (precondition) pairs, from 10 effect and 7 precondition statements; the two polarity law-like sets are not written |
| `stats.yaml` | | each causal set gains `true_statements` |

The lexicon, the renderings of the unchanged sentences, and the word-form request are the same. The full chain runs on the tiny configuration: `define`, `generate`, `wordforms all` (9 seconds), and `render` (203 sentences and 1,100 test items rendered).

## The edits to the specification

Every edit to `docs/specs/WORLD_AND_LANGUAGE.md`, by section. Each is a parenthesis added to the existing sentence, marked with its stage, except the three renderings rewritten with the tense. The design is unchanged.

| Section | Edit | Source |
| --- | --- | --- |
| Lexicon | The tense of `become`: marked when tense is a separate word, not at all under an affix, until stage b3 (as built, stage a7a). | a7a choice 7, Jon's ruling 2 on a7a |
| "Can": what is held fixed | The `ABLE_NOW` rendering written with the tense, `ABLE_NOW(SCENE.8, TIME.2, PAST, ...)`, with the reason (as built, stage a7a). | a7a choice 4, Jon's ruling 1 on a7a |
| States and changes | The two `BECOME` renderings written with the tense (as built, stage a7a). | a7a choice 4, Jon's ruling 1 on a7a |
| States and changes | The state predicate's key is `label` (as built, stage a7b). | a7b choice 1, Jon's ruling 4 on a7a |
| States and changes | A result's derived fluent counts when the event's own changes alone bring it to its value; `caused_by` in the grounding (as built, stage a7a). | a7a reading of the scope, choice 12 |
| Causal statements, Truth | The grounding `{"event_types", "with_entry", "able_bindings", "test": "definition"}`; a statement with no able binding is invalid (as built, stage a7b). | a7b choice 4 |
| Causal statements, Surface | `before` is an `Adv` and the seventeenth base function word (as built, stage a7b). | a7b choices 7 and 9 |
| Causal statements, JSON form | The predicate key `label`; the value in the predicate and the polarity positive; the `causal` record as metadata of a test item (as built, stage a7b). | a7b choices 1, 3, 15 |
| Causal statements, In documents | How the topics were gained; a fluent document holds one causal statement per sentence and the rate does not apply to it; the section `causal` last (as built, stages a7b and a8). | a7b choice 10, a8 choice 1, Jon's ruling 1 on a7b |
| Causal statements, Readings | `causal` takes the place of `generic`; the quantifier readings follow; the `not` never counts as a negated bare plural (as built, stage a7b). | a7b choice 8 |
| Referring content (CG.63) | A pronoun's `descriptive` mark is false (as built, stage a7b). | a7b choice 12, Jon's ruling 5 on a7b |
| Referring content (CG.63) | The brace is followed by one space; a part an asserted mention also contributes is asserted and written once; no braces when nothing describes (as built, stage a7b). | a7b choice 13 |
| Referring content (CG.63) | What the `marked` and the `omitted` renderings parse back to (as built, stage a7b). | a7b choice 13 |
| Referring content (CG.63) | What `test_sets.seen.descriptions: false` counts (as built, stage a7b). | a7b choice 14 |
| Test sets, State sets | A pair is `_changed` when either item changed; `changed_item` on both items (as built, stages a7a and a7b). | a7a choice 13, a7b choice 2, Jon's ruling 5 on a7a |
| Test sets, Causal sets | The builder's levels; `test_sets.changes` gains `polarity` and `event`; every statement paired with every valid false item, capped at the size; `statement`; the non-vacuous `observed` mark; no polarity law-like twin, 41 sets (as built, stages a7b and a8). | a7b choices 15 and 16, a8 choices 2 and 3, Jon's rulings 2 to 4 on a7b |
| Outputs | The `states` and `causal` blocks, `pairs_by_changed_item`, `true_statements` (as built, stages a7a to a8). | a7a choice 14, a7b choices 2 and 17, a8 choice 3 |
| Configuration changes | `test_sets.changes` gains the values `polarity` and `event` (as built, stage a7b). | a7b choice 15 |

Items of the "Notes for stage a8" that needed no edit, because the specification already says what was built: the subject of a causal statement as `{"head": "THING", "event": ..., "role": ...}`; `before` as a word-order setting; the `causal` block's names (covered by the Outputs edit); the guide's figures (the guides, not the specification).

## The notes in the older specifications

Each note is one line under the section's heading, and the three files are not otherwise changed.

| File | Section | What the note says |
| --- | --- | --- |
| `TAXONOMY_GENERATOR.md` | Terms | CAN features are not a feature type; IS and HAS are PROPERTY and PART. |
| | Labels | The new labels; the old ones are the record. |
| | Feature types, CAN | CAN features are the one-place event types of the world package, under `event_types.unary`. |
| | Rules | The section holds for PROPERTY and PART rules; CAN rules are one-place requirements; the matrix form and the agreement test. |
| | The tree, Distinct leaves | Leaves differ on PROPERTY and PART features. |
| | Instances | No CAN features; capacities are derived values of a world run. |
| | Node vectors | The vectors hold ISA, PROPERTY, PART, and scalars. |
| | Configuration | `features.can` and `rules.overrides.can` moved; `property` and `part` keys (stage a6). |
| | Determinism | The one-place requirements draw from `world:requirements`. |
| | Outputs | `base.csv` and `derived/` replace `instances.csv`; `rule_matrices.json`; the classic view. |
| `TAXONOMY_RELATIONS.md` | Labels | The scalar and event-type labels. |
| | Part B: transitive verbs | All of Part B is built by the world package as two-place event types; `verbs` is `event_types.binary`. |
| | Part B, Projections | Capacities in `derived/capacities.csv`; exposure removed; views. |
| | Configuration | The `verbs` block moved without `projections`; the scalar settings stay; `property` and `part` keys. |
| | Determinism | The verb streams are `world:event_tree`, `world:constraints`, `world:pairs`. |
| | Outputs | The relation files and the capacities are a world run's derived values, with `event_type` for `verb`. |
| `CORPUS_GENERATOR.md` | Inputs | The corpus reads a world; `taxonomy` is an error. |
| | Labels | The new labels. |
| | Layer 1, Concepts that get words | The world's concepts, state adjectives, renamed keys. |
| | Layer 1, Function words | `become`, `before`, `can_now`. |
| | Layer 2: propositions | Six levels and causal statements. |
| | Layer 2, Class level | The quantifiers of CG.59 to CG.62. |
| | Layer 2, Event level | Events of histories; aspect per report; `able` and `legal`. |
| | Layer 2, Negation | `negation_rate.state` and `.able_now`. |
| | Layer 2, False propositions and test sets | Law-like items under `NEC`; the split event sets; the state, `able_now`, and causal sets. |
| | Scenes and events | Replaced by the episodes of the world package. |
| | Layer 3, Document types | Feature documents' new topics and causal statements; narratives' new sentences. |
| | Layer 3, Ordering | The template ends with the causal statements. |
| | Layer 3, Mentioning referents | `mention.event_level_weights`. |
| | Layer 4, Phrase structure | States, changes, blocked events, causal statements. |
| | Layer 4, Word order | `grammar.word_order.before`. |
| | Layer 4, Morphology | The tense of a timed sentence under a word and under an affix. |
| | Layer 4, Logical form and surface form | The readings `state`, `able_now`, `causal`; `descriptive`. |
| | Renderings | The new labels, quantifiers, operators, and braces. |
| | Outputs | Histories in `scenes.jsonl`; the world's identity; the new fields and sets. |
| | Configuration | The renamed and new keys. |
| | Python package | `scenes.py` replaced by `world.episodes`. |
| | Decisions to confirm | Decisions 4 and 39 superseded; the quantifier decisions revised. |

## The regenerated runs and their times

Every run was made on October 9, 2026, on Jon's laptop, with the code of this stage, into the default folders of the commands. The world and the corpus were timed alone; the word forms ran beside the corpus test suite.

| Step | Command | Output | Time |
| --- | --- | --- | --- |
| The tiny world | `python -m semantic_world.world define data/world/tiny.yaml` | `runs/world/tiny_seed1` (12 entities, rule set `bd02c67e7286`) | 2.8 s |
| The default world | `python -m semantic_world.world define data/world/default.yaml` | `runs/world/default_seed1` (284 entities, rule set `123a4a6eb9f3`) | 7.8 s |
| The tiny corpus | `python -m semantic_world.corpus generate data/corpus/tiny.yaml` | `runs/corpus/tiny_seed1` (20 documents, 203 sentences, 41 sets, 550 pairs) | 3.2 s |
| The default corpus | `python -m semantic_world.corpus generate data/corpus/default.yaml` | `runs/corpus/default_seed1` (10,000 documents, 99,448 sentences, 470,381 tokens, 10,088 scenes, 41 sets, 15,509 pairs) | 157.8 s (155.9 s in the first run of the stage, before the event index of choice 4) |
| The tiny word forms | `python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml` | `runs/wordforms/corpus_tiny_seed1` | 9.2 s |
| The tiny rendering | `python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1` | 203 sentences and 1,100 test items rendered | 0.3 s |
| The default word forms | `python -m semantic_world.wordforms all data/wordforms/corpus_default.yaml` | `runs/wordforms/corpus_default_seed1` (199 content lexemes, 17 function words, 45 speakers, five embeddings) | 1,818 s (about 30 minutes), beside the corpus test suite |
| The default rendering | `python -m semantic_world.corpus render runs/corpus/default_seed1 --wordforms runs/wordforms/corpus_default_seed1` | 99,448 sentences and 31,018 test items rendered | 2.4 s |

The default world and corpus together take under 3 minutes, against the 15 minutes the stage allows. The chain also ran on the tiny configuration in a fresh copy of the repository, from the commands of the guides alone (see "Acceptance").

**Superseded run folders**, moved aside with the suffix `_pre_a8` or left in place, for Jon to delete:

- `runs/world/default_seed1_pre_a8` and `runs/world/tiny_seed1_pre_a8`: the world runs of stage a7a (the tiny one still held the taxonomy's verb files of stage a5a, which the stage a5b run did not remove).
- `runs/corpus/default_seed1_pre_a8` and `runs/corpus/tiny_seed1_pre_a8`: the corpus runs of stages a7a and a7b.
- `runs/wordforms/corpus_default_seed1_pre_a8`: the word forms of the default corpus of October 1 (the old labels, 173 lexemes, 15 function words), and `runs/wordforms/corpus_tiny_seed1_pre_a8`.
- `runs/corpus_default_wordforms.log`: the log of that October 1 run.
- `runs/wordforms/default_seed1_stage4`: a word-form run of stage 4 of the pipeline, which nothing reads.
- `runs/wordforms/default_seed1` (the word forms of `data/wordforms/default.yaml`, over the taxonomy's meanings, October 1) is not part of the chain and was not regenerated; the word-form guide's figures of the default run come from it. It is listed here as a candidate only.

## Open questions for Jon

1. **The size of the causal sets** (choice 2). With every valid false item paired, the two event-swap sets of the default corpus fill their 500 pairs from 73 and 51 statements, so a statement appears in about seven pairs there, while the role-swap sets hold 42 and 32 pairs from as many statements. Should a set with more pairs than the size keep every statement (a draw stratified by statement) rather than a uniform draw of the pairs? Under the uniform draw, 73 of the 87 effect statements with an event swap appear in `causal_effect_event`.
2. **The statement record** (choice 3) is the true item's propositional rendering, under `statement`. Would a structured record (the `causal` record of the true item with its kind) serve the analyses better, or both?
3. **The default corpus's time budget.** The corpus takes about 2.6 minutes (158 seconds, against about 2 minutes in stage a7b); the enumeration of the causal sets is the one step whose cost grows with the number of true statements times the number of candidates. Is the budget of 15 minutes to stay the rule for the default scale, so that a larger default world would have to cap the enumeration (for example by sampling candidates before judging them)?

## Notes for phase (b)

What the builder of stage b1 (conditional effects) must know about where things stand.

- **Phase (a) is complete.** Every feature of "Phase (a): the world" and "Phase (a): the corpus" is built, the specification says what was built where it departed from the draft (the "(as built, ...)" parentheses), the older specifications point to it, and the guides describe the programs as they run. The datasets in `runs/` are the regenerated default and tiny chains of this stage.
- **Where conditional effects touch the code.** The world side: `python/semantic_world/world/dynamics.py` (`Literal`, `Effect`), `event_types.py` (the drawing of effects and the fix-ups, `apply_event_file`), `definition.py` (`EventTypeRecord.effects`, the loader's validation, the hashed `event_types` table, so a condition changes the rule-set identity), `runtime.py` (`apply`, and `interferes`, which must count what a condition reads), `fixtures.py` (the brute-force evaluator's `apply` and interference test), `history.py` (`replay`), and `tests/fixtures/world/README.md` (the event-type record and the semantics of `apply`). The corpus side: `propositions.py` (`Truth.has_entry` and `_evaluate_causal`, which read `effects` with `role`, `fluent`, and `value`), `facts.py` (`causal_statements` enumerates over the definition's entries), `testsets.py` (`_causal_candidates` and `_causal_group`), `renderings.py` (the formula of a causal statement, `Quantified(nec_all, (BoundEvent, ...), Temporal(...))`, whose restrictor must take the condition), `grammar.py` and `realize.py` (the causal plan's subject, which must take a modifier or a relative clause for the condition), and the oracle `tests/corpus/truth_oracle.py` (`entries`, `causal`, `observed`, which re-read `definition.json`).
- **The acceptance of b1** ("a world with `conditional_share: 0` gives the same definition as before the stage") can be checked against the regenerated runs: `runs/world/default_seed1/definition.json` and `runs/world/tiny_seed1/definition.json` of this stage, with the rule sets `123a4a6eb9f3` and `bd02c67e7286`; a reference copy of the tiny chain is in `/Users/jon/Documents/Projects/semantic-world-reference/a8/`.
- **The causal sets** are enumerated, not drawn (choice 2), and the `observed` mark scans an index of events by type (choice 4). A conditional causal statement as a false item ("a statement without its condition is not `NEC`") goes into the `_lawlike` twin when it held in every scene, which `observed` already judges; the candidates of the causal changes must learn the condition, and `_causal_group` pairs every candidate, so the number of candidates per statement is the cost to watch.
- **The function words** are the seventeen base words ending in `become` and `before`, with `can_now` conditional; stage b3's inflection of function words (Jon's ruling 2 on a7a) is a word-form change and a corpus change (`Builder.function` takes marks today, and the request's `inflect` entries name content lexemes).
- **Timings.** The default world's `define` takes 8 seconds and the default corpus about 2.6 minutes; `pytest tests/corpus` about 15 minutes, the whole suite about 30; the default word forms with audio and embeddings about 30 minutes.
- **The reference runs** of this stage are in `/Users/jon/Documents/Projects/semantic-world-reference/a8/` (the tiny world and the tiny corpus from `main` at 657af39, before the follow-ups).

## Acceptance

Each criterion of the stage prompt, and how it was checked.

- **Fluent documents** hold causal statements alone, one per sentence, up to the drawn length or the statements about the fluent, and the rate changes documents about event types and categories only: `test_feature_documents_about_event_types_and_fluents` and `test_the_causal_statement_rate_is_a_setting`.
- **The causal pairs**: every true item a causal statement of the set's kind, every false item differing by the set's change and false under `NEC`, every `observed` mark agreeing with the scenes by the oracle's replay, every item naming its statement, no set above `test_sets.size`, and the same pairs on every run: `test_false_causal_items_are_false_under_nec_and_observed_marks_agree`, `test_the_causal_sets_pair_every_statement_with_every_valid_false_item`, `test_the_causal_sets_are_capped_at_the_size`, and the one-change test of `test_corpus_item_sets.py`.
- **41 sets**, without the polarity law-like sets: `test_the_layout_of_the_test_sets`, `test_the_tiny_corpus_holds_every_new_thing`, `test_the_generate_command`.
- **The worlds are byte-identical** to the reference and to the previous runs, apart from provenance: `diff -r` on the folders (above).
- **The full chain in a fresh folder**: a copy of the repository without `runs/`, `.venv`, and `.git` (with the Piper voices copied into `runs/wordforms/voices`, as the word-form guide's setup requires), running the four commands of the guides with `PYTHONPATH=python`: every step ran as written (the tiny world with rule set `bd02c67e7286`, the tiny corpus with 41 sets and 550 pairs, 40 lexemes with forms, 203 sentences and 1,100 test items rendered), and so did every other example command of the guides (`view` with both presets and `--include`, `simulate` with and without `--legal --config`, `check-fixtures`, `make-fixtures` into a scratch folder, whose files matched the committed fixtures byte for byte, `hand_world.py`, the taxonomy generator, the two Python examples, and the levers script on one seed and one lever).
- **Every example command** of `WORLD.md`, `TAXONOMY.md`, `CORPUS.md`, and `CLAUDE.md` runs as written: the commands and the Python snippets were run in the fresh copy (the levers script with `--seeds 1 --only baseline`), and every figure in the guides comes from the runs of this stage.
- **The default world and corpus in under 15 minutes**: 8 seconds and 158 seconds, recorded in `WORLD.md`, `CORPUS.md`, and the guides' README.
- **The spec edits and the notes** are listed above by section.
- **Every corpus test passes**: `pytest tests/corpus`, 875 seconds, every test passed, none skipped; the changed expectations are listed above.
- **The full check list** of `CLAUDE.md`: `cargo fmt --all --check`, `cargo clippy` with warnings as errors, `cargo test --workspace --all-features`, `maturin develop --release`, `ruff check python tests`, and `pytest` (1,916 tests passed, none skipped, 1,391 seconds) all pass on the final code.

## Jon's rulings (October 9, 2026, in the stage b1 prompt)

Choices 1 to 9 are approved, and WM.E165 to WM.E173 are decided. The open questions are answered:

1. **The size of the causal sets** (choice 2): a causal set with more pairs than `test_sets.size` keeps every statement. The pairs are drawn stratified by statement, round-robin over the statements, deterministically from the set's stream, until the set is full. Stage b1 builds the change (its follow-up 1).
2. **The statement record** (choice 3): keep `statement`, the true item's propositional rendering, and add beside it a structured record, the true item's causal record with its kind. Stage b1 builds the change (its follow-up 2).
3. **The default corpus's time budget**: no hard time limit on the default run, and no capping of the enumeration until a run actually gets slow. Instead, every stage records the default world's and corpus's run times in the guides, and the proposal file flags any growth of more than half, with its cause; a step is never made cheaper by changing what it produces by default. If a step becomes too slow, it gets a setting (for example a cap on an enumeration) that a study can turn on, with the full version as the default. Stage b1 records the rule in the specification, the guides, and `CLAUDE.md` (its follow-up 3).
