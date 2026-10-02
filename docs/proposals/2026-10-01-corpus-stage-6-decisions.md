# Proposal: decisions for stage 6 of the corpus generator

October 1, 2026. Raised at the end of stage 5 of `docs/specs/CORPUS_GENERATOR.md`, in the orientation for stage 6 (outputs, test sets, statistics, and the `generate` command), and while building stage 6. Status: decided. Jon gave decisions 46 to 48 in the stage 6 prompt. Jon answered the four questions of the orientation as decisions 49 to 52, and added decision 53 and the last part of decision 46. The specification was updated to match.

## Jon's decisions

### 46. Event test items name only the scene, and true and false items have one format

Stage 5 left one question open: how the propositional rendering writes an event-level test item that names no event. A false event has no label, because it did not happen.

**Decision.** Every event-level test item, true or false, uses the same form: `EVENT(SN.8, PAST, SIMPLE, V1.2(R.1, R.2))`, meaning that some event in scene `SN.8` was this one. Documents keep their event labels. Only test items use the scene form, so a true item cannot be recognized by having a label.

More generally, the true and the false items of a test set must not differ in format: every field that a true item has, a false item has too, in the same notation. A test checks the rule for every test set.

The rule covers the model-facing fields only: the sentence, its renderings, and its logical form. Metadata such as the truth label, the grounding, and the `seen` field differ by design. The item schema keeps the model-facing fields and the metadata apart.

### 47. Longer entity narratives

Stage 5 found that an entity narrative had 6 sentences on average, against a drawn range of 5 to 20, because a scene has few events that involve one instance.

**Decision.** The default `entity.scenes` changes from `[1, 3]` to `[2, 5]`.

### 48. Quantifier weights

Stage 5 found that more than half of the class-level sentences take `some`.

**Decision.** A new setting weights the choice of class-level facts by quantifier, so a study can rebalance the mix. The default keeps the earlier behavior: the strongest true quantifier, with no reweighting. Claude Code chooses the setting's name and shape, and records the choice (choices 1 and 2 below).

## Questions of the orientation, and Jon's answers

Claude Code raised four questions before building.

### 49. Which events can be true items?

Claude Code recommended only the events that a document reports in a main clause.

**Decision.** A true event item can be any event that its document reports, in a main clause or in a relative clause.

### 50. How false must a false event be?

The test sentence names neither its scene nor, with the default grammar, its aspect. An event that did not happen in `SN.8` could have happened in `SN.9` of the same document, or in the other aspect. The words of such a false item would be true of the document.

**Decision.** The stricter rule, as recommended. No event with the item's verb, agent, and patient happened in any scene of the item's document, in either aspect. When the item names its verb at a verb-category level, no event of any leaf verb under that category may match either. This rule is stricter than decision 29.

### 51. How is co-occurrence counted?

**Decision.** By words, as recommended: a leaf occurs in a document when its own noun appears, and a pair's score is the number of documents with both nouns. We report Pearson's and Spearman's correlations. We also report the same correlations counted by referents, as a second measure.

### 52. The layout of the test sets

The specification said that the law-like items and the two kinds of false event get "test sets of their own", but not whether that means one set for each change.

**Decision.** One set for each change, as proposed: `class_<change>` and `class_<change>_lawlike`, `instance_<change>`, and `event_<change>_possible` and `event_<change>_impossible`.

### 53. Seen and unseen items

The test-set settings never change the documents, so no test item is held out of the corpus.

**Decision.** The documents stay unchanged. Every test item gets a field that says whether its logical form appears in any training document. A false item never does. `stats.yaml` reports the share of true items that are seen, for each test set.

## What changed in the earlier code

- **Configuration.** `quantifiers.weights` is new, and the default `entity.scenes` is `[2, 5]`. `data/corpus/default.yaml` shows both. `data/corpus/tiny.yaml` sets `test_sets.size: 20`.
- **Propositions.** `scene_of` and `event_of` read a scene label or an event label. The truth tests are unchanged: a proposition that names no event was already judged against its scene.
- **Renderings.** An event-level form with `"event": null` is written with the label of its scene. Stage 5 raised an error for such a form. The rendering parses back to a proposition with its scene and without an event.
- **The planner.** A document keeps its drawn length, and a class-level sentence keeps the strongest true quantifier of its fact. Both are kept in memory for the statistics, and neither is written to `documents.jsonl`. `draw_class_fact` draws one class-level fact for the test sets.
- **The falsifier.** `candidates` and `falsify` now make false events, limit a subject swap to given instances, and take one more condition on the false item.
- **Scenes.** `leaf_similarity` gives the taxonomic similarity of the leaves without the clipping that the scene generator applies.
- **One test.** The test that nearly every definite mention can be told apart allowed 5% of mentions to fail. It now allows 10%. With 2 to 5 scenes, the cast of an entity narrative is larger, and the smallest test world (`still`, 4 instances for each leaf) has few features to tell instances apart. In the default corpus, 3 of 34,216 definite mentions cannot be told apart.

## Choices made by Claude

These are engineering choices made while building stage 6. Each one is the working design unless Jon changes it.

1. **The name and the keys of the weights.** The setting is `quantifiers.weights: {all: 1.0, most: 1.0, some: 1.0, none: 1.0}`. The key for the quantifier `no` is `none`, because YAML reads a bare `no` as the boolean false, as it read `on` before decision 26. A configuration that writes `no` gets an error that names `none`. A key that is left out has the weight 1. At least one weight must be positive.
2. **How the weights act.** A fact is still stated with its strongest true quantifier. A document chooses a fact with a probability proportional to the weight of that quantifier: a candidate is drawn uniformly, and kept at its weight over the largest weight. Five details follow.
   - Equal weights of any size take the unweighted path, so they change no document.
   - The polarity of a fact is drawn first, at the negation rate. So the weight of `none` moves the mix among the negative facts: `no` against "most ... not" and "some ... not".
   - A scalar pole takes no quantifier word, and has the weight 1. Rule statements are not reweighted, because `propositions.rule_statement_rate` sets how often they appear.
   - In a feature document, a chosen fact can give way to the same fact about a restricted subject. When the two quantifiers differ, the restricted fact takes its place with a probability of the ratio of the two weights.
   - A weight of 0 leaves the quantifier's facts out. A sibling contrast with such a quantifier is not stated.
3. **The item schema.** A test item is `{"input": {...}, "meta": {...}}`. `input` holds the model-facing fields: `document` (null at the class level), and the fields of a sentence in `documents.jsonl` without its label. The logical form has no `id`, no `grounding`, and no `rule`. `meta` holds `set`, `pair`, `truth`, `level`, `change`, `seen`, and `grounding`. A class-level item also has `law_like` and `rule`, and an event-level item also has `possible`. Each true item is followed in the file by the false item made from it.
4. **The format check.** `testsets.format_differences` compares the `input` of two items. It reports a field that one item lacks, a field that is null in one item only, a value of another type, and a label in another notation. The notation of a label is the label without its indices, so `C1.3` and `C1.3.2` are written alike. A scene (`SN.8`) and an event (`SN.8.5`) are not written alike. Lists can differ in length, and the words of a rendering can differ. A category term leaves `clauses` out when it has none, and a role swap moves a clause with its category. So the two logical forms must hold as many `clauses`, wherever they stand. `testsets.input_problems` checks one item: its fields, no part of the answer in the logical form, and no event label anywhere. The generator runs both checks on every pair, and stops with an error when one fails. The tests run them again, beside a second check written in the test file.
5. **The scene form in a sentence plan.** The plan of an event-level test item holds the label of the scene where a document's plan holds the label of the event. The `events` field of the item holds the scene's label too. The realizer, the readings, and the reading of a tree need no change.
6. **True class-level items.** A true class-level item is drawn the way an encyclopedic document draws a sentence. The topic is a category, drawn by `documents.topic_level_weights`. The kind of content is a membership fact, a fact about the topic, a fact about a subcategory, a relation fact, or a rule statement, each with the same chance. The polarity, the restriction, and the bare generic are drawn at their rates. The sets of the role swap draw relation facts only, because only a verb has roles to exchange.
7. **True instance-level items.** A narrative document is drawn, then one of its referents, each with the same chance. The fact is drawn as a description is: the polarity at the instance-level negation rate, and the patient of a verb among the document's other referents.
8. **True event-level items.** A document is drawn among the narratives that report an event, then one of the events it reports, each with the same chance. The verb is named at a level drawn by `mention.verb_level_weights`. So an item can name an event at another level of the verb tree than its document does. Such an item is true, and unseen.
9. **The mentions of a test item.** Every noun phrase is a definite mention with a noun. The noun's level is drawn by `mention.level_weights`, and the modifiers are those that tell the referent apart among the participants of the document's scenes. A test item has no pronoun, no relative clause, and no other modifier. Each instance of a pair draws from its own part of the stream, so the two items mention an instance that they share in the same way. An item whose referent has no noun is left out, because "it" would pick out no one.
10. **The size of a set.** A true item is used once in the sets of one level and change. A set can therefore be smaller than `test_sets.size`. For one level and change, the generator draws at most 60 true items for each pair of one set, plus 300. The generator stops earlier after 6 draws for each pair, plus 300, without a new pair. A set that stays empty still gets its file.
11. **Law-like items.** A false item is law-like when it says `all` or `no`, by its quantifier or as a bare generic that means all, its predicate is an IS, HAS, or CAN feature, and every member of its subject set has the value it claims. Only those features are judged by the fixed test. The role swap has no law-like set, because a verb is never judged by the fixed test. Under `quantifiers.all_grounding: observed`, the law-like sets are empty.
12. **What a document states.** For `seen`, a document states the proposition of every main clause. In a sentence about instances, a document also states the proposition of every relative clause, and what every noun phrase says of its referent: its noun (membership), and its modifiers. The restriction of a class-level subject asserts nothing. An event-level proposition is compared without its event label. Any document counts, not only the item's own.
13. **Streams.** The items of each level and change are chosen from their own part of `corpus:tests`. The mentions of an item come from `corpus:mentions`, and the grammar's choices from `corpus:grammar`, in parts named by the item's set and pair. So a grammar setting never changes what a test item says, and a test checks it.
14. **The co-occurrence check.** The pairs are the unordered pairs of leaves. A pair's co-occurrence is the number of documents in which both leaves occur. By referents, an instance counts for its leaf whatever its noun, and a category counts for itself when it is a leaf. The taxonomic similarity is the one in `thematic.csv`, computed for every pair, and a pair with an undefined similarity is left out. The correlations are reported for each document type, for the two encyclopedic types together, for the two narrative types together, and for all documents. The check compares situational documents with the two encyclopedic types together.
15. **`stats.yaml`.** The blocks are `documents`, `sentences`, `quantifiers`, `tokens`, `ambiguity`, `mentions`, `scenes`, `propositions`, `lexicon`, `rule_statements`, `cooccurrence`, `test_sets`, and `lexeme_frequencies`. The quantifier mix is given twice: as stated, where a bare generic counts as `generic`, and by the strongest true quantifier of each fact, which the weights act on. The statistics describe the documents, never the test items, apart from the block `test_sets`.
16. **The output folder.** `corpus.txt` holds the formal rendering until word forms are attached. `config.yaml` holds the resolved configuration and its provenance: the stream seeds, the taxonomy's identity, the git commit, and the package versions. Writing over a folder replaces the test sets of the earlier run. `generate` does not write `wordform_request.yaml` yet: the request belongs to stage 7, with the word-form pipeline's side of it. The command line has `generate` only, and `render` comes with stage 7.
17. **The tiny configuration.** `data/corpus/tiny.yaml` sets `test_sets.size: 20`. With the default of 500, the tiny corpus of 20 documents came with 7 megabytes of test sets.

## Results on the default configuration

These numbers come from `python -m semantic_world.corpus generate data/corpus/default.yaml` (seed 1), which takes 42 seconds and writes 127 megabytes.

| Measure | Value |
| --- | --- |
| Documents, sentences, tokens | 10,000, 92,464, and 456,307 |
| Scenes, and events in them | 10,062 scenes, 83,419 events |
| Distinct propositions | 63,801 |
| Sentences by level | class 50,230, instance 6,702, event 35,532 |
| Share of negative sentences: class level, instance level | 0.105, 0.103 |
| Ambiguous sentences, with the default grammar | 0 |
| Definite mentions that could not be told apart | 3 of 34,216 |
| Rule terms stated | 86 of 103 (13 read a scalar threshold, 4 have no instance) |
| Test sets, and pairs in them | 16 sets, 7,368 pairs |

**Document lengths, in sentences.**

| Document type | Documents | Drawn length (mean) | Achieved length (mean) | Shorter than drawn |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category | 3,003 | 9.98 | 9.98 | 0% |
| Encyclopedic, about a feature | 2,020 | 10.04 | 10.03 | 0.2% |
| Entity narrative | 2,021 | 12.78 | 8.82 | 57% |
| Situational narrative | 2,956 | 12.52 | 8.25 | 63% |

The mean entity narrative has 8.82 sentences. It had 5.97 with `entity.scenes: [1, 3]`.

**The quantifier mix of the 50,230 class-level sentences.**

| Quantifier | As stated | By the strongest true quantifier |
| --- | --- | --- |
| `all` | 5,782 (11.5%) | 11,509 (22.9%) |
| `most` | 4,861 (9.7%) | 9,666 (19.2%) |
| `some` | 26,835 (53.4%) | 26,835 (53.4%) |
| `no` | 1,039 (2.1%) | 2,077 (4.1%) |
| generic | 11,713 (23.3%) | |
| scalar pole | | 143 (0.3%) |

**The co-occurrence check, over the 741 pairs of the 39 leaves.** Each cell gives Pearson's correlation, then Spearman's.

| Documents | Thematic, by words | Taxonomic, by words | Thematic, by referents | Taxonomic, by referents |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category (3,003) | 0.295, 0.448 | 0.477, 0.231 | 0.293, 0.450 | 0.482, 0.224 |
| Encyclopedic, about a feature (2,020) | 0.333, 0.361 | 0.135, 0.093 | 0.333, 0.361 | 0.135, 0.093 |
| Entity narrative (2,021) | 0.796, 0.828 | 0.184, 0.164 | 0.794, 0.833 | 0.196, 0.160 |
| Situational narrative (2,956) | 0.728, 0.771 | 0.149, 0.127 | 0.724, 0.775 | 0.152, 0.126 |
| Both encyclopedic types (5,023) | 0.378, 0.464 | 0.387, 0.210 | 0.377, 0.462 | 0.389, 0.208 |
| Both narrative types (4,977) | 0.792, 0.832 | 0.173, 0.151 | 0.784, 0.833 | 0.181, 0.147 |
| All documents (10,000) | 0.685, 0.735 | 0.327, 0.223 | 0.726, 0.775 | 0.299, 0.211 |

The acceptance test holds, by both measures and both correlations. Situational documents correlate more with thematic relatedness than the encyclopedic documents do (0.728 against 0.378), and less with taxonomic similarity (0.149 against 0.387).

One thing stands out. The taxonomic signal of the encyclopedic documents comes from the category documents alone. A feature document lists the categories that have one feature, and its co-occurrence correlates with taxonomic similarity no more than a narrative's does (0.135 against 0.149 for situational documents).

**The test sets.**

| Test set | Pairs | True items that are seen |
| --- | --- | --- |
| `class_predicate` | 500 | 72.6% |
| `class_predicate_lawlike` | 196 | 52.0% |
| `class_subject` | 500 | 66.4% |
| `class_subject_lawlike` | 172 | 55.8% |
| `class_quantifier` | 500 | 75.2% |
| `class_quantifier_lawlike` | 500 | 47.0% |
| `class_role` | 500 | 28.4% |
| `instance_predicate` | 500 | 55.6% |
| `instance_subject` | 500 | 62.6% |
| `instance_role` | 500 | 1.8% |
| `event_predicate_possible` | 500 | 90.0% |
| `event_predicate_impossible` | 500 | 87.2% |
| `event_subject_possible` | 500 | 82.0% |
| `event_subject_impossible` | 500 | 88.6% |
| `event_role_possible` | 500 | 50.4% |
| `event_role_impossible` | 500 | 79.4% |

Three things stand out.

- **Two law-like sets are not full.** A predicate swap or a subject swap keeps the quantifier, so a law-like false item needs a true `all` or `no` fact about a feature. The default world has about 200 such facts that give one.
- **Capacities with a verb are rarely stated.** Only 1.8% of the true items of `instance_role` are seen, because a description seldom says that one referent can act on another.
- **An event item is unseen when it names its verb at another level.** A document reports a chase, and the item says "hunt". The event sets of the role swap hold transitive events only, whose verbs have categories above them, so their true items are seen less often.

## Still open

**Update, October 1, 2026.** Stage 7 added `wordform_request.yaml` to `generate`.

Nothing new. Two notes for stage 7:

- `generate` does not write `wordform_request.yaml` yet (choice 16).
- The stage 5 proposal lists the event notation of test items as still open. Decision 46 answers that question.
