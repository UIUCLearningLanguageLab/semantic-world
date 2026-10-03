# Decision log

This log lists every design decision made for Semantic World's taxonomy generator, word-form pipeline, connected speech plan, and corpus generator, with who proposed it, who decided it, and where it is explained. The specs and proposals hold the detail; this log is the index.

Started October 1, 2026, from the specs and proposals on `main` at `c6ed01d`, and from the planning conversation with Claude. New decisions are appended to the end of their section.

## How to read the log

**Proposed by** and **Decided by** name one of three parties:

- **Jon**;
- **Claude (chat)**: Claude in the planning conversation, which writes the specs and reviews each stage;
- **Claude Code**: the session that builds the code.

**Status** is one of:

- **Decided**: Jon chose it.
- **Working design**: built as written, but Jon has not ruled on it individually. These are the entries to review.
- **Proposed**: written into a spec that is not built yet, and not yet ruled on.
- **Superseded**: replaced by a later entry, which the row names.

Rows marked ⚑ have an attribution or status that the record does not settle. They are listed again under "Needs Jon's check" at the end.

## Taxonomy generator

Spec: `docs/specs/TAXONOMY_GENERATOR.md`. Foundational choices come from Jon's original request and his answers to the first questions, September 29, 2026.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| TX.1 | 2026-09-29 | Features are of four types: ISA, IS, HAS, and CAN. Uninformative features are wanted. | Jon | Jon | Decided | Planning conversation |
| TX.2 | 2026-09-29 | Rules vary in complexity, in the style of the Shepard, Hovland, and Jenkins types I–VI, with read-once formulas. | Jon | Jon | Decided | Planning conversation |
| TX.3 | 2026-09-29 | A feature's base rate is inversely related to the number of features. | Jon | Jon | Decided | Planning conversation |
| TX.4 | 2026-09-29 | Inheritance roles (defining, characteristic, random) are set per node, and a defining feature stays defining all the way down. | Jon | Jon | Decided | Planning conversation |
| TX.5 | 2026-09-29 | A rule file can give rules explicitly. | Jon | Jon | Decided | Planning conversation |
| TX.6 | 2026-09-29 | Rules never read ISA features. | Jon | Jon | Decided | Planning conversation |
| TX.7 | 2026-09-29 | CAN features never feed rules. | Jon | Jon | Decided | Planning conversation |
| TX.8 | 2026-09-29 | Each category has three vectors: defining, generative, and mean. | Jon | Jon | Decided | Planning conversation |
| TX.9 | 2026-09-29 | Instances are sampled from their leaf, their number is a separate parameter, and they follow the leaf's values rather than forming new subcategories. | Jon | Jon | Decided | Planning conversation |
| TX.10 | 2026-09-29 | Shape parameters can change with depth (depth schedules). | Jon | Jon | Decided | Planning conversation |
| TX.11 | 2026-09-29 | Outputs are CSV and YAML. | Jon | Jon | Decided | Planning conversation |
| TX.12 | 2026-09-29 | Labels are formal (`C1.3.2`), with periods between indices everywhere, features included (`IS.4`). | Jon | Jon | Decided | Planning conversation |
| TX.13 | 2026-09-29 | Rule files may include explicit rules for named output features, in addition to weighted templates. | Claude (chat) | — | Working design | Spec, decision 1 |
| TX.14 | 2026-09-29 | Exact duplicate rules are disallowed by default. | Claude (chat) | — | Working design | Spec, decision 2 |
| TX.15 | 2026-09-29 | Instance labels use periods throughout: `I1.3.2.5`. | Claude (chat) | — | Working design | Spec, decision 3 |
| TX.16 | 2026-09-29 | Free features are numbered before determined features within each type. | Claude (chat) | — | Working design | Spec, decision 4 |
| TX.17 | 2026-09-29 | A Beta-distributed base-rate heterogeneity option exists, off by default. | Claude (chat) | — | Working design | Spec, decision 5 |
| TX.18 | 2026-09-29 | The complexity score is the minimal DNF literal count, reported alongside the SHJ type. | Claude (chat) | — | Working design | Spec, decision 6 |
| TX.19 | 2026-09-29 | A bare list in `branching` is always a range; the per-level form is `{schedule: list, values: [...]}`. | Claude Code | Jon | Decided | `proposals/2026-09-29-taxonomy-branching-per-level-form.md` |
| TX.20 | 2026-09-29 | The variance bound checks a rule's expected true proportion at sampling time, from the base rates, so rules never depend on the tree or the instances. | Claude Code | Jon | Decided | `proposals/2026-09-29-taxonomy-variance-bound.md` |

## Relations

Spec: `docs/specs/TAXONOMY_RELATIONS.md`.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| REL.1 | 2026-09-29 | No relation is defined by category membership (ISA). | Jon | Jon | Decided | Planning conversation |
| REL.2 | 2026-09-29 | Interaction types are limited to the current set; others are documented as future additions. | Jon | Jon | Decided | Planning conversation |
| REL.3 | 2026-09-29 | Verbs have shared base relations and a verb taxonomy. Event schemas are documented as a future addition. | Jon | Jon | Decided | Planning conversation |
| REL.4 | 2026-09-29 | Scalar dimensions are included, and a setting of 0 turns them off. | Jon | Jon | Decided | Planning conversation |
| REL.5 | 2026-09-29 | A relation can hold between members of the same category, but never between an instance and itself. | Jon | Jon | Decided | Planning conversation |
| REL.6 | 2026-09-29 | Verbs with three or more arguments are out of scope. | Jon | Jon | Decided | Planning conversation |
| REL.7 | 2026-09-29 | Scalars and verbs are off by default, so existing configurations are unchanged. | Claude (chat) | — | Working design | Spec, decision 1 |
| REL.8 | 2026-09-29 | Scalars drift by a normal random walk from a standard normal at the superordinates, and have no inheritance roles. | Claude (chat) | — | Working design | Spec, decision 2 |
| REL.9 | 2026-09-29 | Thresholds and margins are quantiles of the model distribution, not of the realized data. | Claude (chat) | — | Working design | Spec, decision 3 |
| REL.10 | 2026-09-29 | A verb's relation is the conjunction of its constraints. Shared base relations are the defining features of the verb tree. | Claude (chat) | — | Working design | Spec, decision 4 |
| REL.11 | 2026-09-29 | The key-lock rule is a named constraint family with its own weight. | Claude (chat) | — | Working design | Spec, decision 5 |
| REL.12 | 2026-09-29 | Default exposure: all agent projections, and a quarter of patient projections. | Claude (chat) | — | Working design | Spec, decision 6 |
| REL.13 | 2026-09-29 | `relation_proportions.csv` covers categories at the same level only. (The corpus computes any pair itself; see CG.21.) | Claude (chat) | — | Working design | Spec, decision 7 |
| REL.14 | 2026-09-29 | Scalar settings never change the tree, the roles, or any free binary feature. Determined features can change, because rules may read threshold literals. | Claude Code | Jon | Decided | `proposals/2026-09-29-taxonomy-scalars-and-binary-features.md` |
| REL.15 | 2026-09-29 | Every verb must hold for 1–30% of leaf pairs, by resampling (`verbs.density`), and every constraint for at least 10% (`constraint_min_density`). | Claude (chat) | Jon | Decided | `proposals/2026-09-29-taxonomy-verb-density.md` |

## Word-form pipeline

Spec: `docs/specs/WORDFORM_PIPELINE.md`.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| WF.1 | 2026-09-29 | Words get distributed sound embeddings computed from audio, combining machine learning with auditory neuroscience. | Jon | Jon | Decided | Planning conversation |
| WF.2 | 2026-09-29 | Word forms are English-like pseudowords. Grammar is a separate step. | Jon | Jon | Decided | Planning conversation |
| WF.3 | 2026-09-29 | CMUdict is the English source; onsets and rimes are sampled by syllable position and stress. | Claude (chat) | — | Working design | Spec, decision 1 |
| WF.4 | 2026-09-30 | Sound patterns are learned from common words only (Zipf ≥ 3, `english_min_zipf`). | Jon | Jon | Decided | `proposals/2026-09-30-wordforms-common-words-and-spelling.md` |
| WF.5 | 2026-09-30 | Spellings follow common English spellings by position. Real words are drawn from the common words, and a pseudoword is never spelled like a common word. | Jon | Jon | Decided | `proposals/2026-09-30-wordforms-common-words-and-spelling.md` |
| WF.6 | 2026-09-29 | Piper's multi-speaker LibriTTS-R voice is the main voice set, and espeak-ng is the second engine. | Claude (chat) | — | Working design | Spec, decision 2 |
| WF.7 | 2026-09-29 | GPL tools are optional extras, imported only where used, and never vendored. | Claude (chat) | — | Working design | Spec, decision 3 |
| WF.8 | 2026-09-29 | The audio cache is the reproducible artifact, with a SHA-256 hash for every clip. | Claude (chat) | — | Working design | Spec, "Synthesis" |
| WF.9 | 2026-09-30 | HuBERT base, layer 8, is the default pretrained embedding; layer 3 is for graded similarity. | Claude Code | Jon | Decided | Spec, decision 4 |
| WF.10 | 2026-09-29 | A word's embedding is the mean over its training-speaker tokens. | Claude (chat) | — | Working design | Spec, decision 5 |
| WF.11 | 2026-09-29 | Until stage 8, exact acoustic control comes from Praat manipulation of synthesized audio. | Claude (chat) | — | Working design | Spec, decision 6 |
| WF.12 | 2026-09-29 | Real English words are available as a source, for comparison with language models. | Claude (chat) | — | Working design | Spec, decision 7 |
| WF.13 | 2026-09-30 | Function words have one syllable and differ from each other by at least two phonemes. The default configuration inflects nothing. | Claude (chat) | — | Working design | Spec, decision 8 |
| WF.14 | 2026-09-30 | Function words are rejected against the common words only; shape weights CV 0.3, CVC 0.4, VC 0.3; glosses are listed most frequent first, and the most frequent half get two phonemes. | Claude Code | Jon | Decided | `proposals/2026-09-30-wordforms-closed-class-shapes-and-joins.md`, decision 1 |
| WF.15 | 2026-09-30 | Sound patterns are learned from uninflected words (`exclude_inflections`). | Claude Code | Jon | Decided | Same, decision 2 |
| WF.16 | 2026-09-30 | An affix that more than 10% of the words cannot take is rejected (`max_skipped`). | Claude Code | Jon | Decided | Same, decision 2 |
| WF.17 | 2026-09-30 | A vowel meeting a vowel at a join is repaired with a glide before the schwa is tried. | Jon | Jon | Decided | Same, decision 2 |
| WF.18 | 2026-09-30 | Default affix shape weights are C 0.5, VC 0.5, V 0. | Jon | Jon | Decided | Same, decision 2 |
| WF.19 | 2026-09-30 | Affixes must differ from each other even after the joining schwa is added (`same_after_schwa`). ⚑ | Claude Code | Jon | Decided | Same, decision 2 |
| WF.20 | 2026-09-30 | A stem and affix pair whose form is a common English word is skipped; rare words and names are only reported. | Claude Code | Jon | Decided | Same, decision 3 |
| WF.21 | 2026-09-30 | English options: real English function words with weak forms, and English suffixes with allomorphy (a Jabberwocky condition). They are not the default. | Jon | Jon | Decided | Same, decision 4 |
| WF.22 | 2026-09-30 | Talker normalization is an option, off by default, because handling speaker variability is part of what learners must do. | Jon | Jon | Decided | Spec, embeddings; planning conversation |
| WF.23 | 2026-09-30 | One content word in five is held out from every trained encoder. ⚑ | Claude Code | — | Working design | Spec, stage 6 |
| WF.24 | 2026-09-30 | CPC stays as a baseline only on isolated words, to be revisited with connected speech. ⚑ | Claude Code | — | Working design | Spec; commit `c7602a5` |
| WF.25 | 2026-10-01 | Under acoustic mapping, the mapped tokens are the word's tokens, and the originals are a labeled control set. | Claude (chat) | Jon | Decided | Spec, decision 10 |
| WF.26 | 2026-10-01 | A target correlation that is not reached is a reported result with a warning, and an error with `assignment.strict: true`. | Claude (chat) | Jon | Decided | Spec, decision 10 |
| WF.27 | 2026-10-01 | Augmentation's formant ratio and pitch shift are measured frame by frame. | Claude (chat) | Jon | Decided | Spec, decision 10 |
| WF.28 | 2026-10-01 | The request is the top-level setting `request`, which replaces `closed_class.request`, and it can list lexemes and meanings. | Claude (chat) | Jon | Decided | Corpus spec, "Word forms for the corpus"; spec, decision 11 |
| WF.29 | 2026-10-01 | The lexemes of a request are assigned to content words: the lexemes of categories by the assignment mode, all others at random, and homonyms share a form. A run with too few content words for its lexemes is an error. (CG.10.) | Claude (chat) | Jon | Decided | Corpus spec, decision 10; spec, decision 11 |
| WF.30 | 2026-10-01 | The assignment comes first, and the inflected forms are then made, synthesized, and embedded. A category's synonyms are assigned by the mode. The request's meanings and `assignment.meanings` cannot both be given. (CG.16.) | Claude Code | Jon | Decided | Corpus spec, decision 16; spec, decision 11 |
| WF.31 | 2026-10-01 | `inflect` entries can name lexemes, and marked forms can take affixes (`W.12.M.2.AF.1`). (CG.12.) | Claude (chat) | Jon | Decided | Corpus spec, decision 12; spec, decision 11 |
| WF.32 | 2026-10-01 | A lexeme gets only a word that can take the affixes of its part of speech, which the request's `takes` lists from the corpus's grammar settings. The inflected forms that a request asks for never change a content word's form, assignment, or embedding, and the trained encoders never train on inflected forms. (CG.56.) | Claude (chat) | Jon | Decided | Corpus spec, decision 56; spec, decision 11 |
| WF.33 | 2026-10-01 | Every pair of branch markers differs by at least two phonemes, as function words do. The minimum is a setting, `assignment.branch_markers.min_distance` (default 2), and a run whose markers cannot be drawn at that distance stops with an error. | Claude (chat) | Jon | Decided | Spec, decision 12 |

## Connected speech (planned, not built)

Spec: `docs/specs/CONNECTED_SPEECH.md`.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| CS.1 | 2026-09-30 | Whole-utterance synthesis with Piper is the main method; joined citation clips are a control condition. | Claude (chat) | — | Proposed | Spec, decision 1 |
| CS.2 | 2026-09-30 | One speaker per document. | Claude (chat) | — | Proposed | Spec, decision 2 |
| CS.3 | 2026-09-30 | Function words take weak forms at a rate of 0.7 inside utterances, and are unstressed. | Claude (chat) | — | Proposed | Spec, decision 3 |
| CS.4 | 2026-09-30 | Contextual embeddings are means over each word's aligned span of a whole-utterance encoder pass. | Claude (chat) | — | Proposed | Spec, decision 4 |
| CS.5 | 2026-09-30 | The layer lives inside the word-form package and reuses its cache and encoders. | Claude (chat) | — | Proposed | Spec, decision 5 |
| CS.6 | 2026-09-30 | Pause placement has three modes (engine, phrase, random); the default adds no pauses beyond the engine's own. | Claude (chat) | Jon | Decided | Spec, decision 6 |
| CS.7 | 2026-09-30 | Speakers are chosen by measured traits from a catalog, and populations can be transformed with Praat to change their variability. | Claude (chat), at Jon's request | Jon | Decided | Spec, decision 7 |
| CS.8 | 2026-09-30 | Register levers are numbers with neutral defaults. Presets, starting with child-directed speech, come from published measurements chosen by Jon. | Claude (chat), at Jon's request | Jon | Decided | Spec, decision 8 |

## Corpus generator

Spec: `docs/specs/CORPUS_GENERATOR.md`. The CG numbers match the spec's numbered decisions.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| CG.1 | 2026-09-29 | `all` and `no` are grounded in fixed features (law-like), with an option for the observed reading. | Claude (chat) | — | Working design | Spec, decision 1 |
| CG.2 | 2026-09-29 | Rule statements use the generic noun "thing" plus the terms of the rule's minimal DNF. | Claude (chat) | — | Working design | Spec, decision 2 |
| CG.3 | 2026-09-29 | Class-level scalar poles compare a category with its parent; instance-level poles compare an instance with the category its noun names. (Extended by CG.22.) | Claude (chat) | — | Working design | Spec, decision 3 |
| CG.4 | 2026-09-29 | Events never change state, and event-level propositions are never negated. | Claude (chat) | — | Working design | Spec, decision 4 |
| CG.5 | 2026-09-29 | Distinguishing modifiers follow Dale and Reiter's incremental algorithm, with one preference order per language. | Claude (chat) | — | Working design | Spec, decision 5 |
| CG.6 | 2026-09-29 | A pronoun is used only when its referent was the only referent, or the subject, of the previous sentence. | Claude (chat) | — | Working design | Spec, decision 6 |
| CG.7 | 2026-09-29 | Function words, affixes, and inflected forms come from the word-form pipeline, through the corpus's request file. | Claude (chat) | — | Working design | Spec, decision 7 |
| CG.8 | 2026-09-29 | Sibling contrasts are two adjacent sentences, not a contrastive construction. | Claude (chat) | — | Working design | Spec, decision 8 |
| CG.9 | 2026-10-01 | The corpus is generated first, and word forms are made for it from a request. The corpus can be produced entirely in the propositional or conceptual rendering. | Claude (chat) | Jon | Decided | Spec, decision 9 |
| CG.10 | 2026-10-01 | Category lexemes get words by the assignment mode; all other lexemes get words at random. Sound differences between parts of speech are a future addition. | Claude (chat) | Jon | Decided | Spec, decision 10 |
| CG.11 | 2026-10-01 | Subject–verb agreement is a parameter, on by default when number is on. Number, tense, and aspect can each be affixes or separate words. | Claude (chat) | Jon | Decided | Spec, decision 11 |
| CG.12 | 2026-10-01 | In branch-marker mode, marked forms can take affixes. | Claude (chat) | Jon | Decided | Spec, decision 12 |
| CG.13 | 2026-10-01 | A homonym is two lexemes that share one word form; every lexeme has exactly one concept. | Claude Code | Jon | Decided | Spec, decision 13 |
| CG.14 | 2026-10-01 | The taxonomy is given as a configuration file with a seed, or as an output folder, which is regenerated in memory and checked against its files. | Claude Code | Jon | Decided | Spec, decision 14 |
| CG.15 | 2026-10-01 | A verb or verb category whose relation holds for every pair, or for no pair, gets no lexeme. | Claude Code | Jon | Decided | Spec, decision 15 |
| CG.16 | 2026-10-01 | The word-form pipeline assigns first, then makes the inflected forms; inflecting lexemes get only words that can take their affixes; synonyms are assigned by the mode. | Claude Code | Jon | Decided | Spec, decision 16 |
| CG.17 | 2026-10-01 | Word-order and morphology settings never change a logical form. Relative clauses and modifier limits belong to the planner. The sentence limit counts content words. | Claude (chat) | Jon | Decided | Spec, decision 17 |
| CG.18 | 2026-10-01 | Variables in the propositional rendering are `X.1`, `X.2`, and so on. | Claude (chat) | Jon | Decided | Spec, decision 18 |
| CG.19 | 2026-10-01 | A class-level proposition needs at least one instance in its subject set; vacuous truths are never generated. | Claude (chat) | Jon | Decided | Spec, decision 19 |
| CG.20 | 2026-10-01 | In a rule statement, a negated IS literal is a subject relative clause. A term that reads a scalar threshold is skipped and counted. (Its skip rule for two relative clauses is superseded by CG.27.) | Claude (chat) | Jon | Decided | Spec, decision 20 |
| CG.21 | 2026-10-01 | Class-level verb proportions come from the relation matrices, for any category pair and any verb or verb category. | Claude (chat) | Jon | Decided | Spec, decision 21 |
| CG.22 | 2026-10-01 | Class-level scalar poles take the generic only; a top-level category is compared with all instances. | Claude (chat) | Jon | Decided | Spec, decision 22 |
| CG.23 | 2026-10-01 | Verb-category words are used in capacity sentences and to name events, at a level drawn by weight. | Claude (chat) | Jon | Decided | Spec, decision 23 |
| CG.24 | 2026-10-01 | Patient projections appear at the class level, by proportion; `all` and `no` only under the observed reading. | Claude (chat) | Jon | Decided | Spec, decision 24 |
| CG.25 | 2026-10-01 | Rule statements are exempt from modifier limits, with an optional cap (`max_literals`, null by default). | Claude (chat) | Jon | Decided | Spec, decision 25 |
| CG.26 | 2026-10-01 | The morphology switch is `enabled`, not `on`. | Claude (chat) | Jon | Decided | Spec, decision 26 |
| CG.27 | 2026-10-01 | A relative clause can join several verb phrases with "and", so all of a rule term's negated literals fit in one clause. | Claude (chat) | Jon | Decided | Spec, decision 27 |
| CG.28 | 2026-10-01 | False `all` or `no` test items that no instance contradicts are marked, and go into a test set of their own. | Claude (chat) | Jon | Decided | Spec, decision 28 |
| CG.29 | 2026-10-01 | A false event item is an event that did not happen, marked possible or impossible, with separate test sets for the two kinds. | Claude (chat) | Jon | Decided | Spec, decision 29 |
| CG.30 | 2026-10-01 | Every instance-level and event-level test item names a document, and is tested as a continuation of it. | Claude (chat) | Jon | Decided | Spec, decision 30 |
| CG.31 | 2026-10-01 | A rule statement's quantifier is `all` or the generic, drawn at the generic rate, so the logical form matches the surface. | Claude (chat) | Jon | Decided | Spec, decision 31 |
| CG.32 | 2026-10-01 | Events are drawn verb first, then participants. | Claude (chat) | Jon | Decided | Spec, decision 32 |
| CG.33 | 2026-10-01 | The propositional rendering is the logical form and is never ambiguous (`ABLE`, `EVENT` with tense and aspect, `EXISTS`, `R.n` and `X.n`); event tense and aspect move to `propositions.events`. The surface may be ambiguous (`grammar.can_rate`), and each sentence records its possible readings. The conceptual rendering has the same structure as the spelled one. | Jon | Jon | Decided | Spec, decision 33 |
| CG.34 | 2026-10-01 | Class-level relative clauses are restrictive, by proportion; "owls that eat mice" means owls that can eat at least one mouse. All three clause kinds are drawn. | Claude (chat) | Jon | Decided | Spec, decision 34 |
| CG.35 | 2026-10-01 | One affix per word; stacked affixes are a future addition. | Claude Code | Jon | Decided | Spec, decision 35 |
| CG.36 | 2026-10-01 | Noun levels apply to instances only; a class-level subject is always named by its own noun. | Claude (chat) | Jon | Decided | Spec, decision 36 |
| CG.37 | 2026-10-01 | Modifiers at `mention.modifier_rate` go on instance mentions only. | Claude (chat) | Jon | Decided | Spec, decision 37 |
| CG.38 | 2026-10-01 | Sibling contrasts have a rate, `documents.sibling_contrast_rate` (default 0.2). | Claude (chat) | Jon | Decided | Spec, decision 38 |
| CG.39 | 2026-10-01 | Aspect is drawn once for each event, so every report of one event agrees. The event carries its aspect, and `scenes.jsonl` gains the key `aspect`. | Claude Code | Jon | Decided | Spec, decision 39 |
| CG.40 | 2026-10-01 | `readings` is one list per sentence, computed from the tree and the lexeme tokens, never from `events` or the logical form. `generic`, `capacity`, and `event` stand for the three levels. A test shows that the tree carries no information about the reading. | Claude Code | Jon | Decided | Spec, decision 40 |
| CG.41 | 2026-10-01 | A class-level subject takes a restriction at `propositions.restriction_rate` (default 0.1). Class-level relative clauses use `mention.relative_clauses`, and are drawn in the proposition layer. | Claude Code | Jon | Decided | Spec, decision 41 |
| CG.42 | 2026-10-01 | A sibling contrast is added only for a sibling where the fact differs. With no differing sibling, no contrast sentence is added. | Claude Code | Jon | Decided | Spec, decision 42 |
| CG.43 | 2026-10-01 | When modifiers cannot tell a referent apart, the mention falls back to the leaf's noun. A mention that still fits another participant is kept, marked as not distinguished, and counted. | Claude Code | Jon | Decided | Spec, decision 43 |
| CG.44 | 2026-10-01 | A scalar pole writes its comparison class in the propositional rendering, at both levels: `SC.1.HIGH(R.1, C1.5)`, and the comparison class of CG.22 for a category, as in `SC.1.HIGH(X.1, C1)`. | Claude Code | Jon | Decided | Spec, decision 44 |
| CG.45 | 2026-10-01 | The JSON logical form holds relative clauses as a `clauses` list on the subject and the patient, and `tense` and `aspect` on event-level forms. The propositional rendering is generated from the JSON logical form, and a test checks that the two agree. | Claude Code | Jon | Decided | Spec, decision 45 |
| CG.46 | 2026-10-01 | Event test items name only the scene: `EVENT(SN.8, PAST, SIMPLE, V1.2(R.1, R.2))`, for true and false items alike. Documents keep their event labels. The true and the false items of a test set never differ in the format of their model-facing fields (the sentence, its renderings, and its logical form). Metadata (the truth label, the grounding, `seen`) differ by design, and an item keeps the two apart. A test checks every test set. | Claude (chat) | Jon | Decided | Spec, decision 46 |
| CG.47 | 2026-10-01 | The default `entity.scenes` is `[2, 5]`, and was `[1, 3]`, so entity narratives are longer. | Claude (chat) | Jon | Decided | Spec, decision 47 |
| CG.48 | 2026-10-01 | `quantifiers.weights` weights the choice of class-level facts by quantifier, so a study can rebalance the mix. The default keeps the strongest true quantifier with no reweighting. (The name and the shape are CG.E84 and CG.E85.) | Claude (chat) | Jon | Decided | Spec, decision 48 |
| CG.49 | 2026-10-01 | A true event-level test item can be any event that its document reports, in a main clause or in a relative clause. | Claude (chat) | Jon | Decided | Spec, decision 49 |
| CG.50 | 2026-10-01 | A false event-level test item is stricter than CG.29: no event with its verb, agent, and patient happened in any scene of its document, in either aspect, and under a verb-category name no event of any verb below the category matches. | Claude (chat) | Jon | Decided | Spec, decision 50 |
| CG.51 | 2026-10-01 | The co-occurrence check counts by words (a leaf occurs when its own noun appears), reports Pearson's and Spearman's correlations, and reports the same correlations counted by referents as a second measure. | Claude (chat) | Jon | Decided | Spec, decision 51 |
| CG.52 | 2026-10-01 | The test sets are `class_<change>` and `class_<change>_lawlike`, `instance_<change>`, and `event_<change>_possible` and `event_<change>_impossible`. | Claude Code | Jon | Decided | Spec, decision 52 |
| CG.53 | 2026-10-01 | No test item is held out of the documents. Every item says whether its logical form appears in any training document (`seen`); a false item never does. `stats.yaml` reports the share of true items that are seen, for each test set. | Claude (chat) | Jon | Decided | Spec, decision 53 |
| CG.54 | 2026-10-01 | Separating the two signals: `stats.yaml` reports the correlation between thematic relatedness and taxonomic similarity over leaf pairs in the world itself, and, for each document type, the partial correlations of co-occurrence with thematic relatedness controlling for taxonomic similarity, and the reverse. They are given by words and by referents, with Pearson's and Spearman's. (The layout is CG.E103.) | Claude (chat) | Jon | Decided | Spec, decision 54 |
| CG.55 | 2026-10-01 | A setting controls the share of relation facts in category-topic documents, and its default keeps the earlier behavior. (The name and the shape are CG.E101 and CG.E102.) | Claude (chat) | Jon | Decided | Spec, decision 55 |
| CG.56 | 2026-10-01 | Which lexemes must take which affixes comes from the grammar settings alone, by part of speech. The request's `inflect` list decides only which inflected forms are made, synthesized, and embedded. The set of inflected forms requested never changes a content word's form, assignment, or embedding, so the trained encoders train on content words and function words only. (Refines CG.16.) | Claude (chat) | Jon | Decided | Spec, decision 56 |
| CG.57 | 2026-10-01 | `generate` writes `wordform_meanings.csv` beside the request, and the request names the file by a path relative to the request file. | Claude Code | Jon | Decided | Spec, decision 57 |
| CG.58 | 2026-10-01 | `render` fills `provenance.wordforms` in `config.yaml`, and the render determinism property exempts that key. | Claude Code | Jon | Decided | Spec, decision 58 |

### Corpus generator: Claude Code's engineering choices

Claude Code makes these while building, and records them in each stage's proposal file. Each is the working design unless Jon changes it.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| CG.E1 | 2026-10-01 | The `taxonomy` setting is `{config: <file>, seed: N}` or `{run: <folder>}`. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 1 |
| CG.E2 | 2026-10-01 | An output folder is checked by regenerating it and comparing every file but `config.yaml` byte for byte. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 2 |
| CG.E3 | 2026-10-01 | The taxonomy's identity is the SHA-256 of its resolved configuration. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 3 |
| CG.E4 | 2026-10-01 | The sentence limit is `mention.max_content_words: 20`. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 4 |
| CG.E5 | 2026-10-01 | Named concepts are drawn at random in the configured proportion; a scalar's two poles are named together; `THING` and function words are always named. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 5 |
| CG.E6 | 2026-10-01 | Lexeme labels number content lexemes first, then synonyms' second lexemes, then function words. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 6 |
| CG.E7 | 2026-10-01 | Homonyms come in pairs, so the share of lexemes sharing a form matches `homonym_rate`. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 7 |
| CG.E8 | 2026-10-01 | Each content concept, `THING` included, gets a second lexeme at `synonym_rate`; function words never do. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 8 |
| CG.E9 | 2026-10-01 | `lexicon.csv` has a `same_form_as` column. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 9 |
| CG.E10 | 2026-10-01 | Negation's position is `after_auxiliary` or `before_auxiliary`. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 10 |
| CG.E11 | 2026-10-01 | A patient projection keeps its adjective even when its verb has no word. | Claude Code | Claude Code | Working design | Stage 1 proposal, choice 11 |
| CG.E12 | 2026-10-01 | `no` always has positive polarity in the logical form and counts as negative; `all` is never negated. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 1 |
| CG.E13 | 2026-10-01 | Documents state the strongest true quantifier: `all`, then `most`, then `some`, with the generic able to replace any of them. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 2 |
| CG.E14 | 2026-10-01 | `some` is true when the proportion is above 0; `exclude_all` limits what documents state, never truth. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 3 |
| CG.E15 | 2026-10-01 | Negative membership ("no penguins are fish") is true when the categories share no instance; the generic of membership always means all. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 4 |
| CG.E16 | 2026-10-01 | Instance-level membership can name the instance's own leaf ("the bird is a penguin"). | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 5 |
| CG.E17 | 2026-10-01 | Scalar poles use the population standard deviation; a class without variation has no poles; a pole in a restriction is relative to the subject's category. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 6 |
| CG.E18 | 2026-10-01 | An instance-level pole records its comparison class, and the planner must name that category. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 7 |
| CG.E19 | 2026-10-01 | The fixed test with a restriction enumerates the cone, and falls back to the taxonomy's local test above 2^20 settings. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 8 |
| CG.E20 | 2026-10-01 | With `generic.means: all`, a projection's generic needs every instance of the subject set. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 9 |
| CG.E21 | 2026-10-01 | A verb sentence needs at least one pair of distinct instances; the patient can take a restriction. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 10 |
| CG.E22 | 2026-10-01 | A rule statement takes `all`. | Claude Code | Claude Code | Superseded by CG.31 | Stage 2 proposal, choice 11 |
| CG.E23 | 2026-10-01 | False items: each swap keeps everything but the one thing it changes, and a false item is never vacuous. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 12 |
| CG.E24 | 2026-10-01 | The grounding record's `proportion` is always the share that has the predicate; `test` names how truth was decided. | Claude Code | Claude Code | Working design | Stage 2 proposal, choice 13 |
| CG.E25 | 2026-10-01 | Scenes ignore the lexicon; the planner leaves out events no word can report. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 1 |
| CG.E26 | 2026-10-01 | Each scene draws from its own part of the stream. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 2 |
| CG.E27 | 2026-10-01 | Participants are drawn without replacement; a zero-weight instance is never drawn, so a scene can be smaller than `scene.size`. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 3 |
| CG.E28 | 2026-10-01 | Taxonomic similarity is computed for every leaf pair, as in `thematic.csv`. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 4 |
| CG.E29 | 2026-10-01 | Events are drawn with probability proportional to their verb's weight. | Claude Code | Claude Code | Superseded by CG.32 | Stage 3 proposal, choice 5 |
| CG.E30 | 2026-10-01 | `scene.verb_weights` takes CAN features and leaf verbs only; a verb category is a configuration error. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 6 |
| CG.E31 | 2026-10-01 | An event keeps its own leaf verb; a verb category names it only when it is mentioned. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 7 |
| CG.E32 | 2026-10-01 | The same event can recur at a later step, but not twice in one step. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 8 |
| CG.E33 | 2026-10-01 | Events are numbered within their scene in time order. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 9 |
| CG.E34 | 2026-10-01 | Two reports of different events are different propositions, even with the same verb and participants. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 10 |
| CG.E35 | 2026-10-01 | A level with weight 0 never names an event's verb; if no level can, the event cannot be reported. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 11 |
| CG.E36 | 2026-10-01 | `scenes.jsonl` lists events by time step, empty steps included. | Claude Code | Claude Code | Working design | Stage 3 proposal, choice 12 |
| CG.E37 | 2026-10-01 | What a sentence says is a sentence plan; the grammar realizes it, and `interpret` gives it back. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 1 |
| CG.E38 | 2026-10-01 | `interpret` takes each noun phrase's referent and each verb phrase's event from the sentence record. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 2 |
| CG.E39 | 2026-10-01 | Tree labels `NP-SBJ` and `NP-OBJ`; the object sits inside the verb phrase only when it neighbors the verb. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 3 |
| CG.E40 | 2026-10-01 | A noun phrase is built outward from the noun: adjectives, determiner, with-phrases, relative clause. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 4 |
| CG.E41 | 2026-10-01 | `has` and `is` count as auxiliaries, and stay together with `not` and the predicate word. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 5 |
| CG.E42 | 2026-10-01 | A joined relative clause doesn't repeat a shared auxiliary ("that are not red and not big"). | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 6 |
| CG.E43 | 2026-10-01 | An object relative follows the clause order: "that the owl chased". | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 7 |
| CG.E44 | 2026-10-01 | Class-level nouns are plural, instances singular; part nouns are never marked; a plural predicate noun takes no "a". | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 8 |
| CG.E45 | 2026-10-01 | An inflected token is its lexeme label joined to the affix's gloss (`L.5-PLURAL`). | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 9 |
| CG.E46 | 2026-10-01 | The grammar's random draws happen in one order whatever the word order, so word order never changes word choice. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 10 |
| CG.E47 | 2026-10-01 | The fixed adjective order is one random ordering of all adjective concepts. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 11 |
| CG.E48 | 2026-10-01 | "The penguin is a penguin" is not a sentence; the subject is named by the category above ("the bird is a penguin"). | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 12 |
| CG.E49 | 2026-10-01 | An instance's relative clause states a capacity; an event's reports another event of the scene; neither repeats the sentence. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 13 |
| CG.E50 | 2026-10-01 | A restriction's negated IS literals form a depth-1 relative clause whatever the depth limit. | Claude Code | Claude Code | Working design | Stage 4 proposal, choice 14 |
| CG.E51 | 2026-10-01 | `grammar.class_can_rate` (0.5) decides whether a class-level capacity says "can". | Claude Code | Claude Code | Superseded by CG.33 | Stage 4 proposal, question A |
| CG.E52 | 2026-10-01 | A category document draws one kind of content for each sentence, each kind with the same chance: membership, a fact about the topic, a fact about a subcategory, or a relation fact. | Claude Code | Jon | Decided | Stage 5 proposal, choice 1 |
| CG.E53 | 2026-10-01 | A document can be shorter than its drawn length, when its events or its facts run out. A document with no sentence is drawn again. | Claude Code | Jon | Decided | Stage 5 proposal, choice 2 |
| CG.E54 | 2026-10-01 | The distractors of a definite mention are all participants of the document's scenes. | Claude Code | Jon | Decided | Stage 5 proposal, choice 3 |
| CG.E55 | 2026-10-01 | A pronoun stands only in a noun phrase of the main clause. | Claude Code | Jon | Decided | Stage 5 proposal, choice 4 |
| CG.E56 | 2026-10-01 | An event-level proposition must have the corpus's tense; a report with the wrong aspect is false; a proposition that names no event is true when an event of its aspect occurred. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 5 |
| CG.E57 | 2026-10-01 | The realizer makes one draw of `can` for every positive capacity, at the class rate or the instance rate, inside relative clauses too. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 6 |
| CG.E58 | 2026-10-01 | A sentence holds each event's label, tense, and aspect in memory; `documents.jsonl` writes the labels, and the logical form holds the tense and the aspect. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 7 |
| CG.E59 | 2026-10-01 | Readings follow the language: a bare verb is a capacity only when `can_rate.instance` is below 1, and an event only when the language has unmarked events. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 8 |
| CG.E60 | 2026-10-01 | A category term holds a list of relative clauses. A class-level clause holds a CAN feature or a verb, and is never negated. A term with an object relative has that one clause. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 9 |
| CG.E61 | 2026-10-01 | With a relative clause in the subject, `all` and `no` are not valid under the law-like reading, for every kind of predicate. A clause on the patient alone changes nothing. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 10 |
| CG.E62 | 2026-10-01 | A restriction and a relative clause are drawn among those that some members satisfy, and not all. A restriction is one literal, negative at the class-level negation rate. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 11 |
| CG.E63 | 2026-10-01 | A class-level clause is an object relative at `object_share`, and otherwise a subject relative with a CAN feature or a verb, each with the same chance. The other category takes its own clause at the rate, up to `max_depth`. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 12 |
| CG.E64 | 2026-10-01 | Restrictions are drawn for the topic or the subcategory of a fact, and for the subject of a fact in a feature document. Membership facts, rule statements, and contrasts take none. A sentence never states what its subject's restriction states. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 13 |
| CG.E65 | 2026-10-01 | The polarity of a class-level fact is drawn first and kept, so the share of negative sentences stays at the negation rate when the positive facts run out. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 14 |
| CG.E66 | 2026-10-01 | A document states a subject and a predicate once for each polarity, and a narrative reports an event once in a main clause. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 15 |
| CG.E67 | 2026-10-01 | A fact has a sibling contrast when it is strong (`all`, `no`, `most`, or a scalar pole) and about the plain topic, and the sibling's fact of the other polarity is strong too. Top-level categories are siblings. A contrast stays after its fact when shuffled. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 16 |
| CG.E68 | 2026-10-01 | The template's sections are membership, defining (`all`, `no`), characteristic (`most`, scalar poles), rarer (`some`), relation, and rule. `shuffle` mixes each unit's template position with a uniform draw. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 17 |
| CG.E69 | 2026-10-01 | A category topic draws its level by weight and then a category; a feature topic is drawn among the features and the verbs with a word; narrative seeds are drawn among all instances. A situational document's topic is its scene. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 18 |
| CG.E70 | 2026-10-01 | A feature document states a rule at `rule_statement_rate` while the topic has rules left, and otherwise says that a category has the feature or lacks it. A verb topic gives relation facts. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 19 |
| CG.E71 | 2026-10-01 | A description follows an event sentence at `instance_description_rate`: of the topic in an entity narrative, and of a participant of the event in a situational narrative, a newly introduced one first. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 20 |
| CG.E72 | 2026-10-01 | In a narrative, an event-level relative clause reports an earlier event of the scene, and an instance-level clause relates the referent to an instance already mentioned. Noun phrases inside a clause are never pronouns. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 21 |
| CG.E73 | 2026-10-01 | The level of the noun is drawn for each mention. The subject of a scalar pole is named by the comparison class, and the subject of "is a penguin" is never named "penguin". | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 22 |
| CG.E74 | 2026-10-01 | A situational narrative has distinguishing modifiers only. An entity narrative also adds one modifier at `modifier_rate`. No modifier states the feature that the sentence's predicate states. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 23 |
| CG.E75 | 2026-10-01 | Referents are told apart by adjectives for IS features and poles, and with-phrases or without-phrases for HAS features, never by a negated IS literal. The preference order is drawn once per language. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 24 |
| CG.E76 | 2026-10-01 | The content-word limit is checked on the sentence plan. A narrative sentence over the limit loses its drawn relative clauses and keeps its distinguishing modifiers. A class-level fact over the limit is not stated. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 25 |
| CG.E77 | 2026-10-01 | Each document draws from its own parts of four streams, named by its label. Scenes and proposition labels are numbered across the corpus. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 26 |
| CG.E78 | 2026-10-01 | Referents and variables are numbered in the order of the logical form (subject, its relative clauses, object), so word order never changes a label or a rendering. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 27 |
| CG.E79 | 2026-10-01 | In the propositional rendering of a sentence about instances, each noun phrase gives its noun, its modifiers, and its clause's propositions; the main proposition comes last; a repeated proposition is written once. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 28 |
| CG.E80 | 2026-10-01 | The rendering parses back to the formula of the JSON logical form: the whole proposition at the class level, and the main clause's proposition plus the others in order for instances. Clause attachment is in the JSON form and the tree. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 29 |
| CG.E81 | 2026-10-01 | A proposition gets one label, `PR.<n>`, numbered by first use across the corpus, and sentences that say the same proposition share it. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 30 |
| CG.E82 | 2026-10-01 | The sentence record adds `distinguished` and `readings`, writes `referents` as objects, and the document holds `referents` from label to instance. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 31 |
| CG.E83 | 2026-10-01 | A sentence's logical form writes the comparison class of a class-level pole (`"class": "C1"`); the class comes from the tree, so it is no part of the proposition that the truth tests judge. | Claude Code | Claude Code | Working design | Stage 5 proposal, choice 32 |
| CG.E84 | 2026-10-01 | The weights are `quantifiers.weights: {all, most, some, none}`. The key for the quantifier `no` is `none`, because YAML reads a bare `no` as false; a left-out key has the weight 1. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 1 |
| CG.E85 | 2026-10-01 | A fact is chosen with a probability proportional to the weight of its strongest true quantifier. Equal weights change no document. The polarity is drawn first; a scalar pole has the weight 1; rule statements are not reweighted; a weight of 0 leaves the quantifier's facts out, contrasts included. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 2 |
| CG.E86 | 2026-10-01 | A test item is `{"input": ..., "meta": ...}`: `input` holds the document and the fields of a sentence without its label, id, grounding, and rule; `meta` holds the set, the pair, the truth label, the level, the change, `seen`, the grounding, and the marks `law_like`, `rule`, and `possible`. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 3 |
| CG.E87 | 2026-10-01 | The format check compares fields, null values, types, and the notation of labels, where a scene and an event are not written alike. The two logical forms must hold as many `clauses`. The generator runs the check on every pair and stops when it fails. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 4 |
| CG.E88 | 2026-10-01 | The plan of an event-level test item holds the scene's label where a document's holds the event's, and the item's `events` field holds the scene's label. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 5 |
| CG.E89 | 2026-10-01 | A true class-level item is drawn as an encyclopedic document draws a sentence: a topic by level weights, then a membership fact, a fact about the topic, a fact about a subcategory, a relation fact, or a rule statement, each with the same chance. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 6 |
| CG.E90 | 2026-10-01 | A true instance-level item is a fact about a referent of a narrative document, drawn as a description is, with the patient of a verb among the document's other referents. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 7 |
| CG.E91 | 2026-10-01 | A true event-level item draws its verb's level anew, so an item can name an event at another level of the verb tree than its document does. Such an item is true and unseen. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 8 |
| CG.E92 | 2026-10-01 | Every noun phrase of a test item is a definite mention with a noun and with distinguishing modifiers only: no pronoun, no relative clause. The two items of a pair mention a shared instance in the same way. An item whose referent has no noun is left out. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 9 |
| CG.E93 | 2026-10-01 | A true item is used once in the sets of one level and change, so a set can be smaller than `test_sets.size`. The draws are limited, and an empty set still gets its file. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 10 |
| CG.E94 | 2026-10-01 | A false item is law-like when it says `all` or `no` (or a generic that means all) of an IS, HAS, or CAN feature and every member of its subject set has the claimed value. The role swap has no law-like set, and the observed reading has no law-like items. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 11 |
| CG.E95 | 2026-10-01 | For `seen`, a document states the proposition of every main clause and, in a sentence about instances, of every relative clause, and what every noun phrase says of its referent (its noun and its modifiers). Any document counts. An event is compared without its label. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 12 |
| CG.E96 | 2026-10-01 | Test items are chosen from parts of `corpus:tests`, and their mentions and grammar choices come from parts of `corpus:mentions` and `corpus:grammar` named by set and pair, so a grammar setting never changes what a test item says. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 13 |
| CG.E97 | 2026-10-01 | Co-occurrence is the number of documents in which both leaves of a pair occur. By referents, an instance counts for its leaf whatever its noun. The check compares situational documents with the two encyclopedic types together. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 14 |
| CG.E98 | 2026-10-01 | `stats.yaml` gives the quantifier mix twice: as stated, and by each fact's strongest true quantifier. A document's drawn length and a sentence's strongest quantifier are kept in memory and not written to `documents.jsonl`. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 15 |
| CG.E99 | 2026-10-01 | `corpus.txt` holds the formal rendering until word forms are attached. Writing over a folder replaces its test sets. `generate` does not write `wordform_request.yaml` until stage 7, and `render` comes with stage 7. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 16 |
| CG.E100 | 2026-10-01 | `data/corpus/tiny.yaml` sets `test_sets.size: 20`. | Claude Code | Claude Code | Working design | Stage 6 proposal, choice 17 |
| CG.E101 | 2026-10-01 | The setting is `documents.relation_fact_share`: null (the default), or a number from 0 to 1. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 1 |
| CG.E102 | 2026-10-01 | With a share, each sentence of a category document draws a relation fact with that probability, and the other kinds of content share the rest, each with the same chance. Null takes the earlier draw, so it changes no document. The setting acts on category documents only, not on feature documents and not on how the test sets draw true items. The share is the chance of a draw, and the share in the documents runs a few points higher at low settings. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 2 |
| CG.E103 | 2026-10-01 | In `stats.yaml`, the co-occurrence check gains `world.thematic_taxonomic` and, under each measure of each group of documents, `partial` with `thematic_given_taxonomic` and `taxonomic_given_thematic`. A partial correlation comes from the three pairwise correlations, over the pairs with a defined taxonomic similarity; Spearman's is the same formula on ranks; it is null when a variable does not vary. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 3 |
| CG.E104 | 2026-10-01 | The request file has the keys `lexemes`, `takes`, `function_words`, `affixes`, `inflect`, and `meanings`. `takes` maps a part of speech to affix glosses. `inflect` has one entry for each affix, `{lexemes: [...], affixes: [gloss]}`, with the lexemes in label order. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 4 |
| CG.E105 | 2026-10-01 | `takes` and `affixes` come from `grammar.morphology` alone: number as an affix gives nouns `PLURAL`, and verbs too when verbs agree; tense as an affix gives verbs `PAST`; aspect as an affix gives verbs `PROGRESSIVE`. Part nouns and adjectives take no affix. An inflection after its word is a suffix, and one before its word is a prefix. `affixes` lists every inflection realized as an affix, even one that no sentence uses. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 5 |
| CG.E106 | 2026-10-01 | `function_words` lists every function word of the language, with those that no document uses, most frequent first by their counts in the documents. Ties keep the lexicon's order. Test items are not counted. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 6 |
| CG.E107 | 2026-10-01 | `inflect` covers the documents and the test items, because test items are rendered too. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 7 |
| CG.E108 | 2026-10-01 | `wordform_meanings.csv` is the taxonomy's `categories_generative.csv`, byte for byte, scalar columns included. The word-form pipeline drops the columns that are not binary. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 8 |
| CG.E109 | 2026-10-01 | `render` works on folders. It checks that every lexeme has a form and that the word-form run was made from the corpus's request, renders everything before it writes anything, and can be repeated or run with another word-form run. The word-form run's identity is its name, its seed, and the SHA-256 of its resolved configuration, without its path. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 9 |
| CG.E110 | 2026-10-01 | In the word-form configuration, `request` is null, the path of a request file, or the lexeme part of a request inline. A resolved configuration holds the request inline, so a run's `config.yaml` gives the same run again. A file under `closed_class.request` is an error that names `request`; a null there is accepted, so earlier run folders still load. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 10 |
| CG.E111 | 2026-10-01 | The request is checked when the configuration is read: lexeme labels are distinct; `same_form_as` names a lexeme that has a form of its own; a `takes` gloss is an affix; an inflect entry gives a lexeme only an affix that `takes` lists for its part of speech; a request file needs the closed class; and the content words are at least as many as the lexemes that need forms of their own. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 11 |
| CG.E112 | 2026-10-01 | A word that homonyms share takes the affixes of both lexemes' parts of speech. A lexeme of a category that shares another lexeme's form takes that form, and is left out of the mode's assignment. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 12 |
| CG.E113 | 2026-10-01 | A word can take an affix when the joined form passes the phonotactic check with the usual repairs, is not a common English word, and is neither a form of the run nor the form of another word with an affix. The rule looks at the word forms and the affixes alone. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 13 |
| CG.E114 | 2026-10-01 | How each mode keeps the rule: an arbitrary assignment draws the words in a seeded random order, and the lexemes that few words suit choose first; the target-correlation search exchanges only words that suit both lexemes; a branch's marked forms take every affix that a lexeme of the branch requires; the lexemes outside the mode draw from the stream part `wordforms:assign:lexemes`, those with the most affixes first. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 14 |
| CG.E115 | 2026-10-01 | In a run with lexemes, the function words are made after the assignment and avoid its forms (marked forms, markers, and every form a word could have with an affix). Their forms depend on their order of frequency in the corpus, and the words of the lexemes must not. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 15 |
| CG.E116 | 2026-10-01 | The trained encoders never train on inflected forms, in every run, with or without a request, and the tokens of inflected forms are encoded in batches of their own. Marked forms still train, because they are the categories' words and do not depend on the inflect list. A run without inflected forms is unchanged. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 16 |
| CG.E117 | 2026-10-01 | The lexemes are assigned in every subcommand, `forms` included. With an embedding as the sound distance, the first pass synthesizes and embeds the content words, and the layers are made again after the assignment. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 17 |
| CG.E118 | 2026-10-01 | `assignment/lexicon.csv` has one row for each lexeme: `lexeme`, `meaning`, `pos`, `word`, `spelling`, `arpabet`, and `assigned` (the mode, `random`, or `same_form`), with `base_word`, `branch`, and `marker` under branch markers. `words.csv` gains `pos` only in a run with lexemes; a form that homonyms share lists both parts of speech; an inflected form takes its stem's part of speech and split. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 18 |
| CG.E119 | 2026-10-01 | `assignment.categories` chooses the lexemes of categories that the mode assigns, and the others get words at random. A request without meanings gives every lexeme a word at random. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 19 |
| CG.E120 | 2026-10-01 | `data/wordforms/corpus_tiny.yaml` makes 60 content words for the tiny corpus's 35 content lexemes, in the arbitrary mode, with the speakers and embeddings of `tiny.yaml`. | Claude Code | Claude Code | Working design | Stage 7 proposal, choice 20 |
| CG.E121 | 2026-10-01 | A drawn relative clause never reports an event with the same verb, agent, and patient as its sentence's own event. The same holds between two clauses of one sentence. (The same event can occur again at a later step, CG.E32.) | Claude (chat) | Jon | Decided | Spec, "Relative clauses" |
| CG.E122 | 2026-10-01 | `data/wordforms/corpus_default.yaml` makes the word forms of the default corpus from its request, with the default word-form settings: 500 content words for 173 content lexemes, 45 speakers, and all five embeddings, in the arbitrary mode. | Claude (chat) | Jon | Decided | Corpus spec, "Word forms for the corpus" |

## Milestone 1 engine

Spec: `docs/specs/MILESTONE_1.md`.

| ID | Date | Decision | Proposed by | Decided by | Status | Source |
| --- | --- | --- | --- | --- | --- | --- |
| M1.1 | 2026-09-29 | A need's `rise_when` gains the value `awake`, so fatigue rises while awake and falls while asleep. ⚑ | Claude Code | — | Working design | `proposals/2026-09-29-rise-when-awake.md` |
| M1.2 | 2026-10-02 | "Mind" replaces "nervous system" everywhere. The data-file key `nervous_system` becomes `mind` in entity types and in the `population` entries of experiments, the Rust type `NervousSystem` becomes `Mind`, and the Python base class becomes `semantic_world.agents.Mind`. The old key is not kept as an alias. | Jon | Jon | Decided | `CONTRACTS.md`, `ENTITY_DEFINITIONS.md`, `specs/MILESTONE_1.md` |

## Needs Jon's check

- **WF.19.** The proposal describes the `same_after_schwa` rule without saying who proposed it.
- **WF.23.** The record doesn't say whether Jon chose the 20% held-out-word share or Claude Code did.
- **WF.24.** Keeping CPC as a baseline came from Claude Code's results; the record doesn't show Jon ruling on it.
- **M1.1.** The proposal recommends `awake` and was built that way, but the record doesn't show Jon's decision.
- **All "Working design" rows.** These were built as written without an individual ruling. Most are small, but TX.13–18, REL.7–13, WF.3 and WF.6–13, and CG.1–8 shape the world and the language.
