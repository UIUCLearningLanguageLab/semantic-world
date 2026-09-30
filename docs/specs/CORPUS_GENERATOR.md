# Corpus generator: propositions, documents, and grammar

Draft, September 29, 2026. This document is the build specification for a Python program that turns the taxonomy generator's world into a corpus of documents written in an artificial language. The specification is written for handoff to Claude Code, or to any developer, and should be complete together with `docs/specs/TAXONOMY_GENERATOR.md`, `docs/specs/TAXONOMY_RELATIONS.md`, and `docs/specs/WORDFORM_PIPELINE.md`.

## Goal

The taxonomy generator defines what is true in the world: categories, features, rules, instances, scalar dimensions, and relations. The corpus generator writes about that world. The corpus generator produces documents made of sentences, and every sentence expresses a proposition whose truth is grounded in the world's data.

The corpus is built in four layers, and every layer's output is kept:

1. **Lexicon.** Concepts from the world get words.
2. **Propositions.** Typed logical forms, each checked against the world.
3. **Documents.** A discourse planner chooses and orders propositions by document type, and decides how each referent is mentioned.
4. **Sentences.** A grammar realizes each proposition as a sentence, with a parse tree.

Every sentence is stored with its words, its parse tree, its logical form, and the grounding of its truth. We can then ask what a model learned about meaning from form alone, and check the answer against ground truth.

The corpus is a controlled source of distributional statistics. The mix of document types decides which words co-occur. Encyclopedic documents put taxonomic neighbors together. Situational documents put thematic partners together. So the document mix is a direct lever on the taxonomic–thematic distinction in `TAXONOMY_RELATIONS.md`.

Out of scope for now:

- event schemas with changing states (see "Future additions" in `TAXONOMY_RELATIONS.md`);
- dialogue, questions, and commands;
- sentence coordination and subordinate clauses other than relative clauses;
- delivering language to agents in the running world.

## Inputs

- **A taxonomy result** (required): a `TaxonomyResult`, or a taxonomy output folder. Verbs and scalar dimensions are used when the result has them. Without verbs, there are no transitive sentences and no thematic scene sampling.
- **A word-form lexicon** (optional): a word-form pipeline run (`WORDFORM_PIPELINE.md`). Without one, the corpus is written in formal tokens only (see "Renderings").

## Labels

Labels follow the project convention: formal labels, indices starting at 1, and periods between indices.

| Object | Label | Example |
| --- | --- | --- |
| Lexeme (a word of the language) | `L.<n>` | `L.42` |
| Document | `D.<n>` | `D.17` |
| Sentence | `D.<n>.<k>` | `D.17.3` is sentence 3 of document 17 |
| Scene | `SN.<n>` | `SN.8` |
| Event | `SN.<n>.<k>` | `SN.8.5` is event 5 of scene 8 |
| Proposition | `PR.<n>` | `PR.310` |

## Layer 1: the lexicon

### Concepts that get words

| Concept | Part of speech | Example gloss |
| --- | --- | --- |
| Every category, at every level | noun | penguin, bird, animal |
| IS feature (free or determined) | adjective | red |
| HAS feature | part noun | arms |
| CAN feature | intransitive verb | swim |
| Verb (leaf of the verb tree) | transitive verb | chase |
| Internal verb category | transitive verb (more general) | hunt |
| Exposed patient projection | adjective | edible |
| Scalar dimension | two adjectives, one for each pole | big, small |
| The generic head noun | noun | thing |
| Function words (below) | function word | the |

For each concept type, a parameter gives the proportion of concepts that get a word (default 1.0). A concept without a word cannot be expressed, and the discourse planner skips any proposition that needs one.

### Function words

`a`, `the`, `all`, `most`, `some`, `no`, `not`, `can`, `is`, `has`, `with`, `without`, `and`, `that` (relative clauses), `it` (pronoun), and, when inflection is realized as separate words, one function word for each inflectional value (see "Morphology"). The set is fixed by the settings. Function words are ordinary lexemes with their own word forms.

### Synonyms and homonyms

The default is one lexeme per concept and one concept per lexeme. Two parameters, both 0 by default, add ambiguity for later experiments:

- `synonym_rate`: the probability that a content concept gets a second lexeme. When a concept has several lexemes, each mention picks one at random.
- `homonym_rate`: the probability that a content lexeme shares its word form with another concept's lexeme. The pair is chosen within the same part of speech with probability `homonym_same_pos`, and across parts of speech otherwise.

The lexicon records which concept or concepts each lexeme and word form expresses.

### Word forms

When a word-form lexicon is given, every lexeme gets a word form. Function words get the shortest forms: one syllable, drawn from the most frequent syllable shapes. Affixes (see "Morphology") need bound forms of one or two phonemes. The word-form pipeline does not yet produce function words, affixes, or inflected forms. "Changes to the word-form pipeline" lists what it needs.

## Layer 2: propositions

A proposition is a logical form with a type, a polarity, and a truth grounding. There are three levels.

### Class level

A class-level proposition says something about a category: "all penguins can swim", "red penguins swim", "penguins are birds".

- **Subject:** a category, optionally restricted by literals: "red penguins" is the category restricted to members with the IS feature for red. A restriction is a set of IS and HAS literals, positive or negative ("with fins", "without fins"), and scalar pole literals ("big").
- **Predicate:** one of IS feature, HAS feature, CAN feature, scalar pole ("are big"), membership in a category above the subject ("are birds"), or a verb with a patient category ("chase fish").
- **Quantifier:** `all`, `most`, `some`, `no`, or `generic` (a bare plural with no quantifier word: "penguins swim").

**Truth grounding.** For one-place predicates:

- `all` is true when the predicate is fixed at 1 for the category, given its defining features, any fixed-by-rule features, and the restriction. The test uses the cone enumeration of the fixed-by-rule test, with the restriction's literals added as fixed. With `quantifiers.all_grounding: observed`, `all` is true instead when every instance in the subject set satisfies the predicate.
- `no` is true when the predicate is fixed at 0, by the same test.
- `most` is true when the proportion of instances in the subject set that satisfy the predicate is at least `quantifiers.most.min_proportion` (default 0.7).
- `some` is true when that proportion is above 0. With `quantifiers.some.exclude_all` on (the default), `some` is used only when `all` is false, following the usual implicature.
- `generic` means whatever `quantifiers.generic.means` says: `all`, `most`, or `some`. "Penguins swim" can therefore mean all, most, or some penguins swim, set by a parameter.

For verbs, the proportion is the category proportion from `relation_proportions.csv`, and `all` means a proportion of 1 over the existing pairs.

For membership ("penguins are birds"), the predicate is true exactly when the predicate category is an ancestor of the subject category. Membership sentences take `all` or `generic` only.

For scalar poles, the comparison class is the subject category's parent: "elephants are big" means the category's mean value lies at least `scalar_adjectives.z` standard deviations above the mean of the parent category's instances.

**Rule statements.** A class-level proposition can state a rule of the world. For a determined feature, each term of the minimal DNF of its rule is a sufficient condition. The proposition takes the generic noun "thing" as its head, the term's literals as its restriction, and the determined feature as its predicate: "things with wings and with feathers can fly". A rule statement is true with `all` by construction. The truth test above confirms it.

### Instance level

An instance-level proposition says something about one instance: "the penguin can swim", "the penguin has stripes", "the penguin is a bird", "the owl can eat the mouse".

- **Subject:** an instance.
- **Predicate:** IS, HAS, or CAN feature; exposed patient projection ("is edible"); scalar pole; membership in any category above the instance's leaf; or a verb capacity with a patient instance ("can eat the mouse").
- **Truth:** read from the instance's values. Verb capacity is `result.relations.holds`. Scalar poles use the category named by the subject's noun as the comparison class: "the big mouse" is big for a mouse, and "the small animal" can be the same mouse.

### Event level

An event-level proposition says that something happened in a scene: "the penguin swims", "the owl chases the mouse".

- **Content:** an event from the scene generator, which records the scene, the time step, the verb, the agent, and the patient if any.
- **Truth:** the event occurred. Events are generated only where the world allows them (see "Scenes and events"), so an event never contradicts a capacity.

Event-level propositions are never negated.

### Negation

Class-level and instance-level propositions can be negative: "penguins cannot fly", "the owl has no fins", "no fish have fur". Negative propositions are true negations, grounded by the same tests with the predicate's value 0. The rate of negative propositions is a parameter for each level. The default is 0.1.

### False propositions and test sets

Training documents contain true propositions only. False propositions go into separate test sets, each item labeled true or false. A false proposition is made from a true one by one minimal change:

- **predicate swap:** replace the predicate with one of the same kind that makes the proposition false;
- **subject swap:** replace the subject with a category or instance of the same level for which the proposition is false;
- **quantifier swap:** replace the quantifier with one that makes the proposition false ("all" for a "most" fact);
- **role swap** (verbs only): exchange agent and patient, when the reversed relation does not hold.

Every false item is checked false by the same truth tests. Test sets come in matched pairs: each false item sits beside the true item it was made from. There is one test set for each proposition level and change type, with a configured size.

## Scenes and events

Event-level sentences and situational documents need things to happen. The scene generator is a simple stand-in until event schemas or world-simulation logs can supply events.

**Participants.** A scene starts with a seed instance. The generator then draws `scene.size` other instances, each with probability proportional to a weighted sum over its leaf and the seed's leaf:

- the thematic relatedness of the two leaves (`thematic.csv`);
- the taxonomic similarity of the two leaves;
- a constant.

The weights are parameters. With verbs off, the thematic weight has no effect.

**Timeline.** A scene has a number of time steps drawn from `scene.steps`. At each step, the number of events is drawn from a Poisson distribution with mean `scene.events_per_step`. Each event is drawn from the pool of possible events among the participants:

- an intransitive event for every participant and every CAN feature the participant has;
- a transitive event for every ordered pair of distinct participants and every verb whose relation holds for the pair.

Draws are weighted by `scene.verb_weights` (uniform by default) and by `scene.transitive_share`. The same event does not occur twice at one time step. Nothing changes state: events do not alter the participants.

## Layer 3: documents

### Document types

The corpus mixes four document types. Their proportions are parameters.

**Encyclopedic, about a category.** The topic is a category, drawn with configured weights over levels. The content pool holds:

- membership facts: the topic's ancestors ("penguins are birds") and its children ("emperor penguins are penguins");
- class-level facts about the topic: defining features (quantified `all` or generic), characteristic features (`most` or generic), and rarer features (`some`);
- class-level facts about the topic's subcategories;
- relation facts with the topic as agent or as patient;
- optionally, contrasts with sibling categories, realized as two adjacent sentences, one about each category. A contrastive construction ("unlike penguins, gulls can fly") is out of scope.

**Encyclopedic, about a feature.** The topic is an IS, HAS, or CAN feature, or a verb. The content pool holds:

- which categories have the feature, at several levels;
- which categories lack it (negations);
- rule statements whose predicate is the topic feature (sufficient conditions for it);
- rule statements in which the topic feature appears in the restriction (what the topic feature makes possible).

**Entity narrative.** The topic is one instance. The generator builds `entity.scenes` scenes seeded at the instance. The document interleaves:

- instance-level facts about the instance;
- the events of its scenes that involve the instance, in time order, scene by scene.

**Situational narrative.** One scene. The document introduces the participants as they first take part in events, narrates the events in time order, and adds instance-level descriptions of participants at a configured rate.

Each document's length is drawn from a configured range for its type.

### Ordering

Encyclopedic documents follow a loose template: membership first, then defining facts, characteristic facts, rarer facts, and relation facts. A shuffle parameter moves from the strict template (0) to a random order (1): each sentence is displaced by a random amount scaled by the parameter. Narratives follow time order, with descriptions inserted near the referent's first mention.

### Mentioning referents

- **Noun level.** A category or instance is named with the noun of a category drawn from its own level and the levels above it. `mention.level_weights` give the weights, with the leaf level heaviest by default. The logical form records which category the noun names.
- **First and later mentions.** In narratives, an instance is introduced with the indefinite determiner ("a penguin") and mentioned later with the definite determiner ("the penguin") or a pronoun.
- **Pronouns.** A later mention becomes "it" at `mention.pronoun_rate`, when the referent was mentioned in the previous sentence and was either the only referent mentioned there or its subject. Coreference chains are recorded, including for ambiguous pronouns.
- **Distinguishing modifiers.** When a scene has two or more participants that the chosen noun fits, a definite mention adds adjectives or with-phrases until it picks out one referent, following the incremental algorithm of Dale and Reiter (1995), with the preference order fixed per language. In situational documents, adjectives therefore do referential work. Elsewhere, modifiers are added at `mention.modifier_rate`, chosen from features true of the referent.

## Layer 4: grammar

### Phrase structure

The grammar below is written in the default English order. Word order parameters (next section) rearrange it.

```
S    → NP VP
NP   → Det N | Q N | N                     (bare noun: generic plurals)
NP   → NP-core AP? WITH* RC?               (modifiers attach inside NP)
AP   → A | AP A                            (up to max_adjectives)
WITH → with N | without N                  (joined by "and"; up to max_with_phrases)
RC   → that VP | that NP V                 (subject and object relatives)
VP   → V | V NP                            (intransitive, transitive)
VP   → can V | can V NP                    (capacity)
VP   → can not V | can not V NP            (negated capacity)
VP   → is A | is not A                     (property, scalar pole, exposed projection)
VP   → has N | has no N                    (part)
VP   → is a N | is not a N                 (membership)
```

Quantifiers occupy the determiner slot: "all penguins", "no fish". The negative class-level quantifier "no" replaces sentence negation: "no fish have fur".

**Relative clauses.** A noun phrase takes a relative clause at `relative_clauses.rate`. The clause is an object relative ("the mouse that the owl eats") with probability `relative_clauses.object_share`, and a subject relative ("the owl that eats mice") otherwise. A relative clause's own noun phrases can take relative clauses up to `relative_clauses.max_depth`. Depth 2 or more produces center embedding in subject position, and with it controllable long-distance dependencies. A relative clause expresses a true proposition of the same level as its sentence, about the same referent, and the logical form records it as a restriction.

**Adjective order.** When a noun takes several adjectives, they appear in a fixed order: a random ordering of adjective concepts drawn once per language. `adjective_order.fixed: false` makes the order random for each phrase.

**Limits.** `max_adjectives` (default 3), `max_with_phrases` (default 2), and `max_sentence_tokens` (default 30). A proposition whose realization would pass a limit is realized with fewer optional modifiers.

### Word order

Each setting has an English default.

| Setting | Values | Default |
| --- | --- | --- |
| `clause` | SVO, SOV, VSO, VOS, OVS, OSV | SVO |
| `determiner` | before, after the noun | before |
| `adjective` | before, after the noun | before |
| `with_phrase` | before, after the noun | after |
| `relative_clause` | before, after the noun | after |
| `adposition` | preposition ("with arms"), postposition ("arms with") | preposition |
| `auxiliary` | before, after the verb (applies to `can`, `is`, `has`) | before |
| `negation` | before, after the verb or auxiliary | after the auxiliary |

### Morphology

Number, tense, and aspect are each off by default. When on, each is realized either as an affix joined to the word form (`realization: affix`, "swim-s") or as a separate function word (`realization: word`), set separately for each.

- **Number:** singular and plural on nouns, with agreement on verbs and auxiliaries. Generic subjects are plural. With number off, nouns and verbs have one form: "all penguin swim".
- **Tense:** present and past. Class-level and instance-level propositions are present. Event-level propositions take `tense.event_tense` (default past).
- **Aspect:** simple and progressive, drawn for event-level propositions at `aspect.progressive_rate`.

With every inflection off, the language is the one in your examples: "the penguin swim".

### Scalar adjectives

Each scalar dimension has two adjectives, one for each pole. An instance counts as "big" when its value is at least `scalar_adjectives.z` standard deviations above the mean of its comparison class, and "small" when at least that far below. The comparison class is the category named by the noun (instance level) or the subject category's parent (class level). Comparatives ("bigger than") are out of scope for now.

## Renderings

Every sentence is written in two renderings:

- **Formal:** lexeme labels with glosses, for example `the/L.1 penguin/L.57 swim/L.203`. The gloss of a content lexeme is its concept's label (`C1.3`, `CAN.2`), so the formal rendering works without word forms.
- **Spelled:** the word forms' readable spellings, when a word-form lexicon is given.

## Outputs

A run writes one folder, by default `runs/corpus/<name>_seed<seed>/`.

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved configuration, all seeds, the taxonomy run's identity (its configuration hash and seed), the word-form run's identity when used, the git commit hash, and package versions. |
| `lexicon.csv` | One row per lexeme: label, part of speech, concept or concepts, word form label, spelling, gloss. |
| `documents.jsonl` | One JSON object per document (schema below). |
| `corpus.txt` | Every document in the spelled rendering (or the formal rendering without word forms): one sentence per line, a blank line between documents. |
| `corpus_formal.txt` | The same in the formal rendering. |
| `scenes.jsonl` | One JSON object per scene: participants, and events by time step. |
| `tests/<level>_<change>.jsonl` | Test sets: matched true and false items with their logical forms and sentences. |
| `stats.yaml` | Counts by document type, proposition level, quantifier, and part of speech; lexeme frequencies; sentence lengths; relative-clause depths; and the co-occurrence check below. |

Each document object holds its label, type, topic, scenes, and sentences. Each sentence holds:

- `label`;
- `tokens`: lexeme labels;
- `words`: word form labels, or null;
- `text`: the spelled rendering, or null;
- `formal`: the formal rendering;
- `tree`: the parse tree, as a nested list of the form `[label, child, ...]`;
- `logical_form`: the proposition (schema below);
- `referents`: for each noun phrase, the instance or category it refers to and the category its noun names;
- `coreference`: the chain each referring noun phrase belongs to.

A logical form, for example:

```json
{"id": "PR.310", "level": "class", "quantifier": "most", "polarity": true,
 "subject": {"category": "C1.3", "restriction": ["IS.4", "not HAS.2"]},
 "predicate": {"kind": "can", "feature": "CAN.3"},
 "grounding": {"proportion": 0.93, "fixed": false, "test": "observed"}}
```

**Co-occurrence check.** `stats.yaml` reports, over pairs of leaves, the correlation of within-document co-occurrence with thematic relatedness and with taxonomic similarity, separately for each document type. The check confirms that the document mix works as a lever: situational documents should correlate more with thematic relatedness, and encyclopedic documents more with taxonomic similarity.

## Changes to the word-form pipeline

The corpus needs three things from `WORDFORM_PIPELINE.md` that its current stages do not provide:

1. **Function words:** word forms of one syllable, drawn from the most frequent syllable shapes, requested by part of speech.
2. **Affixes:** bound forms of one or two phonemes, joined to stems by the phonotactic rules at the join.
3. **Inflected forms:** synthesis and embedding of stem-plus-affix forms, cached like any other word form.

The corpus generator exports a lexicon request (lexemes with part of speech and length constraints). The word-form pipeline answers the request. These additions will be written into `WORDFORM_PIPELINE.md` as a new stage when the corpus work reaches morphology. Until then, the corpus works with inflection realized as separate words, or in the formal rendering.

## Configuration

```yaml
name: default
seed: 1
taxonomy: runs/taxonomy/relations_seed1      # a taxonomy output folder
wordforms: null                              # a word-form run folder, or null for formal tokens only

lexicon:
  named_proportion: {category: 1.0, is: 1.0, has: 1.0, can: 1.0, verb: 1.0, verb_category: 1.0, patient_projection: 1.0, scalar: 1.0}
  synonym_rate: 0.0
  homonym_rate: 0.0
  homonym_same_pos: 0.5

documents:
  count: 10000
  mix: {encyclopedic_category: 0.3, encyclopedic_feature: 0.2, entity: 0.2, situational: 0.3}
  sentences: {encyclopedic_category: [5, 15], encyclopedic_feature: [5, 15], entity: [5, 20], situational: [5, 20]}
  topic_level_weights: {schedule: linear, start: 1, end: 2}   # weights over category levels for category topics
  shuffle: 0.3
  instance_description_rate: 0.2

propositions:
  negation_rate: {class: 0.1, instance: 0.1}
  rule_statement_rate: 0.3                    # share of feature-topic sentences that state rules

quantifiers:
  all_grounding: fixed                        # fixed or observed
  most: {min_proportion: 0.7}
  some: {exclude_all: true}
  generic: {means: most}                      # all, most, or some
  generic_rate: 0.5                           # share of class-level sentences that use a bare generic when it is true

scene:
  size: [2, 6]
  steps: [3, 8]
  events_per_step: 1.5
  transitive_share: 0.5
  verb_weights: uniform
  participant_weights: {thematic: 1.0, taxonomic: 0.5, constant: 0.1}

entity:
  scenes: [1, 3]

mention:
  level_weights: {schedule: linear, start: 1, end: 4}         # heaviest at the leaf level
  pronoun_rate: 0.5
  modifier_rate: 0.3

grammar:
  max_adjectives: 3
  max_with_phrases: 2
  max_sentence_tokens: 30
  adjective_order: {fixed: true}
  relative_clauses: {rate: 0.1, max_depth: 1, object_share: 0.3}
  word_order: {clause: SVO, determiner: before, adjective: before, with_phrase: after, relative_clause: after, adposition: preposition, auxiliary: before, negation: after_auxiliary}
  morphology:
    number: {on: false, realization: affix}
    tense: {on: false, realization: affix, event_tense: past}
    aspect: {on: false, realization: word, progressive_rate: 0.3}

scalar_adjectives: {z: 1.0}

test_sets:
  size: 500                                   # true items per set; each gets one matched false item
  changes: [predicate, subject, quantifier, role]
```

Validation follows the base conventions: unknown keys are errors, and every error names the file and the field.

## Determinism

Use the stream-seed function in `semantic_world.taxonomy.streams`. Streams: `corpus:lexicon`, `corpus:scenes`, `corpus:documents`, `corpus:propositions`, `corpus:mentions`, `corpus:grammar`, and `corpus:tests`. The corpus seed is its own master seed, independent of the taxonomy's seed. Properties, each tested:

- the same taxonomy run, word-form run, configuration, and seed give byte-identical output folders;
- changing the grammar settings never changes the propositions or their order, only their realization;
- changing the test-set settings never changes the documents.

## Python package

Put the generator in `python/semantic_world/corpus/`. Suggested modules: `config.py`, `lexicon.py`, `propositions.py` (logical forms and truth tests), `scenes.py`, `planner.py` (document types, ordering, and mentions), `grammar.py` (phrase structure, word order, and morphology), `realize.py`, `interpret.py` (tree to logical form, for tests), `testsets.py` (not `test_sets.py`, which pytest would collect as a test module), `io.py`, and `__main__.py`. The command line is:

```
python -m semantic_world.corpus data/corpus/default.yaml [--seed N] [--out DIR]
```

Add `data/corpus/default.yaml` and `data/corpus/tiny.yaml` (built on the tiny relations taxonomy, 20 documents).

## Build stages

Work on one branch per stage (`corpus-stage-1`, and so on). Branch stage 1 from the latest taxonomy stage branch, or from `main` if the taxonomy work has been merged. Each stage ends with its tests passing, the full check list in `CLAUDE.md` passing, and a commit.

1. **Configuration and lexicon.** Loading a taxonomy run, concepts, lexemes, function words, synonym and homonym knobs, and the formal rendering. *Accept:* every concept type gets lexemes in the configured proportions; with both knobs at 0, lexemes and concepts are one to one; with the knobs on, the realized rates are within tolerance.
2. **Propositions and truth.** Class-level and instance-level propositions, quantifiers, restrictions, rule statements, negation, and the false-item generator. *Accept:* every generated proposition passes an independent recomputation of its truth from the taxonomy's output files; every false item fails it; each false item differs from its matched true item by exactly one change; rule statements are `all`-true.
3. **Scenes and events.** *Accept:* every event is possible (its CAN feature or relation holds); no event repeats within a time step; with the thematic weight above 0, participants are more thematically related than with it at 0, on average.
4. **Grammar and realization.** Phrase structure, word order, morphology, adjective order, relative clauses, and scalar adjectives. *Accept:* every tree's leaves equal its tokens; `interpret(tree)` recovers the logical form exactly for every sentence; all six clause orders and all two-way settings produce correct trees on a fixed set of propositions; relative-clause depth never exceeds the limit.
5. **Documents.** The four document types, ordering, noun levels, first and later mentions, pronouns, and distinguishing modifiers. *Accept:* every distinguishing definite mention picks out exactly one participant of its scene; every pronoun's chain matches the referent recorded in its logical form; with shuffle 0, encyclopedic documents follow the template order exactly.
6. **Outputs, test sets, statistics, and the command line.** *Accept:* `documents.jsonl` round-trips through a JSON parser with the documented schema; `corpus.txt` matches the spelled renderings; the determinism properties hold; on the default configuration, situational documents' co-occurrence correlates more with thematic relatedness than encyclopedic documents' does, and less with taxonomic similarity.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. `all` and `no` are grounded in fixed features (law-like), not in the instances that happen to exist, with an option for the observed reading.
2. Rule statements use the generic noun "thing" plus the terms of the rule's minimal DNF.
3. Class-level scalar poles compare a category with its parent. Instance-level poles compare an instance with the category its noun names.
4. Events never change state, and event-level propositions are never negated.
5. Distinguishing modifiers follow the incremental algorithm, with one preference order per language.
6. A pronoun is used only when its referent was the only referent, or the subject, of the previous sentence.
7. Function words, affixes, and inflected forms need a new stage in the word-form pipeline, written when the corpus work reaches morphology.
8. Sibling contrasts are two adjacent sentences, not a contrastive construction.

## References

- Dale, R., & Reiter, E. (1995). Computational interpretations of the Gricean maxims in the generation of referring expressions. *Cognitive Science*, 19, 233–263.
