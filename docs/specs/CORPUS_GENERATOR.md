# Corpus generator: propositions, documents, and grammar

Draft, September 29, 2026. Reconciled with the built word-form pipeline on October 1, 2026. This document is the build specification for a Python program that turns the taxonomy generator's world into a corpus of documents written in an artificial language. The specification is written for handoff to Claude Code, or to any developer, and should be complete together with `docs/specs/TAXONOMY_GENERATOR.md`, `docs/specs/TAXONOMY_RELATIONS.md`, and `docs/specs/WORDFORM_PIPELINE.md`.

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
- delivering language to agents in the running world;
- spoken sentences, which are planned in `docs/specs/CONNECTED_SPEECH.md`.

## Inputs

- **A taxonomy** (required): a taxonomy configuration file with a taxonomy seed, or a taxonomy output folder. In Python, a `TaxonomyResult` works too. A configuration file is generated in memory, so nothing needs to be saved on disk. An output folder is regenerated in memory from its `config.yaml`, and the run stops with an error when the regenerated result differs from the folder's files. Verbs and scalar dimensions are used when the result has them. Without verbs, there are no transitive sentences and no thematic scene sampling.
- **A word-form run** (optional): a word-form pipeline run (`WORDFORM_PIPELINE.md`) made from the corpus's own request. The corpus is generated first, without word forms, and the word forms are attached afterwards (see "Word forms for the corpus"). Without a word-form run, the corpus has every rendering except the spelled one (see "Renderings").

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
| Referent, within one document | `R.<n>` | `R.2` is the second referent introduced in its document |

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

Each concept has a concept label, used by the conceptual and propositional renderings. Categories, features, verbs, and projections keep their taxonomy labels (`C1.3.2`, `IS.12`, `HAS.4`, `CAN.3`, `V1.2`, `CANBE.V1.1`). A scalar pole is its dimension's label with `HIGH` or `LOW` (`SC.1.HIGH`). The generic head noun is `THING`, and a function word's concept label is its gloss in capitals (`THE`).

For each concept type, a parameter gives the proportion of concepts that get a word (default 1.0). The number of named concepts of a type is the proportion times the number of concepts, rounded, and the named concepts are drawn at random. The two poles of a scalar dimension are named together. A concept without a word cannot be expressed, and the discourse planner skips any proposition that needs one.

A verb or verb category whose relation holds for every pair of distinct instances, or for no pair, never gets a word. Such a relation says nothing. `stats.yaml` lists these verbs.

Lexeme labels are numbered in this order: the first lexeme of every named content concept, in the order of the table above; then the second lexemes of concepts with synonyms; then the function words. Content lexemes therefore keep their labels when the grammar settings change the set of function words.

### Function words

`a`, `the`, `all`, `most`, `some`, `no`, `not`, `can`, `is`, `has`, `with`, `without`, `and`, `that` (relative clauses), and `it` (pronoun). With verb agreement on, `are` and `have` are added as the plural forms of `is` and `has`. When an inflection is realized as a separate word, one function word is added for each inflectional value, glossed like the affix (`PLURAL`, `PAST`, `PROGRESSIVE`; see "Morphology"). The set is fixed by the settings. Function words are ordinary lexemes with their own word forms.

### Synonyms and homonyms

The default is one lexeme per concept and one concept per lexeme. Two parameters, both 0 by default, add ambiguity for later experiments:

- `synonym_rate`: the probability that a content concept gets a second lexeme. When a concept has several lexemes, each mention picks one at random.
- `homonym_rate`: the probability that a content lexeme shares its word form with another concept's lexeme. The pair is chosen within the same part of speech with probability `homonym_same_pos`, and across parts of speech otherwise.

Every lexeme has exactly one concept. A homonym is two lexemes, with different concepts, that share one word form. The lexicon records each lexeme's concept, and the lexeme whose word form a homonym shares (`same_form_as`).

### Word forms

When a word-form run is given, every lexeme gets a word form. Function words have one syllable each (CV, CVC, or VC), and the most frequent half get two-phoneme shapes. Affixes (see "Morphology") are bound forms of shape C or VC. The word-form pipeline makes every form, and assigns content words to lexemes, from a request the corpus generator writes (see "Word forms for the corpus").

## Layer 2: propositions

A proposition is a logical form with a type, a polarity, and a truth grounding. There are three levels.

### Class level

A class-level proposition says something about a category: "all penguins can swim", "red penguins swim", "penguins are birds".

- **Subject:** a category, optionally restricted by literals: "red penguins" is the category restricted to members with the IS feature for red. A restriction is a set of IS and HAS literals, positive or negative ("with fins", "without fins"), and scalar pole literals ("big").
- **Predicate:** one of IS feature, HAS feature, CAN feature, scalar pole ("are big"), membership in a category above the subject ("are birds"), or a verb with a patient category ("chase fish").
- **Quantifier:** `all`, `most`, `some`, `no`, or `generic` (a bare plural with no quantifier word: "penguins swim").

**Truth grounding.** A class-level proposition needs at least one instance in its subject set: the instances below the subject category that satisfy the restriction. A proposition with an empty subject set is vacuous, and is never generated, in documents or in test sets. For one-place predicates:

- `all` is true when the predicate is fixed at 1 for the category, given its defining features, any fixed-by-rule features, and the restriction. The test uses the cone enumeration of the fixed-by-rule test, with the restriction's literals added as fixed. With `quantifiers.all_grounding: observed`, `all` is true instead when every instance in the subject set satisfies the predicate.
- `no` is true when the predicate is fixed at 0, by the same test.
- `most` is true when the proportion of instances in the subject set that satisfy the predicate is at least `quantifiers.most.min_proportion` (default 0.7).
- `some` is true when that proportion is above 0. With `quantifiers.some.exclude_all` on (the default), `some` is used only when `all` is false, following the usual implicature.
- `generic` means whatever `quantifiers.generic.means` says: `all`, `most`, or `some`. "Penguins swim" can therefore mean all, most, or some penguins swim, set by a parameter.

For verbs, the proportion is the proportion of pairs of distinct instances, an agent from the subject set and a patient from the patient category, for which the relation holds. The proportion is computed from the relation matrices (`result.relations.matrix`), for any pair of categories, at the same level or not, and for any verb or verb category. A verb category's relation is its base relation. The proportions are not limited to the rows of `relation_proportions.csv`. `all` means a proportion of 1 over the existing pairs.

For membership ("penguins are birds"), the predicate is true exactly when the predicate category is an ancestor of the subject category. Membership sentences take `all` or `generic` only.

For scalar poles, the comparison class is the subject category's parent: "elephants are big" means the category's mean value lies at least `scalar_adjectives.z` standard deviations above the mean of the parent category's instances. A top-level category has no parent, and is compared with all instances in the world. A class-level scalar pole takes the generic only, never a quantifier word.

**Rule statements.** A class-level proposition can state a rule of the world. For a determined feature, each term of the minimal DNF of its rule is a sufficient condition. The proposition takes the generic noun "thing" as its head, the term's literals as its restriction, and the determined feature as its predicate: "things with wings and with feathers can fly". A rule statement is true with `all` by construction. The truth test above confirms it. A positive IS literal of the term is an adjective, and a HAS literal is a with-phrase or a without-phrase. A negated IS literal is realized as a subject relative clause: "things with wings that are not red can fly". The same holds for a negated IS literal in any restriction. A term is skipped when it would need more than one relative clause, or when it reads a scalar threshold, which no pole adjective states. `stats.yaml` reports how many terms were skipped, and why.

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
- **Relative clauses and limits.** The planner decides everything that changes what a sentence says. So the planner, and not the grammar, decides which noun phrases take relative clauses (`mention.relative_clauses`), and applies the limits on adjectives, with-phrases, and sentence length (`mention.max_adjectives`, `mention.max_with_phrases`, `mention.max_content_words`). The grammar section describes how they are realized.
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

**Relative clauses.** A noun phrase takes a relative clause at `mention.relative_clauses.rate`. The clause is an object relative ("the mouse that the owl eats") with probability `mention.relative_clauses.object_share`, and a subject relative ("the owl that eats mice") otherwise. A relative clause's own noun phrases can take relative clauses up to `mention.relative_clauses.max_depth`. The planner makes these choices, and the grammar realizes them. Depth 2 or more produces center embedding in subject position, and with it controllable long-distance dependencies. A relative clause expresses a true proposition of the same level as its sentence, about the same referent, and the logical form records it as a restriction.

**Adjective order.** When a noun takes several adjectives, they appear in a fixed order: a random ordering of adjective concepts drawn once per language. `adjective_order.fixed: false` makes the order random for each phrase.

**Limits.** `mention.max_adjectives` (default 3), `mention.max_with_phrases` (default 2), and `mention.max_content_words` (default 20). The planner applies the limits: a proposition that would pass a limit gets fewer optional modifiers. The sentence limit counts content words only (nouns, adjectives, and verbs), because the number of function words depends on the morphology settings, and a grammar setting must never change what a sentence says.

### Word order

Each setting has an English default. The word-order and morphology settings never change a logical form. They change only how a logical form is realized.

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

- **Number:** singular and plural on nouns. Generic subjects are plural. With number off, nouns and verbs have one form: "all penguin swim".
- **Agreement:** with number on, verbs agree with their subjects unless `number.agreement` is false. The verb takes the same affix or word as a plural noun. `number.verb_marks` says which verbs take it: `plural` (the default) or `singular`, as in English ("the penguin swims", "penguins swim"). The auxiliaries `is` and `has` agree by switching to the function words `are` and `have`. `can` does not agree. Agreement holds across relative clauses, so relative clauses in subject position create long-distance dependencies between subject and verb.
- **Tense:** present and past. Class-level and instance-level propositions are present. Event-level propositions take `tense.event_tense` (default past).
- **Aspect:** simple and progressive, drawn for event-level propositions at `aspect.progressive_rate`.

A morphology word stands right after the word it marks, or right before it with `position: before`. With every inflection off, the language is the one in your examples: "the penguin swim".

With English function words (the Jabberwocky option of the word-form pipeline), every function-word gloss must be an English word. So `realization: word` is an error with English function words, because `PLURAL` is not an English word. The English affixes are the suffixes for `PLURAL`, `PAST`, and `PROGRESSIVE` only.

### Scalar adjectives

Each scalar dimension has two adjectives, one for each pole. An instance counts as "big" when its value is at least `scalar_adjectives.z` standard deviations above the mean of its comparison class, and "small" when at least that far below. The comparison class is the category named by the noun (instance level) or the subject category's parent (class level). Comparatives ("bigger than") are out of scope for now.

## Renderings

Every sentence is written in up to four renderings. The first three need no word forms, so a whole corpus can be produced in any of them alone.

- **Formal:** lexeme labels with glosses, for example `the/L.176 C1.3/L.5 CAN.2/L.138` for "the penguin swim". The gloss of a content lexeme is its concept's label (`C1.3`, `CAN.2`), and the gloss of a function word is its English gloss (`the`).
- **Conceptual:** the sentence's words in order, each replaced by its concept label: `THE IS.12 C1.3.2 CAN V1.2 THE C1.4.1` for "the furry dog can chase the cat". An inflected word is its concept label joined to the affix's gloss: `C1.3.2-PLURAL`. The conceptual rendering carries different information from the formal rendering only when synonyms are on, because synonyms share a concept label. Homonyms are two lexemes, so both renderings tell a homonym's two meanings apart.
- **Propositional:** the sentence's logical form, written out as atomic propositions joined by `AND`. Every modifier, with-phrase, relative clause, and noun becomes its own proposition, so one sentence is usually several propositions. Determiners and pronouns disappear, and referents are named by label. Its format is below.
- **Spelled:** the word forms' readable spellings, after word forms are attached.

**The propositional format.** An atomic proposition is a concept label with its arguments: `C1.3.2(R.1)` (membership), `IS.12(R.1)`, `HAS.4(R.1)`, `CAN.3(R.1)`, `SC.1.HIGH(R.1)`, and `V1.2(R.1, R.2)` (the relation holds, so the agent can do it to the patient). `NOT` before an atom negates it. An event-level proposition is prefixed with its event label: `SN.8.5: V1.2(R.1, R.2)`. A class-level proposition is a quantifier with a restrictor and a scope over variables: `MOST(C1.3(X.1) AND IS.4(X.1), CAN.3(X.1))` for "most red penguins can swim", with `ALL`, `MOST`, `SOME`, `NO`, and `GEN` (generic). Variables are labeled `X.1`, `X.2`, and so on, numbered within each sentence in order of first mention, like referents. So a relative clause at any depth can introduce a new variable. Examples:

| Sentence | Propositional rendering |
| --- | --- |
| the furry dog has legs | `C1.3.2(R.1) AND IS.12(R.1) AND HAS.4(R.1)` |
| the dog that chased the cat ran | `C1.3.2(R.1) AND C1.4.1(R.2) AND SN.3.2: V1.2(R.1, R.2) AND SN.3.4: CAN.7(R.1)` |
| things with wings and with feathers can fly | `ALL(HAS.2(X.1) AND HAS.5(X.1), CAN.1(X.1))` |
| owls eat mice | `GEN(C1.2(X.1) AND C1.5(X.2), V2.1(X.1, X.2))` |

Referents are numbered within each document in order of first mention (`R.1`, `R.2`), so a referent keeps its label across sentences and the propositional rendering shows coreference directly. With `renderings.propositional.referents: instance`, referents are named by their taxonomy instance labels instead. The document's `referents` field records each referent's instance either way.

## Outputs

A run writes one folder, by default `runs/corpus/<name>_seed<seed>/`.

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved configuration, all seeds, the taxonomy run's identity (its configuration hash and seed), the word-form run's identity when used, the git commit hash, and package versions. |
| `lexicon.csv` | One row per lexeme: label, part of speech, concept, word form label, spelling, gloss, and, for a homonym, the lexeme whose word form it shares (`same_form_as`). |
| `documents.jsonl` | One JSON object per document (schema below). |
| `corpus.txt` | Every document in the spelled rendering (or the formal rendering without word forms): one sentence per line, a blank line between documents. |
| `corpus_formal.txt` | The same in the formal rendering. |
| `corpus_conceptual.txt` | The same in the conceptual rendering. |
| `corpus_propositional.txt` | The same in the propositional rendering. |
| `wordform_request.yaml` | The request for the word-form pipeline (see "Word forms for the corpus"). |
| `scenes.jsonl` | One JSON object per scene: participants, and events by time step. |
| `tests/<level>_<change>.jsonl` | Test sets: matched true and false items with their logical forms and sentences. |
| `stats.yaml` | Counts by document type, proposition level, quantifier, and part of speech; lexeme frequencies; sentence lengths; relative-clause depths; the verbs that get no word because their relation holds for every pair or for no pair; the rule terms that were skipped; and the co-occurrence check below. |

Each document object holds its label, type, topic, scenes, and sentences. Each sentence holds:

- `label`;
- `tokens`: lexeme labels;
- `words`: word form labels, or null;
- `text`: the spelled rendering, or null;
- `formal`: the formal rendering;
- `conceptual`: the conceptual rendering;
- `propositional`: the propositional rendering;
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

## Word forms for the corpus

The corpus comes first, and the word forms are made for it. The run has three steps:

1. **Generate.** `python -m semantic_world.corpus generate` writes the corpus in the formal, conceptual, and propositional renderings, and writes `wordform_request.yaml`.
2. **Make the word forms.** The word-form pipeline runs with the request (`request:` in its configuration).
3. **Render.** `python -m semantic_world.corpus render` reads the word-form run and adds the word labels, the spelled rendering (which `corpus.txt` then holds), and the word-form columns of `lexicon.csv`. Rendering changes nothing else in the corpus run.

**The request** lists:

- `lexemes`: every content lexeme, with its concept label and part of speech, and, for a homonym, the lexeme whose word form it shares (`same_form_as`);
- `function_words`: the glosses of the function words the corpus uses, most frequent first, by their counts in the generated corpus;
- `affixes`: the glosses and positions of the affixes the corpus uses;
- `inflect`: the lexemes that appear inflected, with the affixes each one takes;
- `meanings`: the taxonomy run's `categories_generative.csv`, for assigning category lexemes. When the request gives meanings, the pipeline's `assignment.meanings` must be null. Giving both is an error.

**The word-form pipeline's part.** The pipeline already makes function words, affixes, and inflected forms from a request (stage 4a of `WORDFORM_PIPELINE.md`), and assigns words to categories (stage 7). Corpus stage 7 extends it:

- the request becomes a top-level `request` setting that replaces `closed_class.request`, and gains `lexemes` and `meanings`;
- lexemes are assigned to content words. Category lexemes are assigned by the configured assignment mode (arbitrary, target correlation, branch markers, or acoustic mapping). A category without a lexeme is left out of the assignment. A category's synonym lexemes are assigned by the mode too, each with the category's meaning vector, so the synonyms of one category share its sound–meaning structure, including its branch marker. All other lexemes get words at random, from the words left over. A homonym gets its partner's word. The assignment is written to `assignment/lexicon.csv` with a `lexeme` column, and `words.csv` gains the part of speech of each assigned word (`pos`);
- a lexeme that must be inflected gets only a word that can take its affixes. Every assignment mode respects that rule, as branch markers already do for their markers. So no inflected form that the corpus needs is ever skipped;
- the pipeline runs in two passes. The words are assigned first. The inflected forms are then made, synthesized, and embedded. The audio cache keeps the second pass cheap, and with the edit distance as the sound distance, no audio is needed before the assignment;
- `inflect` entries can name lexemes, and the pipeline inflects whatever form the lexeme's concept got, including a marked form of branch-marker mode (`W.12.M.2.AF.1`);
- a run whose content words are fewer than the lexemes that need distinct forms is an error. Content words that no lexeme gets are kept; they can serve as novel words in tests.

The default taxonomy with verbs needs about 180 content lexemes, and the default word-form run makes 500 words.

**English options.** The word-form pipeline can give the real English function words, and the English suffixes with English allomorphy (a Jabberwocky condition). The corpus needs nothing extra for them, apart from the restrictions in "Morphology".

Spoken sentences, with coarticulation across word boundaries and reduced function words, are planned in `docs/specs/CONNECTED_SPEECH.md`. Stage 5 of that specification synthesizes the corpus's documents.

## Configuration

```yaml
name: default
seed: 1
taxonomy: {config: data/taxonomy/relations.yaml, seed: 1}   # a taxonomy configuration file and a taxonomy seed (null: the file's seed)
# taxonomy: {run: runs/taxonomy/relations_seed1}            # or a taxonomy output folder, regenerated in memory and checked against its files

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
  max_adjectives: 3
  max_with_phrases: 2
  max_content_words: 20                       # nouns, adjectives, and verbs in one sentence
  relative_clauses: {rate: 0.1, max_depth: 1, object_share: 0.3}

grammar:
  adjective_order: {fixed: true}
  word_order: {clause: SVO, determiner: before, adjective: before, with_phrase: after, relative_clause: after, adposition: preposition, auxiliary: before, negation: after_auxiliary}
  morphology:
    number: {on: false, realization: affix, position: after, agreement: true, verb_marks: plural}   # verb_marks: plural or singular (English)
    tense: {on: false, realization: affix, position: after, event_tense: past}
    aspect: {on: false, realization: word, position: after, progressive_rate: 0.3}

scalar_adjectives: {z: 1.0}

renderings:
  propositional: {referents: local}          # local (R.1, R.2, ... within each document) or instance (taxonomy instance labels)

test_sets:
  size: 500                                   # true items per set; each gets one matched false item
  changes: [predicate, subject, quantifier, role]
```

Validation follows the base conventions: unknown keys are errors, and every error names the file and the field. `taxonomy` takes exactly one of `config` and `run`, and `seed` goes with `config` only. Relative paths are read from the folder the command runs in, as in the word-form pipeline. YAML reads a bare `on` as the boolean true, so the loader accepts either reading of the `on` key in the morphology settings.

## Determinism

Use the stream-seed function in `semantic_world.taxonomy.streams`. Streams: `corpus:lexicon`, `corpus:scenes`, `corpus:documents`, `corpus:propositions`, `corpus:mentions`, `corpus:grammar`, and `corpus:tests`. The corpus seed is its own master seed, independent of the taxonomy's seed. Properties, each tested:

- the same taxonomy run, word-form run, configuration, and seed give byte-identical output folders;
- changing the grammar settings (word order, adjective order, and morphology) never changes any logical form or the order of the sentences, only their realization;
- rendering with a word-form run changes only the word labels, the spelled rendering (and with it `corpus.txt`), and the word-form columns of `lexicon.csv`; the rest of the output folder is byte-identical;
- changing the test-set settings never changes the documents.

## Python package

Put the generator in `python/semantic_world/corpus/`. Suggested modules: `config.py`, `streams.py`, `world.py` (loading the taxonomy), `lexicon.py`, `propositions.py` (logical forms and truth tests), `scenes.py`, `planner.py` (document types, ordering, and mentions), `grammar.py` (phrase structure, word order, and morphology), `realize.py`, `interpret.py` (tree to logical form, for tests), `testsets.py` (not `test_sets.py`, which pytest would collect as a test module), `renderings.py` (formal, conceptual, and propositional), `request.py` (the word-form request), `io.py`, and `__main__.py`. The command line is:

```
python -m semantic_world.corpus generate data/corpus/default.yaml [--seed N] [--out DIR]
python -m semantic_world.corpus render RUN_FOLDER --wordforms WORDFORM_RUN_FOLDER
```

Add `data/corpus/default.yaml` and `data/corpus/tiny.yaml` (built on the tiny relations taxonomy, 20 documents).

## Build stages

Work on one branch per stage (`corpus-stage-1`, and so on). Branch stage 1 from `main`. Each stage ends with its tests passing, the full check list in `CLAUDE.md` passing, and a commit.

1. **Configuration and lexicon.** Loading a taxonomy run, concepts, lexemes, function words, synonym and homonym knobs, and the formal rendering. *Accept:* every concept type gets lexemes in the configured proportions; with both knobs at 0, lexemes and concepts are one to one; with the knobs on, the realized rates are within tolerance.
2. **Propositions and truth.** Class-level and instance-level propositions, quantifiers, restrictions, rule statements, negation, and the false-item generator. *Accept:* every generated proposition passes an independent recomputation of its truth from the taxonomy's output files; every false item fails it; each false item differs from its matched true item by exactly one change; rule statements are `all`-true.
3. **Scenes and events.** *Accept:* every event is possible (its CAN feature or relation holds); no event repeats within a time step; with the thematic weight above 0, participants are more thematically related than with it at 0, on average.
4. **Grammar and realization.** Phrase structure, word order, morphology and agreement, adjective order, relative clauses, scalar adjectives, and the conceptual rendering. *Accept:* every tree's leaves equal its tokens; with agreement on, every verb and auxiliary agrees with its subject, including across relative clauses; `interpret(tree)` recovers the logical form exactly for every sentence; all six clause orders and all two-way settings produce correct trees on a fixed set of propositions; relative-clause depth never exceeds the limit.
5. **Documents.** The four document types, ordering, noun levels, first and later mentions, pronouns, distinguishing modifiers, referent labels, and the propositional rendering. *Accept:* the propositional rendering of every sentence parses back to its logical form; every distinguishing definite mention picks out exactly one participant of its scene; every pronoun's chain matches the referent recorded in its logical form; with shuffle 0, encyclopedic documents follow the template order exactly.
6. **Outputs, test sets, statistics, and the `generate` command.** *Accept:* `documents.jsonl` round-trips through a JSON parser with the documented schema; each corpus text file matches its rendering in `documents.jsonl`; the determinism properties hold; on the default configuration, situational documents' co-occurrence correlates more with thematic relatedness than encyclopedic documents' does, and less with taxonomic similarity.
7. **Word forms.** The request, the word-form pipeline changes in "Word forms for the corpus" (with their tests in the word-form pipeline's suite, and its spec and guide updated), and the `render` command. *Accept:* on the tiny configuration, with a new word-form configuration made for the tiny corpus (`data/wordforms/tiny.yaml` makes too few words, and stays unchanged), generate, make word forms, and render run end to end; every lexeme gets a word form, distinct lexemes get distinct forms unless they are homonyms, and category lexemes follow the assignment mode; function words are ordered by their corpus counts; only the inflections the corpus uses are synthesized; `corpus.txt` matches the spelled renderings; rendering leaves the rest of the output folder byte-identical; with branch markers and plural affixes, marked forms are inflected.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. `all` and `no` are grounded in fixed features (law-like), not in the instances that happen to exist, with an option for the observed reading.
2. Rule statements use the generic noun "thing" plus the terms of the rule's minimal DNF.
3. Class-level scalar poles compare a category with its parent. Instance-level poles compare an instance with the category its noun names.
4. Events never change state, and event-level propositions are never negated.
5. Distinguishing modifiers follow the incremental algorithm, with one preference order per language.
6. A pronoun is used only when its referent was the only referent, or the subject, of the previous sentence.
7. Function words, affixes, and inflected forms come from stage 4a of the word-form pipeline, through the corpus's request file.
8. Sibling contrasts are two adjacent sentences, not a contrastive construction.

Jon decided the following on October 1, 2026:

9. The corpus is generated first, and the word forms are made for it from a request. The corpus can be produced entirely in the propositional or the conceptual rendering, with no word forms.
10. Category lexemes get words by the word-form pipeline's assignment mode. All other lexemes get words at random. Sound differences between parts of speech are a future addition.
11. Subject–verb agreement is a parameter, on by default when number is on. Number, tense, and aspect can each be realized as affixes or as separate words.
12. In branch-marker mode, marked forms can take affixes.

Jon decided the following on October 1, 2026, before stage 1 (`docs/proposals/2026-10-01-corpus-decisions-before-stage-1.md`):

13. A homonym is two lexemes that share one word form. Every lexeme has exactly one concept.
14. The taxonomy is given as a configuration file with a seed, or as an output folder. An output folder is regenerated in memory and checked against its files.
15. A verb or verb category whose relation holds for every pair, or for no pair, gets no lexeme.
16. The word-form pipeline assigns first, and then makes, synthesizes, and embeds the inflected forms. A lexeme that must be inflected gets only a word that can take its affixes. Categories without a lexeme are left out of the assignment, and a category's synonyms are assigned by the mode. The request's `meanings` and `assignment.meanings` cannot both be given.
17. Word-order and morphology settings never change a logical form. Relative clauses and the limits on modifiers belong to the planner (`mention`). The sentence limit counts content words.
18. Variables in the propositional rendering are labeled `X.1`, `X.2`, and so on.
19. A class-level proposition needs at least one instance in its subject set.
20. In a rule statement, a negated IS literal is a subject relative clause. A term that needs more than one relative clause, or that reads a scalar threshold, is skipped and counted.
21. Class-level verb proportions come from the relation matrices, for any category pair and any verb or verb category.
22. Class-level scalar poles take the generic only. A top-level category is compared with all instances in the world.

## Future additions

- **Sound differences between parts of speech.** In English and other languages, nouns and verbs differ in their sound: English nouns tend to have more syllables and initial stress, and verbs more often have final stress (Kelly, 1992; Monaghan, Christiansen, & Chater, 2007). The request already gives each lexeme's part of speech, so the word-form pipeline could later draw each part of speech's forms from its own sound profile.
- **Case marking.** Richer systems of case marking than subject–verb agreement, such as nominative and accusative markers on noun phrases.

## References

- Dale, R., & Reiter, E. (1995). Computational interpretations of the Gricean maxims in the generation of referring expressions. *Cognitive Science*, 19, 233–263.
- Kelly, M. H. (1992). Using sound to solve syntactic problems: The role of phonology in grammatical category assignments. *Psychological Review*, 99, 349–364.
- Monaghan, P., Christiansen, M. H., & Chater, N. (2007). The phonological-distributional coherence hypothesis: Cross-linguistic evidence in language acquisition. *Cognitive Psychology*, 55, 259–305.
