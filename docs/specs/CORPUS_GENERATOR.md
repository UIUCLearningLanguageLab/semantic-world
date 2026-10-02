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

- **Subject:** a category, optionally restricted by literals: "red penguins" is the category restricted to members with the IS feature for red. A restriction is a set of IS and HAS literals, positive or negative ("with fins", "without fins"), and scalar pole literals ("big"). A subject can also take a restrictive relative clause (see "Restricted subjects" below).
- **Predicate:** one of IS feature, HAS feature, CAN feature, exposed patient projection ("are edible"), scalar pole ("are big"), membership in a category above the subject ("are birds"), or a verb with a patient category ("chase fish"). The verb can be a verb or a verb category ("owls hunt mice"). The patient category can take a restriction too.
- **Quantifier:** `all`, `most`, `some`, `no`, or `generic` (a bare plural with no quantifier word: "penguins swim").

**Truth grounding.** A class-level proposition needs at least one instance in its subject set: the instances below the subject category that satisfy the restriction. A proposition with an empty subject set is vacuous, and is never generated, in documents or in test sets. For one-place predicates:

- `all` is true when the predicate is fixed at 1 for the category, given its defining features, any fixed-by-rule features, and the restriction. The test uses the cone enumeration of the fixed-by-rule test, with the restriction's literals added as fixed. With `quantifiers.all_grounding: observed`, `all` is true instead when every instance in the subject set satisfies the predicate.
- `no` is true when the predicate is fixed at 0, by the same test.
- `most` is true when the proportion of instances in the subject set that satisfy the predicate is at least `quantifiers.most.min_proportion` (default 0.7).
- `some` is true when that proportion is above 0. With `quantifiers.some.exclude_all` on (the default), `some` is used only when `all` is false, following the usual implicature. The rule limits what documents state. It does not make a `some` sentence false.
- `generic` means whatever `quantifiers.generic.means` says: `all`, `most`, or `some`. "Penguins swim" can therefore mean all, most, or some penguins swim, set by a parameter.

For each subject and predicate, a document states the strongest true quantifier: `all` before `most` before `some`. A scalar pole in a restriction holds no binary feature fixed, so the fixed test ignores it.

For verbs, the proportion is the proportion of pairs of distinct instances, an agent from the subject set and a patient from the patient category, for which the relation holds. The proportion is computed from the relation matrices (`result.relations.matrix`), for any pair of categories, at the same level or not, and for any verb or verb category. A verb category's relation is its base relation. The proportions are not limited to the rows of `relation_proportions.csv`. `all` means a proportion of 1 over the existing pairs.

For membership ("penguins are birds"), the predicate is true exactly when the predicate category is an ancestor of the subject category. Membership sentences take `all` or `generic` only, and the generic of a membership sentence always means all. A negative membership sentence ("no penguins are fish", "penguins are not fish") is true exactly when the two categories share no instance.

For exposed patient projections ("most mice are edible"), truth for `most`, `some`, and the generic comes from the share of instances in the subject set that have the projection. The share uses the same instance values as the instance-level sentences. A projection has no fixed test, so `all` and `no` are allowed only under the observed reading (`quantifiers.all_grounding: observed`).

For scalar poles, the comparison class is the subject category's parent: "elephants are big" means the category's mean value lies at least `scalar_adjectives.z` standard deviations above the mean of the parent category's instances. A top-level category has no parent, and is compared with all instances in the world. A class-level scalar pole takes the generic only, never a quantifier word. The category's mean value is the mean over its subject set. A comparison class whose values do not vary has no poles.

**Restricted subjects.** A relative clause on a class-level noun phrase is restrictive: the clause narrows the subject set. Three kinds are drawn: a CAN-feature clause ("penguins that can swim"), a subject relative with a verb ("owls that eat mice"), and an object relative ("mice that owls eat"). A clause about another category means at least one member of that category. So "owls that eat mice are big" has as its subject set the owls that can eat at least one mouse, and "mice that owls eat" are the mice that at least one owl can eat. With such a subject, `most`, `some`, and the generic are judged by the share of the subject set. `all` and `no` are allowed only under the observed reading, as for patient projections. A clause on a verb's patient category narrows the patient set in the same way.

Restrictions and relative clauses on class-level noun phrases come only from the proposition layer, where their truth is grounded. In an encyclopedic document, a subject takes one more literal at `propositions.restriction_rate` ("red penguins"), and a relative clause at `mention.relative_clauses.rate`. Both are drawn among those that some members of the category satisfy, and not all, so the restriction does work. The fact is then drawn for the restricted subject, so the sentence is true of it.

**Rule statements.** A class-level proposition can state a rule of the world. For a determined feature, each term of the minimal DNF of its rule is a sufficient condition. The proposition takes the generic noun "thing" as its head, the term's literals as its restriction, and the determined feature as its predicate: "things with wings and with feathers can fly". A rule statement is true with `all` by construction. The truth test above confirms it. A rule statement's quantifier is drawn like that of any class-level proposition: `all`, or the generic at the generic rate. The logical form then matches the surface: "all things with wings and with feathers can fly" is `ALL`, and the bare plural is `GEN`. A rule statement is true under both. A positive IS literal of the term is an adjective, and a HAS literal is a with-phrase or a without-phrase. A negated IS literal is realized as a subject relative clause: "things with wings that are not red can fly". A relative clause can join several verb phrases with "and", so all of a term's negated IS literals go into one relative clause: "things with wings that are not red and not big can fly". The same holds for the negated IS literals of any restriction. A term is skipped when it reads a scalar threshold, which no pole adjective states. A term that no instance satisfies is skipped too, because the statement would be vacuous. `stats.yaml` reports how many terms were skipped, and why.

Rule statements are exempt from the limits on adjectives, with-phrases, and content words (`mention.max_adjectives`, `mention.max_with_phrases`, `mention.max_content_words`). Rules of varied complexity are a core feature of the world, so long terms must still be stated. An optional cap, `propositions.rule_statements.max_literals`, is null by default (no cap). When a cap is set, a term with more literals is skipped and counted.

### Instance level

An instance-level proposition says something about one instance: "the penguin can swim", "the penguin has stripes", "the penguin is a bird", "the owl can eat the mouse".

- **Subject:** an instance.
- **Predicate:** IS, HAS, or CAN feature; exposed patient projection ("is edible"); scalar pole; membership in the instance's leaf or in any category above it; or a verb capacity with a patient instance ("can eat the mouse"), for a verb or a verb category ("can hunt the mouse").
- **Truth:** read from the instance's values. Verb capacity is `result.relations.holds`. Scalar poles use the category named by the subject's noun as the comparison class: "the big mouse" is big for a mouse, and "the small animal" can be the same mouse.

### Event level

An event-level proposition says that something happened in a scene: "the penguin swims", "the owl chases the mouse".

- **Content:** an event from the scene generator, which records the scene, the time step, the verb, the agent, and the patient if any. The proposition's verb is the event's own verb, or a verb category above it (see "Verb level" under "Mentioning referents").
- **Tense and aspect:** every event has a tense and an aspect, and both are part of its logical form. The tense is the same for every event: `propositions.events.tense`, past by default. The aspect is simple or progressive. The aspect is drawn once for each event, at `propositions.events.progressive_rate`, so every report of one event has the same aspect. The aspect is drawn for every event, even when the grammar does not mark aspect. The grammar decides only whether and how tense and aspect are marked (see "Morphology").
- **Truth:** the event occurred. Events are generated only where the world allows them (see "Scenes and events"), so an event never contradicts a capacity. A proposition that names no event (a test item) is true when such an event, of the same aspect, occurred at any time step of its scene. The grounding also says whether the world allows the event (`possible`): the agent has the CAN feature, or the verb's relation holds for the agent and the patient.

Event-level propositions are never negated.

### Negation

Class-level and instance-level propositions can be negative: "penguins cannot fly", "the owl has no fins", "no fish have fur". Negative propositions are true negations, grounded by the same tests with the predicate's value 0. The rate of negative propositions is a parameter for each level. The default is 0.1.

In the logical form, a negative polarity denies the predicate: `most` with a negative polarity says that most members of the subject set lack the predicate ("most penguins can not fly"). The quantifier `no` replaces sentence negation. So `no` always has a positive polarity in the logical form and counts as a negative proposition, and `all` never has a negative polarity. For a negative fact, a document states `no` before "most ... not" before "some ... not".

### False propositions and test sets

Training documents contain true propositions only. False propositions go into separate test sets, each item labeled true or false. A false proposition is made from a true one by one minimal change:

- **predicate swap:** replace the predicate with one of the same kind that makes the proposition false;
- **subject swap:** replace the subject with a category or instance of the same level for which the proposition is false;
- **quantifier swap:** replace the quantifier with one that makes the proposition false ("all" for a "most" fact);
- **role swap** (verbs only): exchange agent and patient, when the reversed relation does not hold.

A predicate swap keeps the predicate's kind, its patient, and its comparison class: a verb is replaced by another verb or verb category, and a category by another category of its level. A quantifier swap keeps the polarity, so a proposition with a negative polarity never becomes a `no` sentence. A false item is never vacuous: its subject set has at least one instance. Every false item is checked false by the same truth tests. Test sets come in matched pairs: each false item sits beside the true item it was made from. There is one test set for each proposition level and change type, with a configured size.

**The sets.** The class-level sets are `class_predicate`, `class_subject`, `class_quantifier`, and `class_role`. Each one has a law-like twin (`class_predicate_lawlike`), except the role swap, because a verb is never judged by the fixed test. The instance-level sets are `instance_predicate`, `instance_subject`, and `instance_role`. The event-level sets are `event_predicate`, `event_subject`, and `event_role`, each split in two (`event_predicate_possible` and `event_predicate_impossible`). `test_sets.changes` chooses the changes. A set holds at most `test_sets.size` pairs. A true item is used once in the sets of one level and change, so a set is smaller when the world runs out of true items.

**True items.** A true class-level item is drawn the way an encyclopedic document draws a sentence. The topic is a category, drawn by `documents.topic_level_weights`. The kind of content is a membership fact, a fact about the topic, a fact about a subcategory, a relation fact, or a rule statement, each with the same chance. The polarity, the restriction, and the bare generic are drawn at their rates. A true instance-level item is a fact about a referent of a narrative document, drawn as a description is.

Seven more rules shape the test sets.

- **Law-like items.** Under the law-like reading, a false `all` or `no` item can be one that no instance contradicts: every penguin in the world swims, but nothing fixes it. Such an item is marked, and goes into a test set of its own. The ordinary test sets then hold only items that observing the instances could decide. Only IS, HAS, and CAN features are judged by the fixed test, so only they give law-like items. Under the observed reading there are none.
- **True events.** A true event-level item is an event that its document reports, in a main clause or in a relative clause. Its verb is named at a level drawn by `mention.verb_level_weights`, like any report.
- **False events.** A false event-level item is an event that did not happen. Each item is marked possible (the world allows the event) or impossible (the world rules it out), and the two kinds go into separate test sets. The rule is stricter than the truth test. No event with the item's verb, agent, and patient happened in any scene of the item's document, in either aspect. When the item names its verb with a verb category, no event of any verb below that category matches either. The test sentence names neither its scene nor, in most languages, its aspect. A weaker rule would give false items whose words are true of the document.
- **Context.** Every instance-level and event-level test item names a narrative document. The item is tested as a continuation of that document, so its noun phrases are definite mentions of referents that the document has already mentioned. A subject swap brings in another referent of the same document. A test item has no pronoun, no relative clause, and no modifier beyond those that tell its referents apart.
- **Scene form.** Every event-level test item, true or false, names only its scene: `EVENT(SN.8, PAST, SIMPLE, V1.2(R.1, R.2))` means that some event of scene `SN.8` was this one. Documents keep their event labels. Only test items use the scene form, so a true item cannot be recognized by having a label.
- **One format.** The true and the false items of a test set never differ in format: every model-facing field that a true item has, a false item has too, in the same notation. The model-facing fields are the document, the sentence, its renderings, its tree, and its logical form. The metadata differ by design: the truth label, the grounding, and the marks. An item keeps the two apart (see "Test items" under "Outputs"). A test checks the rule for every test set, and the generator stops when a pair breaks it.
- **Seen and unseen.** The test-set settings never change the documents, so no item is held out of the corpus. Every item records instead whether its logical form appears in a training document (`seen`). A false item never does. A document states the proposition of every main clause. In a sentence about instances, a document also states the proposition of every relative clause, and what every noun phrase says of its referent: its noun and its modifiers. `stats.yaml` reports the share of true items that are seen, for each test set.

## Scenes and events

Event-level sentences and situational documents need things to happen. The scene generator is a simple stand-in until event schemas or world-simulation logs can supply events.

**Participants.** A scene starts with a seed instance. The generator then draws `scene.size` other instances, each with probability proportional to a weighted sum over its leaf and the seed's leaf:

- the thematic relatedness of the two leaves (`thematic.csv`);
- the taxonomic similarity of the two leaves;
- a constant.

The weights are parameters. With verbs off, the thematic weight has no effect. The instances are drawn without replacement, and the seed is never drawn again. An instance whose weighted sum is 0 is never drawn, so a scene can be smaller than `scene.size`. The taxonomic similarity is the one in `thematic.csv`: the configured metric over the leaves' generative vectors. An undefined or negative similarity counts as 0.

**Timeline.** A scene has a number of time steps drawn from `scene.steps`. At each step, the number of events is drawn from a Poisson distribution with mean `scene.events_per_step`. Each event is drawn from the pool of possible events among the participants:

- an intransitive event for every participant and every CAN feature the participant has;
- a transitive event for every ordered pair of distinct participants and every verb whose relation holds for the pair.

Events are drawn verb first. `scene.transitive_share` decides the kind first: an event is transitive with that probability, when both kinds of event are possible. Then a CAN feature or a leaf verb is chosen among those with at least one possible event in the scene, weighted by `scene.verb_weights` (uniform by default). Then the participants are chosen uniformly among the pairs, or the agents, for which that verb's event is possible. So a verb that holds for many pairs is no more frequent than a verb that holds for few. `scene.verb_weights` maps CAN features and verbs to weights, and a label that is left out has the weight 1. The same event does not occur twice at one time step, but the same event can occur again at a later step. Nothing changes state: events do not alter the participants, so the pool of possible events is the same at every step.

**Aspect.** Each event is progressive with probability `propositions.events.progressive_rate`, and simple otherwise. The aspects of a scene are drawn after its events, from the scene's own part of the stream. So the rate never changes the participants or the events.

**Scenes and the lexicon.** Scenes are a fact about the world, not about the language. The generator uses every CAN feature and every verb (the leaves of the verb tree), with a word or without one, so the lexicon settings never change a scene. The planner leaves out an event that no word can report. Each scene draws from its own part of the `corpus:scenes` stream, named by its label, so a scene depends only on the corpus seed, its number, its seed instance, and the scene settings.

**Labels.** Events are numbered within their scene in time order, and within a time step in the order they were drawn.

## Layer 3: documents

### Document types

The corpus mixes four document types. Their proportions are parameters.

**Encyclopedic, about a category.** The topic is a category, drawn with configured weights over levels. Each sentence draws one kind of content, each kind with the same chance: a membership fact, a fact about the topic, a fact about a subcategory, or a relation fact. The content pool holds:

- membership facts: the topic's ancestors ("penguins are birds") and its children ("emperor penguins are penguins");
- class-level facts about the topic: defining features (quantified `all` or generic), characteristic features (`most` or generic), and rarer features (`some`);
- class-level facts about the topic's subcategories;
- relation facts with the topic as agent or as patient;
- contrasts with sibling categories. At `documents.sibling_contrast_rate` (default 0.2), a fact about the topic is followed by the matching fact about a sibling category: the same predicate, where the sibling differs. A contrast is two adjacent sentences, one about each category ("penguins can not fly", then "gulls can fly"). A fact has a contrast only when it is a strong fact (`all`, `no`, `most`, or a scalar pole) and the sibling's fact of the other polarity is strong too. With no differing sibling, no contrast sentence is added. A contrastive construction ("unlike penguins, gulls can fly") is out of scope.

**Encyclopedic, about a feature.** The topic is an IS, HAS, or CAN feature, or a verb. The content pool holds:

- which categories have the feature, at several levels;
- which categories lack it (negations);
- rule statements whose predicate is the topic feature (sufficient conditions for it);
- rule statements in which the topic feature appears in the restriction (what the topic feature makes possible).

**Entity narrative.** The topic is one instance. The generator builds `entity.scenes` scenes seeded at the instance: 2 to 5 by default. The document interleaves:

- instance-level facts about the instance;
- the events of its scenes that involve the instance, in time order, scene by scene.

**Situational narrative.** One scene. The document introduces the participants as they first take part in events, narrates the events in time order, and adds instance-level descriptions of participants at a configured rate.

In both narratives, an event sentence is followed by a description at `documents.instance_description_rate`. In an entity narrative, the description is about the topic instance. In a situational narrative, the description is about a participant of the event, and a participant that the sentence introduced comes first. The patient of a verb in a description is another participant of the document's scenes.

Each document's length is drawn from a configured range for its type. A document can be shorter than its drawn length: a narrative ends when its events run out, and an encyclopedic document ends when it has nothing more to say. A document with no sentence is drawn again. A document states a proposition once. The polarity of each class-level fact is drawn at the negation rate and kept, so the share of negative sentences does not rise when the positive facts run out.

**Quantifier weights.** A fact is always stated with its strongest true quantifier. `quantifiers.weights` gives a weight to `all`, `most`, `some`, and `none` (the quantifier `no`). A document chooses a fact with a probability proportional to the weight of the fact's strongest true quantifier. So a study can rebalance the mix: a low weight for `some` gives fewer `some` sentences. Equal weights, the default, change nothing. Five details:

- the polarity of a fact is still drawn first, at the negation rate, so the weight of `none` moves the mix among the negative facts;
- a scalar pole takes no quantifier word, and has the weight 1;
- rule statements are not reweighted, because `propositions.rule_statement_rate` sets how often they appear;
- a weight of 0 leaves the quantifier's facts out, sibling contrasts included;
- the key is `none`, because YAML reads a bare `no` as the boolean false.

### Ordering

Encyclopedic documents follow a loose template: membership first, then defining facts (`all` and `no`), characteristic facts (`most`, and scalar poles), rarer facts (`some`), relation facts, and rule statements. A shuffle parameter moves from the strict template (0) to a random order (1): each sentence is displaced by a random amount scaled by the parameter. A sibling contrast stays right after the fact it matches. Narratives follow time order, with descriptions inserted near the referent's first mention.

### Mentioning referents

- **Noun level.** Noun levels apply to instances only. An instance is named with the noun of its own leaf or of any category above it ("the bird"). The level is drawn for each mention. `mention.level_weights` give the weights, with the leaf level heaviest by default. The logical form records which category the noun names. A class-level noun phrase is always named by its own noun, because a higher noun would make a different proposition. Propositions about higher categories come from the proposition layer.
- **Verb level.** An event's verb is named at a level of the verb tree, in the same way: the verb itself, or a verb category above it ("chase" or "hunt"). `mention.verb_level_weights` give the weights, with the leaf level (the verb) heaviest by default. Words for verb categories are also used in class-level and instance-level capacity sentences ("owls hunt mice"), whose truth is the verb category's own base relation.
- **First and later mentions.** In narratives, an instance is introduced with the indefinite determiner ("a penguin") and mentioned later with the definite determiner ("the penguin") or a pronoun.
- **Pronouns.** A later mention becomes "it" at `mention.pronoun_rate`, when the referent was mentioned in the previous sentence and was either the only referent mentioned there or its subject. A pronoun stands only in the main clause, never inside a relative clause. Coreference chains are recorded, including for ambiguous pronouns.
- **Referent labels.** The referents of a document are labeled `R.1`, `R.2`, and so on, in the order of first mention. The order is that of the logical form (the subject, the noun phrases of its relative clause, then the object), so the word order never changes a label.
- **Relative clauses and limits.** The planner decides everything that changes what a sentence says. So the planner, and not the grammar, decides which noun phrases take relative clauses (`mention.relative_clauses`), and applies the limits on adjectives, with-phrases, and sentence length (`mention.max_adjectives`, `mention.max_with_phrases`, `mention.max_content_words`). The grammar section describes how they are realized.
- **Distinguishing modifiers.** When the document's scenes have two or more participants that the chosen noun fits, a definite mention adds adjectives or with-phrases until it picks out one referent, following the incremental algorithm of Dale and Reiter (1995), with the preference order fixed per language. The distractors are all participants of the document's scenes. When the modifiers cannot tell the referent apart, the mention falls back to the noun of the leaf. A mention that still fits another participant is kept, marked as not distinguished, and counted. In situational documents, adjectives therefore do referential work only.
- **Other modifiers.** Modifiers at `mention.modifier_rate` go on instance mentions only, in entity narratives. A modifier is chosen from the features true of the referent. Restrictions on class-level subjects come only from the proposition layer, where their truth is grounded.

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
RC   → that VP and VP ...                  (a subject relative that joins verb phrases)
VP   → V | V NP                            (intransitive, transitive)
VP   → can V | can V NP                    (capacity)
VP   → can not V | can not V NP            (negated capacity)
VP   → is A | is not A                     (property, scalar pole, exposed projection)
VP   → has N | has no N                    (part)
VP   → is a N | is not a N                 (membership)
```

Quantifiers occupy the determiner slot: "all penguins", "no fish". The negative class-level quantifier "no" replaces sentence negation: "no fish have fur".

**`can` and the bare verb.** A positive capacity can be said with `can` or with the bare verb. Both say the same. `grammar.can_rate` gives the share that take `can`: `class` for class-level capacities (default 0.5: "penguins can swim" or "penguins swim"), and `instance` for instance-level capacities (default 1.0: "the penguin can swim"). The rate applies inside relative clauses too. A negative capacity always keeps `can`, because `not` needs it. With an instance rate below 1, an instance capacity can appear without `can`. Together with tense and number off, "the penguin swim" is then ambiguous between a capacity and an event (see "Logical form and surface form").

**The tree.** The parse tree is a nested list, and its leaves are the sentence's tokens. Its labels are `S`; `NP-SBJ` and `NP-OBJ`, the subject and the object; `VP`; `NP-PRD`, the noun of a "has" or "is a" predicate; `AP`; `PP`, a with-phrase; `RC`; and the words `N`, `V`, `A`, `Det`, `Pro`, `P`, `Conj`, `Rel`, `AUX`, and `Neg`. The subject and the object are labeled by their function, so a tree reads the same in every word order. When the verb and the object are neighbors (SVO, SOV, VOS, OVS), the object is inside the verb phrase. When the subject stands between them (VSO, OSV), the object is a daughter of `S`.

**The noun phrase.** A positive IS literal and a scalar pole are adjectives. A HAS literal is a with-phrase or a without-phrase. The negated IS literals share one relative clause. The noun phrase is built outward from the noun: the adjectives, then the determiner, then the with-phrases, then the relative clause. Each one is placed before or after what is already there, by its word-order setting.

**Relative clauses.** A noun phrase takes a relative clause at `mention.relative_clauses.rate`. The clause is an object relative ("the mouse that the owl eats") with probability `mention.relative_clauses.object_share`, and a subject relative ("the owl that eats mice") otherwise. A relative clause's own noun phrases can take relative clauses up to `mention.relative_clauses.max_depth`. The planner makes these choices, and the grammar realizes them. Depth 2 or more produces center embedding in subject position, and with it controllable long-distance dependencies. A relative clause expresses a true proposition of the same level as its sentence, about the same referent, and the logical form records it as a restriction. A subject relative can join several verb phrases with "and": "things that are not red and not big". The negated IS literals of one restriction always share one relative clause. A verb phrase with the same auxiliary as the one before it does not say the auxiliary again. A noun phrase has at most one relative clause. A relative clause holds CAN features, verbs, exposed patient projections, and membership, beside the negated IS literals. It never holds a positive IS literal, a scalar pole, or a HAS literal, which are adjectives and with-phrases. `that` comes first in every relative clause. An object relative is the clause's subject and its verb, in the order of `word_order.clause`.

A drawn relative clause expresses one of these propositions (`mentions.py`, and `facts.py` for a class-level sentence):

- in an event-level sentence, another event of the same scene that the referent takes part in: as its agent (a subject relative, "the dog that chased the cat") or as its patient (an object relative, "the cat that the dog chased");
- in an instance-level sentence, a capacity of the referent: a CAN feature it has, or a verb's relation with another referent of the document, as agent ("the owl that can eat the mouse") or as patient ("the mouse that the owl can eat");
- in a class-level sentence, a restriction of the category: a CAN feature ("penguins that can swim"), a verb with a patient category ("owls that eat mice"), or a verb with an agent category ("mice that owls eat"). The clause narrows the subject set, so it is drawn with the proposition (see "Restricted subjects"). A subject relative holds a CAN feature or a verb, each with the same chance.

When the drawn kind of clause has no true proposition, the other kind is used. A relative clause never repeats what its sentence already says. In a document, an event-level clause reports an earlier event of the same scene.

**Adjective order.** When a noun takes several adjectives, they appear in a fixed order: a random ordering of adjective concepts drawn once per language. `adjective_order.fixed: false` makes the order random for each phrase.

**Limits.** `mention.max_adjectives` (default 3), `mention.max_with_phrases` (default 2), and `mention.max_content_words` (default 20). The planner applies the limits: a proposition that would pass a limit gets fewer optional modifiers. A sentence over the content-word limit is said again without its drawn relative clauses. The modifiers that tell referents apart are kept. The sentence limit counts content words only (nouns, adjectives, and verbs), because the number of function words depends on the morphology settings, and a grammar setting must never change what a sentence says.

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

The auxiliary stands before or after the predicate word: the verb, the adjective, or the predicate noun ("has fins", "is a bird"). `not` stands right after or right before the auxiliary (`negation: after_auxiliary` or `before_auxiliary`). Every negated verb phrase has an auxiliary. The auxiliary, `not`, and the predicate word stay together, and the object stands before or after them by `clause`. In "has no fins", `no` is the determiner of the part noun, and follows `determiner`.

### Morphology

Number, tense, and aspect are each off by default. When on, each is realized either as an affix joined to the word form (`realization: affix`, "swim-s") or as a separate function word (`realization: word`), set separately for each.

- **Number:** singular and plural on nouns. Generic subjects are plural. With number off, nouns and verbs have one form: "all penguin swim". Every class-level noun is plural, and every instance is singular. A part noun ("with fins", "has fur") is never marked. A plural predicate noun takes no `a`: "penguins are birds".
- **Agreement:** with number on, verbs agree with their subjects unless `number.agreement` is false. The verb takes the same affix or word as a plural noun. `number.verb_marks` says which verbs take it: `plural` (the default) or `singular`, as in English ("the penguin swims", "penguins swim"). The auxiliaries `is` and `has` agree by switching to the function words `are` and `have`. `can` does not agree, and the verb after `can` is not marked. A verb that carries a tense or aspect marker takes no agreement marker, as in English "chased". Agreement holds across relative clauses, so relative clauses in subject position create long-distance dependencies between subject and verb. In an object relative, the verb agrees with the clause's own subject.
- **Tense:** present and past. Class-level and instance-level propositions are present. An event's tense is part of its logical form (`propositions.events.tense`, default past). With tense on, a past event is marked. The present is never marked.
- **Aspect:** simple and progressive. An event's aspect is part of its logical form, and is drawn for every event (`propositions.events.progressive_rate`). With aspect on, a progressive event is marked. The simple aspect is never marked.

`grammar.morphology` holds only whether and how number, tense, and aspect are marked. It never decides what tense or aspect an event has.

A morphology word stands right after the word it marks, or right before it with `position: before`. A word takes at most one affix, because a word form has one. So tense and aspect cannot both be affixes when the event tense is the past: one of them must be a word, and the configuration is an error otherwise. Stacked affixes are a future addition. With every inflection off, the language is the one in your examples: "the penguin swim".

With English function words (the Jabberwocky option of the word-form pipeline), every function-word gloss must be an English word. So `realization: word` is an error with English function words, because `PLURAL` is not an English word. The English affixes are the suffixes for `PLURAL`, `PAST`, and `PROGRESSIVE` only.

### Logical form and surface form

The logical form of a sentence is never ambiguous. Capacity versus event, tense, aspect, and number are explicit in it, whatever the grammar settings. The propositional rendering writes the logical form (see "Renderings").

The surface language may be ambiguous. With `grammar.can_rate.instance` below 1 and the tense and the aspect unmarked, a capacity and an event have the same words: "the penguin swim". The two sentences also have the same tree. Only the sentence's record tells them apart: its `events` field and its logical form.

**Readings.** Each sentence records `readings`: the kinds of logical form that its surface string allows. There are three kinds, one for each level: `generic` (a class-level sentence), `capacity` (an instance-level sentence), and `event` (an event-level sentence). A sentence without a verb ("the penguin is red") has the one reading of its level. A noun phrase shows whether it names a category or an instance, so a class-level sentence has the one reading `generic`. A sentence about instances can have the two readings `capacity` and `event`.

Readings are worked out from the tree and the lexeme tokens alone, never from the sentence's `events` or its logical form. The tree carries no information about the reading: a capacity without `can` and an unmarked event have the same tokens and the same tree. Lexemes are read, not word forms, so homonyms do not count as ambiguity. The readings of a sentence are those that every one of its verb phrases allows:

- a verb phrase with `can` or `not`, and one whose predicate word is not a verb ("is red", "has fins", "is a bird"), states a capacity. A relative clause "that is not red" belongs to the restriction of its noun phrase, and allows both readings;
- a verb marked for tense or aspect reports an event;
- a bare verb states a capacity when the language lets a capacity drop `can` (`grammar.can_rate.instance` below 1). A bare verb reports an event when the language has events that no marker shows: the tense is the present or is not marked, and the simple aspect occurs or the aspect is not marked.

`stats.yaml` reports how often sentences are ambiguous.

### Scalar adjectives

Each scalar dimension has two adjectives, one for each pole. An instance counts as "big" when its value is at least `scalar_adjectives.z` standard deviations above the mean of its comparison class, and "small" when at least that far below. The comparison class is the category named by the noun (instance level) or the subject category's parent (class level). Comparatives ("bigger than") are out of scope for now.

## Renderings

Every sentence is written in up to four renderings. The first three need no word forms, so a whole corpus can be produced in any of them alone.

- **Formal:** lexeme labels with glosses, for example `the/L.176 C1.3/L.5 CAN.2/L.138` for "the penguin swim". The gloss of a content lexeme is its concept's label (`C1.3`, `CAN.2`), and the gloss of a function word is its English gloss (`the`). An inflected word is followed by its affix's gloss: `C1.3/L.5-PLURAL`. Its token is `L.5-PLURAL`.
- **Conceptual:** the sentence's words in order, each replaced by its concept label: `THE IS.12 C1.3.2 CAN V1.2 THE C1.4.1` for "the furry dog can chase the cat". An inflected word is its concept label joined to the affix's gloss: `C1.3.2-PLURAL`. The conceptual rendering is the same sentence as the spelled rendering, with the same tree, word order, morphology, and ambiguities. Only the lexical forms differ: concept labels instead of word forms. The conceptual rendering carries different information from the formal rendering only when synonyms are on, because synonyms share a concept label. Homonyms are two lexemes, so both renderings tell a homonym's two meanings apart.
- **Propositional:** the sentence's logical form, written out as atomic propositions joined by `AND`. Every modifier, with-phrase, relative clause, and noun becomes its own proposition, so one sentence is usually several propositions. Determiners and pronouns disappear, and referents are named by label. The propositional rendering is the logical form, and it is never ambiguous. Its format is below.
- **Spelled:** the word forms' readable spellings, after word forms are attached.

**The propositional format.** The propositional rendering is made from the JSON logical form alone, and parses back. A class-level rendering gives back the whole proposition, with its restrictions and relative clauses. A rendering about instances gives back the proposition of the main clause, and the other propositions in order. The noun phrase that a relative clause hangs on is recorded in the JSON form and in the tree. Capacity versus event, tense, aspect, and number are explicit, whatever the grammar settings.

- **Atoms.** An atomic proposition is a concept label with its arguments: `C1.3.2(R.1)` (membership), `IS.12(R.1)`, `HAS.4(R.1)`, and `CANBE.V1.1(R.1)`. `NOT` before an atom negates it.
- **Scalar poles.** A scalar pole names its comparison class as a second argument, at both levels. At the instance level, the comparison class is the category that the noun names: `SC.1.HIGH(R.1, C1.5)`. A class-level pole names the comparison class of decision 22: the subject category's parent, as in `SC.1.HIGH(X.1, C1)`, or `THING` for a top-level category. A pole in a restriction ("big penguins") names the subject's own category: `SC.1.HIGH(X.1, C1.3)`.
- **Capacities.** A capacity is wrapped in `ABLE`: `ABLE(CAN.3(R.1))` and `ABLE(V1.2(R.1, R.2))`, with the agent first. `ABLE` wraps only CAN features and verbs. A negative capacity is `NOT ABLE(...)`.
- **Events.** An event is `EVENT(<event label>, <tense>, <aspect>, <atom>)`. Tense is `PAST` or `PRESENT`, and aspect is `SIMPLE` or `PROGRESSIVE`: `EVENT(SN.8.5, PAST, PROGRESSIVE, V1.2(R.1, R.2))` and `EVENT(SN.8.5, PAST, SIMPLE, CAN.7(R.1))`. The tense and the aspect are always written, even when the surface does not mark them. A test item writes the label of its scene in place of the event label: `EVENT(SN.8, PAST, SIMPLE, V1.2(R.1, R.2))` says that some event of scene `SN.8` was this one.
- **Number.** `R.n` is always one individual, and `X.n` is a variable bound by a quantifier. If a construction ever gives a plural individual referent, we write it `R.n:PL`. None exist yet.
- **Quantifiers.** A class-level proposition is a quantifier with a restrictor and a scope over variables: `MOST(C1.3(X.1) AND IS.4(X.1), ABLE(CAN.3(X.1)))` for "most red penguins can swim", with `ALL`, `MOST`, `SOME`, `NO`, and `GEN` (generic). A verb's patient category stands in the restrictor with a variable of its own, so the quantifier ranges over pairs. Inside a quantifier, a capacity is wrapped in `ABLE` too: `GEN(C1.3(X.1), ABLE(CAN.3(X.1)))`.
- **"At least one".** A relative clause about another category is `EXISTS(X.n, ...)` inside the restrictor: `EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2)))` for "that eat mice". A CAN-feature clause is `ABLE(CAN.3(X.1))` in the restrictor.
- **Variables.** Variables are labeled `X.1`, `X.2`, and so on, numbered within each sentence in the order of the logical form: the subject, its relative clauses, then the patient and its relative clauses. So a relative clause at any depth can introduce a new variable, and the word order never changes the rendering.
- **Order.** In a sentence about instances, each noun phrase gives its noun and its modifiers, then the propositions of its relative clause. The proposition of the main clause comes last. A proposition that would be written twice is written once.

Examples:

| Sentence | Propositional rendering |
| --- | --- |
| the furry dog has legs | `C1.3.2(R.1) AND IS.12(R.1) AND HAS.4(R.1)` |
| the penguin can swim | `C1.3(R.1) AND ABLE(CAN.3(R.1))` |
| the penguin swam | `C1.3(R.1) AND EVENT(SN.8.5, PAST, SIMPLE, CAN.3(R.1))` |
| the dog that chased the cat ran | `C1.3.2(R.1) AND C1.4.1(R.2) AND EVENT(SN.3.2, PAST, SIMPLE, V1.2(R.1, R.2)) AND EVENT(SN.3.4, PAST, SIMPLE, CAN.7(R.1))` |
| the big mouse is edible | `C1.5(R.1) AND SC.1.HIGH(R.1, C1.5) AND CANBE.V2.1(R.1)` |
| all things with wings and with feathers can fly | `ALL(HAS.2(X.1) AND HAS.5(X.1), ABLE(CAN.1(X.1)))` |
| things with wings and with feathers can fly | `GEN(HAS.2(X.1) AND HAS.5(X.1), ABLE(CAN.1(X.1)))` |
| most penguins can not fly | `MOST(C1.3(X.1), NOT ABLE(CAN.1(X.1)))` |
| owls eat mice | `GEN(C1.2(X.1) AND C1.5(X.2), ABLE(V2.1(X.1, X.2)))` |
| owls that eat mice are big | `GEN(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2))), SC.1.HIGH(X.1, C1))` |
| most mice that owls eat are red | `MOST(C1.5(X.1) AND EXISTS(X.2, C1.2(X.2) AND ABLE(V2.1(X.2, X.1))), IS.4(X.1))` |
| penguins that can swim have fins | `GEN(C1.3(X.1) AND ABLE(CAN.3(X.1)), HAS.2(X.1))` |

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
| `wordform_request.yaml` | The request for the word-form pipeline (see "Word forms for the corpus"). Stage 7 adds it. |
| `scenes.jsonl` | One JSON object per scene: its label, its seed instance, its participants (the seed first), and `steps`, a list with one list of events for each time step. An event holds its label, verb, agent, patient (null for an intransitive event), and aspect (`simple` or `progressive`). |
| `tests/<level>_<change>.jsonl` | Test sets: matched true and false items with their logical forms and sentences. The law-like items (`_lawlike`), the possible false events (`_possible`), and the impossible false events (`_impossible`) have test sets of their own. An instance-level or event-level item names its document. The schema is under "Test items" below. |
| `stats.yaml` | Counts by document type, proposition level, quantifier, and part of speech; each document type's achieved length beside its drawn length; the quantifier mix, as stated and by each fact's strongest true quantifier; lexeme frequencies; sentence lengths; relative-clause depths; how often sentences are ambiguous, by their readings; how many definite mentions could not be told apart; the verbs that get no word because their relation holds for every pair or for no pair; the rule terms that were skipped; the co-occurrence check below; and, for each test set, its number of pairs and the share of its true items that are seen. |

Each document object holds its label, type, topic, scenes, referents, and sentences. The topic is a category, a feature or a verb, an instance, or, for a situational narrative, its scene. `referents` maps each referent label to its instance, in the order of first mention. Each sentence holds:

- `label`;
- `tokens`: lexeme labels;
- `words`: word form labels, or null;
- `text`: the spelled rendering, or null;
- `formal`: the formal rendering;
- `conceptual`: the conceptual rendering;
- `propositional`: the propositional rendering;
- `tree`: the parse tree, as a nested list of the form `[label, child, ...]`;
- `logical_form`: the proposition (schema below);
- `referents`: for each noun phrase, the instance or category it refers to (`referent`) and the category its noun names (`noun`, null for a pronoun), in the order of the tree, a node before its children;
- `events`: for each verb phrase, the label of the event it reports, or null, in the same order;
- `coreference`: for each noun phrase, in the order of `referents`, the chain it belongs to: its referent's label (`R.2`), or null for a noun phrase that names a category;
- `distinguished`: for each noun phrase, in the same order: for a definite mention with a noun, whether the noun and the modifiers pick out the referent alone among the participants of the document's scenes. Null for any other noun phrase;
- `readings`: the kinds of logical form that the sentence's surface string allows (see "Logical form and surface form").

A logical form, for example:

```json
{"id": "PR.310", "level": "class", "quantifier": "most", "polarity": true,
 "subject": {"category": "C1.3", "restriction": ["IS.4", "not HAS.2"]},
 "predicate": {"kind": "can", "feature": "CAN.3"},
 "grounding": {"proportion": 0.93, "fixed": false, "test": "observed"}}
```

The subject of an instance-level proposition is `{"instance": "I1.3.2.5"}`, and an instance-level proposition has no quantifier. The predicate takes one of these forms:

| Kind | Predicate |
| --- | --- |
| `is`, `has`, `can` | `{"kind": "can", "feature": "CAN.3"}` |
| `projection` | `{"kind": "projection", "projection": "CANBE.V1.1"}` |
| `scalar` | `{"kind": "scalar", "pole": "SC.1.HIGH"}`, with `"class": "C1.3"` at the instance level: the comparison class, which the subject's noun must name |
| `member` | `{"kind": "member", "category": "C1"}` |
| `verb` | `{"kind": "verb", "verb": "V1.2", "patient": {"category": "C1.5", "restriction": []}}` at the class level, and `"patient": {"instance": "I1.5.1.2"}` at the instance level |

An event-level proposition has the instance-level form, with `"level": "event"`, the scene (`"scene": "SN.8"`), the event it reports (`"event": "SN.8.5"`), and its tense and aspect (`"tense": "past", "aspect": "progressive"`). Its predicate is a CAN feature, or a verb with a patient instance. Its grounding holds the scene, the time step, `possible`, and the test `event`.

**Relative clauses in a category term.** A subject or a patient category with relative clauses has a `clauses` list. A clause is written like a predicate: `{"kind": "can", "feature": "CAN.3"}`, or a verb with the other category. The other category is the `patient` when the head is the agent ("owls that eat mice"), and the `agent` when the head is the patient ("mice that owls eat"):

```json
{"category": "C1.2", "restriction": [],
 "clauses": [{"kind": "verb", "verb": "V2.1", "patient": {"category": "C1.5", "restriction": []}}]}
```

The other category is a category term too, so it can have a restriction and clauses of its own. `clauses` is left out when there is none. In a sentence's logical form, the predicate of a class-level scalar pole also holds its comparison class (`"class": "C1"`, or `"THING"` for a top-level category).

**Mentions.** In a sentence's logical form, the subject of an instance-level or event-level proposition, and the patient of its verb, are mentions:

```json
{"instance": "I1.3.2.5", "referent": "R.1", "noun": "C1.3.2", "restriction": ["IS.12"],
 "clauses": [{"kind": "verb", "verb": "V1.2",
              "patient": {"instance": "I1.4.1.2", "referent": "R.2", "noun": "C1.4.1", "restriction": []},
              "event": "SN.3.2", "tense": "past", "aspect": "simple"}]}
```

`referent` is the referent's label in its document. `noun` is the category that the noun names, or null for a pronoun. `restriction` holds the modifiers. `clauses` holds the propositions of the relative clause. Each one is written like a predicate, with its other mention as `patient` (a subject relative) or `agent` (an object relative), with `"polarity": false` when it is negative, and, at the event level, with the event it reports and its tense and aspect. A determiner is no part of the logical form. The propositional rendering is made from this JSON form alone.

A rule statement also has `"rule": {"feature": "CAN.1", "term": 2}`: the determined feature, and the number of the term of its rule's minimal DNF. A restriction is written in one order: IS literals, HAS literals, then scalar poles, each by index.

The grounding says how the truth was decided (`test`):

- `exact`: the fixed test, by enumerating the cone. `local`: the fixed test for a cone too large to enumerate;
- `observed`: a proportion over the instances of the subject set (`instances`), or over pairs of an agent and a patient (`pairs`);
- `tree`: membership, read from the tree;
- `mean`: a class-level scalar pole, with the subject set's mean (`value`) and the comparison class's label, mean, and standard deviation;
- `value`: an instance's own value.

`proportion` is always the share that has the predicate, whatever the polarity. `fixed` says whether the predicate's value is fixed for the subject, for IS, HAS, and CAN features.

**Test items.** Each line of a test set is one item, and each true item is followed by the false item made from it. An item has two parts:

```json
{"input": {"document": "D.17", "tokens": ["L.175", "L.6", "L.166", "L.175", "L.4"], "words": null, "text": null,
           "formal": "...", "conceptual": "...", "propositional": "C1.1.4(R.1) AND C1.1.2(R.2) AND EVENT(SN.8, PAST, SIMPLE, V3(R.1, R.2))",
           "tree": ["S", "..."], "logical_form": {"level": "event", "scene": "SN.8", "event": null, "...": "..."},
           "referents": ["..."], "events": ["SN.8"], "coreference": ["R.1", "R.2"], "distinguished": [true, true], "readings": ["event"]},
 "meta": {"set": "event_role_possible", "pair": 12, "truth": true, "level": "event", "change": "role",
          "seen": false, "possible": true, "grounding": {"scene": "SN.8", "step": 1, "possible": true, "test": "event"}}}
```

- `input` holds the model-facing fields: the document that the item continues (null at the class level), and the fields of a sentence in `documents.jsonl`, without its label. The logical form has no `id`, no `grounding`, and no `rule`. An event-level form has `"event": null`, and `events` holds the label of the scene.
- `meta` holds the answer and the bookkeeping: the set, the number of the pair, the truth label (`truth`), the level, the change, `seen`, and the grounding. A class-level item also has `law_like`, and `rule`: the rule that a true rule statement states, or null. An event-level item also has `possible`.

The two items of a pair never differ in the format of `input`. They differ in `meta` by design.

**Co-occurrence check.** `stats.yaml` reports, over the unordered pairs of leaves, the correlation of within-document co-occurrence with thematic relatedness and with taxonomic similarity, separately for each document type. A pair's co-occurrence is the number of documents in which both leaves occur. A leaf occurs in a document in one of two ways, and both are reported:

- `words`: the leaf's own noun appears in the document;
- `referents`: a noun phrase of the document refers to an instance of the leaf, whatever its noun ("the bird", "it"), or to the leaf category itself.

Each correlation is given as Pearson's and as Spearman's. The taxonomic similarity is the one in `thematic.csv`, computed for every pair, and a pair with an undefined similarity is left out. The correlations are reported for each document type, for the two encyclopedic types together, for the two narrative types together, and for all documents. The check confirms that the document mix works as a lever: situational documents should correlate more with thematic relatedness than the encyclopedic documents do, and less with taxonomic similarity.

## Word forms for the corpus

The corpus comes first, and the word forms are made for it. The run has three steps:

1. **Generate.** `python -m semantic_world.corpus generate` writes the corpus in the formal, conceptual, and propositional renderings, and writes `wordform_request.yaml`. Stage 6 builds the command, and stage 7 adds the request.
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
  instance_description_rate: 0.2              # the probability that an event sentence of a narrative is followed by a description
  sibling_contrast_rate: 0.2                  # the probability that a fact in a category document is followed by the matching fact about a sibling

propositions:
  negation_rate: {class: 0.1, instance: 0.1}
  rule_statement_rate: 0.3                    # share of feature-topic sentences that state rules
  rule_statements: {max_literals: null}       # a number skips the rule terms with more literals; null: no cap
  restriction_rate: 0.1                       # the probability that a class-level subject takes a restriction ("red penguins")
  events: {tense: past, progressive_rate: 0.3}   # the tense of every event (past or present), and the share of progressive events

quantifiers:
  all_grounding: fixed                        # fixed or observed
  most: {min_proportion: 0.7}
  some: {exclude_all: true}
  generic: {means: most}                      # all, most, or some
  generic_rate: 0.5                           # share of class-level sentences that use a bare generic when it is true
  weights: {all: 1.0, most: 1.0, some: 1.0, none: 1.0}   # how often a document states the facts of each quantifier (none: the quantifier "no")

scene:
  size: [2, 6]
  steps: [3, 8]
  events_per_step: 1.5
  transitive_share: 0.5
  verb_weights: uniform
  participant_weights: {thematic: 1.0, taxonomic: 0.5, constant: 0.1}

entity:
  scenes: [2, 5]                              # the number of scenes of an entity narrative

mention:
  level_weights: {schedule: linear, start: 1, end: 4}         # noun levels, heaviest at the leaf level
  verb_level_weights: {schedule: linear, start: 1, end: 4}    # verb levels for naming events, heaviest at the leaf (the verb)
  pronoun_rate: 0.5
  modifier_rate: 0.3
  max_adjectives: 3
  max_with_phrases: 2
  max_content_words: 20                       # nouns, adjectives, and verbs in one sentence
  relative_clauses: {rate: 0.1, max_depth: 1, object_share: 0.3}

grammar:
  adjective_order: {fixed: true}
  can_rate: {class: 0.5, instance: 1.0}       # share of positive capacities that say "can", at each level
  word_order: {clause: SVO, determiner: before, adjective: before, with_phrase: after, relative_clause: after, adposition: preposition, auxiliary: before, negation: after_auxiliary}
  morphology:
    number: {enabled: false, realization: affix, position: after, agreement: true, verb_marks: plural}   # verb_marks: plural or singular (English)
    tense: {enabled: false, realization: affix, position: after}     # whether and how the past is marked
    aspect: {enabled: false, realization: word, position: after}     # whether and how the progressive is marked

scalar_adjectives: {z: 1.0}

renderings:
  propositional: {referents: local}          # local (R.1, R.2, ... within each document) or instance (taxonomy instance labels)

test_sets:
  size: 500                                   # true items per set; each gets one matched false item
  changes: [predicate, subject, quantifier, role]
```

Validation follows the base conventions: unknown keys are errors, and every error names the file and the field. `taxonomy` takes exactly one of `config` and `run`, and `seed` goes with `config` only. Relative paths are read from the folder the command runs in, as in the word-form pipeline. The morphology switch is `enabled`. It was `on` at first, which YAML reads as the boolean true when written bare. A configuration that still has `grammar.class_can_rate`, `grammar.morphology.tense.event_tense`, or `grammar.morphology.aspect.progressive_rate` gets an error that names the new key. So does `quantifiers.weights.no`, which YAML reads as the boolean false: the key is `none`.

## Determinism

Use the stream-seed function in `semantic_world.taxonomy.streams`. Streams: `corpus:lexicon`, `corpus:scenes`, `corpus:documents`, `corpus:propositions`, `corpus:mentions`, `corpus:grammar`, and `corpus:tests`. The corpus seed is its own master seed, independent of the taxonomy's seed. Each document draws from its own parts of four streams, named by its label (`corpus:documents:D.17`): its type, topic, length, and order from `corpus:documents`, its facts from `corpus:propositions`, its mentions from `corpus:mentions`, and the grammar's choices from `corpus:grammar`. The test sets choose their items from `corpus:tests`, each level and change from its own part. The mentions and the grammar's choices of a test item come from parts of `corpus:mentions` and `corpus:grammar` named by its set and its pair. So a grammar setting never changes what a test item says. Properties, each tested:

- the same taxonomy run, word-form run, configuration, and seed give byte-identical output folders;
- changing the grammar settings (word order, adjective order, and morphology) never changes any logical form or the order of the sentences, only their realization;
- rendering with a word-form run changes only the word labels, the spelled rendering (and with it `corpus.txt`), and the word-form columns of `lexicon.csv`; the rest of the output folder is byte-identical;
- changing the test-set settings never changes the documents.

## Python package

Put the generator in `python/semantic_world/corpus/`. Suggested modules: `config.py`, `streams.py`, `world.py` (loading the taxonomy), `lexicon.py`, `propositions.py` (logical forms and truth tests), `facts.py` (the true propositions a document can state, the rule statements, and the naming of events), `scenes.py`, `planner.py` (document types and ordering), `grammar.py` (sentence plans: what a sentence says, in the form the grammar realizes), `realize.py` (phrase structure, word order, and morphology), `interpret.py` (tree to logical form, for tests), `readings.py` (the readings that a sentence's words allow), `mentions.py` (the sentence plan of a proposition, relative clauses, and the mentions of a document's referents), `logical.py` (the JSON logical form of a sentence), `testsets.py` (not `test_sets.py`, which pytest would collect as a test module), `renderings.py` (formal, conceptual, and propositional), `stats.py` (`stats.yaml`, with the co-occurrence check), `generate.py` (a whole run), `request.py` (the word-form request), `io.py`, and `__main__.py`. The command line is:

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

Jon decided the following on October 1, 2026, before stage 2 (`docs/proposals/2026-10-01-corpus-stage-2-decisions.md`):

23. Words for verb categories are used in two places: in class-level and instance-level capacity sentences ("owls hunt mice"), and in naming events, where each event's verb is named at a level drawn by weight (`mention.verb_level_weights`, heaviest at the leaf). Stage 2 builds the capacity sentences. Stages 3 and 5 build the event half.
24. Patient projections also appear at the class level ("most mice are edible"). Truth for `most`, `some`, and the generic comes from the share of instances in the subject set that have the projection. `all` and `no` are allowed only under the observed reading.
25. Rule statements are exempt from the limits on adjectives, with-phrases, and content words. An optional cap, `propositions.rule_statements.max_literals`, is null by default. When a cap is set, terms above it are skipped and counted.
26. The morphology switch is `enabled`, not `on`.

Jon decided the following on October 1, 2026, before stage 3 (`docs/proposals/2026-10-01-corpus-stage-3-decisions.md`):

27. A relative clause can join several verb phrases with "and": "things with wings that are not red and not big can fly". All of a rule term's negated IS literals go into one relative clause, so such terms are stated. This replaces the skip rule of decision 20 for relative clauses. Terms that read a scalar threshold stay unstated. Stage 4 builds the grammar.
28. A false `all` or `no` test item that no instance contradicts is marked, and such items go into a test set of their own. Ordinary test sets hold only items that observing the instances could decide.
29. A false event-level test item is an event that did not happen in the scene. Each item is marked possible or impossible, and the two kinds go into separate test sets.
30. Every instance-level and event-level test item names a document. The item is tested as a continuation of that document, so its definite noun phrases refer to that document's referents.
31. A rule statement's quantifier is drawn like that of any class-level proposition: `all` or the generic, at the generic rate. "All things with wings ..." is `ALL`, and the bare plural is `GEN`. A rule statement is true under both.

Jon decided the following on October 1, 2026, before stage 4 (`docs/proposals/2026-10-01-corpus-stage-4-decisions.md`):

32. Events are drawn verb first. `scene.transitive_share` decides the kind. Then a CAN feature or leaf verb is chosen among those with at least one possible event in the scene, weighted by `scene.verb_weights`. Then the participants are chosen uniformly among the pairs, or the agents, for which that verb's event is possible.

Jon decided the following on October 1, 2026, before stage 5 (`docs/proposals/2026-10-01-corpus-stage-5-decisions.md`):

33. Logical form versus surface form. An event's tense and aspect are part of its logical form: the settings are `propositions.events: {tense, progressive_rate}`, and `grammar.morphology` keeps only whether and how tense and aspect are marked. Aspect is drawn for every event, even when aspect marking is off. The propositional rendering is the logical form, and it is never ambiguous: a capacity is wrapped in `ABLE`, an event is `EVENT(<event label>, <tense>, <aspect>, <atom>)`, `R.n` is one individual and `X.n` a bound variable, and "at least one" is `EXISTS(X.n, ...)`. The surface language may be ambiguous: `grammar.can_rate: {class: 0.5, instance: 1.0}` replaces `grammar.class_can_rate`, and a negative capacity always keeps `can`. Each sentence records its `readings`, worked out from the lexeme tokens alone, and `stats.yaml` reports how often sentences are ambiguous. The conceptual rendering is the same sentence as the spelled rendering, and only the lexical forms differ.
34. Class-level relative clauses are restrictive, by proportion. All three kinds are drawn: subject relatives with a verb, object relatives, and CAN-feature clauses. A clause about another category means at least one member of it. `most`, `some`, and the generic are judged by the share of the subject set, and `all` and `no` are allowed only under the observed reading.
35. One affix per word. The limit stays, with its configuration error. Stacked affixes are a future addition.
36. Noun levels apply to instances only. An instance can be named by the noun of its own leaf or of any category above it. A class-level subject is always named by its own noun.
37. Modifiers at `mention.modifier_rate` go on instance mentions only. Restrictions on class-level subjects come only from the proposition layer. Distinguishing modifiers are unchanged.
38. Sibling contrasts get a rate: `documents.sibling_contrast_rate` (default 0.2) is the probability that a class-level fact in a category-topic document is followed by the matching fact about a sibling category.
39. Aspect is drawn once for each event, so every report of one event agrees. The event carries its aspect, and `scenes.jsonl` gains the key `aspect`. The aspects are drawn after the events, so the rate never changes a scene.
40. `readings` is one list per sentence, computed from the tree and the lexeme tokens, never from `events` or the logical form. `generic`, `capacity`, and `event` stand for the three levels, and a sentence without a verb has the one reading of its level. A test shows that the tree carries no information about the reading: a capacity and an event with the same tokens have identical trees and identical readings.
41. A class-level subject takes a restriction at `propositions.restriction_rate` (default 0.1). Class-level relative clauses use `mention.relative_clauses`, and are drawn in the proposition layer.
42. A sibling contrast is added only for a sibling where the fact differs. With no differing sibling, no contrast sentence is added.
43. When modifiers cannot tell a referent apart, the mention falls back to the noun of the leaf. A mention that still fits another participant is kept, marked as not distinguished, and counted.
44. A scalar pole writes its comparison class in the propositional rendering, at both levels: `SC.1.HIGH(R.1, C1.5)` for an instance, and the comparison class of decision 22 for a category, as in `SC.1.HIGH(X.1, C1)`.
45. The JSON logical form holds relative clauses as a `clauses` list on the subject and the patient, and `tense` and `aspect` on event-level forms. The propositional rendering is generated from the JSON logical form, and a test checks that the two always agree.

Jon decided the following on October 1, 2026, before stage 6 (`docs/proposals/2026-10-01-corpus-stage-6-decisions.md`):

46. Event test items name only the scene. Every event-level test item, true or false, uses the same form: `EVENT(SN.8, PAST, SIMPLE, V1.2(R.1, R.2))`, meaning that some event in scene `SN.8` was this one. Documents keep their event labels. More generally, the true and the false items of a test set never differ in format: every field that a true item has, a false item has too, in the same notation. The rule covers the model-facing fields: the sentence, its renderings, and its logical form. Metadata such as the truth label, the grounding, and the `seen` field differ by design. An item keeps its model-facing fields (`input`) apart from its metadata (`meta`). A test checks the rule for every test set.
47. Entity narratives are longer. The default `entity.scenes` is `[2, 5]`, and was `[1, 3]`.
48. Quantifier weights. `quantifiers.weights` weights the choice of class-level facts by quantifier, so a study can rebalance the mix. The default keeps the earlier behavior: the strongest true quantifier, with no reweighting. The setting is `{all: 1.0, most: 1.0, some: 1.0, none: 1.0}`, and a fact is chosen with a probability proportional to the weight of its strongest true quantifier.
49. A true event-level test item can be any event that its document reports, in a main clause or in a relative clause.
50. A false event-level test item follows a stricter rule than the truth test. No event with its verb, agent, and patient happened in any scene of its document, in either aspect. When the item names its verb with a verb category, no event of any verb below that category matches either.
51. The co-occurrence check counts by words: a leaf occurs in a document when its own noun appears. It reports Pearson's and Spearman's correlations. It also reports the same correlations counted by referents, as a second measure.
52. The test sets are `class_<change>` and `class_<change>_lawlike`, `instance_<change>`, and `event_<change>_possible` and `event_<change>_impossible`.
53. Seen and unseen items. The documents stay unchanged, and every test item has a field that says whether its logical form appears in any training document. A false item never does. `stats.yaml` reports the share of true items that are seen, for each test set.

## Future additions

- **Sound differences between parts of speech.** In English and other languages, nouns and verbs differ in their sound: English nouns tend to have more syllables and initial stress, and verbs more often have final stress (Kelly, 1992; Monaghan, Christiansen, & Chater, 2007). The request already gives each lexeme's part of speech, so the word-form pipeline could later draw each part of speech's forms from its own sound profile.
- **Case marking.** Richer systems of case marking than subject–verb agreement, such as nominative and accusative markers on noun phrases.
- **Plural events.** Events with several agents. Plural events would allow the event reading of "penguins swim", which today has only the generic reading.
- **Stacked affixes.** More than one affix on a word, for example tense plus agreement on one verb. Today a word takes one affix.

## References

- Dale, R., & Reiter, E. (1995). Computational interpretations of the Gricean maxims in the generation of referring expressions. *Cognitive Science*, 19, 233–263.
- Kelly, M. H. (1992). Using sound to solve syntactic problems: The role of phonology in grammatical category assignments. *Psychological Review*, 99, 349–364.
- Monaghan, P., Christiansen, M. H., & Chater, N. (2007). The phonological-distributional coherence hypothesis: Cross-linguistic evidence in language acquisition. *Cognitive Psychology*, 55, 259–305.
