# Proposal: decisions for stage 5 of the corpus generator

October 1, 2026. Raised at the end of stage 4 of `docs/specs/CORPUS_GENERATOR.md`, in the orientation for stage 5 (documents), and while building stage 5. Status: decided. Jon gave decisions 33 to 38 in the stage 5 prompt, and answered the seven questions of the orientation as decisions 39 to 45. The specification was updated to match. One question for stage 6 is open, under "Still open".

## Jon's decisions

### 33. Logical form versus surface form

Stage 4 left three things to the grammar that change what a sentence means: the tense of an event, its aspect, and whether a bare verb is a capacity or an event.

**Decision.**

- **Where tense and aspect live.** An event's tense and aspect are part of its logical form. The settings are `propositions.events: {tense: past, progressive_rate: 0.3}`. `grammar.morphology` keeps only whether and how tense and aspect are marked. Aspect is drawn for every event, even when aspect marking is off.
- **The propositional rendering is the logical form, and it is never ambiguous.** Capacity versus event, tense, aspect, and number are explicit in it, whatever the grammar settings. A capacity is wrapped in `ABLE`: `ABLE(CAN.3(R.1))`, `ABLE(V1.2(R.1, R.2))`, and, inside a quantifier, `GEN(C1.3(X.1), ABLE(CAN.3(X.1)))`. `ABLE` wraps only CAN features and verbs, and a negative capacity is `NOT ABLE(...)`. An event is `EVENT(<event label>, <tense>, <aspect>, <atom>)`, with `PAST` or `PRESENT` and `SIMPLE` or `PROGRESSIVE`, always written. `R.n` is always one individual, and `X.n` is a variable bound by a quantifier. A plural individual referent would be `R.n:PL`. None exist yet. "At least one" is `EXISTS(X.n, ...)`, inside a restrictor.
- **The surface language may be ambiguous.** `grammar.can_rate: {class: 0.5, instance: 1.0}` replaces `grammar.class_can_rate`. With an instance rate below 1, an instance capacity can appear without "can". Together with tense and number off, "the penguin swim" is then ambiguous between a capacity and an event. A negative capacity always keeps "can", because "not" needs it.
- **Readings.** Each sentence records `readings`: the kinds of logical form its surface string allows (capacity, event, generic). Readings are worked out from the lexeme tokens alone, without the sentence record, so homonyms do not count as ambiguity. `stats.yaml` reports how often sentences are ambiguous.
- **The conceptual rendering** is the same sentence as the spelled rendering, with the same tree, word order, morphology, and ambiguities. Only the lexical forms differ: concept labels instead of word forms.
- **Future addition.** Plural events: events with several agents, which would allow the event reading of "penguins swim".

### 34. Class-level relative clauses are restrictive, by proportion

Stage 4 drew no relative clause for a class-level noun phrase, because the specification did not say what "owls that eat mice are big" means.

**Decision.** All three kinds are drawn: subject relatives with a verb ("owls that eat mice"), object relatives ("mice that owls eat"), and CAN-feature clauses ("penguins that can swim"). A clause about another category means at least one member of it. "Owls that eat mice are big" has as its subject set the owls that can eat at least one mouse: `GEN(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2))), SC.1.HIGH(X.1))`. "Mice that owls eat" is the mice that at least one owl can eat. `most`, `some`, and the generic are judged by the share of the subject set. `all` and `no` are allowed only under the observed reading, as for patient projections (decision 24). The truth oracle of the tests covers these clauses.

### 35. One affix per word

**Decision.** The limit stays, with its configuration error. Stacked affixes (for example, tense plus agreement on one verb) are a future addition.

### 36. Noun levels apply to instances only

**Decision.** An instance can be named by the noun of its own leaf or of any category above it ("the bird"). A class-level subject is always named by its own noun, because a higher noun would make a different proposition. Propositions about higher categories come from the proposition layer.

### 37. Modifiers go on instance mentions only

**Decision.** Modifiers at `mention.modifier_rate` go on instance mentions only. Restrictions on class-level subjects come only from the proposition layer, where their truth is grounded. Distinguishing modifiers are unchanged.

### 38. Sibling contrasts get a rate

**Decision.** `documents.sibling_contrast_rate` (default 0.2) is the probability that a class-level fact in a category-topic document is followed by the matching fact about a sibling category.

## Questions of the orientation, and Jon's answers

Claude Code raised these seven questions before building. Jon chose the recommended option of each, with the conditions noted.

### 39. Is aspect drawn for each event, or for each report?

**Decision.** For each event. The aspect is drawn once, so every report of `SN.8.5` agrees. The event carries its aspect, and `scenes.jsonl` gains the key `aspect`. The aspects of a scene are drawn after its events, from the scene's own part of the stream, so `progressive_rate` never changes the participants or the events.

### 40. How are readings computed, and what do they hold?

A parser that works from the tokens alone, in six clause orders and with center embedding, would be large work.

**Decision.** Readings are computed from the tree and the lexeme tokens, and never from `events` or the logical form. The condition is that the tree carries no information about the reading. A test shows it: a sentence realized once as a capacity and once as an event, with the same tokens, gets identical trees and identical readings. `readings` is one list per sentence. `generic`, `capacity`, and `event` stand for the three levels, and a sentence without a verb ("the penguin is red") has the one reading of its level.

### 41. How often does a class-level subject take a restriction?

Nothing set how often a document says "red penguins swim".

**Decision.** A new setting, `propositions.restriction_rate` (default 0.1). Class-level relative clauses use `mention.relative_clauses`, and are drawn in the proposition layer.

### 42. What is the "matching fact" of a sibling contrast?

**Decision.** The same predicate about a sibling, only where the sibling differs. With no differing sibling, no contrast sentence is added.

### 43. What happens when modifiers cannot tell a referent apart?

Two participants can share every feature.

**Decision.** The mention falls back to the noun of the leaf. A mention that still fits another participant is kept, marked as not distinguished, and counted.

### 44. Does a scalar pole write its comparison class?

`SC.1.HIGH(R.1)` leaves the comparison class implicit, and the propositional rendering must never be ambiguous.

**Decision.** The comparison class is written, at both levels. At the instance level, it is the category that the noun names: `SC.1.HIGH(R.1, C1.5)`. A class-level pole names the comparison class of decision 22: the subject category's parent, as in `SC.1.HIGH(X.1, C1)`. So the example of decision 34 is written `GEN(C1.2(X.1) AND EXISTS(X.2, C1.5(X.2) AND ABLE(V2.1(X.1, X.2))), SC.1.HIGH(X.1, C1))`.

### 45. Where do relative clauses go in the JSON logical form?

**Decision.** The schema is settled in stage 5. A subject and a patient hold a `clauses` list, and an event-level form holds `tense` and `aspect`. The propositional rendering is generated from the JSON logical form, and a test checks that the two always agree.

## What changed in the stage 4 code

- **Configuration.** `propositions.events`, `propositions.restriction_rate`, `documents.sibling_contrast_rate`, and `grammar.can_rate` are new. `grammar.morphology.tense` and `grammar.morphology.aspect` keep `enabled`, `realization`, and `position`. The old keys `grammar.class_can_rate`, `grammar.morphology.tense.event_tense`, and `grammar.morphology.aspect.progressive_rate` give an error that names the new key.
- **Scenes.** An event has an aspect, written to `scenes.jsonl`.
- **Propositions.** An event-level proposition has a tense and an aspect. A category term can have relative clauses, which narrow its subject set. The truth tests judge both.
- **The grammar.** The tense and the aspect of an event are part of the sentence plan, and the realizer no longer draws the aspect. The realizer draws `can` for a positive capacity at `grammar.can_rate`, at both levels. `interpret` reads a bare verb after an instance as an event when the sentence's record gives one, and as a capacity otherwise.
- **Mentions.** Class-level relative clauses are drawn with the proposition (`facts.draw_clause`). The clauses of instance mentions take the document's mentions for their own noun phrases.
- **Tests.** The oracle that recomputes truth from the taxonomy's files judges category terms with relative clauses, with plain loops over the pairs.

## Choices made by Claude

Jon approved the first four choices before the build. The others are engineering choices made while building stage 5. Each one is the working design unless Jon changes it.

1. **The content of a category document.** Each sentence draws one kind of content, each kind with the same chance: a membership fact, a fact about the topic, a fact about a subcategory, or a relation fact. The fact is then drawn among the facts of that kind. A relation fact has the topic as its agent or as its patient, each with the same chance.
2. **Short documents.** A narrative with too few events is shorter than its drawn length. An encyclopedic document is shorter when it has nothing more to say. A document with no sentence is drawn again, with a new type and topic. After 100 such draws, the run stops with an error.
3. **Distractors.** The distractors of a definite mention are all participants of the document's scenes, mentioned or not.
4. **Pronouns.** A pronoun stands only in a noun phrase of the main clause.
5. **Tense and aspect in the truth tests.** An event-level proposition must have the corpus's tense. Another tense is not a valid logical form. A report with the wrong aspect is false. A proposition that names no event (a test item) is true when an event of its aspect occurred.
6. **One draw of `can` for every positive capacity.** The realizer draws once for each positive capacity, at the class rate or the instance rate, inside relative clauses too. The draw is made even at a rate of 0 or 1, so the rate never moves the grammar's other draws.
7. **The sentence's `events` in memory.** A sentence holds, for each verb phrase, the event's label, tense, and aspect. `documents.jsonl` writes the labels, as the specification says, and the logical form holds the tense and the aspect.
8. **Readings follow the language.** A bare verb after an instance has the capacity reading only when `grammar.can_rate.instance` is below 1. A bare verb has the event reading only when the language has unmarked events: the tense is the present or is not marked, and the simple aspect occurs or the aspect is not marked. A verb phrase without `can` in a joined relative clause, after one with `can`, has the capacity reading. A relative clause "that is not red" allows both readings.
9. **Relative clauses in a category term.** A category term holds a list of clauses. A class-level clause holds a CAN feature or a verb, and is never negated. A term with an object relative has that one clause, because a noun phrase has one relative clause. Several subject relatives join with "and".
10. **`all` and `no` with a relative clause.** With a relative clause in the subject, `all` and `no` are not valid under the law-like reading, for every kind of predicate, verbs and membership included. A generic that means `all` is judged over the instances. A relative clause on the patient alone changes nothing: a verb's quantifiers were always judged over pairs. The grounding's `fixed` is computed for the subject without its clauses.
11. **A restriction must do work.** A restriction and a relative clause are drawn among those that some members satisfy, and not all. A restriction is one literal: an IS or HAS literal, negative at the class-level negation rate, or a scalar pole.
12. **The kinds of class-level clause.** A clause is an object relative with probability `mention.relative_clauses.object_share`, and a subject relative otherwise. A subject relative holds a CAN feature or a verb, each with the same chance. When the drawn kind has no clause, the other kind is used. The other category of a verb takes a clause of its own at the rate, up to `max_depth`. A subject with negated IS literals takes a subject relative, which joins them.
13. **Which subjects are restricted.** In a category document, the topic or the subcategory of a fact, and the topic of a relation fact, as agent or as patient. In a feature document, the subject. Membership facts, rule statements, and sibling contrasts take no drawn restriction. A sentence never states a predicate that its subject's restriction already states.
14. **The negation rate is kept.** The polarity of a class-level fact is drawn first, and the fact is drawn among the facts of that polarity. When they run out, the document says something else. So the share of negative sentences stays at the rate.
15. **A document says a thing once.** A document states a subject and a predicate once for each polarity, whatever the quantifier. A narrative reports an event once in a main clause. A relative clause can report an event that a main clause also reports.
16. **Sibling contrasts.** A fact has a contrast when it is strong (`all`, `no`, `most`, or a scalar pole) and is about the plain topic, as subject or as patient. The matching fact is the sibling's strongest fact of the other polarity, and must be strong too. Siblings are the categories with the same parent, and the top-level categories are siblings of each other. A contrast counts toward the document's length, and stays right after its fact when the document is shuffled.
17. **The template.** The sections are membership, defining (`all` and `no`), characteristic (`most`, and scalar poles), rarer (`some`), relation (every verb predicate), and rule. The section of a bare generic is that of the fact it stands for. With `n` units, a unit at template position `i` gets the key `(1 − shuffle) · i / n + shuffle · u`, with `u` uniform, and the units are sorted by key. So `shuffle: 0` is the template, and `shuffle: 1` is a random order.
18. **Topics.** A category topic draws its level by `documents.topic_level_weights`, and then a category of that level, each with the same chance. A feature topic is drawn among the IS, HAS, and CAN features and the verbs and verb categories that have a word, each with the same chance. An entity topic and a scene's seed are drawn among all instances. The topic of a situational document is its scene's label.
19. **Feature documents.** A sentence states a rule at `propositions.rule_statement_rate`, while the topic has rules left. A rule is a sufficient condition for the topic, or a rule whose restriction reads the topic. Otherwise the sentence says that a category has the feature, or lacks it, at the negation rate. For a verb topic, the sentence is a relation fact about a pair of categories.
20. **Descriptions.** An event sentence is followed by a description at `documents.instance_description_rate`. An entity narrative describes its topic. A situational narrative describes a participant of the event, and one that the sentence introduced comes first. The patient of a verb in a description is another participant of the document's scenes.
21. **Relative clauses in narratives.** An event-level clause reports an earlier event of the same scene. An instance-level clause relates the referent to an instance that the document has already mentioned. The noun phrases inside a clause are mentions like any other, and are never pronouns.
22. **Noun levels.** The level of the noun is drawn for each mention, so one referent can be "a bird" and later "the penguin". Three mentions do not draw freely. The subject of a scalar pole is named by the comparison class. The subject of "is a penguin" is never named "penguin". A mention that cannot be told apart falls back to the leaf.
23. **Modifiers.** A situational narrative has distinguishing modifiers only. An entity narrative also adds one modifier at `mention.modifier_rate`: an adjective for an IS feature or a pole, or a with-phrase for a HAS feature, true of the referent. A first mention is indefinite, and takes no distinguishing modifier. No modifier states the feature that the sentence's own predicate states.
24. **The attributes that tell referents apart.** An adjective for an IS feature the referent has, a with-phrase or a without-phrase for a HAS feature, and a pole adjective for a scalar on which the referent is at a pole of the noun's category. A negated IS literal is never used. The preference order is one random ordering of the IS features, the HAS features, and the scalar dimensions, drawn once from `corpus:mentions:preference`. The limits on adjectives and with-phrases hold.
25. **The limit on content words.** The count is made on the sentence plan: nouns, adjectives, and verbs. A narrative sentence over the limit is said again without its drawn relative clauses. Distinguishing modifiers are kept, even over the limit. A class-level fact over the limit is not stated. Rule statements are exempt (decision 25).
26. **Streams.** Each document draws from its own parts of four streams, named by its label. Scenes and proposition labels are numbered across the corpus, so documents are made in order.
27. **The order of labels.** Referents and variables are numbered in the order of the logical form: the subject, the noun phrases of its relative clause, then the object. So the word order never changes a label or a propositional rendering.
28. **The order of the rendering.** In a sentence about instances, each noun phrase gives its noun and its modifiers, then the propositions of its relative clause, each after the noun phrases it brings in. The proposition of the main clause comes last. A proposition that would be written twice is written once. A restrictor that would be empty is written `THING(X.1)`.
29. **What the rendering parses back to.** The rendering parses back to the formula of the JSON logical form. A class-level formula gives back the whole proposition, with its restrictions and relative clauses. A formula about instances gives back the proposition of the main clause, and the other propositions in order. The noun phrase that a relative clause hangs on is in the JSON form and in the tree, and not in the formula, because a conjunction has no such structure.
30. **Proposition labels.** A proposition gets one label, `PR.<n>`, numbered by first use across the corpus. Two sentences that say the same proposition share the label.
31. **The sentence record.** `referents` holds an object with `referent` and `noun` for each noun phrase. `coreference` and `distinguished` are lists in the same order. The document holds `referents`, from referent label to instance.
32. **A class-level pole in the JSON form.** A sentence's logical form writes the comparison class of a class-level pole (`"class": "C1"`), so the rendering can be made from the JSON form alone. The class comes from the tree, so it is no part of the proposition that the truth tests judge.

## Results on the default world

These numbers come from the first 2,000 documents of `data/corpus/default.yaml` (seed 1), which take 4 seconds to plan and realize.

| Measure | Value |
| --- | --- |
| Sentences | 17,324 |
| Scenes, and events in them | 1,339 scenes, 11,273 events |
| Sentences per document (lowest, mean, highest): category | 5, 9.8, 15 |
| Sentences per document: feature | 5, 9.8, 15 |
| Sentences per document: entity | 1, 6.0, 18 |
| Sentences per document: situational | 1, 8.4, 20 |
| Event sentences, and descriptions | 5,917 and 1,131 |
| Class-level sentences that are not rules | 9,669 |
| ... with a restricted subject or patient | 733 |
| ... with a relative clause | 766 |
| ... by quantifier | `some` 5,346, generic 2,191, `most` 1,006, `all` 930, `no` 196 |
| Rule statements | 607 |
| Sibling contrasts | 40 |
| Share of negative sentences: class level, instance level | 0.109, 0.112 |
| Share of progressive events | 0.300 |
| Instance mentions | 10,823: 3,560 indefinite, 5,678 definite, 1,585 pronouns |
| Definite mentions that could not be told apart | 0 |
| Instance mentions with a relative clause | 551 |
| Ambiguous sentences, with the default grammar | 0 |
| Distinct propositions | 14,066 |

Three things stand out.

- **Narratives are short.** An entity narrative has 6 sentences on average, against a range of 5 to 20, because a scene has few events that involve one instance. `scene.events_per_step`, `scene.steps`, and `entity.scenes` raise the number.
- **`some` leads.** More than half of the class-level sentences take `some`. A category has many features that some of its members have, and few that most or all have. The bare generic stands only for `all` and `most`, with `quantifiers.generic.means: most`.
- **Contrasts are rare.** A contrast needs a strong fact about the topic and a strong opposite fact about a sibling. With the default rate of 0.2, that gives 40 contrasts in 646 category documents. At a rate of 1, the first 300 documents have more than 30.

## Still open

- **Stage 6: an event-level test item that names no event.** The notation `EVENT(<event label>, ...)` needs a label, and a false event item has none (decision 29). Options: the scene's label in its place (`EVENT(SN.8, PAST, SIMPLE, ...)`), or a fixed mark for "no such event". Stage 5 gives such a form no rendering, and stops with an error.
