# The corpus generator

The corpus generator writes documents about a world of the world package (`python/semantic_world/world/`), which embeds a taxonomy of the taxonomy generator and adds fluents, event types with preconditions and effects, and time. The documents are in an artificial language. Every sentence states a proposition about the world, and every proposition is checked true against the world's data before it is written. Every sentence is stored with its words, its parse tree, and its logical form, so we always know what a sentence means and why it is true.

The corpus is a controlled source of language for models. Settings decide which kinds of documents the corpus holds, how the language orders and marks its words, and how ambiguous its surface is. Test sets of matched true and false sentences come with every corpus.

This guide covers running the generator, the ideas behind it, the output files, the test sets, the configuration file, and the Python interface. The design is specified in `docs/specs/CORPUS_GENERATOR.md`, and every design decision is listed in `docs/DECISIONS.md`.

**Status.** Complete on the world package (stages a5a to a8 of `docs/specs/WORLD_AND_LANGUAGE.md`): the lexicon, propositions and their truth, scenes as histories of the world, the grammar, the four document types, the test sets, the statistics, and spoken word forms through the word-form pipeline. Every label is the world's own, and nothing translates. Since stage a7a, narratives state states and changes and what a participant could not do, and the test sets include state, `able_now`, and blocked-event sets. Since stage a7b, feature documents state what events do and need (causal statements), the logical form tells descriptions from assertions, and the test sets include causal sets. Since stage a8, a document about a fluent holds one causal statement per sentence, and the causal sets pair every true statement with every valid false item. Since stage b1 the world has conditional effects, which the corpus does not yet state: an effect that applies only under a condition makes no causal statement true until stage b3, and a capped causal set keeps every one of its statements. The world itself is the world package's; `WORLD.md` is its guide.

## Quick start

From the root of the repository (see `README.md` in this folder for setup):

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/tiny.yaml
```

The program prints one line:

```
wrote runs/corpus/tiny_seed1: 20 documents, 203 sentences, 1078 tokens, 17 scenes, 41 test sets with 534 pairs
```

The tiny corpus is about the tiny world, `data/world/tiny.yaml`: the tiny relations taxonomy (6 categories, 12 instances, 1 scalar dimension, 4 one-place and 4 two-place event types) with 4 fluents (`WORLD.md`). It is small enough to read every file by eye. The generator defines the world in memory from its configuration file, so no world run needs to exist first; `world: {run: runs/world/tiny_seed1}` reads a world run instead, and checks it against the regenerated world.

The default corpus is 10,000 documents about the default world, `data/world/default.yaml`:

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/default.yaml
```

It prints `10000 documents, 99233 sentences, 469271 tokens, 10087 scenes, 41 test sets with 15210 pairs`, and takes about 2.6 minutes (157 seconds) on a laptop (October 9, 2026, stage b1; 158 seconds in stage a8). The world itself takes 8 seconds (`WORLD.md`). Every stage records these times, measured with nothing else running, and flags a growth of more than half in its proposal file; there is no hard time limit, and no step is made cheaper by changing what it produces by default (Jon's ruling 3 on stage a8).

Two options change a run without editing the configuration:

- `--seed N` replaces the corpus's master seed. The corpus seed is independent of the world's seed, so the same world can get different corpora.
- `--out DIR` writes the output folder somewhere else. The default is `runs/corpus/<name>_seed<seed>/`.

The example configurations in `data/corpus/` are:

| File | What it makes |
| --- | --- |
| `tiny.yaml` | The tiny corpus above, with 20 pairs per test set. |
| `default.yaml` | The default corpus. Every parameter appears with its default value, so this file is also the reference for defaults. |

## One sentence, four renderings

Every sentence is written four ways. Here is a sentence from an entity narrative of the tiny corpus (`DOC.19.SENT.3`):

| Rendering | The sentence |
| --- | --- |
| Conceptual | `THE PROPERTY.4 CATEGORY.2.1 THAT EVENTTYPE2.1.2 THE SCALARDIM.1.LOW CATEGORY.1.1 EVENTTYPE1.4` |
| Formal | `the/LEXEME.42 PROPERTY.4/LEXEME.10 CATEGORY.2.1/LEXEME.5 that/LEXEME.54 EVENTTYPE2.1.2/LEXEME.32 the/LEXEME.42 SCALARDIM.1.LOW/LEXEME.39 CATEGORY.1.1/LEXEME.2 EVENTTYPE1.4/LEXEME.30` |
| Propositional | `{CATEGORY.2.1(REF.1) AND PROPERTY.4(REF.1) AND CATEGORY.1.1(REF.2) AND SCALARDIM.1.LOW(REF.2, CATEGORY.1.1) AND EVENT(SCENE.16.EVENTINSTANCE.1, PAST, PROGRESSIVE, EVENTTYPE2.1.2(REF.1, REF.2))} EVENT(SCENE.16.EVENTINSTANCE.3, PAST, SIMPLE, EVENTTYPE1.4(REF.1))` |
| Spelled | `muh camascuck clonstanzive urs skence muh tulden sprost cannagzus` |

In English, with made-up glosses, the sentence says "the red owl that was eating the small penguin ran". `CATEGORY.2.1` and `CATEGORY.1.1` are categories, `PROPERTY.4` a property, `SCALARDIM.1.LOW` the low pole of a scalar dimension (small for a `CATEGORY.1.1`), `EVENTTYPE2.1.2` a two-place event type, and `EVENTTYPE1.4` a one-place one. The relative clause reports an earlier event of the same scene, in the progressive; the main clause reports a later one, whose agent is the document's topic.

- **Conceptual.** The sentence's words in order, each replaced by its concept's label. The conceptual rendering is a perfectly tokenized version of the language: it has the same word order, morphology, and ambiguities as the spelled rendering, but every word is a symbol for its meaning.
- **Formal.** Each word as its gloss and its lexeme label (`LEXEME.3`). It differs from the conceptual rendering only when synonyms are on, because two synonyms share a concept but are different lexemes.
- **Propositional.** The logical form. Every noun, modifier, and clause becomes its own proposition, joined by `AND`. Referents are labeled `REF.1`, `REF.2`, and so on within each document, so the rendering shows which noun phrases refer to the same thing. The parts in braces are the descriptions, which identify the referents of the definite mentions; the part after them is what the sentence asserts (see "Descriptions and assertions"). The propositional rendering is never ambiguous.
- **Spelled.** The readable spellings of the language's spoken word forms. These come from the word-form pipeline, after the corpus is made (see "Spoken word forms"). Before that, the spelled rendering is empty, and `corpus.txt` holds the formal rendering.

A model can be trained on any of the four. Comparing the conceptual and the spelled renderings isolates the effect of lexical form: the structure is identical, and only the words differ. The propositional rendering removes the language altogether.

## Concepts

### Labels

Labels follow the project's convention: formal labels, indices starting at 1, and periods between indices.

| Object | Label | Example |
| --- | --- | --- |
| Lexeme (a word of the language) | `LEXEME.<n>` | `LEXEME.42` |
| Document | `DOC.<n>` | `DOC.17` |
| Sentence | `DOC.<n>.SENT.<k>` | `DOC.17.SENT.3` is sentence 3 of document 17 |
| Scene | `SCENE.<n>` | `SCENE.8` |
| Event | `SCENE.<n>.EVENTINSTANCE.<k>` | `SCENE.8.EVENTINSTANCE.5` is event 5 of scene 8, in time order |
| Proposition | `PROP.<n>` | `PROP.120`, the same wherever the proposition is said |
| Referent, within one document | `REF.<n>` | `REF.2` is the second thing mentioned in its document |
| Variable, within one sentence | `VAR.<n>` | `VAR.1` in a quantified proposition |

Categories, features, instances, and event types carry the world's labels (`CATEGORY.1.3.2`, `PROPERTY.12`, `PART.4`, `INSTANCE.1.3.2.5`, `EVENTTYPE1.3` for a one-place event type, `EVENTTYPE2.1.2` for a two-place one, and `EVENTTYPE2.1` for a category of two-place event types). A patient capacity is `CANBE.` before its event type (`CANBE.EVENTTYPE2.1.1`: "edible"). A scalar pole is its dimension with `HIGH` or `LOW` (`SCALARDIM.1.HIGH`: "big"). The generic noun "thing" is `THING`, and a function word's concept is its gloss in capitals (`THE`).

### The lexicon

Every concept of the world gets a word:

| Concept | Part of speech | Example gloss |
| --- | --- | --- |
| Every category, at every level | noun | penguin, bird, animal |
| PROPERTY feature | adjective | red |
| PART feature | part noun | fins |
| Fluent (base or derived) | state adjective | asleep |
| One-place event type | intransitive verb | swim |
| Two-place event type | transitive verb | chase |
| Category of two-place event types | transitive verb, more general | hunt |
| Patient capacity of a two-place event type | adjective | edible |
| Scalar dimension | two adjectives, one per pole | big, small |
| The generic noun | noun | thing |

The language also has 17 function words: `a`, `the`, `all`, `most`, `some`, `no`, `not`, `can`, `is`, `has`, `with`, `without`, `and`, `that`, `it`, `become`, and `before`. Grammar settings can add more (see "Grammar"), and `lexicon.can_words: distinct` adds `can_now`, a word for what a participant could do at a time point (see "States, changes, and what was possible").

The default world gives 216 lexemes: 199 content words and the 17 function words. One patient capacity in four gets a word (`lexicon.named_proportion.patient_projection: 0.25`). A two-place event type, or a category of them, that is able for every pair of instances, or for none, gets no word, because it says nothing. `stats.yaml` lists such event types under `lexicon.event_types_without_word` (none in the default world). Every fluent gets a state adjective (`lexicon.named_proportion.state: 1.0`); the lexeme tells a state adjective from a static one, because it names a fluent.

Two settings, both 0 by default, add lexical ambiguity. `lexicon.synonym_rate` gives a concept a second word. `lexicon.homonym_rate` makes two words of different concepts share one word form. `lexicon.named_proportion` leaves some concepts without a word, so propositions that need them are never stated.

### Propositions and truth

A proposition is a logical form that the generator checks against the world. There are three levels.

**Class level.** A statement about a category: "most penguins can swim", "penguins are birds", "owls hunt mice". A class-level proposition has a subject category, a predicate, and one of six quantifiers: `nec_all`, `all`, `most`, `some`, `no`, and `nec_no`.

- `nec_all` and `nec_no` are the law-like universals: "all penguins swim" is `nec_all` only when the world's rules fix swimming for every penguin, and "penguins are birds" is `nec_all` because the taxonomy says so. `all` and `no` are the extensional universals: every existing penguin happens to swim. A relation fact ("owls eat mice") and a fact about a subject with a relative clause take the extensional universals only.
- `most` and `some` are judged by the share of the category's instances with the predicate. `most` needs more than half, and a document states it only from 70% (`quantifiers.most.usage_min`). `some` is not stated when the language's universal is true (`quantifiers.some.exclude_all`).
- Which quantifiers the words "all" and "no" state is a setting of the language: `quantifiers.universal_words` is `nec` by default, so "all" means `nec_all`; `extensional` makes it mean `all`, and `either` lets it mean both.
- A bare plural ("penguins swim") states `most` by default (`quantifiers.bare_plural.expresses`), and `nec_all` for a membership fact or a rule statement. The logical form always records the real quantifier, and a sentence's `readings` list the quantifiers that its words allow.
- A document states the strongest true quantifier that the language can state: `nec_all` before `all` before `most` before `some`. In the default language the extensional `all` has no word, so a fact that holds of every penguin without a rule is stated as `most`.
- A scalar pole of a category ("penguins are big") has no quantifier: it says that the category's mean stands out against its parent's.

A subject can be restricted: "red penguins", "penguins with fins", "penguins that can swim", "owls that eat mice". A restriction narrows the set of instances that the quantifier counts. "Owls that eat mice" are the owls that can eat at least one mouse.

**Causal statements.** What events do and what they need can be stated outright too (see "Causal statements" below): "things that things catch become caught", "things that catch things are awake before". Like rule statements, they are `nec_all`, because the world's definition guarantees them.

**Rule statements.** The world's rules can be stated outright. Each term of a rule's minimal formula is a sufficient condition, stated with the generic noun "thing". From a feature document of the tiny corpus:

```
ALL PROPERTY.2 THING WITH PART.1 HAS PART.8
THING THAT IS NOT PROPERTY.2 CAN EVENTTYPE1.4
```

A negated PROPERTY feature becomes a relative clause ("that is not red"). A rule statement's quantifier is `nec_all`, said with "all" or with a bare plural, as in the second line. The second line states a term of a one-place event type's requirement, which is a rule of the world package over the taxonomy's features. Rule statements are exempt from the limits on sentence length, because rules of every complexity must be stateable. In the default world, 88 of the 102 rule terms are stated. The rest read a scalar threshold, which no adjective states exactly, or have no instance.

**Instance level.** A statement about one instance: "the penguin can swim", "the penguin has stripes", "the penguin is a bird". Truth is read from the instance's own values.

**Event level.** A report that something happened in a scene: "the owl chased the mouse". Every report has a tense and an aspect (simple or progressive), which the logical form always records, even when the language does not mark them. The event itself has no aspect: a document chooses the aspect of each report, at `documents.progressive_rate`, and both aspects are true of an event that happened. The report's grounding says whether the binding is `able` (the world allows it) and `legal` (its preconditions held at some time point of the scene).

**State, change, and `able_now` levels.** Three levels about one participant of a scene at a time point (see "States, changes, and what was possible" below): a state, "the mouse was asleep"; a change, "the mouse became asleep"; and what a participant could or could not do then, "the owl could not eat the mouse". Their truth is read from the scene's history.

Class-level and instance-level propositions can be negative: "penguins can not fly", "no fish have fur". About 10% are, by default (`propositions.negation_rate`). Events are never negated.

### Scenes and events

Narratives need things to happen, so the generator makes scenes. A scene is an episode of the world package. It starts with a seed instance and adds 2 to 6 other instances. Instances that are thematically related to the seed are more likely to join: a scene with an owl tends to have mice in it. Each scene then runs for 3 to 8 time steps. At each step, events are drawn among the events that are legal: the agent, or the agent and the patient, must be able to take part (the world's requirements), and the event type's preconditions on their fluents must hold. An event's effects change fluents, so later events depend on earlier ones, and the history records every change. `scenes.jsonl` holds the histories: each scene's seed, participants, initial fluent values, and the events of every step with their changes.

Events never contradict the world. The selection policy is `scene.policy`: `uniform_event` (the default) draws uniformly among the legal events, so an event type that is legal for many bindings is drawn more often; `uniform_event_type` draws the kind of event by `scene.transitive_share`, then the event type, then the binding, as the old scene generator did. `scene.event_type_weights` rebalances the event types under either policy. In the default world, about a third of the events of a scene have a patient (31% of the events of the world's statistics episodes): the world's two-place requirements hold for about a sixth of the ordered pairs of entities and its one-place requirements for about half of the entities, by the retuned settings of `data/world/default.yaml` (see `docs/guides/TAXONOMY.md`, "Where CAN features and verbs went"). `examples/two_place_levers.py` reports how each setting moves the share.

### States, changes, and what was possible

Events change state, and since stage a7a narratives say so. After an event sentence, three kinds of sentence can follow, each at its rate:

- **An initial state.** When an event sentence first mentions a participant, a sentence states one of the participant's fluents at the time point before the participant's first reported event, at `documents.initial_state_rate` (default 0.2): "a mouse was asleep". It is negative ("was not asleep") at `propositions.negation_rate.state` (default 0.1). The proposition is `HOLDS(SCENE.8, TIME.2, PAST, BOOLFL.3(REF.1))`, or `NOT BOOLFL.3(REF.1)` inside for a negative state.
- **A result.** At `documents.result_rate` (default 0.5), a sentence states one change that the event's own effects made in its step to a participant of the event: a base fluent the event changed, or a derived fluent that the event's own changes bring to its new value. "The mouse became awake" is `BECOME(SCENE.8, TIME.2, PAST, BOOLFL.3(REF.1))`: the fluent was false at `TIME.2` and true at `TIME.3`; `BECOME(..., NOT BOOLFL.3(REF.1))` is the opposite change. A change says nothing of its cause; its grounding records the event whose own effect made it (`caused_by`). An event that changed nothing with a word has no result sentence (since stage b1 an effect whose condition fails changes nothing either). In the default corpus, 23% of event sentences are followed by a result.
- **A blocked event.** At `documents.blocked_rate` (default 0.1), a sentence says what a participant of the event could not do at that time point: an event type for which the binding is able (the world's requirement holds) and not legal (a precondition fails), "the owl could not eat the mouse", `NOT ABLE_NOW(SCENE.8, TIME.2, PAST, EVENTTYPE2.1.2(REF.1, REF.2))`. At one minus `propositions.negation_rate.able_now` (default 0.8), the sentence instead says what a participant could do and did not do in that step. `ABLE` holds the static facts fixed, and `ABLE_NOW` the state at a time point too.

Mentions stay static: a definite mention never tells referents apart by a fluent. States and changes are realized with the copula or `become` and the state adjective; what was possible is said with "can" (`lexicon.can_words: shared`, the default: "can" expresses `ABLE` and `ABLE_NOW` alike, as in English) or with the function word `can_now` (`distinct`). The tense of such a sentence is marked on its auxiliary when the morphology realizes tense as a separate word, and not at all when tense is an affix, since an affix never joins a function word. A sentence's `readings` say `state` for a state or a change, and, with "can" shared, `capacity` and `able_now` for every sentence about an instance with "can". The propositional rendering writes the scene, the time point, and the tense in every case, so it stays unambiguous.

### Causal statements

Every event type of the world has effects, which set base fluents of its participants, and a precondition, which requires fluent values of them. Since stage a7b the corpus states them at the class level, in feature documents:

- **An effect statement.** "Things that things catch become caught": after every event of the event type, the participant in the stated role has the fluent's value. In the logical form, `NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), AFTER(EVENTVAR.1, BOOLFL.3(VAR.2))))`: `EVENTVAR.1` binds the events of `EVENTTYPE2.1.2`, `VAR.1` is their agent and `VAR.2` their patient, and `AFTER` says what holds at the time point after the event. An effect that sets a fluent to false is `AFTER(EVENTVAR.1, NOT BOOLFL.3(VAR.2))`, "become not caught".
- **A precondition statement.** "Things that catch things are awake before": `NEC(ALL(EVENT(EVENTVAR.1, EVENTTYPE2.1.2(VAR.1, VAR.2)), BEFORE(EVENTVAR.1, BOOLFL.1(VAR.1))))`, what holds at the event's own time point.

A statement is true exactly when the definition has the entry; a statement about a category of event types ("things that things hunt become caught") is true when every event type below the category has it. Statements are about base fluents only, because the definition guarantees nothing about a derived fluent after an event. Since stage b1 an effect may hold only under a condition (`WORLD.md`, "What a world is"); such an effect guarantees nothing after every event, so it makes no effect statement true, and a statement about a category holds only when every event type below it has the entry unconditionally. The conditional causal statements of the specification ("red things that things strike become broken") come in stage b3. Result sentences in narratives are unaffected: they state the changes the history records, which a conditional effect makes only when its condition held. The subject is the generic noun "thing" with a relative clause that names the event: a subject relative for the agent ("things that catch things"), an object relative for the patient ("things that things catch"), and a one-place event type for its agent ("things that sleep"); the other participant is the bare plural "things". An effect's predicate is `become` with the state adjective, and a precondition's the copula with the state adjective and the function word `before`, which stands at the end of the verb phrase or, with `grammar.word_order.before: before_predicate`, at its start. Causal statements are in the present tense and take "all" or the bare plural, by the language's settings for `nec_all`. In the JSON logical form, the subject is `{"head": "THING", "event": "EVENTTYPE2.1.2", "role": "patient"}`, the predicate `{"kind": "effect", "label": "BOOLFL.3", "value": true}` or `{"kind": "precondition", ...}`, and `causal` repeats the definition entry the statement states. A sentence with `become` or `before` and a category as its subject has the reading `causal`.

Feature documents state them. A document about an event type, or a category of event types, draws a causal statement at `propositions.causal_statement_rate` (default 0.5) before its other content: what its events do and need. A document about a fluent, a new kind of topic, holds causal statements alone, one per sentence, until it reaches its drawn length or has stated every statement about its fluent: the event types that set it, clear it, and need it. The rate does not apply to a fluent document (stage a8). Narratives keep their result sentences, so the corpus carries both the general statement and its instances. In the default corpus, 2,389 sentences are causal statements (1,262 effects and 1,127 preconditions); the default world's 25 conditional effects are stated by none of them, which leaves 59 effect statements of the 87 the world had before stage b1.

### Descriptions and assertions

A sentence about instances has two parts (CG.63): the **assertion**, its main clause, and the **descriptions** that identify its referents. In a definite mention ("the furry dog") the noun, the modifiers, and a restrictive relative clause are descriptions; in an indefinite mention ("a penguin") they are asserted, because the sentence introduces the referent; a pronoun has no description. Every mention of the JSON logical form says which it is (`"descriptive": true` or `false`), and the propositional rendering writes the descriptions first, inside braces, then the assertion:

| Sentence | Propositional rendering |
| --- | --- |
| the furry dog has legs | `{CATEGORY.1.3.2(REF.1) AND PROPERTY.12(REF.1)} PART.4(REF.1)` |
| a penguin swam | `CATEGORY.1.3(REF.1) AND EVENT(SCENE.8.EVENTINSTANCE.5, PAST, SIMPLE, EVENTTYPE1.3(REF.1))` |
| it has legs | `PART.4(REF.1)` |

With `renderings.propositional.descriptions: omitted`, the braces and their contents are left out, and the rendering holds the assertion alone. Either way the rendering parses back: the marked form to the full logical form, the omitted form to the logical form with its descriptions removed. The setting changes nothing but the propositional rendering. Whether a description counts as stated, for the `seen` mark of a test item, is `test_sets.seen.descriptions` (default true, which counts everything, as before).

### Documents

The corpus mixes four document types (`documents.mix`):

| Type | Default share | Topic | What it says |
| --- | --- | --- | --- |
| Encyclopedic, category | 30% | A category | Membership, the category's features, its subcategories' features, and relation facts ("owls eat mice"), with contrasts to sibling categories |
| Encyclopedic, feature | 20% | A feature, an event type, or a fluent | Which categories have it, which lack it, the rules it takes part in, and what its events do and need (causal statements); a fluent document holds causal statements alone, one per sentence |
| Entity narrative | 20% | An instance | The instance's features, and the events it takes part in, across 2 to 5 scenes |
| Situational narrative | 30% | A scene | The scene's events in time order, with descriptions of the participants |

Narratives often end early. In the default corpus, about 46% of entity narratives and 43% of situational narratives end before their drawn length, because their scenes run out of events (the follow-up sentences of stage a7a fill some of the length that events alone did not). `stats.yaml` reports the counts.

A category document draws the kind of each sentence (membership, a fact about the topic, a fact about a subcategory, or a relation fact) with the same chance for each kind (`documents.content_kind_weights: equal`). With `proportional`, it draws the kind in proportion to the number of facts of that kind that it can still state; a category has far more relation facts to state than facts about itself, so most sentences of a category document are then relation facts. How a document divides its sentences among kinds of content is a choice of discourse, not a frequency of the world, so the balanced form is the default. Part of an encyclopedic document about category `CATEGORY.1`, in the conceptual rendering:

```
CATEGORY.1.1 IS A CATEGORY.1
ALL CATEGORY.1.1 IS PROPERTY.7
ALL PROPERTY.8 CATEGORY.1.1 HAS PART.4
ALL CATEGORY.1.2 IS A CATEGORY.1
SOME CATEGORY.1 HAS PART.6
SOME CATEGORY.1 IS PROPERTY.1
```

The first sentence is a bare plural that states `nec_all`: the taxonomy makes every `CATEGORY.1.1` a `CATEGORY.1`. The second says "all" for `nec_all`: a rule fixes the property for the subcategory. The third restricts its subject: a rule fixes the part for the `CATEGORY.1.1` that are `PROPERTY.8`. A sibling contrast follows a fact with the same predicate about a sibling of the topic, with the other polarity, at `documents.sibling_contrast_rate`.

The first sentences of a situational narrative of the tiny corpus (`DOC.9`):

```
A CATEGORY.1.1 EVENTTYPE2.1.2 A CATEGORY.1.1
THE SCALARDIM.1.LOW CATEGORY.1.1 IS BOOLFL.1
THE CATEGORY.1 WITH PART.5 AND WITHOUT PART.6 BECOME NOT BOOLFL.3
A CATEGORY.1.2 EVENTTYPE1.2
THE SCALARDIM.1.HIGH CATEGORY.1.1 THAT EVENTTYPE2.1.2 THE SCALARDIM.1.LOW CATEGORY.1.1 EVENTTYPE2.2.2 A CATEGORY.2.1
THE CATEGORY.2.1 IS NOT BOOLFL.4
```

In narratives, an instance is introduced with "a" and mentioned later with "the" or "it". A mention can name an instance by a higher category ("the bird" for a penguin: `CATEGORY.1` for the `CATEGORY.1.1`). When a scene has two things that the noun fits, a definite mention adds adjectives or with-phrases until it picks out one: "the small CATEGORY.1.1" against "the big CATEGORY.1.1", or "the CATEGORY.1 with PART.5 and without PART.6". The second sentence states a participant's initial state, the third a result of the first event (the agent's `BOOLFL.3` became false), and the sixth the initial state of the participant the fifth sentence introduced (see "States, changes, and what was possible"). The fifth sentence reports an event with a relative clause that reports the first. A sentence like "it can EVENTTYPE2.1 a CATEGORY.1.1", at `documents.instance_description_rate`, is a description of a participant: it can hunt a `CATEGORY.1.1`, said with the category of event types `EVENTTYPE2.1`. A blocked sentence, "the CATEGORY.1.1 can not EVENTTYPE2.1 the CATEGORY.2.2", says what a participant could not do at that time point.

**The document mix is a lever.** Which words occur together in a document depends on the document type. Narratives put thematically related things together. Category documents put taxonomic neighbors together, and, with their relation facts, thematically related things as well. In the default corpus:

| Documents | Co-occurrence with thematic relatedness | Co-occurrence with taxonomic similarity |
| --- | --- | --- |
| Entity narratives | 0.69 | 0.20 |
| Situational narratives | 0.58 | 0.20 |
| Category documents | 0.17 | 0.43 |
| Category documents, `relation_fact_share: 0` | 0.06 | 0.46 |
| Feature documents | 0.36 | 0.24 |

The numbers are Pearson's correlations over the 741 pairs of leaf categories, counting a leaf when its own noun appears (the last row comes from a run of 2,500 documents). Narratives carry the thematic signal and category documents the taxonomic one. The relation facts of a category document carry its small thematic signal, which `documents.relation_fact_share: 0` removes; `content_kind_weights: proportional` makes most of a category document's sentences relation facts, and moves its two correlations toward each other. `stats.yaml` also gives Spearman's correlations, the same correlations counted by referents, partial correlations that control each measure for the other, and a `check` block that says whether situational documents' co-occurrence tracks thematic relatedness more than encyclopedic documents' does, and taxonomic similarity less. In the default corpus, both hold, by both correlations and both counts.

### Grammar

The default grammar is English-like and uninflected: "the penguin swim". The settings in `grammar` change how the same propositions are realized, and never what they say. Changing the grammar changes no logical form and no sentence order.

**Word order.** `grammar.word_order.clause` takes any of the six orders (SVO, SOV, VSO, VOS, OVS, OSV). Two-way settings place determiners, adjectives, with-phrases, relative clauses, adpositions, auxiliaries, and negation before or after their head. The same sentence with `clause: SOV`, `adjective: after`, and `determiner: after`:

```
default:   THE CATEGORY.1.2 WITH PART.5 THAT EVENTTYPE1.1 EVENTTYPE1.3
reordered: CATEGORY.1.2 THE WITH PART.5 THAT EVENTTYPE1.1 EVENTTYPE1.3
```

**Morphology.** Number, tense, and aspect are off by default. Each can be turned on, and realized either as an affix or as a separate function word:

```
number and tense as affixes:  ALL CATEGORY.1.1-PLURAL HAVE PART.4
                              THE CATEGORY.1.2 WITH PART.5 THAT EVENTTYPE1.1-PAST EVENTTYPE1.3-PAST
number and tense as words:    ALL CATEGORY.1.1 PLURAL HAVE PART.4
                              THE CATEGORY.1.2 WITH PART.5 THAT EVENTTYPE1.1 PAST EVENTTYPE1.3 PAST
aspect as a word:             THE CATEGORY.1.2 WITH PART.5 THAT EVENTTYPE1.1 PROGRESSIVE EVENTTYPE1.3
```

With number on, verbs agree with their subjects (`number.agreement`), and `is` and `has` become `are` and `have` in the plural. Agreement holds across relative clauses, which creates long-distance dependencies. A word takes at most one affix.

**Ambiguity.** The logical form is never ambiguous, but the surface can be. `grammar.can_rate` gives the share of capacities that say "can". At the class level, it is 0.5 by default, so "penguins can swim" and "penguins swim" both occur. At the instance level, it is 1.0, so "the penguin can swim" always has "can". With the instance rate below 1, and tense unmarked, "the penguin swim" can be a capacity or an event. Each sentence records its possible readings (`readings`), worked out from its words alone, and `stats.yaml` reports how many sentences are ambiguous. Under the default grammar, the only ambiguous sentences are those about an instance with "can", which expresses a capacity (`ABLE`) and what was possible at a time point (`ABLE_NOW`) alike; `lexicon.can_words: distinct` removes the ambiguity with a second word. A class-level sentence also lists the quantifiers its words allow: "all" allows `nec_all` in the default language, and a bare plural allows `nec_all` and `most`.

## Reading the outputs

A run writes one folder:

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved configuration, all seeds, the world's identity (its source, name, seed, configuration hash, and rule-set identity), the git commit, package versions, and, after rendering, the word-form run's identity. |
| `lexicon.csv` | One row per lexeme: `label`, `pos`, `concept`, `word`, `spelling`, `gloss`, and `same_form_as` for a homonym. `word` and `spelling` are filled by rendering. |
| `documents.jsonl` | One JSON object per document, with every sentence's renderings, tree, and logical form. |
| `corpus.txt` | Every document in the spelled rendering (the formal rendering before rendering): one sentence per line, a blank line between documents. |
| `corpus_formal.txt`, `corpus_conceptual.txt`, `corpus_propositional.txt` | The same documents in the other renderings. |
| `scenes.jsonl` | One history per scene: its seed, its participants, their initial fluent values, and the events of each time step with the changes they made. The schema is the world package's (`tests/fixtures/world/README.md`, "Histories"). |
| `tests/<set>.jsonl` | The test sets (see "Test sets"). |
| `stats.yaml` | Counts, lengths, the quantifier mix, the state, result, and blocked sentences (`states`), the causal statements by kind (`causal`), ambiguity, mentions, the co-occurrence check, and the size of each test set, with the share of `changed` items and the counts of pairs by which item changed in each state set, and the number of distinct true statements in each causal set. |
| `wordform_request.yaml`, `wordform_meanings.csv` | The request for the word-form pipeline (see "Spoken word forms"). |

The text files are the easiest way into a corpus. For training a language model, one of the four `corpus*.txt` files is usually all we need.

### `documents.jsonl`

Each document holds its `label`, `type`, `topic`, `scenes`, `referents` (each referent label's instance), and `sentences`. Each sentence holds:

- `label`, `tokens` (lexeme labels), and `words` (word-form labels, after rendering);
- `text`, `formal`, `conceptual`, and `propositional`: the four renderings;
- `tree`: the parse tree, as nested lists. The sentence "the CATEGORY.1.2 with PART.5 that EVENTTYPE1.1 EVENTTYPE1.3" of the tiny language is `["S", ["NP-SBJ", ["Det", "LEXEME.42"], ["N", "LEXEME.3"], ["PP", ...], ["RC", ...]], ["VP", ["V", "LEXEME.29"]]]`. The subject and object are labeled by function, so a tree reads the same in every word order;
- `logical_form`: the proposition as JSON, with its truth `grounding`: how the truth was decided, and the proportion it rests on. The `level` is `class`, `instance`, `event`, `state`, `change`, or `able_now`; a sentence about a time point carries its `scene`, its `time` (`TIME.2`), and its `tense`. A predicate has a `kind` (`property`, `part`, `event_type1`, `event_type2`, `patient_capacity`, `scalar`, `member`, `state`, `effect`, or `precondition`) and names its symbol under `label`; a scalar predicate adds its comparison `class`, a two-place predicate its `patient`, and an effect or a precondition its `value`. Every mention of a sentence about instances says whether it is `descriptive`. A causal statement's subject is an event term (`head`, `event`, `role`), and its `causal` record repeats the definition entry it states;
- `referents`, `events`, `coreference`, and `distinguished`: for each noun phrase, the thing it refers to and the noun that names it; for each verb, the event it reports, or the time point it is about (`SCENE.8.TIME.2`); and whether each definite mention picks out its referent alone;
- `readings`: the kinds of meaning that the sentence's words allow: its level (`generic`, `capacity`, `event`, `state`, `able_now`, or `causal`), and, for a class-level sentence, the quantifiers.

## Test sets

Every corpus comes with test sets of matched pairs. Each true item is followed by a false item made from it by one minimal change. A model can be scored by whether it prefers the true sentence.

| Level | Sets |
| --- | --- |
| Class | `class_predicate`, `class_subject`, `class_quantifier`, `class_role`, and law-like twins of the first three |
| Instance | `instance_predicate`, `instance_subject`, `instance_role` |
| Event | `event_predicate`, `event_subject`, `event_role`, each split into `_possible`, `_blocked`, and `_impossible` |
| State | `state_predicate` and `state_subject`, each split into `_changed` and `_unchanged` |
| Able now | `able_now_predicate` and `able_now_subject`, each split into `_blocked` and `_impossible` |
| Causal | `causal_effect_<change>` and `causal_precondition_<change>`, with the changes `predicate`, `polarity`, `event`, and `role`, and a law-like twin of each but the polarity sets |

The changes are:

- **predicate:** another predicate of the same kind ("penguins can fly" for "penguins can swim");
- **subject:** another subject of the same level;
- **quantifier:** another quantifier ("all" for a "most" fact, or the extensional "all" for a law-like one);
- **role:** agent and patient exchanged ("mice eat owls"), or, for a causal statement, the other role of the event ("things that catch things become caught" for "things that things catch become caught");
- **polarity** (causal statements only): the opposite value ("become not caught");
- **event** (causal statements only): another event type, or category at the same level, that lacks the effect or the literal ("things that things chase become caught").

A pair from `class_quantifier_lawlike` of the tiny corpus, in the conceptual rendering:

```
true:   MOST CATEGORY.1.2 HAS PART.8
false:  ALL CATEGORY.1.2 HAS PART.8
```

Three kinds of item have sets of their own:

- **Law-like items.** A false `nec_all` or `nec_no` item whose extensional twin is true: every `CATEGORY.1.2` has `PART.8`, but no rule fixes it, so "all CATEGORY.1.2 have PART.8" is false as a law. No observation contradicts such an item, so it tests whether a model grasps lawfulness. These items are in the `_lawlike` sets. The pair above is one.
- **Possible, blocked, and impossible events.** A false event is one that never happened in its document's scenes. It is possible when its binding was legal at some time point of the scene (the requirement and the preconditions held, and the event still did not happen), blocked when the world allows the binding (the agent, or the pair, is `able`) but a precondition never held in the scene, and impossible when the world rules the binding out. A possible false event tests memory of the episode, a blocked one knowledge of the preconditions and the states, and an impossible one can be rejected by world knowledge alone.
- **States.** A state item continues a situational narrative and states a fluent of one of its participants at the scene's final time point, `HOLDS(SCENE.8, TIME.6, PAST, BOOLFL.3(REF.1))`. The false item names another fluent that is false of the referent then (a predicate swap), or another participant of which the fluent is false (a subject swap). Each item is marked `changed` when the fluent's value at the final time point differs from its value at `TIME.1`, and a pair whose true or false item changed goes into the `_changed` set: only tracking what the events did can answer it. A pair in `_unchanged` can be answered from the initial state alone. `stats.yaml` gives the share of changed items in each state set (about half in the `_changed` sets of the default corpus).
- **Able now.** An `able_now` item continues a situational narrative and says that a participant could do something at the scene's final time point, `ABLE_NOW(SCENE.8, TIME.6, PAST, EVENTTYPE1.3(REF.1))`. A false item is blocked (the binding is able, and not legal then) or impossible (not able), with the same two changes.
- **Causal statements.** A causal item states an effect or a precondition of an event type, and every false item is false under `NEC`: the definition lacks the entry. A false item that held after (an effect) or before (a precondition) every event of its type in the corpus's scenes, of which there was at least one, is marked `observed` and goes into the `_lawlike` twin of its set, as a law-like class item does: no observation contradicts it. A polarity swap is never observed, because the true effect or precondition guarantees the opposite value, so the polarity sets have no law-like twin. The causal sets are not drawn like the others: every true causal statement of the world is paired with every valid false item of the set's change, so a statement appears in several pairs of a set (stage a8). A set with more than `test_sets.size` such pairs keeps a draw of them stratified by statement (stage b1): the pairs of each statement are shuffled, and the set takes one pair from each statement in turn, round after round, until it is full, so that every statement appears before any appears twice, and every statement of the set is kept as long as there are no more statements than the size. Both items of a pair record the true statement they test, under `statement` (its propositional rendering) and under `statement_record` (its causal record with its `kind`, `effect` or `precondition`), so that an analysis can group a set's pairs by statement; `stats.yaml` counts the distinct statements of each causal set (`true_statements`).
- **Items in context.** An instance-level, event-level, state, or `able_now` item names a narrative document, and is tested as a continuation of it. "The penguin" in the item refers to that document's penguin.

Each item has two parts. `input` holds what a model sees: the document it continues, the sentence's tokens, renderings, tree, and logical form. `meta` holds the answer and the bookkeeping: `truth`, the change, the grounding, `possible` for events, `law_like` for class-level items, `changed` and `changed_item` (which item of the pair changed: `true_item`, `false_item`, `both`, or null) for state items, `observed`, the `causal` record, `statement`, and `statement_record` for causal items, and `seen`. An event item's aspect is drawn for the item, and its false item keeps it. True and false items never differ in the format of `input`, so the format never gives the answer away.

**Seen and unseen.** True test items are not held out of the documents. Instead, `seen` records whether an item's proposition appears in any training document. A false item is never seen. Scoring seen and unseen items separately tells memory apart from generalization. A report counts as seen in either aspect. In the default corpus, the share of true items that are seen runs from 1% (`instance_role`) to 95% (`event_subject_impossible`); the state and `able_now` items are about the scene's final time point, which no document states, so none is seen; every causal item is seen, because the default corpus's feature documents state every causal statement of the world. `stats.yaml` gives the share for each set.

The default corpus has 500 pairs in each set, except the law-like class sets, where the default world runs out of law-like false items at 395 pairs (`class_predicate_lawlike`) and 313 pairs (`class_subject_lawlike`), and the causal sets, which hold every pair of a true statement and a valid false item, capped at 500: the two event-swap sets are full (500 pairs, from 57 and 54 statements, every one kept by the stratified draw, with 195 and 72 pairs in their law-like twins), and the others hold what the world gives, 231 and 232 pairs for the predicate swaps (56 and 30 law-like), 59 and 54 for the polarity swaps, and 26 and 32 for the role swaps (11 and 4 law-like), effects before preconditions. `test_sets.size` sets the size, and `test_sets.changes` chooses the changes (the state and `able_now` sets take the predicate and subject changes). Test-set settings never change the documents.

## Spoken word forms

The corpus is made first, in symbols. The word-form pipeline then makes a spoken word for every lexeme, and `render` attaches the words to the corpus:

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/tiny.yaml
PYTHONPATH=python python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml
PYTHONPATH=python python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1
```

`generate` writes `wordform_request.yaml`: the lexemes with their parts of speech, the function words in order of their frequency in the corpus, the affixes the grammar needs, and the categories' meaning vectors. The word-form pipeline reads the request, makes pseudowords, assigns them to lexemes, and synthesizes them. `render` fills the spelled rendering, `corpus.txt`, the word columns of `lexicon.csv`, and the word-form run's identity in `config.yaml`. It changes nothing else in the corpus run.

For the tiny corpus, the three steps take under a minute (about 9 seconds for the word forms with audio and embeddings, and under a second for the other two). Using `forms` in place of `all` makes the word forms without audio, which is all that `render` needs. The rendered tiny corpus has 1078 tokens. For the default corpus, `data/wordforms/corpus_default.yaml` is the matching word-form configuration: it reads the request in `runs/corpus/default_seed1`, and writes `runs/wordforms/corpus_default_seed1`; its 199 content lexemes and 17 function words take about 30 minutes with audio and all five embeddings, and `render` about 2 seconds (October 9, 2026).

The word-form pipeline decides how sound relates to meaning: arbitrary, correlated at a target, marked by branch, or shaped by features. Its guide, `WORDFORMS.md`, covers this under "Assigning words to meanings" and "Word forms for a corpus". The words a lexeme gets never depend on the number of documents or the test sets, so a corpus can grow without its words changing.

## The configuration file

`data/corpus/default.yaml` lists every parameter with its default. A configuration file only needs the parameters it changes: `tiny.yaml` sets the world, the document count, and the test-set size, and nothing else. Unknown keys are errors, and every error names the file and the field. A key of the old corpus (`taxonomy`, `quantifiers.all_grounding`, `scene.verb_weights`, and the others renamed in stage a5a) is an error that names the new key.

The most useful parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `world` | `{config: data/world/default.yaml, seed: 1}` | The world: a world configuration and seed, defined in memory, or `{run: <folder>}`, a world run folder, which is regenerated in memory and checked against the folder. |
| `documents.count` | 10000 | Number of documents. |
| `documents.mix` | 0.3, 0.2, 0.2, 0.3 | Shares of category, feature, entity, and situational documents. |
| `documents.sentences` | 5–15, 5–20 | Length range for each type. Narratives often end sooner, when their events run out. |
| `documents.content_kind_weights` | equal | How a category document draws the kind of its next sentence: each kind with the same chance, or `proportional` to the facts of each kind left to state. |
| `documents.relation_fact_share` | null | Share of relation facts in category documents; 0 leaves them out. Null leaves the share to `content_kind_weights`. |
| `documents.progressive_rate` | 0.3 | Share of reports in the progressive aspect. With `documents.one_aspect_per_event` (true), a document reports an event in one aspect. |
| `documents.sibling_contrast_rate` | 0.2 | How often a category fact is followed by a contrasting sibling fact. |
| `documents.initial_state_rate` | 0.2 | How often a participant's first mention is followed by a sentence stating one of its fluents at the time point before its first event. |
| `documents.result_rate` | 0.5 | How often an event sentence is followed by a sentence stating a change the event's own effects made. |
| `documents.blocked_rate` | 0.1 | How often an event sentence is followed by a sentence saying what a participant could not (or could) do then. |
| `propositions.negation_rate` | class 0.1, instance 0.1, state 0.1, able_now 0.8 | Share of negative propositions: at the class and instance levels, among initial-state sentences, and among blocked sentences ("could not" rather than "could"). |
| `propositions.causal_statement_rate` | 0.5 | How often a sentence of a feature document about an event type or a category of event types is a causal statement. A document about a fluent holds causal statements alone, whatever the rate. |
| `lexicon.can_words` | shared | Whether "can" expresses both `ABLE` and `ABLE_NOW` (`shared`), or `ABLE_NOW` gets the word `can_now` (`distinct`). |
| `propositions.events.tense` | past | The tense of every report. |
| `quantifiers.universal_words` | nec | What "all" and "no" state: the `nec` quantifiers, the `extensional` ones, or `either`. |
| `quantifiers.most.usage_min` | 0.7 | The share from which a true `most` is stated. |
| `quantifiers.bare_plural.expresses` | [most] | What a bare plural states, besides `nec_all` for membership and rules. |
| `quantifiers.generic_rate` | 0.5 | How often a fact that a word could state is said with a bare plural instead. |
| `quantifiers.weights` | all 1 | Weights that rebalance the quantifier mix. The keys are `nec_all`, `all`, `most`, `some`, `none`, and `nec_none`. |
| `scene.participant_weights` | thematic 1.0, taxonomic 0.5, constant 0.1 | How scene participants are chosen. |
| `scene.policy` | uniform_event | How events are drawn at each step: `uniform_event` or `uniform_event_type`. |
| `scene.event_type_weights` | all 1 | Weights that rebalance the event types. |
| `mention.level_weights` | heaviest at the leaf | How often a mention names a higher category ("the bird"). |
| `mention.pronoun_rate` | 0.5 | How often a later mention becomes "it". |
| `mention.relative_clauses` | rate 0.1, depth 1 | How often noun phrases take relative clauses, and how deep they nest. |
| `grammar.word_order` | English | The word-order settings, with `before` (`after_predicate`): where the function word `before` of a precondition statement stands. |
| `grammar.morphology` | all off | Number, tense, and aspect: `enabled`, `realization` (`affix` or `word`), and `position`. |
| `grammar.can_rate` | class 0.5, instance 1.0 | Share of capacities that say "can". |
| `lexicon.synonym_rate`, `lexicon.homonym_rate` | 0 | Lexical ambiguity. |
| `renderings.propositional.referents` | local | `local` (`REF.1`, `REF.2`) or `instance` (the world's instance labels). |
| `renderings.propositional.descriptions` | marked | Whether the propositional rendering writes the descriptions in braces before the assertion (`marked`) or leaves them out (`omitted`). |
| `test_sets.size` | 500 | Pairs per test set. |
| `test_sets.changes` | all six | The changes: `predicate`, `subject`, `quantifier`, `polarity`, `event`, `role`. |
| `test_sets.seen.descriptions` | true | Whether a description counts as stated, for the `seen` mark. |

## Using the generator from Python

```python
from semantic_world.corpus import load_config, config_from_mapping, generate

config = load_config("data/corpus/tiny.yaml", seed=1)
corpus = generate(config)
corpus.write()                       # writes runs/corpus/tiny_seed1/

document = corpus.documents[5]
document.type, document.topic        # ('situational', 'SCENE.4')
sentence = document.to_json()["sentences"][14]
sentence["conceptual"]               # 'THE SCALARDIM.1.LOW CATEGORY.1.1 BECOME BOOLFL.2'
corpus.stats["ambiguity"]            # the ambiguity counts of stats.yaml
corpus.planner.world                 # the corpus's view of the world (semantic_world.corpus.World)
corpus.planner.world.result          # the WorldResult of the world package behind it
```

`config_from_mapping` builds a configuration from a dictionary, which is convenient for sweeping a parameter:

```python
import yaml

settings = yaml.safe_load(open("data/corpus/tiny.yaml"))
settings["grammar"] = {"word_order": {"clause": "SOV"}}
corpus = generate(config_from_mapping(settings))
```

`generate(config, world)` also takes a `World` that is already in memory, from `load_world(config)`.

## Recipes

- **A taxonomic corpus.** Only encyclopedic documents, with no relation facts: `documents.mix: {encyclopedic_category: 0.6, encyclopedic_feature: 0.4, entity: 0, situational: 0}` and `documents.relation_fact_share: 0`.
- **A thematic corpus.** Only narratives: `documents.mix: {encyclopedic_category: 0, encyclopedic_feature: 0, entity: 0.5, situational: 0.5}`.
- **A verb-final language.** `grammar.word_order: {clause: SOV}`. Add `adjective: after` and `relative_clause: before` for a more consistently head-final language.
- **Agreement and long-distance dependencies.** `grammar.morphology.number: {enabled: true}`, with `mention.relative_clauses: {rate: 0.3, max_depth: 2}`.
- **An English-like morphology.** Number with `verb_marks: singular` ("the penguin swims", "penguins swim"), and tense as an affix.
- **An ambiguous surface.** `grammar.can_rate: {class: 0.5, instance: 0.5}`, with tense and aspect unmarked. In the tiny corpus, 54 of the 203 sentences then allow both a capacity and an event reading. `ambiguity` in `stats.yaml` gives the count.
- **Symbols only.** Train on `corpus_conceptual.txt` or `corpus_propositional.txt`, and skip the word-form pipeline.
- **The extensional universals.** `quantifiers.universal_words: extensional`, for a language in which "all" means every existing instance. The `nec` quantifiers are then stated only by the bare plurals of membership facts and rule statements, which no observation can make law-like, so the law-like test sets are empty.
- **The old scene generator's draws.** `scene.policy: uniform_event_type` on a world with `fluents: {count: 0}` gives scenes with the participants of the corpus's old scene generator, with the kind of event drawn by `scene.transitive_share`.

## Troubleshooting

- **"quantifiers.weights.no: is written none".** YAML reads a bare `no` as the boolean false. In `quantifiers.weights`, the quantifier "no" is written `none`, and `nec_no` is written `nec_none`.
- **"taxonomy: is now world".** The corpus is about a world of the world package. Point `world.config` at a world configuration file (`data/world/default.yaml`), which names the taxonomy.
- **"scene.event_type_weights.EVENTTYPE2.1: not an event type".** A category of event types takes no weight: weight its event types.
- **"entities.csv differs from the regenerated world".** A run folder given as `world.run` is regenerated from its `config.yaml` and compared. Regenerate the run with the current code, or point the corpus at the world's configuration file instead.
- **"grammar.morphology.number.on: is now enabled".** The morphology switch is `enabled`. YAML reads a bare `on` as the boolean true.
- **A document shorter than its range.** Narratives end when their events run out, and encyclopedic documents end when they have nothing more to say. `stats.yaml` reports drawn and achieved lengths for each type.
- **A small test set.** A world can run out of true items for a set. `stats.yaml` reports the number of pairs in each set.
- **"the run was not made from this corpus's request".** `render` needs a word-form run made from this corpus's own `wordform_request.yaml`. A corpus generated again with other settings needs a new word-form run.
- **Too few words.** The word-form pipeline stops when its configuration makes fewer content words than the corpus has content lexemes. The default corpus has 199 content lexemes.
- **Relative paths.** Paths in a configuration are read from the folder the command runs in. Run from the root of the repository, as the examples do.

## Reference

- `docs/specs/CORPUS_GENERATOR.md`: the full design, including the logical-form schema, the propositional notation, and the test-item schema.
- `docs/DECISIONS.md`: every design decision, with who proposed and who decided it.
- `docs/proposals/`: the decisions made during the build, stage by stage.
- `docs/specs/WORLD_AND_LANGUAGE.md`: the world model, the labels, and the corpus's move onto it (stages a5a to a8), with the decisions WM.1 to WM.34 and the engineering choices WM.E1 and following in `docs/DECISIONS.md`.
- `docs/guides/WORLD.md`: the world the corpus is about; `docs/guides/TAXONOMY.md` and `docs/guides/WORDFORMS.md`: the taxonomy inside the world and the word forms that the corpus builds on.
