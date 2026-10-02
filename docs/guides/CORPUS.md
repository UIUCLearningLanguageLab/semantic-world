# The corpus generator

The corpus generator writes documents about the world that the taxonomy generator makes. The documents are in an artificial language. Every sentence states a proposition about the world, and every proposition is checked true against the world's data before it is written. Every sentence is stored with its words, its parse tree, and its logical form, so we always know what a sentence means and why it is true.

The corpus is a controlled source of language for models. Settings decide which kinds of documents the corpus holds, how the language orders and marks its words, and how ambiguous its surface is. Test sets of matched true and false sentences come with every corpus.

This guide covers running the generator, the ideas behind it, the output files, the test sets, the configuration file, and the Python interface. The design is specified in `docs/specs/CORPUS_GENERATOR.md`, and every design decision is listed in `docs/DECISIONS.md`.

**Status.** Complete: the lexicon, propositions and their truth, scenes and events, the grammar, the four document types, the test sets, the statistics, and spoken word forms through the word-form pipeline.

## Quick start

From the root of the repository (see `README.md` in this folder for setup):

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/tiny.yaml
```

The program prints one line:

```
wrote runs/corpus/tiny_seed1: 20 documents, 152 sentences, 870 tokens, 16 scenes, 16 test sets with 258 pairs
```

The tiny corpus is about the tiny relations world of the taxonomy generator: 6 categories, 12 instances, 1 scalar dimension, and a few verbs. It is small enough to read every file by eye. The generator makes the taxonomy in memory from `data/taxonomy/tiny_relations.yaml`, so no taxonomy run needs to exist first.

The default corpus is 10,000 documents about the default world with verbs:

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/default.yaml
```

It prints `10000 documents, 92464 sentences, 455011 tokens, 10062 scenes, 16 test sets with 7368 pairs`, and takes under a minute on a laptop.

Two options change a run without editing the configuration:

- `--seed N` replaces the corpus's master seed. The corpus seed is independent of the taxonomy's seed, so the same world can get different corpora.
- `--out DIR` writes the output folder somewhere else. The default is `runs/corpus/<name>_seed<seed>/`.

The example configurations in `data/corpus/` are:

| File | What it makes |
| --- | --- |
| `tiny.yaml` | The tiny corpus above, with 20 pairs per test set. |
| `default.yaml` | The default corpus. Every parameter appears with its default value, so this file is also the reference for defaults. |

## One sentence, four renderings

Every sentence is written four ways. Here is the first sentence of the tiny corpus:

| Rendering | The sentence |
| --- | --- |
| Conceptual | `A C1.2 THAT V1.1 A C2.2 V2 A C2.1` |
| Formal | `a/L.36 C1.2/L.3 that/L.49 V1.1/L.27 a/L.36 C2.2/L.6 V2/L.31 a/L.36 C2.1/L.5` |
| Propositional | `C1.2(R.1) AND C2.2(R.2) AND EVENT(SN.1.1, PAST, SIMPLE, V1.1(R.1, R.2)) AND C2.1(R.3) AND EVENT(SN.1.2, PAST, SIMPLE, V2(R.1, R.3))` |
| Spelled | `gea stess lur tect gea spurtish lonseetet gea clonstanzive` |

In English, with made-up glosses, the sentence says "a penguin that chased a fish hunted a seal". `C1.2`, `C2.2`, and `C2.1` are categories. `V1.1` is a verb, and `V2` is a verb category: a more general verb, like "hunt" above "chase".

- **Conceptual.** The sentence's words in order, each replaced by its concept's label. The conceptual rendering is a perfectly tokenized version of the language: it has the same word order, morphology, and ambiguities as the spelled rendering, but every word is a symbol for its meaning.
- **Formal.** Each word as its gloss and its lexeme label (`L.3`). It differs from the conceptual rendering only when synonyms are on, because two synonyms share a concept but are different lexemes.
- **Propositional.** The logical form. Every noun, modifier, and clause becomes its own proposition, joined by `AND`. Referents are labeled `R.1`, `R.2`, and so on within each document, so the rendering shows which noun phrases refer to the same thing. The propositional rendering is never ambiguous.
- **Spelled.** The readable spellings of the language's spoken word forms. These come from the word-form pipeline, after the corpus is made (see "Spoken word forms"). Before that, the spelled rendering is empty, and `corpus.txt` holds the formal rendering.

A model can be trained on any of the four. Comparing the conceptual and the spelled renderings isolates the effect of lexical form: the structure is identical, and only the words differ. The propositional rendering removes the language altogether.

## Concepts

### Labels

Labels follow the project's convention: formal labels, indices starting at 1, and periods between indices.

| Object | Label | Example |
| --- | --- | --- |
| Lexeme (a word of the language) | `L.<n>` | `L.42` |
| Document | `D.<n>` | `D.17` |
| Sentence | `D.<n>.<k>` | `D.17.3` is sentence 3 of document 17 |
| Scene | `SN.<n>` | `SN.8` |
| Event | `SN.<n>.<k>` | `SN.8.5` is event 5 of scene 8 |
| Referent, within one document | `R.<n>` | `R.2` is the second thing mentioned in its document |
| Variable, within one sentence | `X.<n>` | `X.1` in a quantified proposition |

Categories, features, instances, and verbs keep their taxonomy labels (`C1.3.2`, `IS.12`, `I1.3.2.5`, `V1.2`). A scalar pole is its dimension with `HIGH` or `LOW` (`SC.1.HIGH`: "big"). The generic noun "thing" is `THING`, and a function word's concept is its gloss in capitals (`THE`).

### The lexicon

Every concept of the world gets a word:

| Concept | Part of speech | Example gloss |
| --- | --- | --- |
| Every category, at every level | noun | penguin, bird, animal |
| IS feature | adjective | red |
| HAS feature | part noun | fins |
| CAN feature | intransitive verb | swim |
| Verb | transitive verb | chase |
| Verb category | transitive verb, more general | hunt |
| Exposed patient projection | adjective | edible |
| Scalar dimension | two adjectives, one per pole | big, small |
| The generic noun | noun | thing |

The language also has 15 function words: `a`, `the`, `all`, `most`, `some`, `no`, `not`, `can`, `is`, `has`, `with`, `without`, `and`, `that`, and `it`. Grammar settings can add more (see "Grammar").

The default world gives 188 lexemes: 173 content words and the 15 function words. A verb whose relation holds for every pair of instances, or for none, gets no word, because it says nothing. `stats.yaml` lists such verbs.

Two settings, both 0 by default, add lexical ambiguity. `lexicon.synonym_rate` gives a concept a second word. `lexicon.homonym_rate` makes two words of different concepts share one word form. `lexicon.named_proportion` leaves some concepts without a word, so propositions that need them are never stated.

### Propositions and truth

A proposition is a logical form that the generator checks against the world. There are three levels.

**Class level.** A statement about a category: "most penguins can swim", "penguins are birds", "owls hunt mice". A class-level proposition has a subject category, a predicate, and a quantifier: `all`, `most`, `some`, `no`, or the bare generic ("penguins swim").

- `most` and `some` are judged by the share of the category's instances with the predicate. `most` needs at least 70% (`quantifiers.most.min_proportion`).
- `all` and `no` are law-like by default: "all penguins swim" is true only when the world's rules fix swimming for every penguin, not merely when every existing penguin happens to swim. `quantifiers.all_grounding: observed` switches to the plain reading.
- The bare generic means `most` by default (`quantifiers.generic.means`).
- A document always states the strongest true quantifier: `all` before `most` before `some`.

A subject can be restricted: "red penguins", "penguins with fins", "penguins that can swim", "owls that eat mice". A restriction narrows the set of instances that the quantifier counts. "Owls that eat mice" are the owls that can eat at least one mouse.

**Rule statements.** The world's rules can be stated outright. Each term of a rule's minimal formula is a sufficient condition, stated with the generic noun "thing". From a feature document of the tiny corpus:

```
ALL IS.4 THING WITHOUT HAS.2 AND WITHOUT HAS.3 AND WITH HAS.5 HAS HAS.7
ALL THING WITH HAS.7 THAT IS NOT IS.6 CAN CAN.4
```

A negated IS feature becomes a relative clause ("that is not red"). Rule statements are exempt from the limits on sentence length, because rules of every complexity must be stateable. In the default world, 86 of the 103 rule terms are stated. The rest read a scalar threshold, which no adjective states exactly, or have no instance.

**Instance level.** A statement about one instance: "the penguin can swim", "the penguin has stripes", "the penguin is a bird". Truth is read from the instance's own values.

**Event level.** A report that something happened in a scene: "the owl chased the mouse". Every event has a tense and an aspect (simple or progressive), which the logical form always records, even when the language does not mark them.

Class-level and instance-level propositions can be negative: "penguins can not fly", "no fish have fur". About 10% are, by default (`propositions.negation_rate`). Events are never negated.

### Scenes and events

Narratives need things to happen, so the generator makes scenes. A scene starts with a seed instance and adds 2 to 6 other instances. Instances that are thematically related to the seed are more likely to join: a scene with an owl tends to have mice in it. Each scene then runs for 3 to 8 time steps. At each step, events are drawn among the things the participants can do: an intransitive event ("the penguin swam") for a CAN feature, and a transitive event ("the owl chased the mouse") for a verb whose relation holds for the pair.

Events never contradict the world: an event happens only where the world allows it. Events are drawn verb first, so a verb that holds for many pairs is no more frequent than one that holds for few. Events do not change anything, so a scene has no plot. Event schemas with changing states are a future addition.

### Documents

The corpus mixes four document types (`documents.mix`):

| Type | Default share | Topic | What it says |
| --- | --- | --- | --- |
| Encyclopedic, category | 30% | A category | Membership, the category's features, its subcategories' features, and relation facts ("owls eat mice"), with contrasts to sibling categories |
| Encyclopedic, feature | 20% | A feature or verb | Which categories have it, which lack it, and the rules it takes part in |
| Entity narrative | 20% | An instance | The instance's features, and the events it takes part in, across 2 to 5 scenes |
| Situational narrative | 30% | A scene | The scene's events in time order, with descriptions of the participants |

Part of an encyclopedic document about category `C1.2`, in the conceptual rendering:

```
C1.2 IS A C1
MOST C1.2 HAS HAS.6
SOME C1.2 THAT CAN.1 HAS HAS.4
C1.2 HAS HAS.8
MOST C1.1 HAS NO HAS.8
C1 V2 C1.2
```

The fifth sentence is a sibling contrast: `C1.2` has `HAS.8`, but most of its sibling `C1.1` lack it.

Sentences from a situational narrative:

```
A C2.1 V1.1 A C2.2
A C1.1 CAN.4
THE C2 WITH HAS.6 V1.1 THE C2 WITHOUT HAS.6
THE C1 WITHOUT HAS.6 CAN.4
THE C2 WITH HAS.6 CAN.4
```

In narratives, an instance is introduced with "a" and mentioned later with "the" or "it". A mention can name an instance by a higher category ("the bird" for a penguin). When a scene has two things that the noun fits, a definite mention adds adjectives or with-phrases until it picks out one: "the C2 with HAS.6" against "the C2 without HAS.6".

**The document mix is a lever.** Which words occur together in a document depends on the document type. Narratives put thematically related things together. Category documents put taxonomic neighbors together. In the default corpus:

| Documents | Co-occurrence with thematic relatedness | Co-occurrence with taxonomic similarity |
| --- | --- | --- |
| Entity narratives | 0.80 | 0.18 |
| Situational narratives | 0.72 | 0.15 |
| Category documents | 0.29 | 0.48 |
| Category documents, `relation_fact_share: 0` | 0.09 | 0.56 |
| Feature documents | 0.33 | 0.13 |

The numbers are Pearson's correlations over the 741 pairs of leaf categories, counting a leaf when its own noun appears. Relation facts carry almost all the thematic signal of category documents. With `documents.relation_fact_share: 0`, category documents become a clean taxonomic condition. `stats.yaml` also gives Spearman's correlations, the same correlations counted by referents, and partial correlations that control each measure for the other.

### Grammar

The default grammar is English-like and uninflected: "the penguin swim". The settings in `grammar` change how the same propositions are realized, and never what they say. Changing the grammar changes no logical form and no sentence order.

**Word order.** `grammar.word_order.clause` takes any of the six orders (SVO, SOV, VSO, VOS, OVS, OSV). Two-way settings place determiners, adjectives, with-phrases, relative clauses, adpositions, auxiliaries, and negation before or after their head. The same sentence with `clause: SOV`, `adjective: after`, and `determiner: after`:

```
default:  A C1.2 THAT V1.1 A C2.2 V2 A C2.1
reordered: C1.2 A THAT C2.2 A V1.1 C2.1 A V2
```

**Morphology.** Number, tense, and aspect are off by default. Each can be turned on, and realized either as an affix or as a separate function word:

```
number and tense as affixes:  MOST C1.2-PLURAL HAVE HAS.6
                              A C1.2 THAT V1.1-PAST A C2.2 V2-PAST A C2.1
number and tense as words:    MOST C1.2 PLURAL HAVE HAS.6
                              A C1.2 THAT V1.1 PAST A C2.2 V2 PAST A C2.1
```

With number on, verbs agree with their subjects (`number.agreement`), and `is` and `has` become `are` and `have` in the plural. Agreement holds across relative clauses, which creates long-distance dependencies. A word takes at most one affix.

**Ambiguity.** The logical form is never ambiguous, but the surface can be. `grammar.can_rate` gives the share of capacities that say "can". At the class level, it is 0.5 by default, so "penguins can swim" and "penguins swim" both occur. At the instance level, it is 1.0, so "the penguin can swim" always has "can". With the instance rate below 1, and tense unmarked, "the penguin swim" can be a capacity or an event. Each sentence records its possible readings (`readings`), worked out from its words alone, and `stats.yaml` reports how many sentences are ambiguous. Under the default grammar, none are.

## Reading the outputs

A run writes one folder:

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved configuration, all seeds, the taxonomy's identity, the git commit, package versions, and, after rendering, the word-form run's identity. |
| `lexicon.csv` | One row per lexeme: `label`, `pos`, `concept`, `word`, `spelling`, `gloss`, and `same_form_as` for a homonym. `word` and `spelling` are filled by rendering. |
| `documents.jsonl` | One JSON object per document, with every sentence's renderings, tree, and logical form. |
| `corpus.txt` | Every document in the spelled rendering (the formal rendering before rendering): one sentence per line, a blank line between documents. |
| `corpus_formal.txt`, `corpus_conceptual.txt`, `corpus_propositional.txt` | The same documents in the other renderings. |
| `scenes.jsonl` | One object per scene: its seed, its participants, and its events by time step. |
| `tests/<set>.jsonl` | The test sets (see "Test sets"). |
| `stats.yaml` | Counts, lengths, the quantifier mix, ambiguity, mentions, the co-occurrence check, and the size of each test set. |
| `wordform_request.yaml`, `wordform_meanings.csv` | The request for the word-form pipeline (see "Spoken word forms"). |

The text files are the easiest way into a corpus. For training a language model, one of the four `corpus*.txt` files is usually all we need.

### `documents.jsonl`

Each document holds its `label`, `type`, `topic`, `scenes`, `referents` (each referent label's instance), and `sentences`. Each sentence holds:

- `label`, `tokens` (lexeme labels), and `words` (word-form labels, after rendering);
- `text`, `formal`, `conceptual`, and `propositional`: the four renderings;
- `tree`: the parse tree, as nested lists. The first sentence above is `["S", ["NP-SBJ", ["Det", "L.36"], ["N", "L.3"], ["RC", ...]], ["VP", ["V", "L.31"], ["NP-OBJ", ...]]]`. The subject and object are labeled by function, so a tree reads the same in every word order;
- `logical_form`: the proposition as JSON, with its truth `grounding`: how the truth was decided, and the proportion it rests on;
- `referents`, `events`, `coreference`, and `distinguished`: for each noun phrase, the thing it refers to and the noun that names it; for each verb, the event it reports; and whether each definite mention picks out its referent alone;
- `readings`: the kinds of meaning that the sentence's words allow.

## Test sets

Every corpus comes with test sets of matched pairs. Each true item is followed by a false item made from it by one minimal change. A model can be scored by whether it prefers the true sentence.

| Level | Sets |
| --- | --- |
| Class | `class_predicate`, `class_subject`, `class_quantifier`, `class_role`, and law-like twins of the first three |
| Instance | `instance_predicate`, `instance_subject`, `instance_role` |
| Event | `event_predicate`, `event_subject`, `event_role`, each split into `_possible` and `_impossible` |

The changes are:

- **predicate:** another predicate of the same kind ("penguins can fly" for "penguins can swim");
- **subject:** another subject of the same level;
- **quantifier:** another quantifier ("all" for a "most" fact);
- **role:** agent and patient exchanged ("mice eat owls").

A pair from `class_quantifier` of the tiny corpus, in the conceptual rendering:

```
true:   SOME C2.1 IS CANBE.V2.2
false:  MOST C2.1 IS CANBE.V2.2
```

Three kinds of item have sets of their own:

- **Law-like items.** Under the law-like reading, "all penguins swim" can be false even when every penguin swims, because no rule fixes it. No observation contradicts such an item, so it tests whether a model grasps lawfulness. These items are in the `_lawlike` sets.
- **Possible and impossible events.** A false event is one that never happened in its document's scenes. It is possible when the world allows it, and impossible when the world rules it out. A possible false event tests memory of the episode. An impossible one can be rejected by world knowledge alone.
- **Items in context.** An instance-level or event-level item names a narrative document, and is tested as a continuation of it. "The penguin" in the item refers to that document's penguin.

Each item has two parts. `input` holds what a model sees: the document it continues, the sentence's tokens, renderings, tree, and logical form. `meta` holds the answer and the bookkeeping: `truth`, the change, the grounding, `possible` for events, `law_like` for class-level items, and `seen`. True and false items never differ in the format of `input`, so the format never gives the answer away.

**Seen and unseen.** True test items are not held out of the documents. Instead, `seen` records whether an item's proposition appears in any training document. A false item is never seen. Scoring seen and unseen items separately tells memory apart from generalization. In the default corpus, the share of true items that are seen runs from 2% (`instance_role`) to 90% (`event_predicate_possible`). `stats.yaml` gives the share for each set.

The default corpus has 500 pairs in each set, except two law-like sets: the default world runs out of law-like false items at 196 and 172 pairs. `test_sets.size` sets the size, and `test_sets.changes` chooses the changes. Test-set settings never change the documents.

## Spoken word forms

The corpus is made first, in symbols. The word-form pipeline then makes a spoken word for every lexeme, and `render` attaches the words to the corpus:

```
PYTHONPATH=python python -m semantic_world.corpus generate data/corpus/tiny.yaml
PYTHONPATH=python python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml
PYTHONPATH=python python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1
```

`generate` writes `wordform_request.yaml`: the lexemes with their parts of speech, the function words in order of their frequency in the corpus, the affixes the grammar needs, and the categories' meaning vectors. The word-form pipeline reads the request, makes pseudowords, assigns them to lexemes, and synthesizes them. `render` fills the spelled rendering, `corpus.txt`, the word columns of `lexicon.csv`, and the word-form run's identity in `config.yaml`. It changes nothing else in the corpus run.

For the tiny corpus, the three steps take under 20 seconds. Using `forms` in place of `all` makes the word forms without audio, which is all that `render` needs. For the default corpus, `data/wordforms/corpus_default.yaml` is the matching word-form configuration: it reads the request in `runs/corpus/default_seed1`, and writes `runs/wordforms/corpus_default_seed1`.

The word-form pipeline decides how sound relates to meaning: arbitrary, correlated at a target, marked by branch, or shaped by features. Its guide, `WORDFORMS.md`, covers this under "Assigning words to meanings" and "Word forms for a corpus". The words a lexeme gets never depend on the number of documents or the test sets, so a corpus can grow without its words changing.

## The configuration file

`data/corpus/default.yaml` lists every parameter with its default. A configuration file only needs the parameters it changes: `tiny.yaml` sets the taxonomy, the document count, and the test-set size, and nothing else. Unknown keys are errors, and every error names the file and the field.

The most useful parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `taxonomy` | `{config: data/taxonomy/relations.yaml, seed: 1}` | The world: a taxonomy configuration and seed, made in memory, or `{run: <folder>}`, a taxonomy output folder. |
| `documents.count` | 10000 | Number of documents. |
| `documents.mix` | 0.3, 0.2, 0.2, 0.3 | Shares of category, feature, entity, and situational documents. |
| `documents.sentences` | 5–15, 5–20 | Length range for each type. Narratives often end sooner, when their events run out. |
| `documents.relation_fact_share` | null | Share of relation facts in category documents; 0 leaves them out. Null gives them an equal chance with the other kinds. |
| `documents.sibling_contrast_rate` | 0.2 | How often a category fact is followed by a contrasting sibling fact. |
| `propositions.negation_rate` | 0.1 | Share of negative propositions, by level. |
| `propositions.events` | past, 0.3 | The tense of every event, and the share of progressive events. |
| `quantifiers.all_grounding` | fixed | `fixed` (law-like) or `observed`. |
| `quantifiers.generic.means` | most | What the bare generic means: `all`, `most`, or `some`. |
| `quantifiers.weights` | all 1 | Weights that rebalance the quantifier mix. The key for "no" is `none`. |
| `scene.participant_weights` | thematic 1.0, taxonomic 0.5, constant 0.1 | How scene participants are chosen. |
| `mention.level_weights` | heaviest at the leaf | How often a mention names a higher category ("the bird"). |
| `mention.pronoun_rate` | 0.5 | How often a later mention becomes "it". |
| `mention.relative_clauses` | rate 0.1, depth 1 | How often noun phrases take relative clauses, and how deep they nest. |
| `grammar.word_order` | English | The word-order settings. |
| `grammar.morphology` | all off | Number, tense, and aspect: `enabled`, `realization` (`affix` or `word`), and `position`. |
| `grammar.can_rate` | class 0.5, instance 1.0 | Share of capacities that say "can". |
| `lexicon.synonym_rate`, `lexicon.homonym_rate` | 0 | Lexical ambiguity. |
| `renderings.propositional.referents` | local | `local` (`R.1`, `R.2`) or `instance` (taxonomy instance labels). |
| `test_sets.size` | 500 | Pairs per test set. |

## Using the generator from Python

```python
from semantic_world.corpus import load_config, config_from_mapping, generate

config = load_config("data/corpus/tiny.yaml", seed=1)
corpus = generate(config)
corpus.write()                       # writes runs/corpus/tiny_seed1/

document = corpus.documents[0]
document.type, document.topic        # ('entity', 'I2.1.3')
sentence = document.to_json()["sentences"][0]
sentence["conceptual"]               # 'A C1.2 THAT V1.1 A C2.2 V2 A C2.1'
corpus.stats["ambiguity"]            # the ambiguity counts of stats.yaml
corpus.planner.result                # the TaxonomyResult the corpus is about
```

`config_from_mapping` builds a configuration from a dictionary, which is convenient for sweeping a parameter:

```python
import yaml

settings = yaml.safe_load(open("data/corpus/tiny.yaml"))
settings["grammar"] = {"word_order": {"clause": "SOV"}}
corpus = generate(config_from_mapping(settings))
```

`generate(config, result)` also takes a `TaxonomyResult` that is already in memory.

## Recipes

- **A taxonomic corpus.** Only encyclopedic documents, with no relation facts: `documents.mix: {encyclopedic_category: 0.6, encyclopedic_feature: 0.4, entity: 0, situational: 0}` and `documents.relation_fact_share: 0`.
- **A thematic corpus.** Only narratives: `documents.mix: {encyclopedic_category: 0, encyclopedic_feature: 0, entity: 0.5, situational: 0.5}`.
- **A verb-final language.** `grammar.word_order: {clause: SOV}`. Add `adjective: after` and `relative_clause: before` for a more consistently head-final language.
- **Agreement and long-distance dependencies.** `grammar.morphology.number: {enabled: true}`, with `mention.relative_clauses: {rate: 0.3, max_depth: 2}`.
- **An English-like morphology.** Number with `verb_marks: singular` ("the penguin swims", "penguins swim"), and tense as an affix.
- **An ambiguous surface.** `grammar.can_rate: {class: 0.5, instance: 0.5}`, with tense and aspect unmarked. In the tiny corpus, 42 of the 152 sentences then allow both a capacity and an event reading. `ambiguity` in `stats.yaml` gives the count.
- **Symbols only.** Train on `corpus_conceptual.txt` or `corpus_propositional.txt`, and skip the word-form pipeline.
- **The observed reading.** `quantifiers.all_grounding: observed`, for a world in which "all" means every existing instance. The law-like test sets are then empty.

## Troubleshooting

- **"quantifiers.weights.no: is written none".** YAML reads a bare `no` as the boolean false. In `quantifiers.weights`, the quantifier "no" is written `none`.
- **"grammar.morphology.number.on: is now enabled".** The morphology switch is `enabled`. YAML reads a bare `on` as the boolean true.
- **A document shorter than its range.** Narratives end when their events run out, and encyclopedic documents end when they have nothing more to say. `stats.yaml` reports drawn and achieved lengths for each type.
- **A small test set.** A world can run out of true items for a set. `stats.yaml` reports the number of pairs in each set.
- **"the run was not made from this corpus's request".** `render` needs a word-form run made from this corpus's own `wordform_request.yaml`. A corpus generated again with other settings needs a new word-form run.
- **Too few words.** The word-form pipeline stops when its configuration makes fewer content words than the corpus has content lexemes. The default corpus has 173 content lexemes.
- **Relative paths.** Paths in a configuration are read from the folder the command runs in. Run from the root of the repository, as the examples do.

## Reference

- `docs/specs/CORPUS_GENERATOR.md`: the full design, including the logical-form schema, the propositional notation, and the test-item schema.
- `docs/DECISIONS.md`: every design decision, with who proposed and who decided it.
- `docs/proposals/`: the decisions made during the build, stage by stage.
- `docs/guides/TAXONOMY.md` and `docs/guides/WORDFORMS.md`: the world and the word forms that the corpus builds on.
