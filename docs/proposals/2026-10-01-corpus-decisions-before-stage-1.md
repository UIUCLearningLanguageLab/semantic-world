# Proposal: decisions before stage 1 of the corpus generator

October 1, 2026. Raised during the orientation for `docs/specs/CORPUS_GENERATOR.md`, before any code was written. Status: decided. Jon answered every question below on October 1, 2026. The specification was updated to match, and decisions 13 to 22 at its end point here.

## Questions that blocked stage 1

### 1. What is a homonym?

The specification read two ways. The request's `same_form_as` and the stage 7 acceptance test described two lexemes that share one word form. `lexicon.csv` ("concept or concepts") and the "Renderings" section described one lexeme with two concepts.

**Decision.** A homonym is two lexemes that share one word form. Every lexeme has exactly one concept. `lexicon.csv` holds one concept per lexeme. The conceptual rendering carries different information from the formal rendering only when synonyms are on, because synonyms share a concept label.

### 2. How is a taxonomy output folder loaded?

No code loads an output folder back into a `TaxonomyResult`, and the truth tests need the live objects: the rules, the cones, and `relations.holds`.

**Decision.** The corpus generator reads the folder's `config.yaml`, regenerates the taxonomy in memory with the folder's seed, and stops with an error when the regenerated result differs from the folder's files. The taxonomy code does not change.

### 3. What can the `taxonomy` setting name?

No taxonomy runs are saved on disk, so a configuration that names an output folder cannot run until someone makes the folder.

**Decision.** The setting names either an output folder or a taxonomy configuration file with a taxonomy seed. `data/corpus/tiny.yaml` and `data/corpus/default.yaml` use configuration files, so they run with nothing saved on disk.

### 4. Which verbs get no word?

In the tiny relations taxonomy, the verb category `V1` has no defining constraints, so its base relation holds for every pair of instances.

**Decision.** A verb or verb category whose relation holds for every pair of distinct instances gets no lexeme. The same goes for one whose relation holds for no pair. `stats.yaml` lists such verbs.

## Stage 7: the word-form pipeline

5. **The order of inflection.** Inflection runs today before the assignment. Two passes are fine: assign first, then make, synthesize, and embed the inflected forms. The cache keeps the second pass cheap, and with the edit distance no audio is needed before the assignment.
6. **Skipped joins.** A lexeme that must be inflected gets only a word that can take its affixes. Every assignment mode must respect that rule, as branch markers already do.
7. **The tiny configuration.** `data/wordforms/tiny.yaml` makes 20 words, and the tiny corpus needs 36 content lexemes. Stage 7 adds a new word-form configuration for the tiny corpus. `tiny.yaml` stays unchanged.
8. **Meanings.** When the request gives meanings, `assignment.meanings` must be null, and giving both is an error. Categories without a lexeme are left out of the assignment. A category's synonym lexemes are also assigned by the mode, using the category's meaning vector, so the synonyms of one category share its sound–meaning structure, including its branch marker.

## Later stages, answered now

- **Stage 4, determinism.** Word-order and morphology settings must never change a logical form. Anything that changes what a sentence says belongs to the planner: relative clauses, and the limits on adjectives and with-phrases. Those settings move from `grammar` to `mention`. The sentence limit counts content words only (see "Choices made by Claude").
- **Stage 5, variables.** Variables in the propositional rendering are labeled `X.1`, `X.2`, and so on, like referents, so any depth works.
- **Stage 2, vacuous truth.** A class-level proposition needs at least one instance in its subject set. Vacuous truths are never generated, in documents or in test sets.
- **Stage 2, rule statements.** A negated IS literal is realized as a subject relative clause ("things with wings that are not red can fly"). A term that would need more than one relative clause, or that reads a scalar threshold, is skipped. The run reports how many terms were skipped.
- **Stage 2, verb proportions.** The proportions are computed from the relation matrices, for any category pair and any verb or verb category. They are not limited to `relation_proportions.csv`.
- **Stage 2, class-level scalar poles.** They take the generic only. A top-level category is compared with all instances in the world.

## Choices made by Claude

Jon left the first choice open. The others are engineering choices made while building stage 1. Each one is the working design unless Jon changes it.

1. **The shape of the `taxonomy` setting.** A mapping with exactly one of `config` and `run`: `{config: data/taxonomy/relations.yaml, seed: 1}` or `{run: runs/taxonomy/relations_seed1}`. `seed` goes with `config` only, and `seed: null` uses the seed in the taxonomy file. Relative paths are read from the folder the command runs in, as in the word-form pipeline.
2. **The check of an output folder.** The regenerated result is written to a temporary folder, and every file except `config.yaml` is compared byte for byte with the folder's file. `config.yaml` is left out because it records the git commit. The error names the first file that differs.
3. **The taxonomy's identity.** The configuration hash is the SHA-256 of the taxonomy's resolved configuration, written as YAML. The seed is part of the resolved configuration.
4. **The sentence limit.** `max_sentence_tokens: 30` becomes `mention.max_content_words: 20`. The number of tokens depends on the morphology settings (a plural word, a past word), so the limit cannot be checked before realization without letting a grammar setting change what a sentence says. A content word is a noun, an adjective, or a verb. The default of 20 is a guess at what 30 tokens allowed, and is easy to change.
5. **Named proportions.** The number of named concepts of a type is the proportion times the number of concepts, rounded half up, and the named concepts are drawn at random. The two poles of a scalar dimension are named together. The generic noun `THING` and the function words are always named. Each type draws from its own part of the `corpus:lexicon` stream, so changing one type's proportion never changes which concepts of another type are named.
6. **Lexeme labels.** Labels are numbered in this order: the first lexeme of every named content concept, then the second lexemes of concepts with synonyms, then the function words. Content lexemes keep their labels when a grammar setting adds a function word (`are`, `have`, `PLURAL`), and first lexemes keep their labels when synonyms are turned on.
7. **Homonym pairs.** Homonyms come in pairs. The number of pairs is `homonym_rate` times the number of content lexemes, divided by 2, rounded at random to a neighboring whole number, so the expected share of content lexemes that share a word form equals `homonym_rate`. For each pair, the first lexeme is drawn at random from the lexemes not yet paired. Its partner is drawn from the unpaired lexemes of other concepts: of the same part of speech with probability `homonym_same_pos`, and of another part of speech otherwise. When the wanted kind of partner does not exist, the other kind is used. The later lexeme of a pair records the earlier one in `same_form_as`.
8. **Synonyms.** Each named content concept, `THING` included, gets a second lexeme with probability `synonym_rate`. Function words never get synonyms.
9. **`lexicon.csv`.** The file gains a `same_form_as` column, so homonyms are visible before word forms are attached.
10. **Negation's position.** `grammar.word_order.negation` takes `after_auxiliary` (the default) or `before_auxiliary`. Every negated verb phrase of the grammar has an auxiliary (`can`, `is`, or `has`), so "the verb or auxiliary" in the word-order table is always the auxiliary. Stage 4 builds the realization, and the reading can change then.
11. **Patient projections of verbs without a word.** A patient projection keeps its adjective even when its verb gets no word. Jon's decision 4 names verbs and verb categories only.

## Still open

These questions were raised in the orientation and not answered. Each one comes back when its stage starts.

- Stage 2: patient projections are predicates at instance level only. Should "mice are edible" exist at class level?
- Stage 2 or 3: where are the words of internal verb categories used?
- Stage 2: can a rule term pass `max_adjectives` or `max_with_phrases`, or is such a term skipped too?
- Stage 6: what makes an event-level test item false, and which instance does an instance-level test item refer to outside a document?
