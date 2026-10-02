# To do: the world-content and language programs

October 2, 2026. This list covers the taxonomy generator, the word-form pipeline, the corpus generator, and the planned connected-speech layer. It lists what remains: checks that Jon has not yet made, work that is specified but not built, known limitations, future additions, and the points where these programs meet the simulation. Each item points to where it is explained. The engine's own work is in `docs/specs/MILESTONE_1.md`.

## Where things stand

- **Taxonomy generator:** built (taxonomy stages 1 to 12a). Guide: `docs/guides/TAXONOMY.md`.
- **Word-form pipeline:** stages 1 to 7 built, plus the changes of corpus stages 7 and 7b. Stage 8 has no specification yet. Guide: `docs/guides/WORDFORMS.md`.
- **Corpus generator:** built (stages 1 to 7 and 7b). Guide: `docs/guides/CORPUS.md`.
- **Full chain:** verified at default scale on October 2, 2026: the default corpus (10,000 documents, 92,464 sentences), word forms for its 173 content lexemes with audio and five embeddings, and the rendered corpus with 14,736 test items.
- **Connected speech:** specified in `docs/specs/CONNECTED_SPEECH.md`, not built.
- **Decisions:** every design decision is in `docs/DECISIONS.md`.

## Checks for Jon

- [ ] Review the "Working design" rows of `docs/DECISIONS.md`. They were built as written, without an individual ruling. The ones that most shape the world and the language are TX.13–18, REL.7–13, WF.3 and WF.6–13, and CG.1–8.
- [ ] Settle the four rows marked ⚑ in `docs/DECISIONS.md` (WF.19, WF.23, WF.24, M1.1), listed under "Needs Jon's check".
- [ ] Verify the Kelly (1992) citation in `docs/specs/CORPUS_GENERATOR.md`, which was written from memory, especially its subtitle. Monaghan, Christiansen, and Chater (2007) was checked.
- [ ] Verify the references of `docs/specs/CONNECTED_SPEECH.md` (Fernald et al., 1989; Kuhl et al., 1997; Rost & McMurray, 2009; Saffran et al., 1996), which were written from memory.
- [ ] Confirm decisions 1 to 5 of `docs/specs/CONNECTED_SPEECH.md` before it is built: whole-utterance synthesis, one speaker per document, the weak-form rate, contextual embeddings, and placement in the word-form package.
- [ ] Choose the published measurements for the child-directed speech preset (`CONNECTED_SPEECH.md`, "Acoustic levers"). The preset values are not set until then.

## Specified, not built

- [ ] **Connected speech** (`docs/specs/CONNECTED_SPEECH.md`, stages 1 to 7): utterance synthesis, alignment, contextual embeddings, the concatenation baseline, pause modes, corpus documents, the speaker catalog, and register levers. Stage 6, the speaker catalog, can be built first, and applies to isolated words as well.
- [ ] **Word-form stage 8, a parametric formant synthesizer** (`docs/specs/WORDFORM_PIPELINE.md`, stage 8). It needs its own specification first. It would give exact control of every acoustic dimension, which Praat manipulation of Piper and espeak-ng audio only approximates.

## First use

- [ ] Train a first baseline model on the default corpus, and score it on the 16 test sets, separately for seen and unseen items. This checks that the corpus and its test sets are usable before larger studies.
- [ ] Compare training on the four renderings (conceptual, formal, propositional, spelled), and with sound embeddings as inputs (`examples/wordforms_lm_inputs.py`).

## Known limitations

- **One synthesis model.** Most voices come from one Piper model, so speaker invariance is easier than in real speech. Real recordings, or more engines, would make the speaker problem harder and more realistic. (Word-form stage reports.)
- **Pitch-range manipulation.** The measured pitch range misses its target by more than 5% for most manipulated clips (about 64% in the stage 5 check). The pitch median, formants, and duration are reliable. (Word-form stage 5 report.)
- **CPC on isolated words.** The CPC encoder tells words apart across speakers barely above chance. It is kept as a baseline, to be revisited with connected speech (`WORDFORM_PIPELINE.md`; WF.24).
- **The HuBERT layer for graded similarity.** The layer whose distances best track phoneme distance moves between layers 3 and 5 from run to run. Layer 8 stays best for word identity (`WORDFORMS.md`, "What the evaluation shows").
- **GPU results.** Learned encoders trained on the Mac's GPU differ slightly between runs. CPU results are bit-identical.
- **Short narratives.** About 57% of entity narratives and 63% of situational narratives end before their drawn length, because their scenes run out of events (`stats.yaml`).
- **Verb frequencies.** Two verbs of the default world get 62% of transitive events, because they are possible in nearly every scene. `scene.verb_weights` can rebalance them.
- **Feature documents.** Encyclopedic documents about a feature carry little taxonomic signal, because features cut across the tree. Category documents with `documents.relation_fact_share: 0` are the clean taxonomic condition (`CORPUS.md`, "Documents").
- **Short law-like test sets.** The default world gives only 196 and 172 pairs for two law-like test sets.
- **Scalar thresholds in rules.** Rule terms that read a scalar threshold (13 of 103 in the default world) are never stated, because no pole adjective states a threshold exactly.

## Future additions

From `docs/specs/TAXONOMY_RELATIONS.md`, "Future additions":

- [ ] Event schemas with mutable states, preconditions, and effects (see "Integration with the simulation").
- [ ] More constraint families: similarity, derived relations (converses, compositions), and context-dependent relations.
- [ ] A symmetry constraint, and relations that hold with a probability.
- [ ] Relations that depend on tree position, to test what nominal structure adds.
- [ ] Verbs with three or more arguments; scalars determined by rules; rules among verb features.

From `docs/specs/CORPUS_GENERATOR.md`, "Future additions" and "Out of scope":

- [ ] Sound differences between parts of speech (nouns and verbs with different sound profiles).
- [ ] Case marking beyond subject–verb agreement.
- [ ] Plural events, which would allow the event reading of "penguins swim".
- [ ] Stacked affixes (tense plus agreement on one verb).
- [ ] Comparatives ("bigger than"), which would also let rules with scalar thresholds be stated.
- [ ] A contrastive construction ("unlike penguins, gulls can fly").
- [ ] Dialogue, questions, and commands; sentence coordination.

## Integration with the simulation

These are the points where the world-content and language programs meet the engine, the agents, and the 3D worlds.

- [ ] **Taxonomy worlds as world content.** The taxonomy generator's categories, instances, features, and verbs could define the entity types and objects of the generative world, in the style of Blocks World and SHRDLU (`README.md`, "Two kinds of world"; `docs/ENTITY_DEFINITIONS.md`). Features would become properties that agents can perceive, and verbs actions that agents can take.
- [ ] **Events from the simulation.** The corpus's scene generator is a stand-in. Events should eventually come from simulation logs: what agents and animals actually did (`CORPUS_GENERATOR.md`, "Scenes and events").
- [ ] **Event schemas and the rule engine.** Verbs with preconditions and effects belong in the engine's rule engine, which already models actions in a PDDL-compatible layer. The relations specification names event schemas as the point where the taxonomy generator meets the world simulation (`TAXONOMY_RELATIONS.md`, "Event schemas").
- [ ] **One symbolic state.** The engine's principle is that pixels, propositions, and text are views of one world state. The corpus's propositional rendering and the engine's propositional sensor should share a notation, or have a documented translation between them.
- [ ] **Language delivered to agents.** Delivering sentences to agents in the running world is out of scope for the corpus generator so far. Word-form clips could be played as sound events that agents hear, and the full wave simulation of sound (`docs/specs/3D_FDTD_Acoustic_Simulation_Implementation_Specification.docx`) could carry them through space.
- [ ] **Animal agents.** The taxonomy's animal categories and their relations (eat, chase) could supply the animal species of the 3D world, with the developing organisms of `docs/specs/developmental-3d-artificial-animal-simulator-spec.md`.

## Housekeeping

- [ ] The old run folder `runs/wordforms/default_seed1_stage4` (about 1.7 GB) can be deleted.
- [ ] `runs/corpus/tiny_seed1` predates the stage 7b clause fix. Generating it again updates it.
- [ ] Optional: a Hugging Face token (`HF_TOKEN`) removes a warning, and speeds up the first download of HuBERT.
