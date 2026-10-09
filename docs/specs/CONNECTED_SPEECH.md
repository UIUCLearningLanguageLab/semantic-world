# Connected speech

Draft, September 30, 2026. Planned, not yet scheduled. This document specifies how Semantic World turns sequences of words into continuous speech, with coarticulation across word boundaries, reduced function words, and sentence prosody. It also plans the acoustic levers for the speaker population and for register, such as child-directed speech, which apply to isolated words as well. It extends `docs/specs/WORDFORM_PIPELINE.md` and takes its sentences from `docs/specs/CORPUS_GENERATOR.md`. The specification is written for handoff to Claude Code, or to any developer.

## Why

The word-form pipeline synthesizes every word alone, in its citation form. Within a word, the synthesis engines produce coarticulation: each sound is shaped by its neighbors. Across words, there is none, because no word is ever spoken next to another. Joining isolated clips into sentences would miss four things:

- **coarticulation across word boundaries:** the end of one word does not shape the start of the next;
- **reduction:** function words keep their full citation forms, so "the" never becomes "thuh";
- **sentence prosody:** no pitch declining across the sentence, and no lengthening at the ends of phrases;
- **hidden boundaries:** the joins between clips mark every word boundary, so a model learning to find words in speech is handed the answer.

The last point matters most. Segmenting continuous speech is a central problem for learners, and a classic topic in statistical learning (Saffran, Aslin, & Newport, 1996). Synthesized connected speech makes that problem real, with a known ground truth.

## What this layer adds

1. **Utterance phoneme strings.** A sentence's words, as one phoneme string with word boundaries marked, and with function words in weak forms where configured.
2. **Utterance synthesis.** Whole sentences synthesized at once, so the engine produces coarticulation and prosody.
3. **Alignment.** The start and end time of every word, and of every phoneme, in every utterance.
4. **Contextual embeddings.** An embedding for every word occurrence, computed from its span inside the utterance, and frame sequences for whole utterances.
5. **A concatenation baseline.** The same sentences built by joining citation clips, as a control condition in which boundaries are easy to find.

## Inputs

- **A word-form run** (required): the word forms, including function words and inflected forms (stage 4a of the word-form pipeline), the speakers, the audio cache, and the voice.
- **Sentences**, from one of two sources:
  - **a corpus run** (`documents.jsonl` from the corpus generator), where each sentence lists its lexemes and each lexeme maps to a word form;
  - **random sequences** from a word-form run: word sequences drawn from a configured bigram model or uniformly, with function words mixed in at a configured rate. Random sequences let the layer be built and tested before the corpus generator exists.

## Weak forms

Every function word gets a weak form beside its citation form. The weak form replaces the vowel with schwa (AH0), or drops the vowel when the result still passes the phonotactic check and still has a vowel somewhere in the syllable. Inside utterances, a function word takes its weak form at the configured rate `weak_form_rate` (default 0.7). Content words keep their citation forms and their stress. Function words are unstressed inside utterances. Every occurrence records whether it used the weak form.

## Utterance synthesis

- **Engines.** Piper is the main engine, because it produces realistic connected speech. Piper takes the whole utterance's phonemes, with spaces between words. espeak-ng is available as a second engine.
- **Speakers.** Each document gets one speaker, the way one person tells one story, drawn from the word-form run's speakers. A speaker's split (training or held out) carries over, so held-out speakers stay held out. With random sequences, each utterance gets a speaker.
- **Rate.** Each document gets a speaking rate, drawn from a configured range.
- **Pauses.** Pause placement is a configuration setting, `pauses.mode`:
  - `engine` (the default): only the pauses the engine produces itself, mostly at sentence ends and at its own phrase breaks;
  - `phrase`: a phrase break at every major phrase boundary of the sentence, given to Piper as punctuation in its phoneme input (this needs the corpus's parse trees; with random sequences, phrase boundaries are drawn at a configured rate);
  - `random`: in addition, a silence at randomly chosen word boundaries, at `pauses.rate`, with durations drawn from `pauses.duration_ms`. Random pauses stand in for the hesitation pauses of real speech.

  Every inserted pause is recorded in `occurrences.csv` with the boundary it follows. Pauses in real speech are an imperfect cue to word boundaries: most boundaries have none, and silences also occur inside words, in the closure before a stop consonant is released. Synthesized speech has the same two properties. Its pauses are more systematic than natural pauses, because the engine never hesitates, which is why the pause modes exist.
- **Documents.** A document is a sequence of utterances with a configured silence between sentences.
- **Cache and reproducibility.** As with words, audio is cached under a hash of the phoneme string, speaker, and settings, and the cache with its hashes is the reproducible artifact.

**Scale.** Sentence audio cannot be reused the way word clips can: every sentence and speaker pair is new audio. The word-form pipeline's stage 2 synthesized about 40 seconds of speech per second of compute on a laptop (27,400 seconds of audio in 12 minutes), so 100,000 sentences of about 2 seconds each would take about an hour and a half. Subsets for quick experiments take minutes.

## Alignment

Every utterance needs the start and end of each word and each phoneme. Two methods, in order of preference:

1. **The engine's own durations.** Recent versions of Piper may return phoneme alignments with the audio. Check the installed version first. When available, these durations are exact for the synthesized audio.
2. **Forced alignment.** The phoneme sequence is known exactly, so alignment is a well-posed problem. A CTC phoneme recognizer that outputs espeak-style IPA, such as `facebook/wav2vec2-lv-60-espeak-cv-ft`, fits the pipeline's IPA table, which follows espeak's conventions. The alignment runs with the known phoneme sequence as the target.

When both are available, stage 2 compares them and reports the boundary differences.

## Contextual embeddings and utterance frames

- **Utterance frames.** Each front end (log-mel, cochleagram) is computed over whole utterances and stored with an index, as the word-form pipeline stores clips. Segmentation models read these frames.
- **Contextual word embeddings.** A pretrained encoder (HuBERT, by default layer 8) runs over the whole utterance. A word occurrence's embedding is the mean of the encoder's frames within the word's aligned span. The same word therefore gets different embeddings in different contexts, as in real speech.
- **Fixed encoders** apply to each word's aligned span of front-end frames, as they apply to isolated clips.

## The concatenation baseline

The same utterances are also built by joining the citation clips of their words, with no gap by default and a configurable crossfade. The baseline has no coarticulation across words, no reduction, and no sentence prosody. Comparing a model's performance on the two conditions measures what connected speech costs a learner.

## Evaluation

- **Silence as a boundary cue.** For each condition and pause mode, a silence detector (a dip below a configured level lasting at least 30 ms) is scored against the aligned word boundaries. Two numbers: **recall**, the proportion of word boundaries that fall at a detected silence; and **precision**, the proportion of detected silences that fall at a word boundary. Silences inside words, such as stop closures, lower precision. The concatenation baseline should have high recall, and connected speech low recall. The report also lists how many detected silences fall inside words, by the phoneme that follows them.
- **Context effects.** For each word, the similarity between its contextual embeddings and its citation embedding, and same-different average precision across contexts and speakers, as in the word-form evaluation.
- **Weak forms.** The same measures separately for function words in weak and citation forms.
- **Intelligibility.** With real English words as the source, a Whisper model transcribes utterances, and the word error rate is reported for each engine.
- **Alignment.** When both alignment methods are available, the median and 95th-percentile boundary differences.

## Acoustic levers: speakers and register

This section plans the levers that change how the speech sounds, apart from what is said. The linguistic side of these differences, such as shorter sentences or more repetition in child-directed speech, belongs to the corpus generator. The speaker levers apply to isolated words (the word-form pipeline) as well as to utterances.

### The speaker population

Today a run draws its Piper speakers at random from the voice's several hundred speakers, and its espeak-ng speakers from configured variants. The plan adds four levers.

**1. A speaker catalog.** Every available speaker speaks a fixed calibration set once: 20 words and 5 sentences. The pipeline measures each speaker's:

- median pitch (F0), in Hz;
- pitch range: the 5th to 95th percentile of F0, in semitones;
- speaking rate, in syllables per second;
- an estimate of vocal tract length, from the average formant frequencies of the calibration vowels.

The results go in `speaker_catalog.csv`. The calibration set is small, so the whole Piper voice takes minutes to measure.

**2. Selection by traits.** `speakers.selection` chooses the run's speakers from the catalog:

- `random`: the current behavior;
- `spread`: speakers that cover the trait space as evenly as possible (farthest-point sampling on standardized traits), for high talker variability;
- `central`: speakers closest to the center of the trait space, for low talker variability;
- `filter`: speakers within configured ranges of each trait, for example only high-pitched voices.

Talker variability matters for learning. For example, Rost and McMurray (2009) found that infants learned a minimal pair only when it was spoken by several talkers. Selection makes high and low variability comparable conditions with the same number of speakers.

**3. Transformation.** `speakers.transform` changes speakers after synthesis, with Praat (the manipulation tools of word-form stage 5):

- a pitch shift, in semitones;
- a formant shift ratio, which models a longer or shorter vocal tract;
- a rate factor.

The transformation can create new virtual speakers from existing ones. It can also scale a whole population's variability: `variability_scale: 0.5` moves every speaker's traits halfway toward the population mean, and 1.5 spreads them out.

**4. Variation within a speaker.** The token perturbations that exist today (small random changes of rate and pitch for each recording) get their ranges set per population, and connected speech adds variation from one utterance to the next.

Every run reports the realized distribution of its speakers' traits, before and after any transformation.

### Register

A register is a set of acoustic settings applied to every utterance, and to isolated words. The default register, `adult`, changes nothing. Each lever is a number whose neutral value leaves the speech unchanged:

| Lever | What it does | Mechanism |
| --- | --- | --- |
| `rate_factor` | Slows or speeds the whole utterance. | the engine's own rate setting |
| `vowel_lengthening` | Stretches vowels only, by a factor. | Praat duration change, on the aligned vowels |
| `stress_enhancement` | Adds duration, pitch excursion (semitones), and loudness (dB) to stressed syllables. | Praat, on the aligned stressed syllables; the stress pattern of every word is known |
| `pitch_shift` | Raises or lowers mean pitch, in semitones. | Praat pitch change |
| `pitch_range_factor` | Expands or compresses pitch movement around each phrase's contour. | Praat pitch change |
| `final_lengthening` | Stretches the last syllable of each phrase. | Praat, on the aligned final syllables |
| pauses | Adds pauses, as in "Pauses" above. | the pause modes |
| `vowel_hyperarticulation` | Moves each vowel's formants away from the center of the vowel space. | **Planned, not buildable yet.** Needs the formant synthesizer (word-form stage 8), or formant editing beyond what Praat's global shift can do. |

The levers that act on vowels, stressed syllables, and phrase ends need phoneme alignments. Utterances get alignments from stage 2 of this specification. Isolated words can be aligned the same way.

**Presets.** A preset is a named set of lever values. `child_directed` will be the first preset. Its values should come from published measurements that compare child-directed with adult-directed speech, for example Fernald et al. (1989) on pitch and Kuhl et al. (1997) on vowels. Jon will choose them, and the configuration records the source of each value. Values are not invented in code.

**Checking a register.** For each register, the evaluation measures the realized features: vowel durations, mean pitch and pitch range, speaking rate, pause rate, and the ratios between stressed and unstressed syllables. It compares them with the preset's targets. With real English words as the source, it also reports Whisper's word error rate, because heavy manipulation can make speech unnatural or unintelligible.

## Labels

**Note (stage a6 of `WORLD_AND_LANGUAGE.md`, October 9, 2026).** The labels below predate the world-and-language refactor. When this layer is built, it takes the labels of that specification's "Labels" section: `UTTERANCE.<n>.SPEAKER.<m>` for `U.<n>.S.<m>`, `DOC.<n>.SENT.<k>.SPEAKER.<m>` for `D.<n>.<k>.S.<m>`, and the word-form labels of stage a6 (`WORD.<n>`, `SPEAKER.<n>`, `FUNCWORD.<n>`, `AFFIX.<n>`) wherever this document names a word form or a speaker. The word occurrence's label is not in that table and takes the same style when the layer is built. The table below keeps the old labels, as the record of the plan.

| Object | Label | Example |
| --- | --- | --- |
| Utterance from random sequences | `U.<n>.S.<m>` | `U.40.S.3` |
| Utterance from a corpus sentence | `<sentence label>.S.<m>` | `D.17.3.S.4` is sentence 3 of document 17, spoken by speaker 4 |
| Word occurrence | `<utterance label>.<position>` | `D.17.3.S.4.2` is the second word of that utterance |

## Outputs

| File | Contents |
| --- | --- |
| `utterances.csv` | One row per utterance: label, document, sentence, speaker, engine, phoneme string, rate, duration, cache path, SHA-256 hash, condition (`connected` or `concatenated`). |
| `occurrences.csv` | One row per word occurrence: label, utterance, position, word-form label, lexeme (with a corpus), kind (content, function, or inflected), weak form used, start and end times, alignment method. |
| `phones.csv` | One row per aligned phoneme: utterance, word occurrence, phoneme, start and end times. |
| `frontends/<name>/` | Utterance frames and index, as in the word-form pipeline. |
| `embeddings/<name>/occurrences.npy` | Contextual embeddings, one row per word occurrence, in `occurrences.csv` order. |
| `eval/connected.csv` | The evaluation measures above. |
| `speaker_catalog.csv` | One row per available speaker: engine, speaker ID or variant, median F0, pitch range, speaking rate, vocal tract length estimate. |
| `eval/speakers.csv`, `eval/register.csv` | The realized speaker traits of the run, before and after transformation, and the realized register features against their targets. |

## Placement and build stages

Put the layer in `python/semantic_world/wordforms/utterances/`, reusing the pipeline's synthesis, cache, front ends, and encoders. Streams: `wordforms:utterances` (speakers, rates, and random sequences) and `wordforms:weak_forms`.

Stages 1 to 4 need only a word-form run with closed-class forms, and can be built before the corpus generator exists. Stage 5 needs the corpus generator's grammar and documents (its stages 4 and 5). Stage 6 needs only Piper and word-form stage 5. Stage 7 needs stage 2 and word-form stage 5.

1. **Utterances from random sequences.** Weak forms, utterance phoneme strings, utterance synthesis, documents, and the cache. *Accept:* every utterance's phoneme string is the concatenation of its words' forms (weak forms where recorded); the cache is reused on a second run; utterances are intelligible by Whisper when the source is real English words (report the word error rate).
2. **Alignment.** Both methods where available. *Accept:* every word occurrence has a start and end within its utterance, in order, without overlap; report the boundary differences between methods.
3. **Contextual embeddings and utterance frames.** *Accept:* occurrence embeddings are reproducible on the CPU; the frame index recovers every utterance exactly.
4. **The concatenation baseline and the evaluation.** *Accept:* the recall and precision of silence as a boundary cue are reported for both conditions and every pause mode, and recall is clearly higher in the concatenated condition; inserted pauses appear at exactly the recorded boundaries; all evaluation measures are reported.
5. **Corpus documents.** Reading `documents.jsonl`, mapping lexemes to word forms, one speaker per document, and labels from the corpus. *Accept:* every sentence of a small corpus run is synthesized, aligned, and embedded, and every occurrence maps back to its lexeme and logical form.
6. **Speaker catalog, selection, and transformation.** Needs Piper and word-form stage 5, not the corpus; it can be built before stages 1 to 5, and applies to isolated words as well. *Accept:* the catalog's traits are reproducible; `spread` selections have a larger trait variance than `central` selections of the same size; a transformation's measured pitch and rate changes are within 5% of their settings; `variability_scale` changes the measured trait variance by the expected factor.
7. **Register levers.** Needs stage 2 (alignment) and word-form stage 5. *Accept:* each lever's measured effect is within a tolerance of its setting, and changes only what it targets (for example, `vowel_lengthening` leaves consonant durations unchanged); the neutral register leaves the audio unchanged; the register report compares every preset with its targets.

## Decisions to confirm

1. Whole-utterance synthesis with Piper is the main method, and joined citation clips are a control condition, not the main method.
2. One speaker per document.
3. Function words take weak forms at a rate of 0.7 inside utterances, and are unstressed.
4. Contextual embeddings are means over each word's aligned span of a whole-utterance encoder pass.
5. The layer lives inside the word-form package and reuses its cache and encoders.
6. Pause placement is a setting with three modes (engine, phrase, random), and the default adds no pauses beyond the engine's own.
7. Speakers are chosen by measured traits from a catalog, and populations can be transformed with Praat to change their variability.
8. Register levers are numbers with neutral defaults. Preset values, starting with child-directed speech, come from published measurements chosen by Jon.

## References

- Fernald, A., Taeschner, T., Dunn, J., Papousek, M., de Boysson-Bardies, B., & Fukui, I. (1989). A cross-language study of prosodic modifications in mothers' and fathers' speech to preverbal infants. *Journal of Child Language*, 16, 477–501.
- Kuhl, P. K., Andruski, J. E., Chistovich, I. A., Chistovich, L. A., Kozhevnikova, E. V., Ryskina, V. L., Stolyarova, E. I., Sundberg, U., & Lacerda, F. (1997). Cross-language analysis of phonetic units in language addressed to infants. *Science*, 277, 684–686.
- Rost, G. C., & McMurray, B. (2009). Speaker variability augments phonological processing in early word learning. *Developmental Science*, 12, 339–349.
- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274, 1926–1928.
