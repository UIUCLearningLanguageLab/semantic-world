# The word-form pipeline

The word-form pipeline makes the spoken words of Semantic World's language. It generates English-like pseudowords, synthesizes each one with many voices, turns the audio into auditory representations, and computes a distributed sound embedding for every recording and every word. The embeddings can serve as input embeddings for language models, contrastive models, and other models of meaning.

This guide covers setup, running the pipeline, the ideas behind each layer, the output files, the evaluation table, and the Python interface. The design is specified in `docs/specs/WORDFORM_PIPELINE.md`.

**Status.** Stages 1 to 7 are complete: word forms, synthesis, auditory front ends, sound embeddings with their evaluation, closed-class forms (function words, affixes, and inflected forms), augmentation with Praat manipulation and the modulation front end, encoders trained on the world's own audio, and sound–meaning assignment (arbitrary, target correlation, branch markers, and acoustic mapping). The pipeline also makes the word forms of a corpus from the corpus generator's request (see "Word forms for a corpus").

## Setup

The pipeline needs more than the taxonomy generator does. Generating word forms needs only two small packages. Synthesis and embeddings need audio and machine-learning packages, a system program, and two model downloads.

**Python packages.** The `speech` extra in `pyproject.toml` lists them, pinned to tested versions. From the root of the repository:

```
python -m pip install -e ".[speech]"
```

That command builds the whole package, which needs the Rust toolchain. Without Rust, install the same packages directly and run with the `python/` folder on the path:

```
python -m pip install numpy polars pyyaml cmudict wordfreq scipy soundfile piper-tts torch transformers praat-parselmouth pyroomacoustics
PYTHONPATH=python python -m semantic_world.wordforms forms data/wordforms/tiny.yaml
```

Check the versions in `pyproject.toml` if anything misbehaves.

**espeak-ng.** The second synthesis engine is a system program. On a Mac:

```
brew install espeak-ng
```

**The Piper voice.** The main synthesis engine needs a voice file of about 80 MB. The pipeline says so when the voice is missing, and gives this command:

```
python -m piper.download_voices en_US-libritts_r-medium --download-dir runs/wordforms/voices
```

**The HuBERT model.** The pretrained embedding downloads `facebook/hubert-base-ls960` from Hugging Face on its first run, into the usual Hugging Face cache.

**Licenses.** Piper, espeak-ng, and praat-parselmouth are GPL-3.0. The repository is Apache-2.0. The pipeline keeps them apart: espeak-ng runs as a separate program, and Piper and parselmouth are imported only in the modules that use them.

**Disk.** A default run uses about 4 GB: the audio cache (1.8 GB), the run folder (1.8 GB, mostly the stored front ends), and the voice.

## Quick start

The tiny configuration runs in under a minute and needs only the fixed embeddings:

```
PYTHONPATH=python python -m semantic_world.wordforms embed data/wordforms/tiny.yaml
```

The full default configuration makes 500 content words and 15 function words, with 45 speakers and 46,350 recordings:

```
PYTHONPATH=python python -m semantic_world.wordforms eval data/wordforms/default.yaml
```

On a recent Mac, the default run takes about 35 minutes the first time: 12 minutes of synthesis, 4 of front ends, about 6 for HuBERT and 12 for the two learned encoders on the Mac's GPU, plus the evaluation. A second run reads everything it can from the cache and the stored files, and takes a fraction of that.

Each subcommand runs the layers up to its name:

| Subcommand | Runs |
| --- | --- |
| `forms` | word forms |
| `synth` | word forms and synthesis |
| `frontends` | through the auditory front ends |
| `embed` | through the sound embeddings |
| `eval` | through the evaluation |
| `assign` | word forms and their assignment to meanings |
| `all` | every layer |
| `check-ipa` | compares the pipeline's IPA table with espeak-ng |
| `check-whisper CONFIG` | transcribes real English words synthesized from their phonemes, as a check of intelligibility |

`--seed N` replaces the master seed, and `--out DIR` changes the output folder. The default folder is `runs/wordforms/<name>_seed<seed>/`.

## The four layers

### 1. Word forms

Word forms are English-like pseudowords. The generator learns English sound patterns from the CMU Pronouncing Dictionary, restricted to its common words (Zipf frequency 3 or above, about 29,000 words) without their regular inflections (about 19,000 words remain). Restricting to common words keeps out the unusual sound clusters of proper names, and leaving out the inflections (*walked*, *walking*, *walks*) keeps the pseudowords from ending in what sounds like an English affix: with the inflections in, 8% of the default words ended in *-ing*.

A word is built syllable by syllable. The number of syllables and the stress pattern are drawn first. Each syllable then gets an onset (the consonants before the vowel) and a rime (the vowel plus the consonants after it), drawn from counts for that position in the word. A candidate is rejected when:

- it contains a sequence of three sounds that never occurs in a common English word;
- it is a real English word, checked against the whole dictionary;
- it is too close to a word already accepted (`min_lexicon_distance`).

Every word has a readable spelling for human readers, such as *jorts*, *mondating*, or *speebic*. No model is given the spelling or the phonemes. They are ground truth for analysis.

A sample of default words: *kipast, dascrux, untood, pronlis, vaulvid, grittergric, dekning, wem, baddanling, threeved, rarny, rezz, saulen, speebic, brate, teckity, nelya, tangmit, frar, cunvic, orviz, swirrashing, duppic, curfing, tarpasher, bressa, varshy*.

The `source` setting can also give real English words (`english`), or a mix of real words and pseudowords (`mixed`). Real words come from the common words.

### 2. Synthesis

Two text-to-speech engines receive the phonemes directly, so English spelling never enters:

- **Piper**, a neural engine. The default voice, `en_US-libritts_r-medium`, has hundreds of speakers. The default run uses 40.
- **espeak-ng**, an older rule-based engine with a more robotic sound. The default run uses 5 voice variants.

Every word is spoken by every speaker, twice by default. The two tokens differ by small seeded changes in rate and pitch. A proportion of speakers (20% by default) is **held out**: held-out speakers never contribute to word embeddings, so they test generalization to new voices.

Most voices come from one Piper model, so the voices are more alike than real speakers are. Generalizing across speakers is therefore easier than in real speech. Real recordings, or more synthesis engines, would make the speaker problem harder and more realistic.

Clips are mono, 16 kHz, trimmed of silence, and set to a common loudness. A clip more than 1.8 times longer than its word's median is synthesized again. Piper sometimes stretches a word, and the retry catches most cases. Words that Piper stretches for most speakers are flagged `long_synthesis` in `words.csv`. The default run flags 6.

As a check of intelligibility, a Whisper speech recognizer correctly transcribes about 86% of real English words synthesized by Piper and about 65% of those synthesized by espeak-ng.

**The audio cache is the reproducible artifact.** Piper cannot reproduce a clip exactly on demand, and one espeak-ng variant varies slightly between runs. So every clip is stored once in the cache (`runs/wordforms/cache/`), under a hash of its phonemes, speaker, and settings, and every later step reads from the cache. `tokens.csv` records each clip's SHA-256 hash. For any dataset used in a paper, archive the cache together with the run folder. Regenerating from the configuration alone will not give identical audio.

### 3. Auditory front ends

Each front end turns a clip into a sequence of frames:

| Front end | What it is | Default |
| --- | --- | --- |
| `waveform` | the audio itself | not stored, because it would repeat the cache |
| `logmel` | log-mel spectrogram, the standard input of speech recognition models | 80 bands, 10 ms frames |
| `cochleagram` | a model of the cochlea: a bank of filters spaced like the ear's frequency resolution, with compression | 64 channels, 50 Hz to 8 kHz, 10 ms frames |
| `modulation` | spectrotemporal modulation: how fast the cochleagram changes in time (rates, in Hz) and how finely it ripples across frequency (scales, in cycles per octave), after the cortical model of Chi, Ru, and Shamma | off by default; rates 2–32 Hz, scales 0.25–4 cycles per octave, 8 frequency bands (200 channels) |

### 4. Sound embeddings

Each embedding gives one vector per token (recording). A word's embedding is the mean of its tokens from training speakers. The default configuration has five embeddings:

| Name | Encoder | Dimensions |
| --- | --- | --- |
| `cochleagram_fixed` | the cochleagram averaged in 10 time bins, reduced by principal components | 256 |
| `logmel_fixed` | the same, from the log-mel spectrogram | 256 |
| `hubert_base` | layer 8 of the pretrained HuBERT speech model, averaged over time | 768 |
| `contrastive_logmel` | a small convolutional encoder trained on the run's own clips so that tokens of the same word lie together (supervised by word identity) | 128 |
| `cpc_logmel` | a small self-supervised encoder trained by predicting its own future frames (contrastive predictive coding), with no word labels; a weak baseline: it tells words apart across speakers far worse than the others | 128 |

The fixed embeddings involve no learning. HuBERT was trained on human speech, so its embeddings stand for an adult English listener. Every output labels HuBERT as pretrained. The two learned encoders train on the training speakers' clean tokens only, never on inflected forms, are seeded from the run's seed, and are frozen afterwards; `meta.yaml` holds their training report, and `embed` runs them on new forms. On the default run (29,880 training tokens: 415 training words, content and function, from 36 training speakers, twice each) the contrastive encoder trains in about 1 minute and the CPC encoder in about 10 minutes on a laptop's Apple GPU; the GRU makes the CPC encoder the slow one.

**Held-out words.** One word in five (`training.held_out_word_proportion`) is held out from everything that is trained: the two learned encoders and the fixed encoders' projection. Held-out words still get audio and embeddings, and `words.csv` marks them (`split`). The evaluation gives every measure for all words, the training words, and the held-out words (`word_split`); the held-out rows say how an encoder does on words it has never seen.

**Talker normalization.** Any embedding can take `talker_normalization: true`, which subtracts each speaker's mean embedding (from that speaker's training-word tokens) from the speaker's tokens. It is off by default: it uses speaker identity, which a learner does not get for free, and handling speaker variability is part of what learners must do. The evaluation reports every embedding both ways (`talker_normalized`).

## Closed-class forms

Content words are an open class. A corpus also needs function words and affixes, a closed class with short, simple forms. The `closed_class` section of the configuration asks for them; `closed_class: null` gives a run of content words only.

- **Function words** have one syllable of a simple shape (CV, CVC, or VC), drawn from the sounds of common English monosyllables. A function word is never a common English word, never a content word, and never within one phoneme of another function word. The glosses are listed most frequent first, and the most frequent half get two-phoneme forms. Each one has a gloss for human readers, such as `the`, and the label `F.<n>`. The default configuration makes 15: for example *iss* (the), *muh* (and), *rau* (a), *gea* (is), *eep* (that). With `source: english`, each gloss is its English word instead, with its weak forms recorded.
- **Affixes** are bound forms of shape C, VC, or V with an unstressed vowel, such as `-AH0` or `-L`. They are never synthesized alone. An affix that more than 10% of the words cannot take is drawn again. The default makes three suffixes, glossed `PLURAL`, `PAST`, and `PROGRESSIVE`; `position: prefix` makes a prefix. Their labels are `AF.<n>`. With `source: english`, the three glosses are the English suffixes *-s*, *-ed*, and *-ing*, with English allomorphy (*cats*, *dogs*, *buses*).
- **Inflected forms** join a stem to an affix, labeled `W.<n>.AF.<m>`. When the plain join fails the phonotactic check, a glide goes between a vowel and a vowel (*acoo* + `AH0` becomes *acoowa*), and otherwise a schwa goes between stem and affix (*rosk* + `L` becomes *roskal*); `words.csv` records the repair (`join`: `none`, `schwa`, or `glide`). A pair that fails even with the schwa, or whose form is a common English word, is skipped and listed in `summary.yaml`. Inflecting 500 words with 3 affixes quadruples the audio, so the default inflects nothing (`inflect: []`), and the tiny configuration inflects every word with every affix.

The request can come from a separate YAML file instead, named by the top-level `request` setting. The corpus generator writes such a file. It lists `function_words` (glosses), `affixes` (glosses and positions), and `inflect` entries; the shapes and the other settings stay in the configuration. A request can also list the lexemes of a corpus (see "Word forms for a corpus"). The setting was `closed_class.request` in earlier versions, and a configuration that still names a file there gets an error that says so.

Closed-class forms never change a content word. They come from their own random stream, their audio is synthesized after the content words, the projection of a fixed embedding is fitted on content words only, and the content words' evaluation rows are the same with and without them. `words.csv` gains the columns `kind` (`content`, `function`, or `inflected`), `gloss`, `stem`, `affix`, and `join` (how an inflected form was joined: `none`, `schwa`, or `glide`), and a run with closed-class forms also writes `affixes.csv`.

The evaluation reports every measure for each kind separately and for all forms together (the `kind` column), each kind on its own sample of tokens. Inflected forms add the **stem AUC**: the probability that an inflected form's embedding is closer to its own stem's than to another stem's. It says how visible morphology is in an embedding. On the tiny configuration, the fixed embeddings reach 0.84 to 0.90.

## Augmentation and acoustic manipulation

Augmentation makes new tokens from cached clips, with seeded transformations. It is off by default; an `augmentation` section with a list of recipes turns it on:

```yaml
augmentation:
  recipes:
    - {name: noisy, noise: {kinds: [babble, speech], snr_db: [0, 20]}}
    - {name: room, reverberation: {rt60: [0.2, 0.8]}}
    - {name: shifted, speed_pitch: {speed: [0.9, 1.1], pitch_semitones: [-2, 2]}}
    - {name: voice, manipulation: {pitch_median_hz: [100, 250], formant_shift_ratio: [0.85, 1.2]}}
  proportion: 0.5     # each recipe is applied to half the tokens
  speakers: train     # all, train, or held_out
```

Each recipe applies its transformations in the order manipulation (Praat's "Change gender": pitch median, pitch range, formant shift, duration), speed and pitch, reverberation (a simulated room), and noise (white, pink, speech-shaped, or babble from other tokens), with every value drawn from its range. An augmented token is labeled after its source (`W.12.S.3.2.A.1` is recipe 1 applied to that token), keeps the source's word and speaker, and records the recipe and the drawn values in the `augmentation` column of `tokens.csv`. The `achieved` column holds each transformation's target beside the value measured right after it (median pitch, pitch range, formant ratio, duration, reverberation time, signal-to-noise ratio), and `summary.yaml` counts the tokens that miss a target by more than 5% and 10%. The formant ratio and the pitch shift of `speed_pitch` are measured frame by frame, over the frames that are voiced before and after the change. Augmented clips live in the cache like any clip. They go through the front ends and embeddings like any token; `SoundEmbeddings.token_augmented` marks them. Word embeddings leave augmented tokens out unless `word_embeddings: {tokens: all}`.

With augmentation on, the evaluation gives every measure for clean tokens, augmented tokens, both, and each recipe (the `tokens` column), plus a robustness measure: how well a token retrieves its own word's clean embedding (`robustness_ap`, with `robustness_top1` for the share of tokens whose nearest word is their own).

The Praat tools in `semantic_world.wordforms.praat` (`measure_pitch`, `change_pitch`, `change_duration`, `manipulate`, `measure_pitch_shift`, `measure_formants`, `measure_formant_shift`) work on any clip, whole or within a time range, for later work on connected speech. `praat-parselmouth` is GPL-3.0 and is imported only inside that module.

## What the evaluation shows

`eval/embeddings.csv` evaluates every embedding on a sample of 5,000 tokens. The main measures:

- **Same-different average precision.** Given pairs of tokens, how well does embedding distance pick out pairs of the same word? Reported within one speaker, across training speakers, and with held-out speakers. Chance is about 0.001 to 0.002.
- **Fidelity.** Does embedding distance track sound distance, measured as phoneme edit distance? Three versions: the rank correlation over all word pairs (`fidelity_spearman`), and two neighbor tests (`fidelity_auc` and `fidelity_auc_1v2`). The neighbor tests ask whether words one phoneme apart are closer than words three or more phonemes apart, or two phonemes apart. Chance is 0.5 for the neighbor tests.

Results for the default run, for content words, without talker normalization:

| Embedding | Within speaker | Across speakers | Held-out speakers | Spearman | Neighbors, 1 vs 3+ | Neighbors, 1 vs 2 |
| --- | --- | --- | --- | --- | --- | --- |
| cochleagram_fixed | 0.218 | 0.053 | 0.049 | 0.169 | 0.949 | 0.844 |
| logmel_fixed | 0.290 | 0.089 | 0.091 | 0.154 | 0.956 | 0.830 |
| hubert_base, layer 8 | 0.600 | 0.566 | 0.559 | 0.155 | 0.987 | 0.887 |
| contrastive_logmel | 0.947 | 0.954 | 0.935 | 0.181 | 0.991 | 0.897 |
| cpc_logmel | 0.192 | 0.009 | 0.012 | −0.027 | 0.873 | 0.813 |

Four things stand out:

- **The fixed embeddings show the lack-of-invariance problem.** They identify a word within one speaker, but barely across speakers. Raw spectra depend on the voice. That makes the fixed embeddings a meaningful baseline, not a failure.
- **HuBERT generalizes across voices** and captures fine sound similarity (0.89 for words one phoneme apart against words two apart).
- **The contrastive encoder identifies words almost perfectly across voices**, including the held-out words it never trained on (0.98 across speakers among the 100 held-out words). It is supervised by word identity.
- **The CPC encoder is weak on isolated words.** It identifies words within one speaker, but across speakers it is near chance. The spec keeps it as a baseline, to be revisited with connected speech.

Rows for the held-out words (`word_split: held_out`) cover 100 words instead of 500, so chance is higher (about 0.006 to 0.010), and so is average precision. Compare each row with its own chance columns, not with the rows for all words.

Talker normalization raises same-different precision across speakers for the fixed embeddings (0.053 to 0.091 for the cochleagram) and for HuBERT (0.566 to 0.644), and changes the contrastive encoder little.

The evaluation also sweeps every HuBERT layer. Word identity across speakers peaks at layers 7 to 9. The rank correlation with edit distance peaks in the early layers: at layer 3 in the stage 4 sweep, and at layer 4 in the current default run (0.21, with layers 3 and 5 at 0.19). The spec's guidance: use layer 3 when graded similarity across the whole range of phoneme distances matters, and layer 8 otherwise. Sweep rows (`basis: sweep`) build word embeddings from the sample's tokens only, so compare layers within the sweep, not against stored rows.

Every result appears twice, with and without the `long_synthesis` words. Leaving them out changes no number by more than 0.006.

## Output files

| File | Contents |
| --- | --- |
| `config.yaml` | The resolved configuration, all seeds, and each model's name and revision. |
| `summary.yaml` | Counts, rejections, synthesis statistics, duration retries, and the flagged words. |
| `words.csv` | One row per word: `label`, `arpabet`, `ipa`, `espeak`, `spelling`, `syllables`, `stress`, `log_probability` (under the English sound model), `english_neighbors` (English words one phoneme away), `nearest_english`, `lexicon_neighbors` (content words one phoneme away), `real_word`, `long_synthesis`, `kind`, `gloss`, `stem`, `affix`, `join` (how an inflected form was joined: `none`, `schwa`, or `glide`), and `weak_forms` (an English function word's other pronunciations). Content words come first, then function words, then inflected forms. |
| `affixes.csv` | One row per affix: `label`, `gloss`, `position`, `arpabet`, and `ipa`. Written when the run has closed-class forms. |
| `speakers.csv` | One row per speaker: engine, voice, speaker ID or variant, and `split` (`train` or `held_out`). |
| `tokens.csv` | One row per token: its word and speaker, synthesis settings, rate and pitch perturbations, `augmentation` (the recipe and drawn values of an augmented token), `mapping` and `control` (acoustic mapping: a mapped token's record, and true for an unmapped original), duration, retries, level, cache path, and SHA-256 hash. |
| `frontends/<name>/` | `frames.npy` (all tokens' frames, one after another), `index.csv` (each token's first frame and frame count), and `meta.yaml`. |
| `embeddings/<name>/` | `tokens.npy` (one row per token, in `tokens.csv` order), `types.npy` (one row per word, in `words.csv` order), and `meta.yaml`. |
| `eval/embeddings.csv` | The evaluation table. |
| `assignment/` | `lexicon.csv` (each meaning's word, or, with a corpus's request, each lexeme's word), `summary.yaml` (the sound–meaning correlation and its null distribution), and, with branch markers, `markers.csv`. Written when the run assigns words to meanings. |

Labels follow the project convention: words `W.12`, speakers `S.3`, and tokens `W.12.S.3.2` (token 2 of word 12 by speaker 3). Function words are `F.2`, affixes `AF.1`, and inflected forms `W.12.AF.1`; their tokens extend the labels the same way.

## Using the embeddings from Python

`SoundEmbeddings` loads one embedding of a finished run:

```python
from semantic_world.wordforms import SoundEmbeddings

emb = SoundEmbeddings.load("runs/wordforms/default_seed1", "hubert_base")
emb.types          # word embeddings: 515 x 768, in the order of emb.words
emb.tokens         # token embeddings: 46,350 x 768
emb.token_words    # each token's row in emb.words
emb.token_speakers # each token's row in emb.speakers
emb.token_held_out # True for tokens of held-out speakers
emb.token_mapped   # True for tokens changed by acoustic mapping
emb.control        # the control tokens of a run with acoustic mapping, held apart
emb.words          # the word table, as a polars data frame
emb.word_kinds     # each word's kind: content, function, inflected, or marked
tensors = emb.to_torch()
```

**Novel words.** `embed` and `embed_types` take new word forms as ARPAbet strings and run them through the same synthesis, front end, and frozen encoder:

```python
vectors = emb.embed(["B L IH1 K S T", "S N AO1 R P"])        # forms x speakers x dimensions
word_vectors = emb.embed_types(["B L IH1 K S T"])            # forms x dimensions
```

A form already in the lexicon returns its stored embedding. A new form is synthesized once, stored in the cache, and read from the cache afterwards. Synthesizing new forms needs the engines installed.

**Two example scripts** show the plumbing for the pipeline's main purpose:

- `examples/wordforms_lm_inputs.py`: a small transformer language model that takes sound embeddings through a learned projection in place of an embedding table. The same projected embeddings form the output layer, so the model can score a word it has never seen from its sound alone.
- `examples/wordforms_contrastive.py`: a CLIP-style model that aligns sound embeddings with meaning vectors from the taxonomy generator, and tests retrieval for held-out speakers and novel words.

Both run on the tiny configuration in under half a minute on a CPU. They are demonstrations, not experiments.

## Assigning words to meanings

With `assignment.meanings` set to a CSV file, the `assign` subcommand assigns content words to meanings and reports the correlation between sound distance and meaning distance, with a null distribution from random reassignments. The file's first column holds the meaning labels, and the other columns hold 0 or 1 features. The taxonomy generator's `categories_generative.csv` works as it is. A column with other numbers, such as a scalar dimension, is dropped and listed in the summary (`non_binary: error` stops the run instead). `categories` chooses the meanings that get words: `all`, `leaves`, or a list of labels.

```yaml
assignment:
  mode: target_correlation
  meanings: runs/taxonomy/default_seed1/categories_generative.csv
  target_correlation: {target: 0.3}
```

There are four modes:

- `arbitrary`: a random assignment. Sound says nothing about meaning, and the correlation is near 0.
- `target_correlation`: the assignment starts random, and words are exchanged while each exchange moves the correlation toward `target_correlation.target`. The summary says whether the target was reached within the tolerance. A target that the words cannot give is not an error: the closest value is reported, with `reached: false` and a warning in the summary and on the command line. `assignment.strict: true` makes a miss larger than the tolerance an error. With 500 words and 56 meanings, targets up to about 0.75 are reached.
- `branch_markers`: the words of each branch of the taxonomy share a marker syllable, at the start (`position: initial`) or the end (`final`) of the word. `depth: 1` marks the top branches (`CATEGORY.1`, `CATEGORY.2`, and so on), and `depth: 2` the branches below them. A marker is joined like an affix, with the same glide and schwa repairs. The marked forms are new words of kind `marked` (`W.12.M.2` is word 12 with marker 2). They are synthesized and embedded like any word, and their stem AUC is reported. Two markers differ by at least two phonemes (`branch_markers.min_distance: 2`), so that two branches never sound alike: markers such as *ri-* and *rih-* are not drawn together. When the markers cannot be drawn that far apart, the run stops with an error that names the setting.
- `acoustic_mapping`: the assignment is random, and semantic features then change the sound of a word's recordings. Each mapping names a feature column, a property, and an amount:

```yaml
assignment:
  mode: acoustic_mapping
  meanings: runs/taxonomy/default_seed1/categories_generative.csv
  acoustic_mapping:
    - {feature: PROPERTY.3, property: pitch, amount: 2.0}       # semitones
    - {feature: PROPERTY.7, property: formants, amount: 1.1}    # ratio; above 1 is a shorter vocal tract
    - {feature: PROPERTY.9, property: duration, amount: 1.2}    # factor
    - {feature: PROPERTY.12, property: tilt, amount: -3.0}      # decibels per octave; negative is duller
```

Every clean token of a word whose meaning has the feature gets a changed copy, labeled with `.M` (`W.12.S.3.2.M`). The mapped tokens are the mapped word's tokens from then on: they make the word's embedding, trained encoders and augmentation use them, and `SoundEmbeddings` returns them. The unmapped originals stay in the run as a control set (`control` is true in `tokens.csv`). The control tokens are used for nothing but comparison: `SoundEmbeddings` holds them apart in `emb.control`, and the evaluation reports the two sets as `mapped` and `control` in the `tokens` column. The `mapping` column of `tokens.csv` holds a mapped token's source, meaning, and mappings, and the `achieved` column holds each amount beside the shift measured on the changed clip. The summary reports the measured shifts, not the amounts: their mean and spread, how many tokens miss the amount by more than 5% and 10%, and the correlation between the feature and the measured shift. Acoustic mapping needs `praat-parselmouth`.

The sound distance is the phoneme edit distance, or, for the first two modes, the cosine distance of an embedding (`sound_distance: hubert_base`). The meaning distance is `hamming`, `cosine`, or `jaccard`. The assignment is written to `assignment/lexicon.csv` and `assignment/summary.yaml`, with `assignment/markers.csv` for branch markers. `assign` runs the word forms and the assignment alone, and `all` runs every layer with the assignment in place, so marked forms and mapped tokens are synthesized, embedded, and evaluated.

## Word forms for a corpus

The corpus generator (`docs/specs/CORPUS_GENERATOR.md`) writes its documents first, without word forms, and asks the pipeline for them. The chain has three steps:

```
python -m semantic_world.corpus generate data/corpus/tiny.yaml
python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml
python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1
```

`generate` writes `wordform_request.yaml` in the corpus's folder, with `wordform_meanings.csv` beside it. The word-form configuration names the request:

```yaml
request: runs/corpus/tiny_seed1/wordform_request.yaml
wordforms: {count: 60}
assignment: {mode: arbitrary}
```

`data/wordforms/corpus_tiny.yaml` is the example. `data/wordforms/corpus_default.yaml` does the same for the default corpus (`runs/corpus/default_seed1`), with the default word-form settings: 500 words, 45 speakers, and all five embeddings. On a laptop, the three steps take under 20 seconds for the tiny corpus. `forms` in place of `all` makes the word forms without audio, which is all that `render` needs, in about 2 seconds.

The request lists:

- `lexemes`: the content words of the corpus's language, each with its concept and its part of speech. A homonym names the lexeme whose form it shares (`same_form_as`);
- `takes`: for each part of speech, the affixes that its lexemes' words must be able to take;
- `function_words`, `affixes`, and `inflect`, as in any request. The function words come in order of their frequency in the corpus, and an `inflect` entry names lexemes;
- `meanings`: the meaning vectors of the categories, in a file beside the request.

**Every lexeme gets a form.** The lexemes of categories get their words by the assignment mode, from the request's meanings: `arbitrary`, `target_correlation`, `branch_markers`, or `acoustic_mapping`. `assignment.meanings` must then be null. The synonyms of a category are assigned one by one, and share its branch marker. Every other lexeme (features, verbs, and so on) gets a word at random from the words that are left. A homonym shares its partner's form. The words that no lexeme gets stay in the run, and can serve as novel words.

**Affixes.** A lexeme gets only a word that can take the affixes of its part of speech (`takes`): the joined form must pass the sound-pattern check, must not be a common English word, and must not sound like another form of the run. So every inflected form that the corpus needs exists. The inflected forms are made after the assignment, from whatever form each lexeme got. With branch markers, the plural of a category's word is the inflected form of its marked form (`W.12.M.2.AF.1`). Only the inflected forms that the corpus uses are made.

**What cannot change a lexeme's word.** The inflected forms that a corpus uses, and the order of its function words, depend on its documents and its test items. The words of the lexemes do not: the affix requirement comes from `takes`, which the corpus's grammar settings fix, and the function words are made after the assignment. Two requests that differ only in their `inflect` lists give byte-identical content words, assignments, and content-word embeddings.

**Outputs.** `assignment/lexicon.csv` has one row for each lexeme: `lexeme`, `meaning` (its concept), `pos`, `word`, `spelling`, `arpabet`, and `assigned` (the mode's name, `random`, or `same_form`), with `base_word`, `branch`, and `marker` under branch markers. `words.csv` gains a `pos` column: the part of speech of each assigned word, and both parts of speech for a form that homonyms share. `summary.yaml` counts the lexemes by how they were assigned.

The lexemes are assigned in every subcommand, `forms` included. With an embedding as the sound distance, the pipeline runs in two passes: the content words are synthesized and embedded, the lexemes are assigned, and the layers are made again with every form. The audio cache keeps the second pass cheap.

## Configuration

`data/wordforms/default.yaml` lists every parameter with its default, and `data/wordforms/tiny.yaml` shows a small configuration. The most useful settings:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `wordforms.count` | 500 | Number of words. |
| `wordforms.source` | pseudowords | `pseudowords`, `english`, or `mixed`. |
| `wordforms.syllables` | `{1: 0.3, 2: 0.5, 3: 0.2}` | Weights over syllable counts. |
| `wordforms.min_lexicon_distance` | 1 | Minimum phoneme distance between words. 1 allows minimal pairs, and 2 forbids them. |
| `wordforms.english_min_zipf` | 3.0 | Frequency floor for the words the sound patterns are learned from. Null uses the whole dictionary. |
| `wordforms.exclude_inflections` | true | Leave regular inflections (*-s*, *-ed*, *-ing*, *-er*, *-est*, *-ly*) out of those words. |
| `synthesis.engines.piper.speakers` | 40 | Number of Piper speakers. |
| `synthesis.engines.espeak.variants` | m1, m3, m7, f2, f4 | espeak-ng voice variants. |
| `synthesis.tokens_per_speaker` | 2 | Recordings of each word by each speaker. |
| `synthesis.held_out_speaker_proportion` | 0.2 | Share of speakers held out. |
| `synthesis.cache_dir` | `runs/wordforms/cache` | Where the audio lives. |
| `embeddings` | five embeddings | A list; each entry names an encoder, a front end or model, and its settings. For HuBERT, `layer` picks the layer, and `store_layers: true` keeps every layer (about 1.8 GB more). |
| `request` | null | A request file, such as a corpus's `wordform_request.yaml` (see "Word forms for a corpus"). |
| `closed_class` | 15 function words, 3 suffixes, no inflection | The closed-class request and its settings (see "Closed-class forms"). `null` gives content words only. `inflect: [{words: all, affixes: [PLURAL]}]` inflects every word with one affix. `function_words.source: english` and `affixes.source: english` use the English forms. |
| `frontends.modulation` | null | The modulation front end: `rates`, `scales`, and `bands`. |
| `augmentation` | null | Augmentation recipes (see "Augmentation and acoustic manipulation"). |
| `training.held_out_word_proportion` | 0.2 | Share of the content words held out from every trained encoder. |
| `assignment` | arbitrary, no meanings | The assignment of words to meanings (see "Assigning words to meanings"). |
| `word_embeddings.tokens` | clean | Which training-speaker tokens make a word's embedding: `clean` leaves augmented tokens out, `all` includes them. |
| `device` | auto | `cpu`, `cuda`, `mps`, or `auto`. CPU results are bit-identical across runs; GPU results differ by about 1e-6. |

To turn an engine off, set it to null, for example `espeak: null` under `synthesis.engines`. Leaving an engine out keeps it on, with its defaults.

## Troubleshooting

- **"the Piper voice ... is not in runs/wordforms/voices".** Run the download command the message gives.
- **"espeak-ng is not installed".** `brew install espeak-ng`, or set `synthesis.engines.espeak` to null.
- **Relative paths.** The cache and voice paths are relative to the folder the command runs in. Run from the root of the repository, as the examples do.
- **Tests skip.** Tests never download anything. They skip with a message when the voice or a model is missing.
- **"the request has N lexemes that need distinct forms".** The run makes fewer content words than the corpus has lexemes. Raise `wordforms.count`.
- **"closed_class.request: is now the top-level key request".** Move the request's path to `request:` at the top of the configuration.
- **A gloss reads as a boolean.** YAML reads a bare `no`, `yes`, `on`, or `off` as a boolean. Write such a gloss in quotes: `"no"`.

## Reference

- `docs/specs/WORDFORM_PIPELINE.md`: the full design, including the planned stage 8, a parametric formant synthesizer.
- `docs/specs/CONNECTED_SPEECH.md`: the plan for whole utterances, pauses, speakers, and register.
- `docs/proposals/`: decisions made during the build.
