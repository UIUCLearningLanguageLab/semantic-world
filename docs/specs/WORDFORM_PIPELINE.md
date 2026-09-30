# Word-form pipeline

Draft, September 29, 2026. This document is the build specification for a Python pipeline that generates the spoken word forms of Semantic World's language, and turns them into audio, auditory representations, and distributed sound embeddings. The specification is written for handoff to Claude Code, or to any developer, and should be complete without access to any other planning material.

## Goal

We want a distributed vector for the sound of every word. The vectors should come from sound, in the way modern speech models and the auditory system get them, and not from hand-coded phonological features. With such vectors:

- novel words are easy to introduce: we generate a new form, synthesize the form, and embed the audio;
- sound–meaning correspondence can be controlled and measured;
- models of meaning can take sound embeddings as their input embeddings.

The immediate goal is the last one. Sound embeddings should be usable as input embeddings for standard language models and for contrastive learning models as soon as possible. The build stages are ordered so that stage 4 delivers usable embeddings. Later stages add realism and control.

The pipeline has four layers. Each layer is computed from the one before, and every layer after the first offers several interchangeable options.

1. **Word forms.** English-like pseudowords, generated as phoneme sequences. The phoneme sequences are ground truth for analysis only. No model is given them.
2. **Synthesis.** Audio for every word, from several speakers, with several tokens per speaker.
3. **Auditory front ends.** Waveform, log-mel spectrogram, cochleagram, and later spectrotemporal modulation.
4. **Sound embeddings.** One vector per token and one vector per word, from fixed, pretrained, or learned encoders.

Out of scope:

- grammar and sentences (a separate specification will cover them);
- delivering sound to agents in the running world (the world's ears will later play the cached clips as sound events);
- the Rust engine. The pipeline is a standalone Python program and does not change `docs/CONTRACTS.md`.

## Labels

Labels follow the taxonomy generator's convention: formal labels, indices starting at 1, and periods between indices.

| Object | Label | Example |
| --- | --- | --- |
| Word | `W.<n>` | `W.12` |
| Speaker | `S.<n>` | `S.3` |
| Token (one recording) | `W.<n>.S.<m>.<k>` | `W.12.S.3.2` is token 2 of word 12 by speaker 3 |
| Function word | `F.<n>`, with a gloss | `F.2`, gloss `the` |
| Affix | `AF.<n>`, with a gloss | `AF.1`, gloss `PLURAL` |
| Inflected form | `W.<n>.AF.<m>` | `W.12.AF.1` is word 12 with affix 1 |

Tokens of function words and inflected forms extend their labels the same way: `F.2.S.3.1`, `W.12.AF.1.S.3.2`.

## Layer 1: word forms

### English source

The English source is the CMU Pronouncing Dictionary (CMUdict), in ARPAbet with stress marks.

The sound patterns of English are learned from common, uninflected words only. The common words are the CMUdict words whose Zipf frequency in the `wordfreq` package is at least `english_min_zipf` (3.0 by default). CMUdict holds many proper names, loanwords, and rare words, and their sound sequences do not belong in English-like pseudowords. With `english_min_zipf: null`, every CMUdict word is a common word. See `docs/proposals/2026-09-30-wordforms-common-words-and-spelling.md`. The pattern words are the common words without their regular inflections: a word is dropped when it is another CMUdict word plus plural or third-person -s or -es, past -ed, -ing, comparative -er or -est, or adverbial -ly, allowing for a dropped final e, a doubled consonant, and y to i, and when its pronunciation is the base's pronunciation plus the ending's sounds. Otherwise the pseudowords end in what sounds like an English affix (decided September 30, 2026: before the change, 8% of the default words ended in -ing and 19% in a consonant plus s or z). With `exclude_inflections: false`, the inflections stay. The default drops 9,651 of the 28,872 common words.

The pipeline syllabifies every CMUdict pronunciation by maximal onset, allowing only onsets that begin at least 1 in every 6,300 pattern words (4 words with the default, and 16 words with the whole dictionary without its inflections). From the syllabified pattern words, the pipeline counts, by type frequency:

- onsets, by syllable position (initial, medial, final) and stress;
- rimes (the vowel plus the coda, kept together as one unit), by syllable position and stress;
- phoneme trigrams, with word boundaries, over whole pronunciations.

Keeping the rime as a unit preserves the constraints between vowels and codas in English.

Rejecting real words, and counting English neighbors, still use the whole dictionary. Real words for the `english` and `mixed` sources come from the pattern words.

### Generating pseudowords

A word form is built syllable by syllable:

1. Draw the number of syllables from a configured distribution.
2. Draw the stress pattern. Multisyllabic words take initial stress with a configured probability, and otherwise stress on a random other syllable.
3. For each syllable, draw an onset and a rime from the counts for that syllable's position and stress.

A candidate is rejected when:

- it contains a phoneme trigram that occurs in no pattern word (the phonotactic check);
- it is the pronunciation of a CMUdict word, or lies closer to one than `min_english_distance` (phoneme edit distance), when real words are excluded;
- it lies closer than `min_lexicon_distance` to a word form already accepted.

With `min_lexicon_distance: 1`, minimal pairs are allowed. With 2, no two words differ by only one phoneme.

The source can also be `english` (real words, drawn by syllable count from the pattern words) or `mixed` (a configured proportion of each). Real words let us compare the sound embeddings of real words with language-model embeddings of the same words.

### Word-form statistics

Every word form records:

- ARPAbet, IPA, and espeak-ng phoneme strings;
- a readable spelling;
- the number of syllables and the stress pattern;
- the phonotactic log probability under the trigram model;
- English neighbors: the number of CMUdict words at phoneme edit distance 1, and the nearest CMUdict word;
- lexicon neighbors: the number of other word forms at edit distance 1.

The readable spelling comes from a table of common English spellings for each phoneme, chosen by position in the word. For example, the vowel of "my" is written "y" at the end of a word, and "i" with a silent "e" before a final consonant. A consonant is doubled after a stressed short vowel. The table is a data file, and the contexts that the table refers to are defined in code. The spelling is for human readers only. A real English word keeps its dictionary spelling. When a pseudoword's spelling equals the spelling of a pattern word, the pseudoword gets its next-best spelling instead (for example, "roum" and not "room" for `R UH1 M`). When two words would get the same spelling, the later word's spelling gets a numeric suffix.

### Phoneme mappings

The pipeline needs two mapping tables from ARPAbet: one to IPA (for Piper), and one to espeak-ng's phoneme mnemonics (for espeak-ng). Both tables live in data files, not code. As a sanity check of the IPA table, the pipeline compares the mapped IPA of a sample of CMUdict words with the IPA that espeak-ng produces from the spelled words, and reports the agreement.

## Layer 2: synthesis

### Engines

Two text-to-speech engines receive phoneme input directly, so no English spelling is involved:

- **Piper**, a neural engine (package `piper-tts`, from the `OHF-Voice/piper1-gpl` repository). Piper accepts raw phonemes with the syntax `[[ phonemes ]]`, and synthesis settings through a `SynthesisConfig`. The default voice is `en_US-libritts_r-medium`, a multi-speaker English model. Its many speakers give us realistic variability between speakers. Synthesis settings: `noise_scale`, `length_scale`, `noise_w`, and the speaker ID.
- **espeak-ng**, a rule-based formant engine, called as an external program. espeak-ng accepts phoneme mnemonics inside `[[ ]]`. Voice variants (for example `en-us+m3`, `en-us+f2`), pitch, and rate vary the speakers.

The installed Piper API should be checked when stage 2 starts, because the API changed at version 1.3.0.

`piper.voice_dir` is the folder that holds the Piper voice files. The voice is downloaded once, with `python -m piper.download_voices`, and the pipeline never downloads a voice by itself. `espeak.speakers` is the number of espeak-ng speakers. The default is one speaker for each variant, and with more speakers the variants repeat, each time with a newly drawn pitch and rate.

A third engine, a parametric formant synthesizer, comes in stage 8. Until then, exact acoustic control comes from manipulating the Piper and espeak-ng audio (see "Augmentation and acoustic manipulation").

### Speakers and tokens

- A configured number of speakers is drawn for each engine. A configured proportion of speakers is held out: held-out speakers never train a learned encoder, and never contribute to word embeddings. Held-out speakers test generalization to new voices.
- Every word is synthesized by every speaker, with a configured number of tokens per speaker. Tokens differ through the engine's own variability and through small seeded perturbations of rate and pitch.
- A neural engine now and then stretches a word far beyond its usual length. After synthesis, the pipeline compares each clip's duration with the median duration of the same word across all tokens. A clip longer than `duration_check.max_ratio` times its word's median (1.8 by default) is synthesized again with a new perturbation seed, up to `duration_check.max_tries` tries in all (5 by default). When no try passes, the shortest try is kept. `tokens.csv` records the number of tries, and the run's summary records how many clips were tried again and how many still exceed the limit.
- Piper also stretches a few words for most speakers, and the retry rule does not catch those words. `words.csv` marks them with the flag `long_synthesis`. A word is flagged when the median duration of its Piper tokens is more than `long_synthesis_ratio` times (1.6 by default) the median for words with the same number of syllables. The flag does not change the audio. Later analyses can leave the flagged words out.

### Audio

Audio is mono, 16 kHz, 32-bit float in memory, and FLAC on disk. Leading and trailing silence is trimmed with a configured threshold, keeping a configured margin. The trim margin holds within 16-bit rounding: a clip is trimmed before it is stored as 16-bit audio, so about 0.6% of stored clips exceed the margin, most of them by under 5 milliseconds and the longest by 90 milliseconds in the default run. Every clip is then scaled to a target RMS level (`level.rms_db`, in dB relative to full scale, −24 by default), so that clips from different engines and speakers are equally loud. A peak guard limits the scaling: when the scaled clip's peak would exceed `level.max_peak` (0.9 by default), the clip is scaled to that peak instead. Synthesis happens once. Audio goes into a cache folder, indexed by a hash of the phoneme string, engine, speaker, and settings. Every later step reads from the cache.

### Reproducibility

Neural synthesis may not be bit-reproducible, because the noise inside a Piper model may not be controllable by a seed. Stage 2 tests whether the same settings reproduce identical audio for each engine, and records the result. When an engine is not bit-reproducible, the cached audio, identified by its SHA-256 hash, is the reproducible artifact, and the run records the hashes. Every perturbation the pipeline itself applies is seeded.

## Layer 3: auditory front ends

Each front end turns a clip into a matrix of frames by channels. The front ends are:

| Front end | Description | Default settings |
| --- | --- | --- |
| `waveform` | The trimmed audio itself, for encoders that take waveforms. | 16 kHz |
| `logmel` | Log-mel spectrogram, the standard input of speech recognition models. | 80 mel bands, 25 ms windows, 10 ms hops |
| `cochleagram` | A model of the cochlea: a bank of filters spaced on the ERB scale, envelope extraction, power-law compression, and downsampling. | 64 channels, 50 Hz to 8 kHz, compression exponent 0.3, 100 frames per second |
| `modulation` (stage 5) | Spectrotemporal modulation, after the cortical model of Chi, Ru, and Shamma (2005): two-dimensional Gabor filters over the cochleagram at configured temporal rates and spectral scales. | rates 2–32 Hz, scales 0.25–8 cycles per octave |

The cochleagram is a NumPy implementation, inside the package, of the filter bank of `pycochleagram` and `chcochleagram`, both from the McDermott lab. Neither package is on PyPI, so the pipeline does not depend on them. A test compares the implementation with saved outputs of `chcochleagram` (`tests/wordforms/fixtures/cochleagram_reference.npz`). The 64 channels are 62 band-pass filters plus the low-pass and high-pass filters that complete the bank.

Front-end output is stored as one float32 array per front end, with all clips concatenated along the frame axis, plus an index of each token's first frame and number of frames. That layout loads with memory mapping and needs no extra dependency.

## Layer 4: sound embeddings

### Encoders

Each embedding configuration names an encoder, and the encoder's output is one vector per token. The encoders are:

- **Fixed.** A front end's frames are averaged within a configured number of equal time bins, and the bins are concatenated. An optional principal component projection, fitted on training-speaker tokens, reduces the dimension. No learning is involved beyond the projection.
- **Pretrained.** A pretrained speech model from Hugging Face `transformers`: HuBERT (`facebook/hubert-base-ls960`), wav2vec 2.0, WavLM, or the Whisper encoder. The configuration names the model and the layer. The layer's frame outputs are mean-pooled over the clip. Pretrained models were trained on human speech, so pretrained embeddings stand for an adult English listener. Every output labels them as pretrained.

  The default layer of HuBERT base is layer 8. In the stage 4 layer sweep (default configuration, 500 words, 45 speakers), layer 8 has the highest same-different average precision across training speakers (0.57) and with held-out speakers (0.57), against 0.39 and 0.32 for layer 6. Use layer 3 when graded similarity across the whole range of phoneme distances matters (it has the highest rank correlation with edit distance); otherwise use layer 8. In the sweep, the Spearman correlation between embedding distance and phoneme edit distance is 0.25 for layer 3 and 0.13 for layer 8. The neighbor AUC measures favor the later layers: distance 1 against distance 3 or more gives 0.945 for layer 3 and 0.990 for layer 8, and distance 1 against distance 2 gives 0.840 for layer 3 and 0.892 for layer 8.

  A run stores the configured layer only. With `store_layers: true`, the run also stores the pooled output of every layer (`layers.npy`), which is large (1.8 GB for the default configuration).
- **Learned on the world's audio** (stage 6). Two kinds:
  - a contrastive acoustic word encoder, trained so that tokens of the same word from different speakers lie close together (in the manner of Kamper et al., 2016). The contrastive encoder uses word identity as supervision;
  - a self-supervised encoder trained by prediction alone, in the manner of contrastive predictive coding (van den Oord et al., 2018), with no word labels. Its frame outputs are mean-pooled like a pretrained model's.

  Learned encoders train on training speakers only, and are frozen afterwards.

### Word embeddings

A word's embedding is the mean of its tokens' embeddings over training speakers. Token embeddings are kept too, because token variability is part of what models should face.

### Evaluation

Every embedding configuration is evaluated in the same way:

- **Same-different average precision.** Over pairs of tokens, rank pairs by embedding distance, and compute the average precision for detecting pairs of the same word. Reported for pairs within a speaker, across training speakers, and involving held-out speakers. This is the standard evaluation for acoustic word embeddings.
- **Phonological fidelity.** Two measures on the word embeddings. The first is the correlation (Pearson and Spearman) between embedding distance and phoneme edit distance across word pairs. The second is the neighbor AUC: for a word, the probability that a word at phoneme edit distance 1 is closer in embedding space than a word at distance 3 or more, averaged over the words that have a neighbor at distance 1. The neighbor AUC is near its ceiling for every embedding, so a harder version stands beside it: distance 1 against distance 2 exactly, averaged over the words that have neighbors at both distances. The table gives the number of words behind each AUC. The correlation measures graded similarity over the whole lexicon. The two AUC measures ask whether minimal pairs sound alike.
- **Layer sweep.** For pretrained models, the evaluation runs over every layer, so the default layer can be chosen from evidence. The sweep runs the model on the evaluation sample, so a run does not have to store every layer. In a sweep row, a word's embedding is the mean over the word's training-speaker tokens in the sample.

Embedding distance is cosine distance. A run with many tokens has too many pairs, so the evaluation uses a seeded sample of 5,000 tokens, drawn from the `wordforms:eval` stream. Every embedding is evaluated on the same sample. Chance for the average precision is the proportion of same-word pairs.

When some words are flagged `long_synthesis`, every row of the evaluation is given twice: for all words, and without the flagged words.

### Novel words

`SoundEmbeddings.embed(forms, speakers)` runs a new word form through the same pipeline: synthesis, the front end, and the frozen encoder. The result for a form already in the lexicon must equal the stored embedding for the same speaker and settings, within numerical tolerance. `embed` must look up the audio cache before synthesizing, because Piper cannot reproduce a clip on demand.

## Closed-class forms: function words, affixes, and inflected forms

The corpus generator (`docs/specs/CORPUS_GENERATOR.md`) needs three kinds of form beside content words. Content words are an open class: a language keeps adding them. Function words and affixes are a closed class: a small, fixed set used constantly. Real languages give closed-class items short, simple forms, and the pipeline does the same.

### The request

The closed-class forms a run needs are listed in a closed-class request, given in the configuration or in a separate YAML file named there. The corpus generator will write such a file. The request lists glosses, which are names for human readers, and says which words to inflect:

```yaml
function_words: [the, and, a, is, that, it, with, not, all, can, has, "no", some, most, without]   # most frequent first
affixes:
  - {gloss: PLURAL, position: suffix}
  - {gloss: PAST, position: suffix}
  - {gloss: PROGRESSIVE, position: suffix}
inflect:
  - {words: all, affixes: [PLURAL]}
```

`words` in an `inflect` entry is `all`, `none`, or a list of word labels. With `closed_class: null`, a run has content words only, as before.

### Function words

A function word has one syllable. Its shape is drawn from configured weights over simple shapes: consonant and vowel (CV), CVC, VC, and V. The default weights are CV 0.3, CVC 0.4, and VC 0.3; the shape V has no form that is not an English word. The onset is at most one consonant, and so is the coda. Consonants and vowels are drawn from the common-word counts that content words use, restricted to these simple shapes, so function words use the most frequent sounds of English. The request lists the function words in order of frequency, most frequent first, and the most frequent half of them (rounded up) get two-phoneme shapes, CV or VC, as English gives its most frequent words its shortest forms. A candidate function word is rejected when:

- it fails the phonotactic check;
- it is a common English word (Zipf 3 or above). Content words are checked against the whole dictionary, but the whole dictionary leaves too few short forms (every single vowel is a word, and 22 CV forms remain). A function word may therefore sound like a rare word or a name, and the run's summary lists such cases;
- it lies closer than `function_words.min_distance` (default 2) to another function word;
- it is identical to a content word.

The default distance of 2 keeps function words from being minimal pairs of each other. Function words are short and frequent, so confusions between them would be costly for a learner.

With `function_words.source: english`, each gloss takes its English word instead: the CMUdict citation pronunciation (the first pronunciation with primary stress), with the dictionary's other pronunciations recorded in `words.csv` as `weak_forms`, for connected speech later. A gloss that is not in the dictionary is an error that names it.

Function words are synthesized and embedded like content words, by every speaker. Synthesis gives the citation form, spoken alone. The reduced forms of running speech ("the" as "thuh") are out of scope until sentences are synthesized as wholes.

### Affixes

An affix is a bound form: it never occurs alone, and it is never synthesized alone. Its shape is drawn from configured weights over C (one consonant, like English -s), VC (like -ing), and V (like -y). Suffixes are the default, and `position: prefix` makes a prefix. Two affixes must differ in at least one phoneme. An affix's vowel is unstressed. An affix that more than `affixes.max_skipped` (default 0.1) of the content words cannot take, even with the schwa below, is rejected and drawn again, so that no affix leaves large gaps in the paradigm.

With `affixes.source: english`, the glosses PLURAL, PAST, and PROGRESSIVE become the English suffixes with English allomorphy: -s is `IH0 Z` after a sibilant, `S` after another voiceless consonant, and `Z` otherwise; -ed is `IH0 D` after `T` or `D`, `T` after another voiceless consonant, and `D` otherwise; -ing is `IH0 NG`. `affixes.csv` lists the allomorphs. Any other gloss, or a prefix, is an error that names it.

**Joining.** An inflected form is the stem's phonemes followed by the affix's phonemes (or the reverse, for a prefix). When the join creates a sequence that fails the phonotactic check, an unstressed schwa (AH0) is inserted between stem and affix. English does the same with the plural of "bus". The rule gives affixes a simple, learnable variant, and the word table records where it applied. When a form still fails the check after the schwa is inserted, that stem and affix pair is skipped and reported. An English affix takes its allomorph for the stem instead, with no schwa and no phonotactic check, because the check's trigrams come from uninflected words. A pair whose form is a common English word (Zipf 3 or above) is skipped and reported too; a form that is only a rare word or a name is kept and listed in the summary.

### Inflected forms

Every stem and affix pair named by an `inflect` entry becomes an inflected form, labeled `W.<n>.AF.<m>`. Inflected forms are synthesized, cached, and embedded like any word, by every speaker. Inflection multiplies synthesis: inflecting 500 words with 3 affixes quadruples the audio. The default configuration therefore inflects nothing, and the tiny configuration inflects every word with every affix.

### Independence from content words

Adding closed-class forms never changes a content word's form, audio, or embedding. Closed-class forms draw from their own stream, `wordforms:closed_class`. A token's rate and pitch perturbation must not depend on which other tokens exist in the run. If the current synthesis draws perturbations in a way that depends on the other tokens, stop and write a proposal before changing it, because changing it would change the audio of existing runs.

### Evaluation by kind

The evaluation reports every measure for each kind of form separately (`kind`: `content`, `function`, or `inflected`), as well as for all forms together. It adds one measure for inflected forms, the **stem AUC**: the probability that an inflected form's embedding is closer to its own stem's embedding than to a randomly chosen other stem's. The stem AUC measures how visible morphology is in each embedding. A high stem AUC means a model reading the embeddings could notice that "blick" and "blicks" share a stem.

## Using the embeddings in other models

The Python interface is a `SoundEmbeddings` object with:

- `types`: an array of word embeddings, one row per word;
- `tokens`: an array of token embeddings, with arrays giving each token's word and speaker;
- `words`: the word table;
- `embed(forms, speakers)`: token embeddings for new forms, one for each form and speaker;
- `embed_types(forms)`: word embeddings for new forms, the mean over every token of every training speaker, as for the words of the lexicon;
- `to_torch()`: the arrays as PyTorch tensors.

Two example scripts go in `examples/`, as plumbing demonstrations rather than experiments:

- `wordforms_lm_inputs.py`: a small transformer language model from `transformers` that takes sound embeddings through a learned linear projection, in place of an embedding table, using `inputs_embeds`. The same projected sound embeddings also serve as the output matrix, with tied weights. With tied weights, the model can score a novel word it has never seen, from the word's sound alone. The training sequences are random walks over a toy bigram model, because grammar comes later.
- `wordforms_contrastive.py`: a CLIP-style model that aligns token sound embeddings with meaning vectors, such as the taxonomy generator's category vectors, under an arbitrary assignment of words to meanings. The script reports retrieval accuracy for held-out speakers and for novel words.

## Sound–meaning assignment

Stage 7 assigns word forms to meanings. The meanings are any table of IDs with binary feature vectors, such as the taxonomy generator's `categories_generative.csv`. Which meanings receive words is set in the configuration. The default is every category.

The assignment modes are:

- **Arbitrary.** A seeded random assignment. Stage 4 already includes this mode, for the examples.
- **Target correlation.** Word forms are assigned to meanings so that the correlation between sound distance and meaning distance approaches a configured target. The assignment starts random and swaps pairs of words while each swap moves the correlation toward the target. Sound distance is either phoneme edit distance or a named embedding's distance. Meaning distance is Hamming, cosine, or Jaccard distance on the feature vectors.
- **Branch markers.** Word forms for the members of a taxonomic branch share a marker syllable, at a configured depth and position (initial or final). Markers make the systematicity morphological.
- **Acoustic mapping.** Configured semantic features shift configured acoustic properties of every token of a word, by a configured amount: median pitch, formant scaling, duration, or spectral tilt. Acoustic mapping models sound symbolism of the frequency-code and bouba/kiki kinds. Acoustic mapping is applied with the manipulation tools of stage 5.

Every assignment reports the sound–meaning correlation, with a null distribution from random reassignments. Monaghan, Christiansen, and Fitneva (2011) argued that arbitrary vocabularies help learners individuate words, while systematic vocabularies help learners learn categories. These assignment modes let us test that claim across model types.

## Augmentation and acoustic manipulation

Stage 5 adds seeded transformations of cached audio:

- additive noise at a configured signal-to-noise range: white, pink, speech-shaped, or babble built from other tokens in the lexicon;
- reverberation with synthetic room impulse responses (`pyroomacoustics`);
- speed and pitch perturbation;
- acoustic manipulation through Praat (`praat-parselmouth`): pitch median, pitch range, formant shift ratio, and duration, as in Praat's "Change gender" command.

Augmented tokens are new tokens with their own records. Augmentation is off by default.

## Configuration

A run is defined by one YAML file. Unknown keys are errors, and every error names the file and the field. The example shows the main parameters with their defaults.

```yaml
name: default
seed: 1

wordforms:
  source: pseudowords            # pseudowords, english, or mixed
  mixed_proportion_english: 0.5
  count: 500
  syllables: {1: 0.3, 2: 0.5, 3: 0.2}
  initial_stress_probability: 0.8
  exclude_real_words: true
  min_english_distance: 1
  min_lexicon_distance: 1
  english_min_zipf: 3.0          # learn sound patterns from common words; null uses all of CMUdict
  exclude_inflections: true      # leave regular inflections out of the words the patterns come from

synthesis:
  cache_dir: runs/wordforms/cache
  sample_rate: 16000
  trim: {threshold_db: -40, margin_ms: 20}
  level: {rms_db: -24, max_peak: 0.9}
  duration_check: {max_ratio: 1.8, max_tries: 5}   # null turns the check off
  long_synthesis_ratio: 1.6                        # null turns the flag off
  tokens_per_speaker: 2
  held_out_speaker_proportion: 0.2
  engines:
    piper:  {voice: en_US-libritts_r-medium, voice_dir: runs/wordforms/voices, speakers: 40, noise_scale: 0.667, length_scale: 1.0, noise_w: 0.8}
    espeak: {voice: en-us, variants: [m1, m3, m7, f2, f4], speakers: 5, pitch: [35, 65], rate: [150, 190]}
  token_perturbation: {rate: 0.05, pitch_semitones: 0.5}

frontends:
  waveform: {store: false}       # true also stores the waveforms, which repeat the audio cache
  logmel: {n_mels: 80, window_ms: 25, hop_ms: 10}
  cochleagram: {channels: 64, low_hz: 50, high_hz: 8000, compression: 0.3, frame_rate: 100}

embeddings:
  - {name: cochleagram_fixed, encoder: fixed, frontend: cochleagram, time_bins: 10, pca_dims: 256}
  - {name: logmel_fixed, encoder: fixed, frontend: logmel, time_bins: 10, pca_dims: 256}
  - {name: hubert_base, encoder: pretrained, model: facebook/hubert-base-ls960, layer: 8, pooling: mean, store_layers: false}

closed_class:                    # null: content words only
  request: null                  # a request file (see "Closed-class forms"); the keys below give the request inline
  function_words:
    glosses: [the, and, a, is, that, it, with, not, all, can, has, "no", some, most, without]   # most frequent first
    source: pseudo               # english: each gloss's English pronunciation
    shapes: {CV: 0.3, CVC: 0.4, VC: 0.3}
    min_distance: 2
  affixes:
    items: [{gloss: PLURAL, position: suffix}, {gloss: PAST, position: suffix}, {gloss: PROGRESSIVE, position: suffix}]
    source: pseudo               # english: the English suffixes with their allomorphs
    shapes: {C: 0.4, VC: 0.4, V: 0.2}
    epenthesis: true
    max_skipped: 0.1             # reject an affix that more than this share of the words cannot take
  inflect: []                    # for example [{words: all, affixes: [PLURAL]}]

augmentation: null
assignment: {mode: arbitrary, meanings: null}
device: auto                     # cpu, cuda, mps, or auto
```

Also add `data/wordforms/tiny.yaml` (20 words, 3 speakers per engine, 1 token per speaker, fixed embeddings only) for fast tests.

## Determinism

Use the stream-seed function in `semantic_world.taxonomy.streams` (SHA-256 of the master seed and the stream name). The streams are `wordforms:generate`, `wordforms:speakers`, `wordforms:synthesis`, `wordforms:augment`, `wordforms:train`, `wordforms:assign`, `wordforms:eval`, and `wordforms:closed_class`. The principal component projection of the fixed encoder is exact and needs no stream. The `wordforms:eval` stream draws the evaluation's sample of tokens. Changing the speakers never changes the word forms, and changing the embeddings never changes the audio. Pretrained encoders run in evaluation mode. On the CPU, embeddings must be identical across runs. On a GPU, embeddings must match within a tolerance.

## Outputs

A run writes one folder, by default `runs/wordforms/<name>_seed<seed>/`, with audio in the shared cache.

| File | Contents |
| --- | --- |
| `config.yaml` | The fully resolved configuration, all seeds, the git commit hash (with a flag for uncommitted changes), and package versions, including every model's name and revision. |
| `words.csv` | One row per word: label, ARPAbet, IPA, espeak-ng string, spelling, syllables, stress, phonotactic log probability, English neighbors, nearest English word, lexicon neighbors, whether the word is a real English word, whether the word's Piper synthesis is unusually long (`long_synthesis`). With closed-class forms, function words and inflected forms also get rows, and the table gains `kind` (`content`, `function`, or `inflected`), `gloss` (function words), `stem`, `affix`, and `epenthesis` (inflected forms), and `weak_forms` (the other dictionary pronunciations of an English function word). |
| `affixes.csv` | One row per affix: label, gloss, position, ARPAbet, IPA. An English affix lists its allomorphs. Written when the run has closed-class forms. |
| `speakers.csv` | One row per speaker: label, engine, voice, speaker ID or variant, pitch and rate settings, training or held out. |
| `tokens.csv` | One row per token: label, word, speaker, synthesis settings, perturbations, augmentation, duration, number of tries, peak and RMS level, cache path, SHA-256 hash. |
| `frontends/<name>/frames.npy`, `index.csv`, `meta.yaml` | Front-end frames, the frame index, and settings. |
| `embeddings/<name>/tokens.npy`, `types.npy`, `meta.yaml` | Token and word embeddings, and settings. A fixed encoder with a projection also writes `projection.npz`. A pretrained encoder with `store_layers: true` also writes `layers.npy`. |
| `eval/embeddings.csv` | Evaluation results for every stored embedding (`basis` is `stored`), and for every layer of each pretrained model (`basis` is `sweep`), for all words and without the `long_synthesis` words (`word_set`). |
| `assignment/lexicon.csv`, `assignment/summary.yaml` | Word-to-meaning assignment, and the sound–meaning correlation with its null distribution. |

## Python package

Put the pipeline in `python/semantic_world/wordforms/`. Suggested modules:

```
python/semantic_world/wordforms/
  __init__.py        # load_config, SoundEmbeddings
  __main__.py        # command line with subcommands: forms, synth, frontends, embed, eval, assign, all
  config.py
  english.py         # CMUdict, syllabification, counts, trigram model
  generate.py        # pseudowords, filters, statistics
  spelling.py        # readable spellings by position in the word
  phonemes.py        # loading the mapping tables
  synth/             # piper.py, espeak.py, cache.py, and later formant.py
  augment.py         # noise, reverberation, perturbation, Praat manipulation
  frontends.py
  encoders/          # fixed.py, pretrained.py, learned.py
  evaluate.py
  assign.py
  io.py
data/wordforms/      # default.yaml, tiny.yaml, arpabet_ipa.yaml, arpabet_espeak.yaml, spelling.yaml
examples/            # wordforms_lm_inputs.py, wordforms_contrastive.py
```

Module names are recommendations. Speech dependencies go in an optional extra, `speech`, so the rest of the package installs without them: `torch`, `transformers`, `soundfile`, `scipy`, `cmudict`, `wordfreq`, `piper-tts`, and later `pyroomacoustics` and `praat-parselmouth`. Pin versions to the minor version, as the rest of `pyproject.toml` does. espeak-ng is a system program (`brew install espeak-ng`). Tests that need a missing tool are skipped with a message naming the tool, and the stage report lists every skip.

### Licenses

The repository is Apache-2.0. `piper-tts` (since version 1.3.0), espeak-ng, and `praat-parselmouth` are GPL-3.0. The pipeline keeps GPL tools out of the core: espeak-ng runs as an external program, and `piper-tts` and `praat-parselmouth` are imported only inside the modules that use them, within the optional extra. The pipeline never vendors GPL code. Whether that arrangement is enough is a licensing judgment for Jon (see "Decisions to confirm"). If it is not, Kokoro (an Apache-2.0 neural engine that accepts phoneme input) is the fallback for Piper.

## Build stages

Work on a branch for each stage (`wordforms-stage-1`, and so on), each branched from the previous stage's branch unless Jon has merged it into `main`. Each stage ends with its tests passing, the full check list in `CLAUDE.md` passing, and a commit.

Stages 1–4 are the fast path to usable embeddings. Stage 4a, closed-class forms, comes after stage 4 and before stage 5; it is described after the list.

1. **Word forms.** CMUdict loading, syllabification, counts, the trigram model, the generator, the filters, statistics, spelling, and the two mapping tables. *Accept:* the syllabifier matches a hand-checked list of 30 words; every generated form passes the phonotactic check; with real words excluded, no form is a CMUdict pronunciation; every pair of forms respects `min_lexicon_distance`; the syllable counts match the configured distribution within tolerance; the same seed gives identical word tables; the agreement between the IPA table and espeak-ng's IPA is reported.
2. **Synthesis.** Both engines, speakers, tokens, trimming, and the cache. *Accept:* every clip is mono, 16 kHz, unclipped, and trimmed; a second run reads the cache without synthesizing; the reproducibility of each engine is tested and recorded; a Whisper model (`openai/whisper-small.en`) transcribes 200 common real English words synthesized from their phonemes, and the accuracy is reported for each engine (provisional thresholds: 80% for Piper, 60% for espeak-ng; if a threshold is missed, report the miss rather than tuning to pass).
3. **Front ends.** Waveform, log-mel, and cochleagram, with the concatenated storage. *Accept:* shapes and frame rates match the settings; a 1 kHz tone peaks in the cochleagram channel nearest 1 kHz; the index recovers every token's frames exactly.
4. **Embeddings, evaluation, and the interface.** Fixed and pretrained encoders, word embeddings, the evaluation with the layer sweep, `SoundEmbeddings`, arbitrary assignment, and the two examples. *Accept:* same-different average precision is above chance for every embedding; `embed` reproduces stored embeddings within tolerance; both examples run end to end on the tiny configuration on a CPU in under 10 minutes. Report the evaluation table to Jon at the end of stage 4.
5. **Augmentation, manipulation, and the modulation front end.** *Accept:* achieved signal-to-noise ratios are within 0.5 dB of their targets; Praat manipulation moves the measured median pitch to within 5% of its target; the modulation front end responds most strongly to a test sound at its matching rate and scale.
6. **Learned encoders.** The contrastive acoustic word encoder and the self-supervised encoder, trained on the world's own audio. *Accept:* training on a CPU with a fixed seed is reproducible; evaluation results, including held-out speakers, are reported beside the fixed and pretrained encoders.
7. **Sound–meaning assignment.** Target correlation, branch markers, and acoustic mapping. *Accept:* arbitrary assignments have a mean correlation near 0 over seeds; target-correlation assignments reach their target within a tolerance or report the closest value reached; every word in a branch carries the branch marker; acoustic mapping produces the configured shifts, as measured.
8. **Parametric formant synthesizer.** A Klatt-style synthesizer driven by the same phoneme sequences, with every acoustic dimension controllable. The detailed design will be written after stages 1–7, and may need its own specification. Do not start stage 8 without it.

**Stage 4a: closed-class forms.** Function words, affixes, joining, inflected forms, the request file, and the evaluation by kind. *Accept:* every function word has one syllable and an allowed shape, passes the phonotactic check, is not an English word, and respects the minimum distances; every affix has an allowed shape; every inflected form passes the phonotactic check, and a schwa is inserted exactly where the plain join fails it; with `closed_class: null`, every existing output is unchanged except for added columns; turning closed-class forms on changes no content word's form, audio, or embedding; the stem AUC is reported for every embedding. Report the function words with their spellings, the affixes, and 20 sampled inflected forms, so Jon can judge them.

## Decisions to confirm

These choices were made while writing this specification. Each one is the working design unless Jon changes it.

1. CMUdict is the English source, and the generator samples onsets and rimes by syllable position and stress. The sound patterns are learned from common words only (decided September 30, 2026).
2. Piper's multi-speaker LibriTTS-R voice is the main voice set, and espeak-ng is the second engine.
3. GPL tools are optional extras, imported only where used, and never vendored.
4. HuBERT base, layer 8, is the default pretrained embedding. Jon chose layer 8 from the stage 4 layer sweep on September 30, 2026. Layer 3 is for graded similarity across the whole range of phoneme distances.
5. A word's embedding is the mean over its training-speaker tokens.
6. Until stage 8, exact acoustic control comes from Praat manipulation of synthesized audio.
7. Real English words are available as a source, for comparison with language-model embeddings.
8. Function words have one syllable of a simple shape and differ from each other by at least two phonemes. Affixes join with an inserted schwa where the plain join would be illegal. The default configuration makes function words and affixes but inflects nothing.
9. Decided September 30, 2026, after stage 4a (`docs/proposals/2026-09-30-wordforms-closed-class-shapes-and-joins.md`): the sound patterns come from uninflected words; function words are rejected against the common words only, with the default weights CV 0.3, CVC 0.4, VC 0.3, and the most frequent half get two phonemes; an affix that more than 10% of the words cannot take is rejected; a stem and affix pair whose form is a common English word is skipped; and `source: english` gives English function words (with weak forms) and English affixes (with allomorphy).

## References

- Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal analysis of complex sounds. *Journal of the Acoustical Society of America*, 118, 887–906.
- Kamper, H., Wang, W., & Livescu, K. (2016). Deep convolutional acoustic word embeddings using word-pair side information. *ICASSP 2016*.
- Monaghan, P., Christiansen, M. H., & Fitneva, S. A. (2011). The arbitrariness of the sign: Learning advantages from the structure of the vocabulary. *Journal of Experimental Psychology: General*, 140, 325–347.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.
