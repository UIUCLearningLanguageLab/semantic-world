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
| Branch marker | `M.<k>` | `M.2` is the marker of the second branch |
| Marked form | `W.<n>.M.<k>` | `W.12.M.2` is word 12 with marker 2 |
| Inflected marked form | `W.<n>.M.<k>.AF.<m>` | `W.12.M.2.AF.1` is that marked form with affix 1 |

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
| `modulation` (stage 5) | Spectrotemporal modulation, after the cortical model of Chi, Ru, and Shamma (2005): two-dimensional Gabor filters over the cochleagram at configured temporal rates and spectral scales, averaged in frequency bands. | rates 2–32 Hz, scales 0.25–4 cycles per octave (up to half the cochleagram's channels per octave), 8 bands; off by default |

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

  Learned encoders train on training speakers only, and are frozen afterwards. They never train on inflected forms, so the inflected forms that a run asks for never change the embedding of a content word or of a function word (decided October 1, 2026). For the same reason, the tokens of inflected forms are encoded in batches of their own.

  Both encoders read a front end's frames (`frontend`, log-mel by default), standardized by channel with statistics from the training tokens, through a stack of one-dimensional convolutions (`layers`, `hidden`, `kernel`). The contrastive encoder pools the convolutions over time (mean and maximum) and projects to `dims`; its loss is a supervised contrastive loss with a `temperature`, over batches of `batch_size` tokens drawn as pairs of tokens of the same word (from different speakers when the word has them). The CPC encoder projects the convolutions to latent frames of `dims`, summarizes them with a GRU of `hidden` units, and predicts the latents 1 to `steps_ahead` frames ahead against `negatives` other frames, drawn from the whole batch (`negatives_from: batch`, the default) or from the same clip (`clip`); its embedding is the mean of the context frames, of `hidden` numbers (`embedding_from: context`, the default), or of the latent frames (`latents`). A time-boxed comparison on the default run (September 30, 2026) set these defaults: the context beat the latents, batch negatives beat clip negatives, and 30 epochs were no better than 10. No CPC setting gave word embeddings near the fixed baselines across speakers, so the CPC encoder stands as a self-supervised baseline, not as a recommended embedding. CPC is trained on isolated words here, which leave little to predict; revisit it with continuous speech once `docs/specs/CONNECTED_SPEECH.md` is built. `train_on` chooses the training tokens: the clean tokens of training speakers (`clean`, the default) or all of them (`all`), in both cases without the held-out words. Training runs for `epochs` epochs with AdamW at `learning_rate`, seeded from the `wordforms:train` stream by the embedding's name, with PyTorch's deterministic algorithms, on the configured device; on the CPU, the same seed gives the same weights and embeddings. The stored embedding holds the frozen model (`model.pt`), the frame statistics, and a training report (epochs, steps, seconds, device, the loss of each epoch). The default configuration trains one of each on the log-mel front end (decided September 30, 2026).

### Held-out words

A configured share of the content words (`training.held_out_word_proportion`, 0.2 by default) is held out from the training of every trained encoder: the contrastive and CPC encoders, and the principal component projection of the fixed encoders. The held-out words are a seeded draw from the `wordforms:train` stream. An inflected form is held out with its stem, and a function word is never held out, because a closed class is heard in full. Held-out words still get tokens and embeddings, from the frozen encoders, and `words.csv` records each word's `split` (`train` or `held_out`). The evaluation reports every measure for all words, for the training words, and for the held-out words (`word_split`). The held-out rows are the fair test of an encoder on novel words (decided September 30, 2026).

### Talker normalization

Each embedding has an optional setting, `talker_normalization` (off by default for every embedding): each speaker's mean token embedding is subtracted from that speaker's tokens. The mean is taken over the speaker's clean tokens of training content words, so held-out words and closed-class forms never enter it. The speaker means are stored with the embedding, and `embed` subtracts them for new forms. Whatever the setting, the evaluation reports every embedding both ways (`talker_normalized`), and `configured` marks the variant that the run stores. Normalization is an option and not the default for two reasons. It uses speaker identity, which a learner does not get for free. And handling speaker variability is part of what learners must do, so embeddings that hand a model normalized speech hide part of the problem (decided September 30, 2026).

### Word embeddings

A word's embedding is the mean of its tokens' embeddings over training speakers, without augmented tokens; `word_embeddings.tokens: all` includes the augmented tokens of training speakers (decided September 30, 2026). Token embeddings are kept too, augmented ones included, because token variability is part of what models should face.

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

The closed-class forms a run needs are listed in a request, given in the configuration (`closed_class`) or in a separate YAML file named by the top-level `request` setting. The corpus generator writes such a file (`wordform_request.yaml`). The request lists glosses, which are names for human readers, and says which words to inflect:

```yaml
function_words: [the, and, a, is, that, it, with, not, all, can, has, "no", some, most, without]   # most frequent first
affixes:
  - {gloss: PLURAL, position: suffix}
  - {gloss: PAST, position: suffix}
  - {gloss: PROGRESSIVE, position: suffix}
inflect:
  - {words: all, affixes: [PLURAL]}
```

`words` in an `inflect` entry is `all`, `none`, or a list of word labels. With `closed_class: null`, a run has content words only, as before. A request file's `function_words`, `affixes`, and `inflect` take the place of `closed_class.function_words.glosses`, `closed_class.affixes.items`, and `closed_class.inflect`, which must then be left out. The shapes and the other settings stay in the configuration. The key was `closed_class.request` before October 1, 2026: a configuration that still gives a file there gets an error that names `request`.

A request can also list lexemes, which the pipeline assigns to content words. See "Requests with lexemes" under "Sound–meaning assignment".

### Function words

A function word has one syllable. Its shape is drawn from configured weights over simple shapes: consonant and vowel (CV), CVC, VC, and V. The default weights are CV 0.3, CVC 0.4, and VC 0.3; the shape V has no form that is not an English word. The onset is at most one consonant, and so is the coda. Consonants and vowels are drawn from the common-word counts that content words use, restricted to these simple shapes, so function words use the most frequent sounds of English. The request lists the function words in order of frequency, most frequent first, and the most frequent half of them (rounded up) get two-phoneme shapes, CV or VC, as English gives its most frequent words its shortest forms. A candidate function word is rejected when:

- it fails the phonotactic check;
- it is a common English word (Zipf 3 or above). Content words are checked against the whole dictionary, but the whole dictionary leaves too few short forms (every single vowel is a word, and 22 CV forms remain). A function word may therefore sound like a rare word or a name, and the run's summary lists such cases;
- it lies closer than `function_words.min_distance` (default 2) to another function word;
- it is identical to a content word.

The default distance of 2 keeps function words from being minimal pairs of each other. Function words are short and frequent, so confusions between them would be costly for a learner.

With `function_words.source: english`, each gloss takes its English word instead: the CMUdict citation pronunciation (the first pronunciation with primary stress), with the dictionary's other pronunciations recorded in `words.csv` as `weak_forms`, for connected speech later. A gloss that is not in the dictionary is an error that names it.

Function words are synthesized and embedded like content words, by every speaker. Synthesis gives the citation form, spoken alone. The reduced forms of running speech ("the" as "thuh") are out of scope until sentences are synthesized as wholes. `docs/specs/CONNECTED_SPEECH.md` plans that layer, including weak forms of function words, alignment, and contextual embeddings.

### Affixes

An affix is a bound form: it never occurs alone, and it is never synthesized alone. Its shape is drawn from configured weights over C (one consonant, like English -s), VC (like -ing), and V (like -y); the default weights are C 0.5 and VC 0.5, so a bare vowel is available by setting but never drawn by default. Suffixes are the default, and `position: prefix` makes a prefix. Two affixes must differ in at least one phoneme, and one must not be the other with the joining schwa added (`L` and `AH0 L`), or a stem plus one, repaired with the schwa, would sound like the stem plus the other. An affix's vowel is unstressed. An affix that more than `affixes.max_skipped` (default 0.1) of the content words cannot take, even with the schwa below, is rejected and drawn again, so that no affix leaves large gaps in the paradigm.

With `affixes.source: english`, the glosses PLURAL, PAST, and PROGRESSIVE become the English suffixes with English allomorphy: -s is `IH0 Z` after a sibilant, `S` after another voiceless consonant, and `Z` otherwise; -ed is `IH0 D` after `T` or `D`, `T` after another voiceless consonant, and `D` otherwise; -ing is `IH0 NG`. `affixes.csv` lists the allomorphs. Any other gloss, or a prefix, is an error that names it.

**Joining.** An inflected form is the stem's phonemes followed by the affix's phonemes (or the reverse, for a prefix). When the join creates a sequence that fails the phonotactic check, the join is repaired. Where a vowel meets a vowel (a vowel-initial suffix after a vowel-final stem, or a vowel-final prefix before a vowel-initial stem), a glide is inserted: Y after a front vowel (IY, IH, EY, EH, AE), W after a back or rounded vowel (UW, UH, OW, AO, AW), and the configured default (`affixes.glide`, Y) otherwise. Otherwise, or when the glide does not help, an unstressed schwa (AH0) is inserted between stem and affix. English does the same with the plural of "bus". The repairs give affixes simple, learnable variants, and the word table records which one applied (`join`: `none`, `schwa`, or `glide`). When a form still fails the check after the repairs, that stem and affix pair is skipped and reported. An English affix takes its allomorph for the stem instead, with no repair and no phonotactic check, because the check's trigrams come from uninflected words. A pair whose form is a common English word (Zipf 3 or above) is skipped and reported too; a form that is only a rare word or a name is kept and listed in the summary.

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

The assignment section of the configuration names the meanings table (`meanings`), which meanings receive words (`categories`: `all`, `leaves`, or a list of IDs), and the two distances (`sound_distance`: `edit` or the name of an embedding; `meaning_distance`: `hamming`, `cosine`, or `jaccard`). A leaf is an ID that no other ID extends with a period. A column of the meanings table that holds other values than 0 and 1, such as a scalar dimension, is dropped and listed in the summary (`non_binary: drop`), or stops the run (`non_binary: error`). The draws come from the `wordforms:assign` stream. Only content words are assigned.

**Target correlation.** Each step proposes to exchange one meaning's word with another word, either another meaning's word or a word without a meaning, and keeps the exchange when the correlation moves toward the target. The search stops within `tolerance` of the target, or after `max_swaps` proposals. The summary gives the correlation reached, whether the target was reached (`reached`), the starting correlation, and the numbers of proposals and accepted exchanges. A target that the words cannot give is not an error by default: the closest value reached is reported, with the gap to the target (`gap`) and a warning, in the summary and on the command line. With `assignment.strict: true`, a gap larger than the tolerance stops the run with an error. With an embedding's distance as the sound distance, the embeddings are computed before the assignment.

**Branch markers.** A branch is the first `depth` parts of a meaning's ID (`C1.2.3` is in the branch `C1` at depth 1 and in `C1.2` at depth 2). A meaning above the depth has no branch, and its word is unmarked. Each branch draws one marker: an unstressed syllable of the configured shape (`CV`, `VC`, or `CVC`), sampled from the unstressed syllables at the beginnings (`initial`) or the ends (`final`) of the pattern words. Markers are labeled `M.<k>`. A marker joins a word as an affix does, before the word or after the word, with the same glide and schwa repairs. The marked form must pass the phonotactic check, and must not be a common English word or another form of the run. A marker is never a function word or an affix of the run, with or without the joining schwa, and no two markers differ only by that schwa. Two markers also differ by at least `branch_markers.min_distance` phonemes (2 by default), as two function words do. Markers that differ in one phoneme alone, such as `R IY0` and `R IH0`, would be nearly impossible to tell apart when spoken. With a CV or a VC marker and the default distance, no two markers share a consonant or a vowel. When no marker of the shape is that far from the markers of the other branches, the run stops with an error that says so. Not every word can take every marker, so a branch's meanings get words that can take the branch's marker, drawn in a seeded random order, and the assignment is otherwise random. A marker that too few free words can take is drawn again. A marked form is a new word form of kind `marked`, labeled `W.<n>.M.<k>`, with its base word as its stem and with the base word's training or held-out split. Marked forms are synthesized, embedded, and evaluated like inflected forms, so the assignment comes before the synthesis. The correlation is reported for the marked forms, beside the correlation of the unmarked base words (`unmarked_correlation`). Branch markers and acoustic mapping come before the synthesis, so their sound distance is the phoneme edit distance.

**Acoustic mapping.** The words are assigned at random. Each entry of `acoustic_mapping` names a feature column of the meanings table, a property, and an amount: `pitch` (semitones added to the median pitch), `formants` (a ratio, above 1 for a shorter vocal tract), `duration` (a factor), or `tilt` (decibels per octave added to the spectral slope, around 1 kHz). Every clean token of a word whose meaning has the feature is changed, with each of the meaning's mappings applied in the listed order. The changed token is a new token labeled `<source token>.M`, with the same word and speaker. The `mapping` column of `tokens.csv` holds the mapped token's source, meaning, and mappings. Pitch, formants, and duration use the Praat tools of stage 5, and the tilt is a gain that rises or falls with the logarithm of frequency. Each change is measured right after it is made, and the `achieved` column of `tokens.csv` holds the amount beside the measured shift, as for augmented tokens. The analysis uses the measured shifts, never the amounts: for each mapping, the summary gives the mean and standard deviation of the measured shifts, the numbers of tokens that miss the amount by more than 5% and by more than 10%, and the correlation between the feature and the measured shift over every clean token of the assigned words (`feature_correlation`; a token of a word without the feature has no shift).

The mapped tokens are the mapped word's tokens. The mapped tokens make the word's embedding, they are the tokens that trained encoders, talker normalization, and the fixed encoders' projection use, they are what `SoundEmbeddings` gives a learner (`embed` returns a mapped word's mapped tokens), and they are the tokens that augmentation is applied to, so the mapping comes before the augmentation. The unmapped originals stay in the run as a **control set**, marked in the `control` column of `tokens.csv`, so that the two can be compared. Control tokens go through the front ends and the encoders, and are used for nothing else: they are in no word embedding, no training set, and no augmentation. `SoundEmbeddings` leaves the control tokens out of its arrays and holds them apart in `control`. In the evaluation, a mapped word's clean tokens are its mapped tokens, and two more token sets compare the two: `mapped` (the mapped tokens) and `control` (their unmapped originals). A token of a word that no mapping changes is neither mapped nor a control token.

The pitch shift and the formant shift are measured frame by frame, over the frames that are voiced in both clips, with the frames matched by their relative time. The pitch shift is the median over those frames of the pitch after over the pitch before, in semitones (`measure_pitch_shift` in `praat.py`). A change of pitch can change which frames Praat finds voiced, so the difference between the two clips' median pitches is a much noisier measure of the same shift. The formant shift is the median ratio of the first three formants (`measure_formant_shift`). A shifted voice needs a shifted analysis ceiling, so the analysis of the changed clip is repeated over a range of ceiling scales, and the scale that the frames' ratios agree with most closely is used. The measure does not use the amount that the manipulation aimed at.

### Requests with lexemes

The corpus generator (`docs/specs/CORPUS_GENERATOR.md`, "Word forms for the corpus") is generated before its word forms, and asks for them with a request file. Beside the function words, the affixes, and the inflect entries, the request lists:

```yaml
lexemes:
  - {label: L.1, concept: C1, pos: noun}
  - {label: L.7, concept: IS.1, pos: adjective}
  - {label: L.23, concept: CAN.1, pos: intransitive_verb}
  - {label: L.40, concept: IS.9, pos: adjective, same_form_as: L.1}   # a homonym
takes:
  noun: [PLURAL]
  intransitive_verb: [PLURAL, PAST]
inflect:
  - {lexemes: [L.1, L.23], affixes: [PLURAL]}
meanings: wordform_meanings.csv   # relative to the request file
```

- `lexemes`: the content words of the corpus's language. Each has a label, the label of its concept, a part of speech, and, for a homonym, the lexeme whose form it shares.
- `takes`: for each part of speech, the affixes that its lexemes' words must be able to take.
- `inflect` entries can name `lexemes` in place of `words`.
- `meanings`: a meanings table for the lexemes whose concepts are categories. With `meanings` in the request, `assignment.meanings` must be null, and giving both is an error.

**Every lexeme gets a form.** A lexeme whose concept is in the meanings table is assigned by the configured mode: arbitrary, target correlation, branch markers, or acoustic mapping. `assignment.categories` chooses among them, as before. The synonyms of one category are assigned one by one, each with the category's meaning vector, so they share its sound–meaning structure, including its branch marker. Every other lexeme gets a word at random, from the words that are left, drawn from the part `wordforms:assign:lexemes` of the assignment stream. A homonym gets the form of the lexeme it names, and is left out of the mode. A run whose content words are fewer than the lexemes that need forms of their own is an error. The words that no lexeme gets are kept, and can serve as novel words.

**Affixes.** A lexeme gets only a word that can take every affix that `takes` lists for its part of speech, and a word that homonyms share takes the affixes of both. A word can take an affix when the joined form passes the phonotactic check, with the glide and the schwa repairs, is not a common English word, and is no other form that the run has or could make. Every mode respects the rule. An arbitrary assignment draws the words in a seeded random order, and the lexemes that few words suit choose first. The target-correlation search exchanges only words that suit both lexemes. A branch's marked forms take every affix that a lexeme of the branch requires. So no inflected form that a request asks for is ever skipped. The requirement comes from `takes` alone, never from `inflect`: the corpus generator fills `takes` from its grammar settings, and `inflect` from the inflected forms that its documents and test items happen to use. An inflect entry that gives a lexeme an affix that `takes` does not list for its part of speech is an error.

**Two passes.** The assignment comes first. The inflected forms are then made from whatever form each lexeme got, a marked form included (`W.12.M.2.AF.1`), and are synthesized and embedded with the rest. With the edit distance as the sound distance, no audio is needed before the assignment. With an embedding's distance, the first pass synthesizes and embeds the content words, the lexemes are assigned, and the layers are made again with every form. The audio cache keeps the second pass cheap. The lexemes are assigned in every subcommand, `forms` included, because the run's forms depend on the assignment.

**What the inflected forms cannot change.** The set of inflected forms that a request asks for never changes a content word's form, its assignment, or its embedding. Two requests that differ only in their inflect lists give byte-identical content words, assignments, tokens, and embeddings (on the CPU). A test checks it.

**Function words.** A request lists its function words in order of frequency in the corpus, and the forms of function words depend on that order. In a run with lexemes, the function words are therefore made after the assignment, and avoid its forms: a function word is no marked form, no marker, and no form that a word could have with an affix. So the words of the lexemes depend on the request's lexemes, `takes`, affixes, and meanings, and on nothing that the documents of a corpus change.

**Outputs.** `assignment/lexicon.csv` has one row for each lexeme: `lexeme`, `meaning` (its concept), `pos`, `word`, `spelling`, `arpabet`, and `assigned` (the mode, `random`, or `same_form`), with `base_word`, `branch`, and `marker` under branch markers. `words.csv` gains the column `pos`: the part of speech of the lexeme that a form was assigned to, with both parts of speech for a form that homonyms share. An inflected form has the part of speech and the training split of its stem. The summary's `lexemes` block counts the lexemes by how they were assigned, and gives, for each affix, how many lexemes require it and how many words can take it. A run without lexemes writes the files it wrote before, byte for byte.

Every assignment reports the sound–meaning correlation, with a null distribution from random reassignments. In a run with lexemes, the correlation is over the lexemes that the mode assigned. Monaghan, Christiansen, and Fitneva (2011) argued that arbitrary vocabularies help learners individuate words, while systematic vocabularies help learners learn categories. These assignment modes let us test that claim across model types.

## Augmentation and acoustic manipulation

Stage 5 adds seeded transformations of cached audio:

- additive noise at a configured signal-to-noise range: white, pink, speech-shaped, or babble built from other tokens in the lexicon;
- reverberation with synthetic room impulse responses (`pyroomacoustics`);
- speed and pitch perturbation;
- acoustic manipulation through Praat (`praat-parselmouth`): pitch median, pitch range, formant shift ratio, and duration, as in Praat's "Change gender" command.

Augmented tokens are new tokens with their own records. Augmentation is off by default.

The configuration lists **recipes**. A recipe names the transformations it applies, in the order manipulation, speed and pitch, reverberation, noise, and gives each setting as a range that a value is drawn from (or one number). Every recipe is applied to a seeded share (`proportion`) of the eligible tokens (`speakers`: all, training, or held-out speakers). An augmented token keeps its source's word and speaker, takes the label `<source token>.A.<recipe number>` (`W.12.S.3.2.A.1`), and records its recipe and every drawn value in the `augmentation` column of `tokens.csv`. Its clip goes in the audio cache under a hash of the source clip and the drawn values. The draws come from a substream of `wordforms:augment` named by the source token and the recipe, so augmenting one token never changes another. The signal-to-noise ratio is set by the ratio of powers over the whole clip, before the mixture is leveled like every other clip. Speech-shaped noise takes the long-term average spectrum of the run's own clips; babble adds a configured number of other tokens. Reverberation simulates a shoebox room with random dimensions, source, and microphone positions for the drawn reverberation time. The walls' absorption comes from a calibration of the simulation (Eyring's formula with a fitted correction for the room's shape), which puts the measured reverberation time within 10% of the target for rooms with sides of 3 to 6 m; with Sabine's formula the measured time was 25 to 40% too long. Augmented tokens go through the front ends, the embeddings, and the evaluation like any token; `SoundEmbeddings.token_augmented` marks them, and `embed` returns synthesized tokens only. Every transformation's target is checked: the value aimed at and the value measured right after the transformation (median pitch, pitch range, formant ratio, duration, reverberation time, signal-to-noise ratio) go in the `achieved` column of `tokens.csv`, and the run's summary counts, for each quantity, the tokens that miss their target by more than 5% and by more than 10%. The formant ratio, and the pitch shift of the speed and pitch transformation, are measured frame by frame, over the frames that are voiced before and after the change (`measure_formant_shift` and `measure_pitch_shift`, described under "Sound–meaning assignment"). The mean formants and the median pitch of a changed clip also move with the frames that Praat finds voiced, and were much noisier measures of the same shifts.

With augmentation on, the evaluation reports every measure separately for the clean tokens, the augmented tokens, and both together (`tokens`: `clean`, `augmented`, `all`), and once more for each recipe (`recipe:<name>`). It adds a **robustness** measure: how well a token retrieves its own word's clean embedding (the mean of the word's clean training tokens), as the same-different average precision over pairs of a sampled token and a word embedding (`robustness_ap`, chance one over the number of words) and the share of tokens whose nearest word embedding is their own (`robustness_top1`). A run without augmentation has the `clean` rows alone.

The Praat tools (`praat.py`) work on any clip, whole or within a time range in seconds, so that later work on connected speech (`docs/specs/CONNECTED_SPEECH.md`) can apply them to an aligned span: measuring the median pitch and pitch range; moving the pitch median and scaling the range through Praat's pitch tier; stretching a span through the duration tier; and Praat's "Change gender" (formant shift, pitch median, pitch range, duration) on a span, spliced back. Praat's pitch analysis uses a floor of 75 Hz and a ceiling of 600 Hz.

The modulation front end (`frontends.modulation`) filters the cochleagram with two-dimensional Gabor filters at the configured temporal rates (Hz) and spectral scales (cycles per octave), and averages the magnitude of the response within a configured number of frequency bands, so a frame has rates × scales × bands channels. The cochleagram's 64 channels sample the frequency axis about 8.6 times per octave, so scales above 4.3 cycles per octave cannot be resolved with the default cochleagram: the default scales stop at 4 (0.25, 0.5, 1, 2, 4), and finer scales need more cochleagram channels. The front end is off by default.

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
  modulation: null               # for example {rates: [2, 4, 8, 16, 32], scales: [0.25, 0.5, 1, 2, 4], bands: 8}

embeddings:
  - {name: cochleagram_fixed, encoder: fixed, frontend: cochleagram, time_bins: 10, pca_dims: 256}
  - {name: logmel_fixed, encoder: fixed, frontend: logmel, time_bins: 10, pca_dims: 256}
  - {name: hubert_base, encoder: pretrained, model: facebook/hubert-base-ls960, layer: 8, pooling: mean, store_layers: false}
  - {name: contrastive_logmel, encoder: learned, kind: contrastive, frontend: logmel, dims: 128, hidden: 128, layers: 3, kernel: 5, epochs: 20, batch_size: 64, learning_rate: 0.001, temperature: 0.1, train_on: clean}
  - {name: cpc_logmel, encoder: learned, kind: cpc, frontend: logmel, dims: 64, hidden: 128, layers: 3, kernel: 5, epochs: 10, batch_size: 32, learning_rate: 0.001, steps_ahead: 8, negatives: 32, negatives_from: batch, embedding_from: context, train_on: clean}

request: null                    # a request file, such as the corpus generator's wordform_request.yaml (see "The request" and "Requests with lexemes")

closed_class:                    # null: content words only
  function_words:
    glosses: [the, and, a, is, that, it, with, not, all, can, has, "no", some, most, without]   # most frequent first
    source: pseudo               # english: each gloss's English pronunciation
    shapes: {CV: 0.3, CVC: 0.4, VC: 0.3}
    min_distance: 2
  affixes:
    items: [{gloss: PLURAL, position: suffix}, {gloss: PAST, position: suffix}, {gloss: PROGRESSIVE, position: suffix}]
    source: pseudo               # english: the English suffixes with their allomorphs
    shapes: {C: 0.5, VC: 0.5, V: 0}   # a bare vowel suffix is available by setting, never drawn by default
    epenthesis: true
    glide: Y                     # the glide between two vowels at a join when neither vowel decides
    max_skipped: 0.1             # reject an affix that more than this share of the words cannot take
  inflect: []                    # for example [{words: all, affixes: [PLURAL]}]; an entry of a request with lexemes can name lexemes

word_embeddings: {tokens: clean} # clean: training-speaker tokens without augmentation; all: with
training: {held_out_word_proportion: 0.2}   # content words that no trained encoder sees
augmentation: null               # off; a recipe list turns it on, for example:
#  recipes:
#    - {name: noisy, noise: {kinds: [white, pink, speech, babble], snr_db: [0, 20], babble_voices: 6}}
#    - {name: room, reverberation: {rt60: [0.2, 0.8], room_m: [3, 8]}}
#    - {name: shifted, speed_pitch: {speed: [0.9, 1.1], pitch_semitones: [-2, 2]}}
#    - {name: voice, manipulation: {pitch_median_hz: [100, 250], pitch_range_factor: [0.5, 2], formant_shift_ratio: [0.85, 1.2], duration_factor: [0.8, 1.25]}}
#  proportion: 1.0                # the share of the eligible tokens each recipe is applied to
#  speakers: all                  # all, train, or held_out
assignment:
  mode: arbitrary                # arbitrary, target_correlation, branch_markers, or acoustic_mapping
  meanings: null                 # a CSV file of IDs with 0 and 1 feature columns; null: no assignment
  categories: all                # which meanings get words: all, leaves, or a list of IDs
  non_binary: drop               # a column with other values than 0 and 1: drop (and report) or error
  sound_distance: edit           # edit (phoneme edit distance) or the name of an embedding
  meaning_distance: hamming      # hamming, cosine, or jaccard
  null_samples: 1000
  strict: false                  # true: a target correlation that is not reached within the tolerance is an error
  target_correlation: {target: 0.3, tolerance: 0.01, max_swaps: 20000}
  branch_markers: {depth: 1, position: initial, shape: CV, min_distance: 2}   # position: initial or final; shape: CV, VC, or CVC; min_distance: the fewest phonemes by which two markers differ
  acoustic_mapping: []           # for example [{feature: IS.3, property: pitch, amount: 2.0}]
device: auto                     # cpu, cuda, mps, or auto
```

Also add `data/wordforms/tiny.yaml` (20 words, 3 speakers per engine, 1 token per speaker, fixed embeddings only) for fast tests. `data/wordforms/corpus_tiny.yaml` makes the word forms of the tiny corpus (`data/corpus/tiny.yaml`) from its request: 60 words, the same speakers and embeddings. `data/wordforms/corpus_default.yaml` makes the word forms of the default corpus (`data/corpus/default.yaml`) with the default settings: 500 words, 45 speakers, and all five embeddings.

In a resolved configuration (`config.yaml`), a request file's function words, affixes, and inflect entries are written inline under `closed_class`, and its lexemes, `takes`, and meanings under `request`. So a run's `config.yaml` gives the same run again without the request file.

## Determinism

Use the stream-seed function in `semantic_world.taxonomy.streams` (SHA-256 of the master seed and the stream name). The streams are `wordforms:generate`, `wordforms:speakers`, `wordforms:synthesis`, `wordforms:augment`, `wordforms:train`, `wordforms:assign`, `wordforms:eval`, and `wordforms:closed_class`. The principal component projection of the fixed encoder is exact and needs no stream. The `wordforms:eval` stream draws the evaluation's sample of tokens. Changing the speakers never changes the word forms, and changing the embeddings never changes the audio. Pretrained encoders run in evaluation mode. On the CPU, embeddings must be identical across runs. On a GPU, embeddings must match within a tolerance.

## Outputs

A run writes one folder, by default `runs/wordforms/<name>_seed<seed>/`, with audio in the shared cache.

| File | Contents |
| --- | --- |
| `config.yaml` | The fully resolved configuration, all seeds, the git commit hash (with a flag for uncommitted changes), and package versions, including every model's name and revision. |
| `words.csv` | One row per word (with `split`, `train` or `held_out`, for the trained encoders, and, in a run that assigns the lexemes of a request, `pos`): label, ARPAbet, IPA, espeak-ng string, spelling, syllables, stress, phonotactic log probability, English neighbors, nearest English word, lexicon neighbors, whether the word is a real English word, whether the word's Piper synthesis is unusually long (`long_synthesis`). With closed-class forms, function words and inflected forms also get rows, and the table gains `kind` (`content`, `function`, `inflected`, or, with branch markers, `marked`), `gloss` (function words), `stem`, `affix`, and `join` (inflected forms: `none`, `schwa`, or `glide`), and `weak_forms` (the other dictionary pronunciations of an English function word). |
| `affixes.csv` | One row per affix: label, gloss, position, ARPAbet, IPA. An English affix lists its allomorphs. Written when the run has closed-class forms. |
| `speakers.csv` | One row per speaker: label, engine, voice, speaker ID or variant, pitch and rate settings, training or held out. |
| `tokens.csv` | One row per token: label, word, speaker, synthesis settings, perturbations, augmentation (the recipe and drawn values of an augmented token), achieved (each transformation's target and measured value), mapping (the source, meaning, and mappings of a token changed by acoustic mapping), control (true for the unmapped original of a mapped token), duration, number of tries, peak and RMS level, cache path, SHA-256 hash. |
| `frontends/<name>/frames.npy`, `index.csv`, `meta.yaml` | Front-end frames, the frame index, and settings. |
| `embeddings/<name>/tokens.npy`, `types.npy`, `meta.yaml` | Token and word embeddings, and settings. A fixed encoder with a projection also writes `projection.npz`. A pretrained encoder with `store_layers: true` also writes `layers.npy`. A learned encoder writes its frozen model, `model.pt`, and its training report in `meta.yaml`. |
| `eval/embeddings.csv` | Evaluation results for every stored embedding (`basis` is `stored`), and for every layer of each pretrained model (`basis` is `sweep`), for all words and without the `long_synthesis` words (`word_set`), by kind of form (`kind`) and, with augmentation or acoustic mapping, by set of tokens (`tokens`). |
| `assignment/lexicon.csv`, `assignment/summary.yaml` | Word-to-meaning assignment, and the sound–meaning correlation with its null distribution. In a run with the lexemes of a request, `lexicon.csv` has one row for each lexeme, with a `lexeme` column. With branch markers, `lexicon.csv` adds each meaning's base word, branch, and marker, and `assignment/markers.csv` lists the markers. With acoustic mapping, the summary holds the measured shifts of each mapping. |

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
  lexemes.py         # assigning the lexemes of a request, and inflecting their forms
  mapping.py         # acoustic mapping of the tokens of assigned words
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

**Stage 4a: closed-class forms.** Function words, affixes, joining, inflected forms, the request file, and the evaluation by kind. *Accept:* every function word has one syllable and an allowed shape, passes the phonotactic check, is not an English word, and respects the minimum distances; every affix has an allowed shape; every inflected form passes the phonotactic check, and a glide or a schwa is inserted exactly where the plain join fails it; with `closed_class: null`, every existing output is unchanged except for added columns; turning closed-class forms on changes no content word's form, audio, or embedding; the stem AUC is reported for every embedding. Report the function words with their spellings, the affixes, and 20 sampled inflected forms, so Jon can judge them.

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
9. Decided September 30, 2026, after stage 4a (`docs/proposals/2026-09-30-wordforms-closed-class-shapes-and-joins.md`): the sound patterns come from uninflected words; function words are rejected against the common words only, with the default weights CV 0.3, CVC 0.4, VC 0.3, and the most frequent half get two phonemes; an affix that more than 10% of the words cannot take is rejected; a vowel meeting a vowel at a join is repaired with a glide before the schwa is tried; a stem and affix pair whose form is a common English word is skipped; and `source: english` gives English function words (with weak forms) and English affixes (with allomorphy).
10. Decided October 1, 2026, after stage 7: under acoustic mapping, the mapped tokens are the mapped word's tokens for word embeddings and every learner-facing output, and the unmapped originals are kept as a labeled control set; a target correlation that is not reached is a reported result with a warning, and an error with `assignment.strict: true`; and the augmentation's formant ratio and pitch shift are measured frame by frame, like the shifts of acoustic mapping.

11. Decided October 1, 2026, with stage 7 of the corpus generator (`docs/specs/CORPUS_GENERATOR.md`, decisions 9, 10, 12, 16, and 56, and `docs/proposals/2026-10-01-corpus-stage-7-decisions.md`): the request is the top-level setting `request`, and can list lexemes and meanings; the lexemes are assigned to content words, categories by the assignment mode and all others at random, with homonyms sharing a form; the assignment comes first, and the inflected forms are then made from the lexemes' forms, marked forms included; a lexeme gets only a word that can take the affixes of its part of speech, which the request's `takes` lists; the inflected forms that a request asks for never change a content word's form, assignment, or embedding, so the trained encoders never train on inflected forms; and a run with too few content words for its lexemes is an error.

12. Decided October 1, 2026: two branch markers differ by at least `assignment.branch_markers.min_distance` phonemes, 2 by default, and a run whose markers cannot be drawn at that distance stops with an error.

## References

- Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal analysis of complex sounds. *Journal of the Acoustical Society of America*, 118, 887–906.
- Kamper, H., Wang, W., & Livescu, K. (2016). Deep convolutional acoustic word embeddings using word-pair side information. *ICASSP 2016*.
- Monaghan, P., Christiansen, M. H., & Fitneva, S. A. (2011). The arbitrariness of the sign: Learning advantages from the structure of the vocabulary. *Journal of Experimental Psychology: General*, 140, 325–347.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv:1807.03748.
