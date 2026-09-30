# Proposal: common-word sound patterns and natural spellings for word forms

September 30, 2026. Raised by Jon while reviewing stage 1 of `docs/specs/WORDFORM_PIPELINE.md`. Status: decided. Jon chose both changes on September 30, 2026. Built as stage 1a, on the branch `wordforms-stage-1a`.

## Question 1: which words teach the generator English sound patterns?

Stage 1 learned the onset counts, the rime counts, and the phoneme trigrams from every word of the CMU Pronouncing Dictionary. The dictionary holds about 126,000 words, and most of them are proper names, loanwords, and rare words. Their sound sequences are legal in the stage 1 generator, so the pseudowords contain sequences that no common English word has.

Jon measured the problem on the default configuration (500 words, seed 1). A word counts when it contains a phoneme trigram, with word boundaries, that occurs in no CMUdict word with a Zipf frequency of 3.0 or above in the `wordfreq` package. 144 of the 500 words count. Examples: `HH AA1 F S K` (the trigram `F S K`), `S V AH0 N AO1 D` (`S V AH`), `EH1 HH IH0 T S` (a word starting `EH HH`), and `T R IY1 N K EY0` (`IY N K`).

### Options

1. **Learn the patterns from common words only.** A new parameter sets the smallest Zipf frequency of the words that teach the patterns.
2. **Keep the whole dictionary, and weight the counts by word frequency.** Rare sequences become rare in the pseudowords, but stay legal.
3. **Leave the design.**

### Decision

Option 1.

### Design

A new parameter, `wordforms.english_min_zipf`, has the default 3.0. The value `null` keeps the whole dictionary, and gives exactly the stage 1 word forms.

```yaml
wordforms:
  english_min_zipf: 3.0          # null uses all of CMUdict
```

The **pattern words** are the CMUdict words whose Zipf frequency is at least `english_min_zipf`. The frequency comes from `wordfreq`'s large English list. With the default, 28,872 of the 125,916 dictionary words are pattern words.

The pattern words alone give:

- the onset counts and the rime counts, by syllable position and stress;
- the phoneme trigrams, for the phonotactic check and the phonotactic log probability;
- the legal onsets of the syllabifier.

The whole dictionary still gives:

- the rejection of real words (`exclude_real_words` and `min_english_distance`);
- the count of English neighbors, and the nearest English word.

So a pseudoword is never the pronunciation of any dictionary word, rare or common.

**The legal onsets.** Stage 1 called an onset legal when the onset begins at least 20 dictionary words. The count of 20 kept the onsets of a few names (`T L`, `D M`) from splitting words like "atlas" wrongly. With a smaller set of pattern words, a fixed count of 20 is too strict. The rule is now a proportion: an onset is legal when the onset begins at least 1 in every 6,300 pattern words. The proportion gives 20 words for the whole dictionary, as in stage 1, and 5 words for the default pattern words. With the default, the onsets `T S`, `S V`, `SH N`, `SH M`, `SH L`, `SH W`, and `G Y` are no longer legal, and `N Y` and `V Y` become legal. "Pizza" is now syllabified `P IY1 T . S AH0`.

**The dependency.** `wordfreq` joins `cmudict` in the `speech` extra, pinned to its minor version (`wordfreq>=3.1,<3.2`). The `cmudict` pin is tightened to its minor version too (`cmudict>=1.1,<1.2`).

### Result

| Default configuration, seed 1 | Whole dictionary (stage 1) | Common words (stage 1a) |
| --- | --- | --- |
| Words with a trigram found in no common word | 144 of 500 | 0 of 500 |
| Distinct trigrams in the model | 19,687 | 11,360 |
| Legal onsets | 71 | 66 |
| Candidates rejected by the phonotactic check | 110 | 339 |

The count is exactly 0, not only near 0, because the phonotactic check now uses the same trigrams as the measure.

## Question 2: how should the readable spellings be built?

Stage 1 spelled each phoneme with one fixed string. The results were readable but unnatural: `greeoe`, `hokingk`, `doosoes`, `dratimee`.

### Decision

The spelling of a phoneme is chosen by the phoneme's position in the word, from common English spellings. The spellings are still for human readers only, and no model is given them. When two words would get the same spelling, the later word's spelling still gets a numeric suffix.

### Design

The table is `data/wordforms/spelling.yaml`. The table gives each phoneme a spelling for each context that matters. The contexts are defined in `python/semantic_world/wordforms/spelling.py`. The main rules are:

- **Long vowels by position.** The vowel of "my" is written "y" at the end of a word, "i" with a silent "e" before a final consonant ("time"), "i" in an open syllable ("tiger"), and "igh" elsewhere. The vowels of "day", "go", and "cute" follow the same pattern ("day", "cake", "paiper"; "go", "home"; "few", "cute", "muzic").
- **A plural or past ending after a silent "e".** One more consonant may follow the final consonant when the two agree in voicing ("times", "liked").
- **Doubling.** A consonant is doubled between a stressed short vowel and a vowel ("happy", "kitten"). At the end of a word after a stressed short vowel, we write "ck", "tch", "dge", "ll", "ff", "ss", and "zz".
- **K.** K is written "c" before "a", "o", "u", "l", "r", and "t", and "k" elsewhere. `K W` is "qu", and `K S` is "x".
- **Vowels before R.** "car", "story", "cair", "feer", and "ur" for the stressed vowel of "bird".
- **Unstressed endings.** "le" ("apple"), "en" ("kitten"), "um" ("album"), "us", "et" ("basket"), "age" ("village"), "ive" ("active"), and "y" ("funny").
- **Final clusters.** "dance", "else", "sleeve", "lives".

A stress-specific entry (`AH0`) overrides the plain entry (`AH`) context by context.

Real English words (the sources `english` and `mixed`) keep their dictionary spellings.

### Result

Stage 1 spellings and stage 1a spellings of the same forms:

| Phonemes | Stage 1 | Stage 1a |
| --- | --- | --- |
| `HH AA1 K IH0 NG K` | hokingk | hockink |
| `D R AE1 T IH0 M IY0` | dratimee | drattimy |
| `AH0 P R UW0 S T OW1 L` | aproostoel | aproostole |
| `L AO1 R SH AH0` | lawrsha | lorsha |
| `B EH1 B IH0 K T` | bebikt | bebbict |
| `K L AA1 M S K ER0 Z IH0 V` | klomskerziv | clomskerzive |

As a check of the table as a whole, we spelled the first pronunciation of every common English word (26,920 plain words with a Zipf frequency of 3.0 or above). The speller gives the dictionary spelling for 24% of them. English spelling is too irregular for a high figure, but the figure shows that the rules follow common patterns.

Known weak spots: "igh" before some consonant clusters (`P AY1 N T` gives "pighnt"), the long "o" before `S T` (`K L OW1 S T` gives "cloced"), and an unstressed first syllable before a single consonant (`HH AH0 L OW1` gives "halo").

## An open point

The sources `english` and `mixed` still draw real words uniformly from the whole dictionary, so most of the real words drawn are rare names. Drawing real words from the pattern words would give common words. The change is not part of this proposal, because Jon's decision covers the sound patterns only.
