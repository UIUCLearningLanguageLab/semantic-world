# Proposal: the shapes of function words, and the stem and affix pairs that cannot join

September 30, 2026. Raised while building stage 4a of `docs/specs/WORDFORM_PIPELINE.md` (closed-class forms) on the branch `wordforms-stage-4a`. Status: open. The stage is built to the specification as written. Each question below says what the specification's rules produce, and what could change. Nothing here changes a content word.

## Question 1: the shapes of function words

The specification draws a function word's shape from the weights `{CV: 0.4, CVC: 0.3, VC: 0.2, V: 0.1}`, and rejects a candidate that is a dictionary word, checked against the whole CMU Pronouncing Dictionary. The whole dictionary holds about 126,000 words, including proper names and interjections, so the short shapes have few forms left:

| Shape | Forms that pass the phonotactic check | Not in the whole dictionary | Not among the common words (Zipf 3 or above) |
| --- | --- | --- | --- |
| V | 12 | 0 | 0 |
| CV | 213 | 22 | 37 |
| VC | 189 | 55 | 76 |
| CVC | 3,018 | 811 | 1,479 |

Two consequences:

- **The shape V never occurs.** Every vowel that can stand alone is a word ("a", "eye", "oh", "owe", "ah", "aw", "err", "uh"). The weight 0.1 on V can never be honored.
- **The CV forms are odd.** Almost every CV form that is not a word ends in a stressed lax vowel, because English does not end stressed syllables in lax vowels except in a few interjections ("uh", "yeah", "nah") that give the trigram model its permission: `M AH1` ("muh"), `K AE1` ("cae"), `F AA1` ("fah"), `T AO1` ("taw"). The pool is 22 forms, and with `min_distance: 2` no two of them may share a vowel, so the same handful ("voy", "chur", "taw", "fah", "zay") turns up in run after run.

### What the build does now

The generator follows the rules, with one fallback: when a drawn shape has no form left, the shape is drawn again from the other shapes. The run's summary records the redraws (`closed_class.function_words.shape_redraws`), and the counts in the table above (`candidates`). In the default configuration, seed 1, the 15 function words are 5 CV, 5 CVC, and 5 VC.

### Options

1. **Keep the rules, and change the default weights** to what the dictionary allows, for example `{CV: 0.2, CVC: 0.5, VC: 0.3}` and no V. The CV forms stay as they are.
2. **Reject function words against the common words only** (the pattern words, Zipf 3 or above), as the spellings already do. That admits 37 CV forms, including natural ones ("chee", "ger", "jer", "nay", "zoe"), and 76 VC forms. A function word could then sound like a rare dictionary word or a name ("Zoe"), but never like a common word.
3. **Allow an unstressed vowel** in CV and V function words, as English does ("the", "a"), so that `DH AH0` shapes become possible. The synthesized citation form would be an unstressed syllable spoken alone, which the engines may render oddly.
4. **Leave the design.** The shape V is silently redrawn, and the CV forms are what they are.

### Recommendation

Option 2, with option 1's weights: reject function words against the common words, and set the default weights to `{CV: 0.3, CVC: 0.4, VC: 0.3}`. Option 2 keeps the reasoning of the common-word decision of September 30 (the pattern words are the English that matters), and the weights stop promising a shape that cannot occur. Function words would be checked differently from content words, which stay checked against the whole dictionary; the specification would say so.

## Question 2: the stem and affix pairs that cannot join

The specification joins a stem and an affix, inserts a schwa when the plain join fails the phonotactic check, and skips the pair when the join fails with the schwa too. With every default word inflected by every default affix (1,500 pairs), the skipped pairs are:

| Seed | Affixes | Pairs made | Pairs skipped |
| --- | --- | --- | --- |
| 1 | `AH0`, `N`, `S` | 1,250 | 250 (17%) |
| 2 | `IY0`, `N`, `NG` | 1,124 | 376 (25%) |
| 3 | `AH0`, `T`, `D` | 1,319 | 181 (12%) |

Two causes account for most of the skips:

- **Stems that end in `IH0 NG`.** 43 of the 500 default words end in `IH0 NG`, because `IH NG` is the most frequent unstressed final rime of English (it is the progressive suffix). No common word continues `NG` with a schwa, `N`, `S`, or `T` at the end, so such a stem takes no suffix at all: 43 × 3 = 129 of seed 1's 250 skips. With the schwa, the trigram `NG AH #` is the problem, not the affix.
- **An affix that few stems can take.** The suffix `NG` (seed 2) fails for 245 of 500 stems: `NG` follows only a few vowels in English, and the schwa does not help, because `AH NG` occurs word-finally in no common word.

Another epenthetic vowel would rescue few pairs (45 of 250 in seed 1 with `IH0`, `IY0`, or `ER0`), except when the affix is the problem (268 of 376 in seed 2).

### What the build does now

The rules as specified. Skipped pairs are listed in the run's summary (`closed_class.inflected.skipped`) with the reason, and the counts are printed by the command line.

### Options

1. **Accept the gaps.** Real morphology has gaps and suppletion. Report them, as now.
2. **Reject an affix that too many stems cannot take.** When an affix is drawn, count the content words that could take it (with the schwa), and reject a candidate that fails for more than a configured share of them (for example 10%). Seed 2's `NG` would be rejected and drawn again. The stems that end in `IH0 NG` would still be skipped.
3. **Keep the content words from ending in an affix's phonemes.** That is, treat affixes as a constraint on the content words. It would change the content words, against the independence rule, and it decides morphology before the affixes are drawn.
4. **Check the join more loosely.** For example, require only the bigrams across the boundary, not the trigrams. It would let the inflected forms contain sequences that no common word has.

### Recommendation

Option 2, with the threshold as a configuration setting (`closed_class.affixes.max_skipped: 0.1`). The stems ending in `IH0 NG` remain a gap in the paradigm (about 9% of the default words), and a later change to the content-word generator could take it up if Jon wants inflectable stems.

## Question 3: inflected forms and function words that sound like English words

The specification rejects a function word that is a dictionary word, but says nothing about an inflected form that is one. In the runs above, 36, 26, and 20 inflected forms (of 1,250, 1,124, and 1,319) are the pronunciation of a dictionary word, nearly all of them names ("issa", "ison", "haven", "rella", "rardon"). The run's summary lists them (`closed_class.english_words`), and `nearest_english` in `words.csv` shows the word.

### Options

1. **Report them, as now.**
2. **Skip the pair**, like a pair that cannot join.
3. **Report only when the word is a common word** (Zipf 3 or above). In the runs above, "haven" would count and "issa" would not.

### Recommendation

Option 1. An inflected form is a stem plus an affix, and the stem is a pseudoword; that the combination happens to be a name in the dictionary is not what the phonotactic rules are protecting against. The report is enough.
