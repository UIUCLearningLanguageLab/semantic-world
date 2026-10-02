# Proposal: decisions for stage 7 of the corpus generator

October 1, 2026. Raised in the orientation for stage 7 of `docs/specs/CORPUS_GENERATOR.md` (word forms: the request, the word-form pipeline changes, and the `render` command), and while building stage 7. Status: decided. Jon gave decisions 54 and 55 in the stage 7 prompt, answered two questions of the orientation as decisions 57 and 58, and changed the plan with decision 56. The corpus specification, the word-form specification (`docs/specs/WORDFORM_PIPELINE.md`), and the word-form guide were updated to match.

## Jon's decisions

### 54. Separating the two signals

Stage 6 reported how co-occurrence correlates with thematic relatedness and with taxonomic similarity. The two are themselves correlated in a world, so each correlation carries some of the other.

**Decision.** `stats.yaml` reports:

- the correlation between thematic relatedness and taxonomic similarity over leaf pairs in the world itself;
- for each document type, partial correlations: co-occurrence with thematic relatedness controlling for taxonomic similarity, and the reverse.

They are reported by words and by referents, with Pearson's and Spearman's.

### 55. A relation-fact share for encyclopedic documents

Stage 6 found that the taxonomic signal of the encyclopedic documents comes from the category documents, and a category document also states relation facts, which bring thematic partners together.

**Decision.** A new setting controls the share of relation facts in category-topic documents. Its default keeps today's behavior. Claude Code chooses the setting's name and shape, and records the choice (choices 1 and 2 below). The co-occurrence correlations and partial correlations are reported on the default configuration, and with relation facts turned off in category documents (see "Results").

### 56. The affix requirement comes from the grammar settings

Decision 16 says that a lexeme that must be inflected gets only a word that can take its affixes. Claude Code planned to read "must be inflected" from the request's `inflect` list, which holds the inflected forms that the documents and the test items use. Then a test-set setting, or the number of documents, could change a lexeme's word, and with it the spelled documents.

**Decision.**

- Which lexemes must take which affixes comes from the grammar settings alone, by part of speech. With number on and realized as an affix, every noun lexeme gets a word that can take `PLURAL`. With agreement on, every verb lexeme gets one too.
- The request's `inflect` list (documents plus test items) decides only which inflected forms are made, synthesized, and embedded.
- The set of inflected forms requested must not change any content word's form, assignment, or embedding. The learned encoders trained on inflected forms, so they now train on content words and function words only. A test requests two different inflection lists and compares the content words' outputs.

Decision 16's wording in the specification was updated to match.

## Questions of the orientation, and Jon's answers

### 57. How does the request give its meanings?

A corpus's taxonomy is generated in memory, so no `categories_generative.csv` exists on disk. An absolute path, or a path inside the run folder, would also break the property that the same run gives byte-identical folders wherever they are written.

**Decision.** As recommended. `generate` writes `wordform_meanings.csv` beside the request, and the request names the file by a path relative to the request file.

### 58. Does `render` touch `config.yaml`?

The outputs table says that `config.yaml` holds the word-form run's identity. The determinism property listed only the word labels, the spelled rendering, `corpus.txt`, and `lexicon.csv` as changing.

**Decision.** As recommended. `render` fills `provenance.wordforms`, and the determinism property exempts that key.

### Who proposed what

Jon also settled the attribution of earlier decisions. Decisions 54, 55, and 56 are proposed by Claude (chat) and decided by Jon. Decisions 57 and 58 took Claude Code's recommendation unchanged. In `docs/DECISIONS.md`, CG.46, CG.47, CG.48, and CG.53 are now proposed by Claude (chat). So are CG.49, CG.50, and CG.51, where Jon changed or extended the stage 6 recommendation: any event that a document reports (CG.49), the verb-category clause (CG.50), and the second measure by referents (CG.51). CG.52 took the recommendation unchanged, and stays with Claude Code.

## What changed in the earlier code

**The corpus generator.**

- **Configuration.** `documents.relation_fact_share` is new. `data/corpus/default.yaml` shows it.
- **The planner.** A category document draws its kind of content by the share, when one is set.
- **Statistics.** The co-occurrence check gains `world` and, for each group and measure, `partial`. `stats.partial_correlation` computes one.
- **The output folder.** `generate` writes `wordform_request.yaml` and `wordform_meanings.csv`. `request.py` and `render.py` are new, and the command line has `render`.
- **Nothing else.** With the default settings, the documents, the lexicon, the scenes, and the test sets of the default corpus are byte-identical to stage 6's.

**The word-form pipeline.**

- **Configuration.** The top-level `request` replaces `closed_class.request`. The request can list `lexemes`, `takes`, and `meanings`, and an inflect entry can name `lexemes`. `data/wordforms/default.yaml` moves the key. `data/wordforms/corpus_tiny.yaml` is new.
- **A new module, `lexemes.py`.** It assigns the lexemes of a request, and inflects the forms that the lexemes got.
- **The assignment.** `assign_branch_markers` takes the affixes that each row's form must take, and its rows can repeat a meaning (synonyms). Its random draws are unchanged for a run without lexemes. `needs_embeddings` and `run_assignment` look for the meanings in the request too.
- **Closed-class forms.** `add_inflected` makes the inflected forms of the second pass, and `add_function_words` makes the function words after the assignment, in a run with lexemes.
- **The run.** `run_forms` assigns the lexemes, unless the sound distance is an embedding's. The command line then assigns after the first pass.
- **Embeddings.** The learned encoders leave inflected forms out of their training tokens, and encode the tokens of inflected forms in batches of their own.
- **Outputs.** `words.csv` gains `pos`, and `assignment/lexicon.csv` has one row for each lexeme, both only in a run with lexemes.
- **One existing test.** The test of the request file now names the file with `request`, and checks the error for the old key.

**A run without a request is unchanged.** `python -m semantic_world.wordforms all data/wordforms/tiny.yaml` writes the same bytes before and after stage 7: every file of the run folder, apart from `config.yaml`, whose resolved configuration now has `request: null` at the top and no `closed_class.request`. So do `forms` and `synth` on `data/wordforms/default.yaml` (500 words and 46,350 tokens). A test holds the hashes of the two word tables.

## Choices made by Claude

These are engineering choices made while building stage 7. Each one is the working design unless Jon changes it.

1. **The name of the share.** The setting is `documents.relation_fact_share`: null (the default), or a number from 0 to 1.
2. **How the share acts.** With a share, each sentence of a category document draws a relation fact with that probability, and the other kinds of content (a membership fact, a fact about the topic, a fact about a subcategory) share the rest, each with the same chance. Null takes the earlier draw, one kind among the others with the same chance, so it changes no document. A share of 0 turns relation facts off. The setting acts on category documents only. A feature document about a verb still states relation facts, and the test sets draw their true items as before. The share is the chance of each draw. The share in the documents runs a few points higher at low settings, because the other kinds of fact run out sooner: 0.13 at a setting of 0.1, 0.29 at 0.25, 0.62 at 0.6, and 0.79 at 0.8. The default gives 0.34.
3. **The layout of the new statistics.** The co-occurrence check gains `world: {thematic_taxonomic: {pearson, spearman}}`. Under each measure (`words`, `referents`) of each group of documents, `partial` holds `thematic_given_taxonomic` and `taxonomic_given_thematic`, each with `pearson` and `spearman`. The groups are those of stage 6: the four document types, the two encyclopedic types together, the two narrative types together, and all documents. A partial correlation is computed from the three pairwise correlations, over the pairs with a defined taxonomic similarity. Spearman's is the same formula on the ranks. A partial correlation is null when a variable does not vary, or when the controlled variable determines one of the two.
4. **The request file.** The keys are `lexemes`, `takes`, `function_words`, `affixes`, `inflect`, and `meanings`, in that order. `takes` maps a part of speech to the glosses of its affixes, and a part of speech without an affix is left out. `inflect` has one entry for each affix, `{lexemes: [...], affixes: [gloss]}`, with the lexemes in label order.
5. **`takes` and `affixes`.** Both come from `grammar.morphology` alone. Number as an affix gives nouns `PLURAL`, and verbs too when verbs agree. Tense as an affix gives verbs `PAST`. Aspect as an affix gives verbs `PROGRESSIVE`. "Verbs" are the intransitive and the transitive verbs. Part nouns and adjectives take no affix, because the grammar never marks them. An inflection that stands after its word is a suffix, and one before its word is a prefix. `affixes` lists every inflection that is realized as an affix, even one that no sentence uses: with the event tense set to the present and tense marked by an affix, `PAST` is made and never used.
6. **Function words.** `function_words` lists every function word of the language, with those that no document uses, so every lexeme gets a form. The order is by counts in the documents, most frequent first, and ties keep the lexicon's order. Test items are not counted, so the test-set settings never change the order.
7. **`inflect` covers the test items.** A test item is rendered like a sentence, so its inflected forms must exist.
8. **The meanings file.** `wordform_meanings.csv` is the taxonomy's `categories_generative.csv`, byte for byte, scalar columns included. The word-form pipeline drops the columns that are not binary (`assignment.non_binary: drop`).
9. **`render`.** `render` works on folders, not on a corpus in memory. A content lexeme's form is the one in the word-form run's `assignment/lexicon.csv`. A function word's form is the run's function word with the same gloss. An inflected token is the form `<word>.<affix>`. `render` stops with an error when a lexeme has no form, when the run has a lexeme that the corpus lacks, when a lexeme's concept differs, or when an inflected form is missing. Everything is rendered before anything is written, so an error leaves the folder as it was. A corpus can be rendered again, or with another word-form run of its request. The word-form run's identity is its name, its seed, and the SHA-256 of its resolved configuration without the provenance. The run's path is left out, so the corpus folder does not depend on where the word forms lie.
10. **The `request` setting of the word-form pipeline.** `request` is null, the path of a request file, or the lexeme part of a request inline (`lexemes`, `takes`, `meanings`). A resolved configuration holds the request inline: the function words, affixes, and inflect entries under `closed_class`, as before, and the lexemes under `request`. So a run's `config.yaml` gives the same run again without the request file. A file named under `closed_class.request` is an error that names `request`. A null there is accepted, so run folders written before stage 7 still load.
11. **Checks on a request.** They are made when the configuration is read, and an error names the request file and the field. Lexeme labels are distinct. `same_form_as` names another lexeme, which has a form of its own. A gloss in `takes` is the gloss of an affix. An inflect entry gives a lexeme only an affix that `takes` lists for its part of speech, because only those are sure to join. A request file needs the closed class (`closed_class` not null). The request's meanings and `assignment.meanings` cannot both be given. The content words are at least as many as the lexemes that need forms of their own.
12. **Homonyms.** A word that homonyms share takes the affixes of both lexemes' parts of speech. A lexeme of a category that shares another lexeme's form takes that form, and is left out of the mode's assignment. The summary counts such lexemes.
13. **When a word can take an affix.** The joined form passes the phonotactic check, with the glide and the schwa repairs, is not a common English word, and is neither a form of the run nor the form of another word with an affix. The rule looks at the word forms and the affixes alone, so it does not depend on which words end up assigned or inflected. So no two forms of a run sound the same.
14. **How each mode keeps the rule.** An arbitrary assignment draws the words in a seeded random order, and the lexemes that the fewest words suit choose first. The target-correlation search starts from such an assignment, and exchanges only words that suit both lexemes. With branch markers, a branch's marked forms take every affix that a lexeme of the branch requires, and neither a marked form nor its inflected forms repeat another form. The lexemes outside the mode draw from the part `wordforms:assign:lexemes` of the assignment stream, and those whose words must take the most affixes choose first. When the words run out, the run stops with an error that says to raise `wordforms.count`.
15. **Function words come after the assignment.** A request orders its function words by their counts in the documents, and the form of a function word depends on its place in that order. Branch markers must not sound like function words, so the number of documents could have changed a marker, and with it the words of a whole branch. In a run with lexemes, the function words are therefore made after the assignment, and avoid its forms: a function word is no marked form, no marker, and no form that a word could have with an affix. The words of the lexemes then depend on the request's lexemes, `takes`, affixes, and meanings alone. A run without lexemes makes its function words where it did before.
16. **The trained encoders and inflected forms.** The contrastive and the CPC encoders never train on inflected forms. The rule holds in every run, with or without a request. A batch pads its clips to the longest one, which changes the last bits of an embedding, so the tokens of inflected forms are also encoded in batches of their own. Marked forms still train: they are the categories' words in branch-marker mode, and they do not depend on the inflect list. A run without inflected forms is unchanged. A run with inflected forms and a learned encoder gives slightly different learned embeddings than before stage 7. No configuration in `data/` is such a run.
17. **When the lexemes are assigned.** In every subcommand, `forms` included, because the run's forms depend on the assignment. With an embedding as the sound distance, the first pass synthesizes and embeds the content words, the lexemes are assigned, and the layers are made again with every form.
18. **Output files.** `assignment/lexicon.csv` has one row for each lexeme: `lexeme`, `meaning` (its concept), `pos`, `word`, `spelling`, `arpabet`, and `assigned` (the mode, `random`, or `same_form`), with `base_word`, `branch`, and `marker` under branch markers. `words.csv` gains `pos` only in a run with lexemes. A form that homonyms of two parts of speech share lists both. An inflected form takes the part of speech and the training split of its stem. The summary's `lexemes` block counts the lexemes by how they were assigned.
19. **Lexemes outside the meanings.** `assignment.categories` chooses the lexemes of categories that the mode assigns, and the others get words at random. A request without meanings gives every lexeme a word at random.
20. **The configuration for the tiny corpus.** `data/wordforms/corpus_tiny.yaml` makes 60 content words for the tiny corpus's 35 content lexemes, in the arbitrary mode, with the speakers and the embeddings of `tiny.yaml`. Its `request` names `runs/corpus/tiny_seed1/wordform_request.yaml`, so the corpus is generated first.

## Results

### The tiny configuration, end to end

| Step | Command | Time |
| --- | --- | --- |
| Generate | `python -m semantic_world.corpus generate data/corpus/tiny.yaml` | 0.5 s |
| Make the word forms | `python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml` | 15.0 s with an empty audio cache, 11.6 s with a warm one |
| Render | `python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1` | 0.2 s |

The corpus has 20 documents, 152 sentences, and 516 test items. The word-form run makes 60 content words and 15 function words, and 450 tokens from 6 speakers. All 35 content lexemes get words: 6 lexemes of categories by the mode, and 29 at random. `forms` in place of `all` gives `render` all it needs in 2 seconds.

Some rendered sentences, beside their conceptual renderings:

| Spelled | Conceptual |
| --- | --- |
| gea stess lur tect gea spurtish lonseetet gea clonstanzive | `A C1.2 THAT V1.1 A C2.2 V2 A C2.1` |
| rezz vessen | `IT CAN.4` |
| rau stess fah yarsh cannagzus rau clonstanzive urs clobbage ut fah pame | `THE C1.2 WITH HAS.5 V2.2 THE C2.1 WITHOUT HAS.4 AND WITH HAS.7` |
| gea vake sprost nisitch | `A SC.1.LOW C1.1 CAN.1` |
| gea stess nisitch | `A C1.2 CAN.1` |

A second run checked morphology and branch markers with audio. The tiny corpus was generated with number and tense as affixes, and its word forms were made with branch markers (80 content words). The run made 17 function words, 2 affixes, 6 marked forms, and 25 inflected forms, 6 of them of marked forms (`W.33.M.1.AF.1`, "riadassinal", from "riadassin"), in 22 seconds. The past of "cunshal" is "cunshaln", and of "hikshet" is "hiksheten", with the joining schwa. The stem AUC of the 25 inflected forms is 0.94 for the cochleagram embedding and 0.90 for the log-mel embedding.

### The two signals, on the default configuration

These numbers come from `python -m semantic_world.corpus generate data/corpus/default.yaml` (seed 1, 10,000 documents, 41 seconds), over the 741 pairs of the 39 leaves. Every pair has a taxonomic similarity.

**In the world itself**, thematic relatedness and taxonomic similarity are nearly independent: Pearson's correlation is 0.108, and Spearman's is 0.091. So the partial correlations stay close to the plain ones.

Each cell gives Pearson's correlation, then Spearman's. "Thematic" and "taxonomic" are the correlations of co-occurrence with each variable. "Thematic, given taxonomic" is the partial correlation with thematic relatedness controlling for taxonomic similarity, and the last column is the reverse.

**By words, the default configuration.**

| Documents | Thematic | Taxonomic | Thematic, given taxonomic | Taxonomic, given thematic |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category (3,003) | 0.295, 0.448 | 0.477, 0.231 | 0.278, 0.441 | 0.469, 0.213 |
| Encyclopedic, about a feature (2,020) | 0.333, 0.361 | 0.135, 0.093 | 0.323, 0.356 | 0.105, 0.065 |
| Entity narrative (2,021) | 0.796, 0.828 | 0.184, 0.164 | 0.795, 0.827 | 0.164, 0.159 |
| Situational narrative (2,956) | 0.728, 0.771 | 0.149, 0.127 | 0.724, 0.769 | 0.103, 0.089 |
| Both encyclopedic types (5,023) | 0.378, 0.464 | 0.387, 0.210 | 0.367, 0.457 | 0.376, 0.190 |
| Both narrative types (4,977) | 0.792, 0.832 | 0.173, 0.151 | 0.790, 0.831 | 0.144, 0.135 |
| All documents (10,000) | 0.685, 0.735 | 0.327, 0.223 | 0.692, 0.736 | 0.350, 0.231 |

**By referents, the default configuration.**

| Documents | Thematic | Taxonomic | Thematic, given taxonomic | Taxonomic, given thematic |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category | 0.293, 0.450 | 0.482, 0.224 | 0.277, 0.443 | 0.474, 0.205 |
| Encyclopedic, about a feature | 0.333, 0.361 | 0.135, 0.093 | 0.323, 0.356 | 0.105, 0.065 |
| Entity narrative | 0.794, 0.833 | 0.196, 0.160 | 0.792, 0.833 | 0.183, 0.152 |
| Situational narrative | 0.724, 0.775 | 0.152, 0.126 | 0.720, 0.773 | 0.109, 0.087 |
| Both encyclopedic types | 0.377, 0.462 | 0.389, 0.208 | 0.366, 0.455 | 0.378, 0.188 |
| Both narrative types | 0.784, 0.833 | 0.181, 0.147 | 0.782, 0.833 | 0.156, 0.129 |
| All documents | 0.726, 0.775 | 0.299, 0.211 | 0.732, 0.777 | 0.323, 0.223 |

**With relation facts turned off in category documents** (`documents.relation_fact_share: 0`). The feature documents and the narratives are the same documents as before, so their rows do not change. The rows that change, by words:

| Documents | Thematic | Taxonomic | Thematic, given taxonomic | Taxonomic, given thematic |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category | 0.092, 0.189 | 0.559, 0.259 | 0.039, 0.172 | 0.555, 0.247 |
| Both encyclopedic types | 0.256, 0.371 | 0.486, 0.215 | 0.235, 0.361 | 0.477, 0.196 |
| All documents | 0.645, 0.722 | 0.401, 0.247 | 0.661, 0.724 | 0.437, 0.263 |

And by referents:

| Documents | Thematic | Taxonomic | Thematic, given taxonomic | Taxonomic, given thematic |
| --- | --- | --- | --- | --- |
| Encyclopedic, about a category | 0.095, 0.273 | 0.554, 0.228 | 0.042, 0.260 | 0.550, 0.212 |
| Both encyclopedic types | 0.258, 0.380 | 0.483, 0.206 | 0.237, 0.370 | 0.474, 0.186 |
| All documents | 0.706, 0.768 | 0.353, 0.230 | 0.718, 0.771 | 0.394, 0.251 |

Four things stand out.

- **The two signals are already separate in the default world.** Thematic relatedness and taxonomic similarity correlate at about 0.1, so controlling for one moves the other's correlation by about 0.05 at most.
- **Relation facts carry the thematic signal of the category documents.** In the default configuration, 9,494 of the 29,968 sentences of the category documents are relation facts. Without them, the thematic partial correlation of the category documents falls from 0.278 to 0.039 (Pearson's, by words), and the taxonomic one rises from 0.469 to 0.555. A category document is then close to a pure taxonomic source.
- **Spearman's correlations move less.** Without relation facts, Spearman's thematic partial correlation of the category documents is still 0.17 by words and 0.26 by referents. Most leaf pairs never co-occur in a category document, so the ranks hold many ties, and the rank correlation is a blunt measure there.
- **The feature documents keep a thematic signal.** A feature document about a verb lists the categories that take part in it, and the setting does not touch those documents. Their thematic partial correlation stays at 0.32.

## Notes for Jon

- **An older proposal is out of date in one place.** `docs/proposals/2026-10-01-corpus-decisions-before-stage-1.md`, item 6 under "Stage 7: the word-form pipeline", says that a lexeme that must be inflected gets only a word that can take its affixes. Decision 56 now says where the requirement comes from. The same proposal says that the tiny corpus needs 36 content lexemes. It has 35. Neither file was edited.
- **Branch markers can sound alike.** In the tiny run with branch markers, the two markers are `R IY0` and `R IH0` ("ri-" and "rih-"). The marker rule asks only that two markers differ, and that they do not differ by the joining schwa alone. A minimum distance between markers would be a change to the word-form pipeline's stage 7 design.
- **The word-form guide lists `epenthesis` as a column of `words.csv`** in its section on closed-class forms. The column is `join`. The sentence is older than stage 7, and was left as it is.

## Still open

Nothing new.
