# User guides

These guides are for people who run Semantic World's programs: generating datasets, reading the outputs, and using them in models. They describe what the programs actually do, checked against real runs.

The specifications in `docs/specs/` are a different kind of document. Specifications are written for the people and agents who build the programs. When a guide and a specification disagree, the guide describes the current behavior, and the decided proposals in `docs/proposals/` usually explain the difference.

## The guides

| Guide | Program | Status |
| --- | --- | --- |
| [`WORLD.md`](WORLD.md) | The world package: entities, fluents (Boolean and numeric), event types with requirements, preconditions, and effects, comparisons, places and place event types, episodes, histories, and views | Phases (a) and (b) complete; phase (c) begun (stage c1, places; stage c2a, pair fluents from an event file; stage c2b, generated pair fluents in `tiny_full.yaml` and `default_full.yaml`; stage c3a, locations as given by hand, in `tiny_place_chain.yaml`) |
| [`TAXONOMY.md`](TAXONOMY.md) | The taxonomy generator: categories, instances, features, rules, and scalar dimensions (the part of a world that is fixed), for entities or for places | Complete |
| [`CORPUS.md`](CORPUS.md) | The corpus generator: documents in an artificial language about a world, their renderings, test sets, and spoken word forms | Complete |
| [`WORDFORMS.md`](WORDFORMS.md) | The word-form pipeline: spoken word forms, synthesis, auditory front ends, and sound embeddings | Stages 1–7 complete |

## The full chain

Together, the programs make the disembodied simulation: a world, propositions about the world, and language about the world, without the 3D simulation. The chain works now, and its outputs are usable for training and testing models.

1. `define` builds the world: the taxonomy generator makes the categories, instances, and features in memory, and since stage c1 the places from a second taxonomy run, and the world package adds fluents, event types, and place event types, writes the definition, and runs the statistics episodes.
2. `generate` writes the corpus about the world: the documents, the four renderings, the test sets, the statistics, and a request for word forms. It defines the world in memory from the same configuration file, so step 1 is needed only when the world's files are wanted on their own.
3. The word-form pipeline reads the request and makes a spoken word for every lexeme.
4. `render` attaches the spoken words to the corpus.

At default scale, from the root of the repository with `.venv` active (see "Setup" below):

```
python -m semantic_world.world define data/world/default.yaml
python -m semantic_world.corpus generate data/corpus/default.yaml
python -m semantic_world.wordforms all data/wordforms/corpus_default.yaml
python -m semantic_world.corpus render runs/corpus/default_seed1 --wordforms runs/wordforms/corpus_default_seed1
```

| Step | Output folder | What the step writes | Time at default scale |
| --- | --- | --- | --- |
| `define` | `runs/world/default_seed1/` | The embedded taxonomy run, the place taxonomy run, the definition, the entities and the places, the derived values, and the world's statistics | 7 seconds (6.9 s in stage c3a, 6.9 s in stage c2b, 6.8 s in stage c2a, 6.4 s in stage c1, 6.3 s in stage b3b, 6.1 s in stage b3a, 6.7 s in stage b2, 7.6 s in stage b1, 7.8 s in stage a8); 7 seconds for `default_full.yaml`, the default world with four pair fluents (`WORLD.md`, "The full worlds") |
| `generate` | `runs/corpus/default_seed1/` | Documents, the four renderings, test sets, statistics, and the word-form request | 2.4 minutes (142 s in stage c3a, 142 s in stages c2b and c2a, 134 s in stage c1, 141 s in stage b3b, 133 s in stage b3a, 158 s in stage b2, 157 s in stage b1, 158 s in stage a8) |
| `wordforms all` | `runs/wordforms/corpus_default_seed1/` | Word forms, audio, auditory front ends, sound embeddings, and their evaluation | about 30 minutes |
| `render` | `runs/corpus/default_seed1/` | The spelled rendering (`corpus.txt`) and the word columns of `lexicon.csv` | 3 seconds |

Every stage records these times, measured with nothing else running, and its proposal file flags a growth of more than half, with its cause. There is no hard time limit, and no step is made cheaper by changing what it produces by default: a step that becomes too slow gets a setting that a study can turn on, with the full version as the default (Jon's ruling 3 on stage a8).

The default chain ran on October 21, 2026 (stage c2b: the world and the corpus regenerated, with the same definition, identity, and corpus as stages c1 and c2a, because the default world has no pair fluents until stage c4; the corpus was last rendered on October 16, 2026, stage c1, with the word forms of stage b3b, whose request did not change; the word forms themselves last ran on October 15, 2026, stage b3b): a world of <!--figure:default.world.entities|d-->284<!--/figure--> entities, <!--figure:default.world.fluents_total|d-->8<!--/figure--> fluents, <!--figure:default.world.numeric.count|d-->2<!--/figure--> numeric fluents, and <!--figure:default.world.event_types.leaves|d-->40<!--/figure--> event types with <!--figure:default.world.conditional.total|d-->10<!--/figure--> conditional and <!--figure:default.world.numeric.effects.total|d-->24<!--/figure--> numeric effects, and <!--figure:default.world.places.places|d-->12<!--/figure--> places with <!--figure:default.world.places.place_event_types|d-->4<!--/figure--> place event types; <!--figure:default.corpus.documents-->10,000<!--/figure--> documents with <!--figure:default.corpus.sentences-->100,918<!--/figure--> sentences; spoken words for <!--figure:default.corpus.lexemes.content|d-->203<!--/figure--> content lexemes and <!--figure:default.corpus.lexemes.function|d-->18<!--/figure--> function words with five sound embeddings; and <!--figure:default.corpus.test_sets.pairs-->16,457<!--/figure--> pairs of test items in <!--figure:default.corpus.test_sets.count|d-->46<!--/figure--> sets. The tiny configurations (`data/world/tiny.yaml`, `data/corpus/tiny.yaml`, and `data/wordforms/corpus_tiny.yaml`) run the same chain in under a minute. With `forms` in place of `all`, the word-form step skips the audio, which `render` does not need.

The taxonomy generator also runs on its own, when only the world's categories and feature vectors are needed (`TAXONOMY.md`), and the world package runs episodes of a world without a corpus (`WORLD.md`, "simulate").

## Setup

The generators are Python programs. They need Python 3.11 or later and three packages:

```
python -m pip install numpy polars pyyaml
```

We run the generators from the root of the repository, with the repository's `python/` folder on the Python path:

```
cd semantic-world
PYTHONPATH=python python -m semantic_world.taxonomy data/taxonomy/tiny.yaml
```

The simulation engine in `crates/` is written in Rust and is separate. The generators do not need Rust.

A conda or virtual environment keeps these packages apart from other projects. Any environment works.

The corpus generator needs the same three packages. The word-form pipeline needs more: its setup is in `WORDFORMS.md`.

The repository also has its own virtual environment, `.venv`, which the full check list and Claude Code use. It holds every package, the speech packages included, and the project is installed in it, so `PYTHONPATH=python` is not needed there. We activate it in each new terminal, from the root of the repository:

```
source .venv/bin/activate
```

The prompt then starts with `(.venv)`. A run that fails with `No module named 'polars'` is usually a run in another environment.

## Where outputs go

Every run writes one folder, by default under `runs/`. The `runs/` folder is ignored by git, so outputs never enter the repository. Every output folder contains a `config.yaml` that records the complete configuration and the random seeds, so any run can be reproduced exactly.
