# User guides

These guides are for people who run Semantic World's programs: generating datasets, reading the outputs, and using them in models. They describe what the programs actually do, checked against real runs.

The specifications in `docs/specs/` are a different kind of document. Specifications are written for the people and agents who build the programs. When a guide and a specification disagree, the guide describes the current behavior, and the decided proposals in `docs/proposals/` usually explain the difference.

## The guides

| Guide | Program | Status |
| --- | --- | --- |
| [`WORLD.md`](WORLD.md) | The world package: entities, fluents, event types with requirements, preconditions, and effects, episodes, histories, and views | Phase (a) complete |
| [`TAXONOMY.md`](TAXONOMY.md) | The taxonomy generator: categories, instances, features, rules, and scalar dimensions (the part of a world that is fixed) | Complete |
| [`CORPUS.md`](CORPUS.md) | The corpus generator: documents in an artificial language about a world, their renderings, test sets, and spoken word forms | Complete |
| [`WORDFORMS.md`](WORDFORMS.md) | The word-form pipeline: spoken word forms, synthesis, auditory front ends, and sound embeddings | Stages 1–7 complete |

## The full chain

Together, the programs make the disembodied simulation: a world, propositions about the world, and language about the world, without the 3D simulation. The chain works now, and its outputs are usable for training and testing models.

1. `define` builds the world: the taxonomy generator makes the categories, instances, and features in memory, and the world package adds fluents and event types, writes the definition, and runs the statistics episodes.
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
| `define` | `runs/world/default_seed1/` | The embedded taxonomy run, the definition, the entities, the derived values, and the world's statistics | 8 seconds (7.6 s in stage b1, 7.8 s in stage a8) |
| `generate` | `runs/corpus/default_seed1/` | Documents, the four renderings, test sets, statistics, and the word-form request | 2.6 minutes (157 s in stage b1, 158 s in stage a8) |
| `wordforms all` | `runs/wordforms/corpus_default_seed1/` | Word forms, audio, auditory front ends, sound embeddings, and their evaluation | about 30 minutes |
| `render` | `runs/corpus/default_seed1/` | The spelled rendering (`corpus.txt`) and the word columns of `lexicon.csv` | 2 seconds |

Every stage records these times, measured with nothing else running, and its proposal file flags a growth of more than half, with its cause. There is no hard time limit, and no step is made cheaper by changing what it produces by default: a step that becomes too slow gets a setting that a study can turn on, with the full version as the default (Jon's ruling 3 on stage a8).

The default chain ran on October 9, 2026 (the world and the corpus in stage b1, the word forms in stage a8, whose lexicon is unchanged): a world of 284 entities, 8 fluents, and 40 event types with 25 conditional effects; 10,000 documents with 99,233 sentences; spoken words for 199 content lexemes and 17 function words with five sound embeddings; and 15,210 pairs of test items in 41 sets. The tiny configurations (`data/world/tiny.yaml`, `data/corpus/tiny.yaml`, and `data/wordforms/corpus_tiny.yaml`) run the same chain in under a minute. With `forms` in place of `all`, the word-form step skips the audio, which `render` does not need.

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
