# User guides

These guides are for people who run Semantic World's programs: generating datasets, reading the outputs, and using them in models. They describe what the programs actually do, checked against real runs.

The specifications in `docs/specs/` are a different kind of document. Specifications are written for the people and agents who build the programs. When a guide and a specification disagree, the guide describes the current behavior, and the decided proposals in `docs/proposals/` usually explain the difference.

## The guides

| Guide | Program | Status |
| --- | --- | --- |
| [`TAXONOMY.md`](TAXONOMY.md) | The taxonomy generator: categories, instances, features, rules, scalar dimensions, and verbs | Complete |
| [`WORDFORMS.md`](WORDFORMS.md) | The word-form pipeline: spoken word forms, synthesis, auditory front ends, and sound embeddings | Stages 1–7 complete |
| [`CORPUS.md`](CORPUS.md) | The corpus generator: documents in an artificial language, their renderings, test sets, and spoken word forms | Complete |

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
