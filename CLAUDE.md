# Instructions for Claude Code

This file is read at the start of every Claude Code session in this repository.

## What the project is

Semantic World is an artificial world for comparing cognitive models. A simulated world supplies input to agents. Different cognitive models run "in the head" of the agents. We compare which kinds of models learn which tasks best, and why. See `README.md` for the overview.

## What to read, in order

1. `docs/specs/MILESTONE_1.md` — the current build specification: scope, world content, build stages, and acceptance tests. Start here.
2. `docs/CONTRACTS.md` — the ten contracts between the engine, the world, the agents, and the viewer. The contracts are the source of truth for every interface and file format.
3. `docs/ENTITY_DEFINITIONS.md` — how bodies, sensors, actuators, and minds are defined.
4. `docs/specs/TAXONOMY_GENERATOR.md` — the taxonomy feature generator: a standalone Python program in `python/semantic_world/taxonomy/` that builds datasets of categories, instances, and binary features with recorded rules. It does not use the Rust engine.
5. `docs/specs/TAXONOMY_RELATIONS.md` — the extension of the generator with scalar dimensions, transitive verbs, and a verb taxonomy. The decided proposals in `docs/proposals/` are part of both taxonomy specifications.
6. `docs/specs/WORDFORM_PIPELINE.md` — the word-form pipeline: a Python program in `python/semantic_world/wordforms/` that makes spoken word forms, synthesizes them, and builds sound embeddings.
7. `docs/specs/CORPUS_GENERATOR.md` — the corpus generator: a Python program in `python/semantic_world/corpus/` that writes documents about the taxonomy's world in an artificial language, with a parse tree and a logical form for every sentence. Its numbered decisions, and the proposals they point to, are part of the specification.
8. `docs/specs/CONNECTED_SPEECH.md` — the plan for spoken sentences: whole utterances, alignment, pauses, speakers, and register. Planned, not built.
9. `docs/specs/WORLD_AND_LANGUAGE.md` — the world-and-language refactor: one world model of state and change for both simulation modes (fluents, event types with preconditions and effects, time steps), a new package `python/semantic_world/world/`, and the corpus rebuilt on that model. Built in phases and stages; its decisions to confirm are reviewed by Jon before stage a1 starts. Drafted, not built.
10. `docs/DECISIONS.md` — the log of every design decision of the taxonomy, word-form, and corpus programs, and of the world model. Append new decisions to it.
11. `docs/guides/` — user guides for the taxonomy generator (`TAXONOMY.md`), the word-form pipeline (`WORDFORMS.md`), and the corpus generator (`CORPUS.md`).
12. `docs/ENVIRONMENT_SURVEY.md` — background only: why the stack was chosen.

The project's planning documents (decisions, to-do lists, research goals, and the first study) live in a private folder that is not available in this repository. Everything needed to build is in `docs/`. If something seems missing, ask rather than guess.

## Working rules

- **Work through the build stages in order.** Finish one stage, with its tests passing, before starting the next. Commit at the end of each stage.
- **What is approved.** The contract additions in `MILESTONE_1.md` are approved, and are also recorded in `CONTRACTS.md`. Items marked **Proposed** in `CONTRACTS.md` are the working design for milestone 1.
- **The contracts are binding.** Never change an interface, file format, or data-file schema in a way that disagrees with `docs/CONTRACTS.md`. If the implementation shows that a contract is wrong or incomplete, stop and write a proposal in `docs/proposals/` (the question, the options, and a recommendation), then ask Jon.
- **Ask about design, decide engineering.** Questions that change the contracts, the world's content, or what agents are given go to Jon. Small engineering choices inside a crate do not.
- **No world content in engine code.** Entity types, numbers, actions, and rules belong in `data/` files. The engine runs any world and hard-codes none. Survival-specific logic that cannot be written as data goes in a named rule module, with its parameters in data.
- **Determinism is a requirement.** Follow the determinism section of `MILESTONE_1.md`: named random streams from one master seed, no hash-map iteration order in simulation logic, and state-hash tests.
- **The world gives no reward.** `step` never returns reward. Reward is computed on the agent's side.
- **Privileged access is labeled.** Only scripted agents and tests may read the full world state, through calls clearly marked as privileged.
- **Tests before done.** Every stage lists acceptance tests. Write the tests, run the tests, and report the results honestly, including anything skipped.

## Commands

Keep this list current. The commands assume the stable Rust toolchain is on the path (`. ~/.cargo/env` if `rustup` did not add it) and the project's virtual environment is active (`source .venv/bin/activate`, created with `uv venv .venv --python 3.12` and filled with `uv pip install -e ".[dev]"`).

```
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
maturin develop --release        # build the Python module into the active virtual environment
ruff check python tests
pytest
```

Other commands:

```
cargo run -p sw-schema -- schemas                                # regenerate schemas/*.json after changing the data-file types
cargo run -p sw-schema -- check data/experiments/m1_smoke.yaml   # load and resolve an experiment
python -m semantic_world.taxonomy data/taxonomy/default.yaml [--seed N] [--out DIR]   # run the taxonomy generator
pytest tests/taxonomy                                            # the taxonomy generator's tests alone
python -m semantic_world.wordforms forms data/wordforms/default.yaml [--seed N] [--out DIR]   # generate word forms
python -m semantic_world.wordforms synth data/wordforms/tiny.yaml [--seed N] [--out DIR]      # word forms and their audio
python -m semantic_world.wordforms frontends data/wordforms/tiny.yaml [--seed N] [--out DIR]  # also the auditory front ends
python -m semantic_world.wordforms all data/wordforms/tiny.yaml [--seed N] [--out DIR]        # every layer: forms, audio, front ends, embeddings, evaluation
python examples/wordforms_lm_inputs.py                            # sound embeddings as language-model inputs (tiny configuration)
python examples/wordforms_contrastive.py                          # sound embeddings aligned with meaning vectors (tiny configuration)
python -m semantic_world.wordforms check-ipa                      # compare the IPA table with espeak-ng
python -m semantic_world.wordforms check-whisper data/wordforms/default.yaml   # transcribe real words synthesized from phonemes
python -m piper.download_voices en_US-libritts_r-medium --download-dir runs/wordforms/voices   # fetch the Piper voice once
pytest tests/wordforms                                           # the word-form pipeline's tests alone
python -m semantic_world.corpus generate data/corpus/default.yaml [--seed N] [--out DIR]   # generate a corpus: documents, test sets, statistics, and the word-form request
python -m semantic_world.wordforms all data/wordforms/corpus_tiny.yaml                     # the word forms of the tiny corpus, from its request (after generate data/corpus/tiny.yaml)
python -m semantic_world.corpus render runs/corpus/tiny_seed1 --wordforms runs/wordforms/corpus_tiny_seed1   # attach the word forms: word labels and the spelled rendering
python -m semantic_world.wordforms all data/wordforms/corpus_default.yaml                  # the word forms of the default corpus (after generate data/corpus/default.yaml); "forms" in place of "all" makes the word forms without audio
pytest tests/corpus                                              # the corpus generator's tests alone
```

The word-form pipeline needs the `speech` extra (`uv pip install -e ".[dev,speech]"`) and, for the espeak-ng engine and the IPA check, the system program espeak-ng (`brew install espeak-ng`).

The taxonomy generator needs only NumPy, polars, and PyYAML: `PYTHONPATH=python python -m semantic_world.taxonomy data/taxonomy/tiny.yaml` runs without the Rust core built.

This is the full check list. A stage is done when the full check list passes locally.

## Git

- The repository's local git config already sets Jon's commit identity. Do not change it.
- Commit at the end of each build stage, with a message naming the stage.
- Do not push, rewrite history, or force-push unless Jon asks. Jon pushes to GitHub and opens pull requests.
- Work on a branch for each stage (for example, `m1-stage-3-core`), not directly on `main`.
- Large files never go into git: run outputs go in `runs/` (ignored), and large assets stay outside the repository.

## Writing style for documents

Jon reads and edits the documents. Follow his style in everything written in `docs/`:

- Short sentences. One idea per sentence.
- "We" rather than "you" in explanations.
- Repeat the noun rather than pointing back with "this", "that", or "it" when the referent is more than a sentence away.
- American spelling.
- No hard wrapping. Each paragraph, list item, and table row is one line.
- Periods and commas go outside closing quotation marks.

Never edit an existing document in `docs/` except for the specific change needed, and say which document changed and why.
