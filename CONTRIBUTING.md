# Contributing to Semantic World

Semantic World is developed in the Learning and Language Lab at the University of Illinois Urbana-Champaign. This guide covers how to set up, how to make a change, and the few rules that keep the project's parts independent.

## Before starting

Read `README.md`, then `docs/CONTRACTS.md`. The current build plan is `docs/specs/MILESTONE_1.md`.

## Setup

These steps apply once the first build stage exists.

1. Install Rust through `rustup` (the stable toolchain).
2. Install Python 3.11 or later, and create a virtual environment.
3. Install the build tool and the development dependencies: `pip install maturin` and then `pip install -e ".[dev]"`.
4. Build the Python module: `maturin develop --release`.
5. Run the tests: `cargo test --workspace --all-features` and `pytest`.

## Making a change

1. Create a branch from `main`.
2. Make the change, with tests.
3. Run the full check list in the "Commands" section of `CLAUDE.md`.
4. Open a pull request. Describe what changed and why, and link any issue.
5. Continuous integration must pass before merging.

Use your own name and email for commits.

## Rules that keep the project's parts independent

- **The contracts in `docs/CONTRACTS.md` are binding.** A change to an interface or a file format needs a written proposal in `docs/proposals/` and Jon's approval before any code changes.
- **World content lives in `data/`, not in engine code.** The engine runs any world and hard-codes none.
- **Runs must be reproducible.** Every source of randomness uses a named stream derived from the run's seed.
- **The world gives no reward.** Reward is computed on the agent's side, from the agent's internal state.

## Assets and data

- Every 3D model, texture, or sound needs an open license that allows use in a public repository. Record the source and license of every asset.
- Large files do not go into git. Run outputs go in `runs/`, which git ignores. Large assets and data are stored outside the repository.

## Writing documents

Documents in `docs/` use short sentences, American spelling, and no hard line wrapping. See the style section of `CLAUDE.md`.

## License

Semantic World is licensed under the Apache License 2.0. By contributing, contributors agree that their contributions are licensed under the same license.
