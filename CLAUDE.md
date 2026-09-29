# Instructions for Claude Code

This file is read at the start of every Claude Code session in this repository.

## What the project is

Semantic World is an artificial world for comparing cognitive models. A simulated world supplies input to agents. Different cognitive models run "in the head" of the agents. We compare which kinds of models learn which tasks best, and why. See `README.md` for the overview.

## What to read, in order

1. `docs/specs/MILESTONE_1.md` — the current build specification: scope, world content, build stages, and acceptance tests. Start here.
2. `docs/CONTRACTS.md` — the ten contracts between the engine, the world, the agents, and the viewer. The contracts are the source of truth for every interface and file format.
3. `docs/ENTITY_DEFINITIONS.md` — how bodies, sensors, actuators, and nervous systems are defined.
4. `docs/ENVIRONMENT_SURVEY.md` — background only: why the stack was chosen.

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
