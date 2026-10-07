# Proposal: decisions for stage a1 of the world-and-language refactor

October 7, 2026. Raised in the orientation for stage a1 of `docs/specs/WORLD_AND_LANGUAGE.md` ("Matrices and derived values"), and while building the stage. Status: working design. Every choice below is Claude Code's unless Jon changes it. The choices are logged as WM.E1 and following in `docs/DECISIONS.md`.

## Orientation

### What was read

- `CLAUDE.md`: the working rules, the commands, and the full check list.
- `docs/specs/WORLD_AND_LANGUAGE.md` in full, with attention to "Architecture", "The world definition" ("Rules: Boolean form and matrices (REL.18)", "Rule-set identity and derived values (REL.16)", "Labels"), "A principle for defaults", "Phase (a): Python package", "Phase (a): build stages", and "Decisions".
- `docs/specs/TAXONOMY_GENERATOR.md` ("Rules", "Fixed by rule", "Outputs") and `docs/specs/TAXONOMY_RELATIONS.md` ("Threshold literals in rules").
- `python/semantic_world/taxonomy/`: `rules.py` (the `Rule`, `Threshold`, and `RuleSet` classes, and `compute_features`, which evaluates every rule by truth-table lookup in layer order), `boolean.py` (`TruthTable` and `minimal_dnf`, a Quine–McCluskey minimum cover with deterministic ties), `expressions.py` (the printed form, `Gt` threshold literals), `io.py` (the output folder), `generate.py`, `features.py`, `streams.py`.
- The tests that enumerate the taxonomy's output folder (`test_taxonomy_regression.py`, `test_taxonomy_outputs.py`, `test_taxonomy_density.py`, `test_taxonomy_relation_outputs.py`, `test_taxonomy_scalars.py`, `test_taxonomy_verb_tree.py`), and `python/semantic_world/corpus/world.py`, which checks a taxonomy run folder file by file.

### How the rules stand today

Every rule is an output feature, an ordered list of inputs (free or determined IS and HAS features, and threshold literals `SC.<n> > θ` on scalars), and a truth table. `RuleSet.compute` evaluates the rules in the taxonomy's layer order by table lookup, vectorized over instances. The minimal DNF of each truth table is already computed for the complexity score (`min_dnf_literals`), as sorted prime implicants. The rules are written to `rules.yaml` with their expressions and truth tables. Nothing today writes matrices, a rule-set identity, or derived values apart from the full `instances.csv`.

### Plan for the stage

1. A new package `python/semantic_world/world/` with `errors.py`, `matrices.py` (rules to two threshold layers, dependency layers, evaluation, the agreement test, the sparse record), `identity.py` (canonical JSON and the rule-set identity), and `derived.py` (the manifest and the loader that refuses a mismatched identity). The package imports only NumPy, polars, and PyYAML.
2. A small adapter in the taxonomy, `taxonomy/rule_matrices.py`, that turns the taxonomy's `RuleSet` into the world package's literal table, symbols, rule records, and matrices, and runs the agreement test over the instances.
3. `generate()` builds the matrices and runs the agreement test at the end of every run. A disagreement is an error that fails the run.
4. `write_result` adds `rule_matrices.json`, `derived/static_features.csv`, and `derived/manifest.yaml`. Every existing file stays byte-identical.
5. Tests in `tests/world/`, the regression test extended, and the tests that enumerate output folders updated for the added entries.
6. Documentation: this file, `docs/DECISIONS.md`, `docs/guides/TAXONOMY.md`, and `CLAUDE.md`.

### Reading of the scope

Every rule in `rules.yaml` is covered: determined IS, HAS, and CAN features. Verb constraints (two-place requirements) and event types are stage a2. `derived/static_features.csv` holds the determined IS and HAS features, as the spec's table says ("determined PROPERTY and PART"); CAN features stay in `instances.csv` and become `capacities.csv` in stage a2. The agreement test still covers the CAN rules.

### A conflict found in the orientation

`python/semantic_world/world.py` already exists: it is the milestone 1 Python wrapper of the Rust core (`World`, `make`, `Action`, contract 3). A module and a package cannot share the name `semantic_world.world`, and the spec's ruling WM.3 gives the name to the new package. The wrapper is named by nothing in `docs/CONTRACTS.md` or `docs/specs/MILESTONE_1.md`; it is imported by `semantic_world/__init__.py` (lazily), `semantic_world/agents/`, and one test. See choice 1.

### Questions

No question of the orientation changes the design, a file format decided in the spec, or what the generators produce. The details that the spec leaves to the build (the exact keys of the sparse layers, the literal table, the manifest) are listed below as choices, and are easy to change in stage a2 if Jon prefers another shape.

## Engineering choices

Each choice is numbered. The log in `docs/DECISIONS.md` refers to them as WM.E1 and following.

1. **The engine wrapper is renamed.** `python/semantic_world/world.py`, the milestone 1 wrapper of the Rust core, becomes `python/semantic_world/engine.py`, so that `semantic_world.world` is the new package, as WM.3 decides. `semantic_world.World`, `semantic_world.make`, and `semantic_world.Action` still load lazily and are unchanged for callers. No contract or specification names the wrapper's module.
2. **The Boolean module is shared.** `taxonomy/boolean.py` (truth tables, the SHJ types, the minimal DNF) moves to `python/semantic_world/common/boolean.py`, a package for code that neither the taxonomy nor the world owns. The move breaks an import cycle: the world's matrices read truth tables, and the taxonomy imports the world's matrices. Every import was updated; nothing else in the module changed. `expressions.py` stays in the taxonomy until a later stage needs it elsewhere.
3. **Modules of the world package in stage a1.** `errors.py` (`WorldError`, `AgreementError`, `DerivedError`), `identity.py` (canonical JSON and the rule-set identity), `matrices.py` (REL.18), and `derived.py` (the manifest and the loader). The spec puts the identity and the identity-checking loader in `definition.py`; that module arrives in stage a2 with the definition itself, and will import these two.
4. **The matrix form is generic.** `matrices.py` knows nothing of the taxonomy's classes. A `LiteralSpec` is a key (the variable name rules use) and a mapping that says what the literal reads; a `RuleSpec` is an output, its input keys, a truth table, and the printed expression. The adapter `taxonomy/rule_matrices.py` maps the taxonomy's `RuleSet` into these forms, builds the matrices, computes the identity, and runs the agreement test over the instances. Stage a2 maps requirements the same way.
5. **Dependency layers are computed from the rules.** A rule's layer is one more than the highest layer of any rule whose output it reads, with base literals at layer 0. The taxonomy's own layer numbers (`layer` in `rules.yaml`) are a property of the generator and are not used, so a CAN rule that reads only free features is in layer 1 of `rule_matrices.json` even though the taxonomy places CAN rules above every IS and HAS layer. Within a layer, rules keep the taxonomy's order.
6. **The literal table holds what the rules read.** One entry per distinct input of any rule: features by matrix position, then threshold literals by scalar and threshold. Each entry has `index`, `key`, `kind` (`feature` or `threshold`), and `role`, then `feature`, or `scalar`, `operator` (always `>`), and `threshold`. `role` is `null` for a fact of the entity itself; stage a2 uses `agent` and `patient` for bindings. A derived feature that another rule reads is a literal too; its column is overwritten as its layer is computed.
7. **The sparse layer record.** Each layer lists its `terms` (`literals`, `complemented`, `threshold`) and its `outputs` (`output`, `terms`, `threshold`). Term indices are local to the layer. The thresholds are written although they follow from the literal count (always the count, and always 1 for outputs), so a reader of the file sees the matrix form without computing anything.
8. **Symbols.** One entry per non-ISA feature and per scalar: `label`, `kind` (`is`, `has`, `can`, or `scalar`), `derived`, `fluent`, and `arity`. ISA features are not symbols: the tree determines them and no rule reads them. The kind `can` is transitional; CAN features leave the taxonomy in stage a5.
9. **Rule records.** `output`, `inputs` (literal indices), `truth_table` (the bit string), and `expression`. The taxonomy's identity hashes these with `event_types: []`, so a world run's identity (stage a2) will differ from its taxonomy's, as it should: the world's rule set is larger.
10. **What `rule_matrices.json` holds.** `version`, `rule_set_id`, `symbols`, `literals`, `rules`, and `layers`: the tables of `definition.json` apart from `event_types`. The spec asks for the `layers` format; the other tables are included so that the identity can be recomputed from the file alone, which a test does.
11. **Canonical JSON and the file form.** A small serializer in `identity.py` writes both: sorted keys, no whitespace, ASCII escapes, and floats with 17 significant digits (a `.0` is appended when the digits carry no point or exponent, so a float reads as a float). The indented file form keeps the given key order, and writes a container whose elements are all scalars, or scalars and lists of scalars, on one line, so a term or a symbol is one line. Python's `json` module cannot write 17 significant digits, hence the serializer.
12. **The agreement test.** Expected values are the determined columns of the instance matrix, the values that `instances.csv` carries, so the test compares the matrices with what was written. The exhaustive check evaluates one rule's two layers alone over the settings of its own inputs. A disagreement raises `AgreementError`, which names the rule and either the number of disagreeing entities and the first row, or the number of disagreeing input settings and the first setting. `generate()` raises it before anything is written, and the taxonomy command line prints `error: ...` and exits with status 1.
13. **`derived/static_features.csv`.** `label`, then every determined IS and HAS feature in matrix order, as the spec's table says. No `leaf` column, and no CAN columns: the capacities file of stage a2 takes those. The values are the same bits as in `instances.csv`.
14. **The manifest.** `derived/manifest.yaml` is `version: 1` and `files: {<name>: {rule_set_id: <hex>}}`. The loader refuses a file that is not listed, listed but missing, or listed with another identity; the last error names the file path and both identities. The manifest has no top-level identity, so there is one place where each file's identity is recorded.
15. **Tests that walk run folders.** The regression test now requires every golden file to be byte-identical and the added entries to be exactly the stage a1 files. The golden hash file of the relations configurations gains entries for the three new files, keyed by their paths relative to the run folder; every old hash is unchanged. The corpus's check of a taxonomy run folder, and the tests that compare two run folders, walk files recursively, so `derived/` is checked like the rest.
16. **The result object.** `TaxonomyResult.matrices` holds the matrices and the identity (`TaxonomyResult.rule_set_id`). The field has a default of `None` so that existing constructor calls keep working.
17. **Nothing existing records the identity.** `config.yaml` and `summary.yaml` stay byte-identical, so the identity appears only in `rule_matrices.json` and the manifest. Stage a5, which regenerates everything, could add it to `summary.yaml`.

## Open questions

None blocks the stage. Each is an easy change in stage a2 if Jon prefers another answer.

1. **`role: null`.** Is `null` the right marker for a fact of the entity itself in a one-place rule, or should one-place rules name a role (`agent`) so that every literal has a role?
2. **The size of `rule_matrices.json`.** Should the taxonomy write the full tables (choice 10), or only `literals` and `layers` as the spec's wording suggests, with `rules.yaml` as the Boolean form?
3. **CAN columns during stages a1 to a4.** `derived/static_features.csv` leaves out CAN features (choice 13). Should it carry them until `capacities.csv` exists?
4. **Layer numbering.** Should `rule_matrices.json` use dependency layers (choice 5), or the taxonomy's layer numbers, so that the two files agree?
