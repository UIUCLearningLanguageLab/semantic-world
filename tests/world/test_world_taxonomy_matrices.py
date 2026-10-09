"""Stage a1 on the taxonomy: the agreement test over every example configuration, the
``rule_matrices.json`` file, the rule-set identity, and the derived folder with its manifest."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from semantic_world.common.boolean import TruthTable
from semantic_world.taxonomy import TaxonomyResult, config_from_mapping, generate, load_config
from semantic_world.taxonomy.io import BASE_FILE, STATIC_FEATURES_FILE, WORLD_FILES
from semantic_world.taxonomy.rule_matrices import (
    build_rule_matrices,
    check_rule_agreement,
    literal_values,
)
from semantic_world.taxonomy.rules import RuleSet
from semantic_world.world import (
    AgreementError,
    DerivedError,
    RuleMatrices,
    canonical_json,
    check_agreement,
    load_derived_csv,
    read_json,
    read_manifest,
    rule_set_id,
)
from semantic_world.world.matrices import settings_array

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
CONFIGURATIONS = ("tiny", "default", "rule_file", "tiny_relations", "relations")


@pytest.fixture(scope="module", params=CONFIGURATIONS)
def run(request: pytest.FixtureRequest) -> TaxonomyResult:
    return generate(load_config(DATA / f"{request.param}.yaml"))


@pytest.fixture(scope="module")
def folder(run: TaxonomyResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run.write(tmp_path_factory.mktemp("run") / run.config.name)


# ---------------------------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------------------------


def test_every_rule_is_covered_and_agrees_on_every_instance(run: TaxonomyResult) -> None:
    assert run.matrices is not None
    outputs = run.matrices.matrices.outputs
    assert sorted(outputs) == sorted(rule.output.label for rule in run.rules.rules)
    assert {r.output.type for r in run.rules.rules} <= {"is", "has"}
    report = check_rule_agreement(run.matrices, run.rules, run.instances)
    assert report.rules == len(run.rules.rules)
    assert report.entities == len(run.instances)
    assert report.exhaustive_rules == sum(1 for r in run.rules.rules if r.arity <= 12)
    assert report.settings == sum(2**r.arity for r in run.rules.rules if r.arity <= 12)


def test_matrices_reproduce_compute_features(run: TaxonomyResult) -> None:
    """The matrix form equals the truth-table evaluation over the instances and over fresh
    random objects, with every derived literal computed from scratch."""
    assert run.matrices is not None
    rng = np.random.default_rng(0)
    features = run.features
    free = rng.integers(0, 2, size=(300, len(features.free)), dtype=np.uint8)
    scalars = rng.normal(size=(300, features.scalar_count)) if features.scalar_count else None
    by_tables = run.rules.compute(free, scalars)
    values = np.zeros((300, len(features)), dtype=np.uint8)
    values[:, features.free_positions] = free
    base = literal_values(run.matrices, run.rules, values, scalars)
    out = run.matrices.matrices.evaluate(base)
    for rule in run.rules.rules:
        assert np.array_equal(out[rule.output.label], by_tables[:, rule.output.position]), (
            rule.output.label
        )


def test_threshold_literals_are_computed_before_the_first_layer() -> None:
    result = generate(load_config(DATA / "relations.yaml"))
    assert result.matrices is not None
    thresholds = [lit for lit in result.matrices.literals if lit.reads["kind"] == "threshold"]
    assert thresholds, "the relations configuration has rules that read scalars"
    for literal in thresholds:
        assert literal.reads["operator"] == ">"
        assert literal.key == f"{literal.reads['scalar']}>{literal.reads['threshold']:.4f}"
    base = literal_values(
        result.matrices, result.rules, result.instances.values, result.instances.scalars
    )
    index = result.matrices.matrices.literal_index(thresholds[0].key)
    scalar = int(thresholds[0].reads["scalar"].split(".")[1])
    expected = result.instances.scalars[:, scalar - 1] > thresholds[0].reads["threshold"]
    assert np.array_equal(base[:, index].astype(bool), expected)


def _with_rule_replaced(rules: RuleSet, index: int, **changes) -> RuleSet:
    replaced = list(rules.rules)
    replaced[index] = dataclasses.replace(replaced[index], **changes)
    return dataclasses.replace(rules, rules=tuple(replaced))


def test_constant_rules_give_their_constant() -> None:
    """Two rules of the tiny taxonomy are replaced by constants (the taxonomy's own layer rules
    never sample a constant); the matrices give the constants on every instance."""
    result = generate(load_config(DATA / "tiny.yaml"))
    rules = result.rules
    first, second = rules.rules[0], rules.rules[1]
    changed = _with_rule_replaced(rules, 0, table=TruthTable.constant(first.arity, True))
    changed = _with_rule_replaced(changed, 1, table=TruthTable.constant(second.arity, False))
    matrices = build_rule_matrices(changed)
    true_rule = matrices.matrices.rule(first.output.label)
    false_rule = matrices.matrices.rule(second.output.label)
    assert true_rule.is_constant_true and true_rule.terms[0].threshold == 0
    assert false_rule.is_constant_false
    values = result.instances.values
    scalars = result.instances.scalars if result.features.scalar_count else None
    out = matrices.matrices.evaluate(literal_values(matrices, changed, values, scalars))
    assert out[first.output.label].tolist() == [1] * len(result.instances)
    assert out[second.output.label].tolist() == [0] * len(result.instances)
    base = literal_values(matrices, changed, values, scalars)
    report = check_agreement(matrices.matrices, matrices.rules, base)
    assert report.rules == len(rules.rules)


def test_a_disagreement_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    module = sys.modules["semantic_world.taxonomy.generate"]

    def broken(matrices, rules, instances):
        raise AgreementError("the matrix form of the rule for PROPERTY.7 disagrees (test)")

    monkeypatch.setattr(module, "check_rule_agreement", broken)
    with pytest.raises(AgreementError, match="PROPERTY.7"):
        generate(load_config(DATA / "tiny.yaml"))


def test_the_command_line_reports_an_agreement_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from semantic_world.taxonomy.__main__ import main

    module = sys.modules["semantic_world.taxonomy.generate"]

    def broken(matrices, rules, instances):
        raise AgreementError("the matrix form of the rule for PROPERTY.7 disagrees (test)")

    monkeypatch.setattr(module, "check_rule_agreement", broken)
    code = main([str(DATA / "tiny.yaml"), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "error: the matrix form of the rule for PROPERTY.7" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------------------------
# rule_matrices.json
# ---------------------------------------------------------------------------------------------


def test_rule_matrices_json_holds_the_definition_tables(run: TaxonomyResult, folder: Path) -> None:
    assert run.matrices is not None
    data = read_json(folder / "rule_matrices.json")
    assert list(data) == ["version", "rule_set_id", "symbols", "literals", "rules", "layers"]
    assert data["version"] == 1 and data["rule_set_id"] == run.rule_set_id
    labels = [s["label"] for s in data["symbols"]]
    assert labels == list(run.features.labels) + list(run.features.scalar_labels)
    derived = {s["label"] for s in data["symbols"] if s["derived"]}
    assert derived == {r.output.label for r in run.rules.rules}
    assert all(not s["fluent"] and s["arity"] == 1 for s in data["symbols"])
    assert [r["output"] for r in data["rules"]] == [r.output.label for r in run.rules.rules]
    for entry, rule in zip(data["rules"], run.rules.rules, strict=True):
        assert entry["truth_table"] == rule.table.bit_string()
        assert entry["expression"] == str(rule.expression)
        keys = [data["literals"][i]["key"] for i in entry["inputs"]]
        assert keys == list(
            next(s for s in run.matrices.rules if s.output == rule.output.label).inputs
        )
    assert [lit["index"] for lit in data["literals"]] == list(range(len(data["literals"])))


def test_rule_matrices_json_rebuilds_and_agrees(run: TaxonomyResult, folder: Path) -> None:
    assert run.matrices is not None
    data = read_json(folder / "rule_matrices.json")
    rebuilt = RuleMatrices.from_record(data["literals"], data["layers"])
    assert rebuilt.layers == run.matrices.matrices.layers
    scalars = run.instances.scalars if run.features.scalar_count else None
    base = literal_values(run.matrices, run.rules, run.instances.values, scalars)
    check_agreement(rebuilt, run.matrices.rules, base)


def test_layers_are_in_dependency_order(run: TaxonomyResult, folder: Path) -> None:
    data = read_json(folder / "rule_matrices.json")
    computed: set[int] = set()
    literal_feature = {lit["index"]: lit.get("feature") for lit in data["literals"]}
    derived = {s["label"] for s in data["symbols"] if s["derived"]}
    for layer in data["layers"]:
        for term in layer["terms"]:
            for literal in term["literals"]:
                feature = literal_feature[literal]
                if feature in derived:
                    assert feature in computed, f"{feature} is read before it is computed"
            assert term["threshold"] == len(term["literals"]) == len(term["complemented"])
        for output in layer["outputs"]:
            assert output["threshold"] == 1
            computed.add(output["output"])
    assert computed == derived


def test_floats_in_the_file_have_17_significant_digits() -> None:
    result = generate(load_config(DATA / "tiny_relations.yaml"))
    assert result.matrices is not None
    text = json.dumps(result.matrices.record())  # not the file: check the writer instead
    from semantic_world.world import to_json

    written = to_json(result.matrices.record(), indent=1, sort_keys=False)
    thresholds = [
        lit.reads["threshold"] for lit in result.matrices.literals if "threshold" in lit.reads
    ]
    assert thresholds
    for value in thresholds:
        assert format(value, ".17g") in written or f"{format(value, '.17g')}.0" in written
    assert text  # json.dumps also round-trips, but writes the shortest form


# ---------------------------------------------------------------------------------------------
# Rule-set identity
# ---------------------------------------------------------------------------------------------


def test_identity_is_stable_across_runs(run: TaxonomyResult) -> None:
    again = generate(load_config(DATA / f"{run.config.name}.yaml"))
    assert again.rule_set_id == run.rule_set_id
    assert again.matrices.record() == run.matrices.record()


def test_identity_is_recomputed_from_the_file(run: TaxonomyResult, folder: Path) -> None:
    data = read_json(folder / "rule_matrices.json")
    assert rule_set_id(data["symbols"], data["literals"], data["rules"]) == data["rule_set_id"]
    assert len(data["rule_set_id"]) == 64


def test_identity_changes_when_any_one_rule_changes() -> None:
    result = generate(load_config(DATA / "relations.yaml"))
    base = build_rule_matrices(result.rules).rule_set_id
    seen = {base}
    for index, rule in enumerate(result.rules.rules):
        bits = list(rule.table.bits)
        bits[0] ^= 1
        changed = _with_rule_replaced(
            result.rules, index, table=TruthTable(rule.arity, tuple(bits))
        )
        identity = build_rule_matrices(changed).rule_set_id
        assert identity not in seen, rule.output.label
        seen.add(identity)


def test_identity_changes_when_a_threshold_changes() -> None:
    result = generate(load_config(DATA / "relations.yaml"))
    base = build_rule_matrices(result.rules).rule_set_id
    index, rule = next((i, r) for i, r in enumerate(result.rules.rules) if r.thresholds)
    threshold = rule.thresholds[0]
    moved = dataclasses.replace(threshold, threshold=threshold.threshold + 0.0001)
    inputs = tuple(moved if item is threshold else item for item in rule.inputs)
    changed = _with_rule_replaced(result.rules, index, inputs=inputs)
    assert build_rule_matrices(changed).rule_set_id != base


def test_identity_does_not_depend_on_the_tree_or_the_instances() -> None:
    """Rules come from their own streams, so another instance count keeps the identity."""
    base = load_config(DATA / "tiny.yaml").resolved()
    base.pop("provenance", None)
    a = generate(config_from_mapping(base))
    b = generate(config_from_mapping({**base, "instances": {**base["instances"], "per_leaf": 7}}))
    assert a.rule_set_id == b.rule_set_id
    assert len(a.instances) != len(b.instances)


def test_different_seeds_give_different_identities() -> None:
    a = generate(load_config(DATA / "tiny.yaml", seed=1))
    b = generate(load_config(DATA / "tiny.yaml", seed=2))
    assert a.rule_set_id != b.rule_set_id


def test_canonical_json_of_the_record_is_deterministic(run: TaxonomyResult) -> None:
    assert run.matrices is not None
    record = run.matrices.record()
    assert canonical_json(record) == canonical_json(json.loads(json.dumps(record)))


# ---------------------------------------------------------------------------------------------
# derived/
# ---------------------------------------------------------------------------------------------


def test_static_features_holds_the_determined_property_and_part_features(
    run: TaxonomyResult, folder: Path
) -> None:
    """``derived/static_features.csv`` holds every determined feature, which ``base.csv`` no
    longer carries; together the two files are every static feature of every instance."""
    frame = pl.read_csv(folder / "derived" / STATIC_FEATURES_FILE)
    determined = [f.label for f in run.features.determined]
    assert determined == [f.label for f in run.features.features if not f.free]
    assert frame.columns == ["label"] + determined
    assert frame["label"].to_list() == list(run.instances.labels)
    for label in determined:
        position = run.features[label].position
        assert frame[label].to_list() == run.instances.values[:, position].tolist(), label
    base = pl.read_csv(folder / BASE_FILE)
    free = [f.label for f in run.features.free]
    assert base.columns == ["label", "leaf", *free, *run.features.scalar_labels]
    assert not set(free) & set(determined)
    assert all(c.startswith(("PROPERTY.", "PART.")) for c in determined + free)


def test_manifest_tags_the_file_with_the_runs_identity(run: TaxonomyResult, folder: Path) -> None:
    derived = folder / "derived"
    assert read_manifest(derived) == {STATIC_FEATURES_FILE: run.rule_set_id}
    frame = load_derived_csv(derived, STATIC_FEATURES_FILE, run.rule_set_id)
    assert len(frame) == len(run.instances)
    other = generate(load_config(DATA / "tiny.yaml", seed=99))
    with pytest.raises(DerivedError) as info:
        load_derived_csv(derived, STATIC_FEATURES_FILE, other.rule_set_id)
    assert run.rule_set_id in str(info.value) and other.rule_set_id in str(info.value)
    assert "static_features.csv" in str(info.value)


def test_every_stage_a1_file_is_written(folder: Path) -> None:
    for name in WORLD_FILES:
        assert (folder / name).is_file(), name


def test_exhaustive_check_uses_the_rules_own_input_settings(run: TaxonomyResult) -> None:
    assert run.matrices is not None
    for spec in run.matrices.rules:
        if len(spec.inputs) <= 12:
            got = run.matrices.matrices.evaluate_rule(
                spec.output, spec.inputs, settings_array(len(spec.inputs))
            )
            assert got.tolist() == list(spec.table.bits), spec.output
