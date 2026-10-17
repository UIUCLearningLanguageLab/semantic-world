"""Stage c1 of ``docs/specs/WORLD_PHASE_C.md``: ``labels.kind``. With ``place``, the generator
writes ``PLACECATEGORY``, ``PLACE``, ``ISA.PLACECATEGORY``, ``PLACEPROPERTY``, ``PLACEPART``, and
``PLACESCALARDIM`` in place of the entity labels, and nothing else changes: the same settings
and seed give the same run, label for label. With the default, ``entity``, every output is
byte-identical to the output before the stage (``test_taxonomy_golden.py`` checks the golden
hashes)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from semantic_world.taxonomy import ConfigError, config_from_mapping, generate, load_config
from semantic_world.taxonomy.config import Labels

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "taxonomy"
ENTITY_LABEL = re.compile(r"(?<![A-Z])(CATEGORY|INSTANCE|PROPERTY|PART|SCALARDIM)\.\d")
PLACE_LABEL = re.compile(
    r"(?<![A-Z])(PLACECATEGORY|PLACE|PLACEPROPERTY|PLACEPART|PLACESCALARDIM)\.\d"
)
PREFIXES = {
    "CATEGORY": "PLACECATEGORY",
    "INSTANCE": "PLACE",
    "PROPERTY": "PLACEPROPERTY",
    "PART": "PLACEPART",
    "SCALARDIM": "PLACESCALARDIM",
}


def relabel(text: str) -> str:
    return ENTITY_LABEL.sub(lambda m: PREFIXES[m.group(1)] + "." + m.group(0)[-1], text)


def test_the_kind_defaults_to_entity_and_is_validated() -> None:
    config = config_from_mapping({})
    assert config.labels == Labels.for_kind("entity")
    assert config.labels.category == "CATEGORY" and config.labels.scalar == "SCALARDIM"
    assert config.resolved()["labels"] == {"kind": "entity"}
    place = config_from_mapping({"labels": {"kind": "place"}})
    assert place.labels.kind == "place"
    assert (place.labels.category, place.labels.instance) == ("PLACECATEGORY", "PLACE")
    assert (place.labels.property, place.labels.part) == ("PLACEPROPERTY", "PLACEPART")
    assert place.labels.scalar == "PLACESCALARDIM"
    with pytest.raises(ConfigError, match="labels.kind"):
        config_from_mapping({"labels": {"kind": "thing"}})
    with pytest.raises(ConfigError, match="labels.colour"):
        config_from_mapping({"labels": {"colour": "red"}})
    with pytest.raises(ValueError):
        Labels.for_kind("thing")


@pytest.mark.parametrize("name", ["tiny_relations", "tiny"])
def test_a_place_run_is_the_entity_run_relabeled(name: str, tmp_path: Path) -> None:
    data = yaml.safe_load((DATA / f"{name}.yaml").read_text(encoding="utf-8"))
    entity = generate(config_from_mapping(data, source="<entity>"))
    place = generate(config_from_mapping({**data, "labels": {"kind": "place"}}, source="<place>"))
    entity_folder = entity.write(tmp_path / "entity")
    place_folder = place.write(tmp_path / "place")
    files = sorted(p.relative_to(entity_folder) for p in entity_folder.rglob("*") if p.is_file())
    assert files == sorted(
        p.relative_to(place_folder) for p in place_folder.rglob("*") if p.is_file()
    )
    for relative in files:
        if relative.name == "config.yaml":
            continue
        # the labels are symbols of the hashed tables, so the identity moves with them
        expected = relabel((entity_folder / relative).read_text(encoding="utf-8")).replace(
            entity.rule_set_id, place.rule_set_id
        )
        found = (place_folder / relative).read_text(encoding="utf-8")
        if relative.suffix == ".yaml":
            # the longer labels wrap YAML lines elsewhere; the data is what must match
            assert yaml.safe_load(found) == yaml.safe_load(expected), relative
        else:
            assert found == expected, relative
        assert not ENTITY_LABEL.search(found), relative
    assert place.rule_set_id != entity.rule_set_id  # the symbols' labels are hashed
    assert [c.label for c in place.tree.categories][:2] == ["PLACECATEGORY.1", "PLACECATEGORY.1.1"]
    assert place.instances.labels[0] == "PLACE.1.1.1"
    assert place.features.scalar_labels == tuple(
        f"PLACESCALARDIM.{i}" for i in range(1, place.config.scalars.count + 1)
    )
    for rule in place.rules.rules:
        for threshold in rule.thresholds:
            assert threshold.label.startswith("PLACESCALARDIM.")
    isa = [c for c in place.frames()["categories_generative.csv"].columns if c.startswith("ISA.")]
    assert isa and all(c.startswith("ISA.PLACECATEGORY.") for c in isa)


@pytest.mark.parametrize("name", ["places_default", "places_tiny"])
def test_the_place_configurations_load_and_run(name: str) -> None:
    config = load_config(DATA / f"{name}.yaml")
    assert config.labels.kind == "place"
    result = generate(config)
    assert all(label.startswith("PLACE.") for label in result.instances.labels)
    assert all(f.label.startswith("PLACEPROPERTY.") for f in result.features.features)
    assert result.features.scalar_labels[0] == "PLACESCALARDIM.1"


def test_a_place_rule_file_names_place_labels(tmp_path: Path) -> None:
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        yaml.safe_dump(
            {
                "templates": [{"family": "literal", "weight": 1}],
                "explicit": [
                    {
                        "output": "PLACEPROPERTY.4",
                        "expression": "PLACEPROPERTY.1 AND PLACESCALARDIM.1 > 0.25",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data = yaml.safe_load((DATA / "places_tiny.yaml").read_text(encoding="utf-8"))
    data["rules"] = {"source": "file", "file": str(rules)}
    result = generate(config_from_mapping(data, source=str(tmp_path / "places.yaml")))
    rule = result.rules.rule_for("PLACEPROPERTY.4")
    assert str(rule.expression) == "PLACEPROPERTY.1 AND PLACESCALARDIM.1 > 0.2500"
    data["rules"]["file"] = str(rules)
    rules.write_text(
        yaml.safe_dump(
            {"explicit": [{"output": "PLACEPROPERTY.4", "expression": "SCALARDIM.1 > 0.25"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown scalar SCALARDIM.1"):
        generate(config_from_mapping(data, source=str(tmp_path / "places.yaml")))
