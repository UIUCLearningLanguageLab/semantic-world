"""The world model: one definition of state and change for both simulation modes.

The specification is ``docs/specs/WORLD_AND_LANGUAGE.md``. The package is built in stages. Stage
a1 holds rules as threshold matrices (REL.18), the rule-set identity, and derived values with
their manifest (REL.16). The package runs with Python alone: NumPy, polars, and PyYAML.
"""

from semantic_world.world.derived import (
    DERIVED_DIR,
    MANIFEST_FILE,
    check_derived_file,
    load_derived_csv,
    read_manifest,
    write_manifest,
)
from semantic_world.world.errors import AgreementError, DerivedError, WorldError
from semantic_world.world.identity import (
    DEFINITION_VERSION,
    canonical_json,
    read_json,
    rule_set_id,
    to_json,
    write_json,
)
from semantic_world.world.matrices import (
    MAX_EXHAUSTIVE_INPUTS,
    AgreementReport,
    Layer,
    LiteralSpec,
    RuleMatrices,
    RuleMatrix,
    RuleSpec,
    Term,
    build_matrices,
    check_agreement,
    dependency_layers,
    evaluate_by_tables,
)

__all__ = [
    "DEFINITION_VERSION",
    "DERIVED_DIR",
    "MANIFEST_FILE",
    "MAX_EXHAUSTIVE_INPUTS",
    "AgreementError",
    "AgreementReport",
    "DerivedError",
    "Layer",
    "LiteralSpec",
    "RuleMatrices",
    "RuleMatrix",
    "RuleSpec",
    "Term",
    "WorldError",
    "build_matrices",
    "canonical_json",
    "check_agreement",
    "check_derived_file",
    "dependency_layers",
    "evaluate_by_tables",
    "load_derived_csv",
    "read_json",
    "read_manifest",
    "rule_set_id",
    "to_json",
    "write_json",
    "write_manifest",
]
