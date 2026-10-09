"""The world model: one definition of state and change for both simulation modes.

The specification is ``docs/specs/WORLD_AND_LANGUAGE.md``. The package is built in stages. Stage
a1 holds rules as threshold matrices (REL.18), the rule-set identity, and derived values with
their manifest (REL.16). Stage a2 is the world generator (``define``). Stage a3 is the runtime
(``runtime.py``: ``derive``, ``able``, ``legal``, ``legal_bindings``, ``apply``), the loader of a
definition from its files (``RuntimeDefinition``), and the conformance fixtures
(``fixtures.py``). Stage a4 adds episodes and histories. Stage a5b moves the one-place and
two-place event types here from the taxonomy (``unary.py``, ``event_tree.py``,
``constraints.py``, ``projections.py``, ``relation_stats.py``, bundled by ``statics.py``). The
package runs with Python alone: NumPy, polars, and PyYAML.
"""

from semantic_world.world.derived import (
    DERIVED_DIR,
    MANIFEST_FILE,
    check_derived_file,
    load_derived_csv,
    read_manifest,
    write_manifest,
)
from semantic_world.world.errors import (
    AgreementError,
    DefinitionError,
    DerivedError,
    IllegalEventError,
    InterferenceError,
    StepError,
    WorldError,
)
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
    "DefinitionError",
    "DerivedError",
    "IllegalEventError",
    "InterferenceError",
    "StepError",
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
