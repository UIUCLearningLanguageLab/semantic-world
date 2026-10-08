"""Errors of the world package. Every error names what went wrong and where."""

from __future__ import annotations


class WorldError(Exception):
    """An error the world package raises on purpose."""


class AgreementError(WorldError):
    """The matrix form of a rule and its truth table disagree (REL.18)."""


class DerivedError(WorldError):
    """A derived file cannot be used: its manifest is missing or malformed, or its rule-set
    identity differs from the rule set in use (REL.16)."""


class DefinitionError(WorldError):
    """A definition record cannot be loaded: it is malformed, its tables disagree, or its
    rule-set identity is not the identity of its tables."""


class StepError(WorldError):
    """``apply`` refuses a step. ``kind`` names the kind of error as the conformance fixtures
    write it: ``illegal`` or ``interference``."""

    kind = ""


class IllegalEventError(StepError):
    """An event of a step is not legal in the state: its event type is unknown or a category,
    its binding is malformed, its requirement does not hold, or its precondition does not hold."""

    kind = "illegal"


class InterferenceError(StepError):
    """Two events of one step interfere: both write the same fluent of the same entity, one
    writes a base fluent that the other's precondition reads (through the cone of every derived
    fluent the precondition reads), or the same event occurs twice."""

    kind = "interference"
