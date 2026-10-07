"""Errors of the world package. Every error names what went wrong and where."""

from __future__ import annotations


class WorldError(Exception):
    """An error the world package raises on purpose."""


class AgreementError(WorldError):
    """The matrix form of a rule and its truth table disagree (REL.18)."""


class DerivedError(WorldError):
    """A derived file cannot be used: its manifest is missing or malformed, or its rule-set
    identity differs from the rule set in use (REL.16)."""
