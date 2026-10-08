"""Label translation between the taxonomy's current labels and the labels of the world model.

Until stage a5 relabels the taxonomy and the corpus, the world package reads taxonomy results
that carry the old labels (``C1.3``, ``I1.3.2``, ``IS.4``, ``HAS.12``, ``SC.2``, ``CAN.3``,
``V1.2``, ``VF.5``, ``K.VF.5``, ``a.``, ``p.``) and writes the labels of "Labels" in
``docs/specs/WORLD_AND_LANGUAGE.md`` (``CATEGORY.1.3``, ``INSTANCE.1.3.2``, ``PROPERTY.4``,
``PART.12``, ``SCALARDIM.2``, ``EVENTTYPE1.3``, ``EVENTTYPE2.1.2``, ``EVENTFEAT.5``,
``CONSTRAINT.EVENTFEAT.5``, ``agent.``, ``patient.``). This module is the one place where the
two vocabularies meet. Stage a5 removes it.
"""

from __future__ import annotations

import re

from semantic_world.world.errors import WorldError

ROLE_PREFIXES = {"a": "agent", "p": "patient"}
_ROLE_PREFIXES_BACK = {new: old for old, new in ROLE_PREFIXES.items()}

_PATH = r"\d+(?:\.\d+)*"
_INDEX = r"\d+"

# (pattern over an old label, replacement with the new label). Order matters: longer prefixes
# first. ``CAN.<n>`` names a CAN feature, which becomes a one-place event type.
_OLD_TO_NEW: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(f"^{pattern}$"), replacement)
    for pattern, replacement in (
        (rf"ISA\.C({_PATH})", r"ISA.CATEGORY.\1"),
        (rf"CANBE\.V({_PATH})", r"CANBE.EVENTTYPE2.\1"),
        (rf"CAN\.V({_PATH})", r"CAN.EVENTTYPE2.\1"),
        (rf"ACTUAL_CANBE\.V({_PATH})", r"ACTUAL_CANBE.EVENTTYPE2.\1"),
        (rf"ACTUAL_CAN\.V({_PATH})", r"ACTUAL_CAN.EVENTTYPE2.\1"),
        (rf"CAN\.({_INDEX})", r"EVENTTYPE1.\1"),
        (rf"K\.VF\.({_INDEX})", r"CONSTRAINT.EVENTFEAT.\1"),
        (rf"K\.V({_PATH})", r"CONSTRAINT.EVENTTYPE2.\1"),
        (rf"VF\.({_INDEX})", r"EVENTFEAT.\1"),
        (rf"IS\.({_INDEX})", r"PROPERTY.\1"),
        (rf"HAS\.({_INDEX})", r"PART.\1"),
        (rf"SC\.({_INDEX})", r"SCALARDIM.\1"),
        (rf"C({_PATH})", r"CATEGORY.\1"),
        (rf"I({_PATH})", r"INSTANCE.\1"),
        (rf"V({_PATH})", r"EVENTTYPE2.\1"),
    )
)

_NEW_TO_OLD: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(f"^{pattern}$"), replacement)
    for pattern, replacement in (
        (rf"ISA\.CATEGORY\.({_PATH})", r"ISA.C\1"),
        (rf"CANBE\.EVENTTYPE2\.({_PATH})", r"CANBE.V\1"),
        (rf"CAN\.EVENTTYPE2\.({_PATH})", r"CAN.V\1"),
        (rf"CAN\.EVENTTYPE1\.({_INDEX})", r"CAN.\1"),
        (rf"CONSTRAINT\.EVENTFEAT\.({_INDEX})", r"K.VF.\1"),
        (rf"CONSTRAINT\.EVENTTYPE2\.({_PATH})", r"K.V\1"),
        (rf"EVENTFEAT\.({_INDEX})", r"VF.\1"),
        (rf"EVENTTYPE1\.({_INDEX})", r"CAN.\1"),
        (rf"EVENTTYPE2\.({_PATH})", r"V\1"),
        (rf"PROPERTY\.({_INDEX})", r"IS.\1"),
        (rf"PART\.({_INDEX})", r"HAS.\1"),
        (rf"SCALARDIM\.({_INDEX})", r"SC.\1"),
        (rf"CATEGORY\.({_PATH})", r"C\1"),
        (rf"INSTANCE\.({_PATH})", r"I\1"),
    )
)

_UNCHANGED = {"TRUE", "FALSE", "AND", "OR", "NOT", "XOR", "label", "leaf"}


def _apply(table: tuple[tuple[re.Pattern[str], str], ...], label: str, direction: str) -> str:
    for pattern, replacement in table:
        if pattern.match(label):
            return pattern.sub(replacement, label)
    raise WorldError(f"cannot translate the {direction} label {label!r}")


def translate(label: str) -> str:
    """A taxonomy label in the world's vocabulary. ``CAN.<n>`` is a feature in the taxonomy and
    becomes the one-place event type ``EVENTTYPE1.<n>``; see :func:`capacity_label` for the
    capacity column. Labels that are already new, and the keywords, pass through."""
    if label in _UNCHANGED or is_new(label):
        return label
    return _apply(_OLD_TO_NEW, label, "taxonomy")


def untranslate(label: str) -> str:
    """A world label in the taxonomy's vocabulary (for parsing with the taxonomy's tools)."""
    if label in _UNCHANGED:
        return label
    return _apply(_NEW_TO_OLD, label, "world")


def is_new(label: str) -> bool:
    """Whether a label is already in the world's vocabulary."""
    return any(pattern.match(label) for pattern, _ in _NEW_TO_OLD)


def capacity_label(old_column: str) -> str:
    """The capacity column for a taxonomy column: ``CAN.3`` becomes ``CAN.EVENTTYPE1.3``, and
    ``CAN.V1.2`` becomes ``CAN.EVENTTYPE2.1.2``."""
    if re.fullmatch(rf"CAN\.{_INDEX}", old_column):
        return f"CAN.EVENTTYPE1.{old_column[4:]}"
    return translate(old_column)


def translate_column(old_column: str) -> str:
    """A column name of a taxonomy table: feature labels, capacities, ``label``, ``leaf``."""
    if old_column in _UNCHANGED:
        return old_column
    return capacity_label(old_column)


_TOKEN = re.compile(
    r"(?<![\w.])(a|p|agent|patient)\.([A-Za-z_]+(?:\.[\w]+)*)|(?<![\w.])([A-Za-z_]+(?:\.[\w]+)*)"
)


def _translate_tokens(text: str, label_function, role_table: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        role, roled, plain = match.group(1), match.group(2), match.group(3)
        if role is not None:
            return f"{role_table.get(role, role)}.{label_function(roled)}"
        if plain in _UNCHANGED or re.fullmatch(r"[A-Za-z_]+", plain):
            return plain
        return label_function(plain)

    return _TOKEN.sub(replace, text)


def translate_expression(text: str) -> str:
    """An expression or key in the taxonomy's vocabulary (``(a.HAS.4 AND NOT p.IS.7) OR
    SC.2 > 0.4127``) in the world's (``(agent.PART.4 AND NOT patient.PROPERTY.7) OR
    SCALARDIM.2 > 0.4127``)."""
    return _translate_tokens(text, translate, ROLE_PREFIXES)


def untranslate_expression(text: str) -> str:
    """The inverse of :func:`translate_expression`, for parsing with the taxonomy's parser."""
    return _translate_tokens(text, untranslate, _ROLE_PREFIXES_BACK)
