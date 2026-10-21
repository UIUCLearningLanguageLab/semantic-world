"""Pair fluents between entities (stage c2a of ``docs/specs/WORLD_PHASE_C.md``, "Pair fluents").

A pair fluent ``PAIRFL.<n>`` is a Boolean fluent of two arguments, stored as the set of ordered
pairs for which it holds at a time point. Its profile is its symmetry (``symmetric``: ``(x, y)``
holds exactly when ``(y, x)`` holds; ``asymmetric``: never both; ``general``: no constraint)
and whether it is ``functional`` (at most one pair in each row, and for a symmetric functional
pair fluent at most one partner in each row and each column). Every pair fluent is irreflexive.
A pair fluent between entities is never total (WM.38); the total functional profile is the
location fluents' alone, toward places (stage c3).

Since stage c2b the world draws ``pair_fluents.count`` pair fluents between entities from
``world:pair_fluents``, in label order: each one's symmetry from ``pair_fluents.symmetry``,
whether it is functional at ``pair_fluents.functional_rate``, and its initial rate uniformly
from ``pair_fluents.initial_rates``. An override ``pair_fluents.fluents.PAIRFL.<n>`` or an event
file's ``pair_fluents`` entry replaces the drawn values of that pair fluent after the draws are
made, so that no other pair fluent's draws shift (as ``fluents.py`` draws an overridden Boolean
fluent's rate before replacing it); an event file's entries must be among ``PAIRFL.1`` to
``PAIRFL.<count>``. The initial pairs are drawn from ``world:pair_initial`` at each pair
fluent's initial rate, respecting the profile ("Initial pairs between entities"): a symmetric
pair fluent draws each unordered pair at the rate and writes both directions; an asymmetric one
draws each unordered pair at the rate and then one direction; a general one draws each ordered
pair at the rate; a functional one gives each entity, at the rate, one partner drawn uniformly
among the entities that still keep the profile. The pairs are written to ``initial_pairs.csv``,
one ordered pair per row (both directions of a symmetric pair).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.rules import _draw
from semantic_world.world.errors import WorldError

PAIR_PREFIX = "PAIRFL"
PAIR_FLUENT_KIND = "pair_fluent"
"""The symbol kind of a pair fluent in ``definition.json``."""
SYMMETRIES = ("symmetric", "asymmetric", "general")
INITIAL_PAIRS_FILE = "initial_pairs.csv"
INITIAL_PAIRS_COLUMNS = ("pair_fluent", "first", "second")
"""The columns of ``initial_pairs.csv``: the pair fluent's label, and the labels of the first
and the second argument of one ordered pair that holds at ``TIME.1``."""


def pair_label(index: int) -> str:
    return f"{PAIR_PREFIX}.{index}"


def is_pair_label(label: str) -> bool:
    prefix, _, number = label.partition(".")
    return prefix == PAIR_PREFIX and number.isdigit()


@dataclass(frozen=True)
class PairFluent:
    """One pair fluent between entities and its profile."""

    label: str
    index: int
    symmetry: str
    functional: bool
    initial_rate: float
    total: bool = False
    """Always False for a pair fluent between entities; a location fluent (stage c3) is total."""

    def __post_init__(self) -> None:
        if self.symmetry not in SYMMETRIES:
            raise WorldError(f"{self.label}: unknown symmetry {self.symmetry!r}")
        if not 0.0 <= self.initial_rate <= 1.0:
            raise WorldError(f"{self.label}: the initial rate is not between 0 and 1")

    @property
    def symmetric(self) -> bool:
        return self.symmetry == "symmetric"

    @property
    def asymmetric(self) -> bool:
        return self.symmetry == "asymmetric"

    def symbol_record(self) -> dict[str, Any]:
        """The symbol of ``definition.json``: a fluent of arity 2 with its profile and its
        initial rate, all hashed into the rule-set identity."""
        return {
            "label": self.label,
            "kind": PAIR_FLUENT_KIND,
            "derived": False,
            "fluent": True,
            "arity": 2,
            "symmetry": self.symmetry,
            "functional": self.functional,
            "total": self.total,
            "initial_rate": float(self.initial_rate),
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> PairFluent:
        label = str(data["label"])
        prefix, _, number = label.partition(".")
        if prefix != PAIR_PREFIX or not number.isdigit():
            raise WorldError(f"{label} is not a pair fluent label")
        return cls(
            label,
            int(number),
            str(data["symmetry"]),
            bool(data["functional"]),
            float(data["initial_rate"]),
            bool(data.get("total", False)),
        )


@dataclass(frozen=True)
class PairFluents:
    """The pair fluents of a world and their initial pairs, one Boolean matrix per pair fluent,
    rows the entities and columns the targets (the entities in this stage, the places for a
    location fluent in stage c3)."""

    fluents: tuple[PairFluent, ...]
    initial_pairs: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.initial_pairs) != len(self.fluents):
            raise WorldError("every pair fluent needs its initial pairs")
        object.__setattr__(self, "_by_label", {f.label: f for f in self.fluents})
        object.__setattr__(self, "_index", {f.label: i for i, f in enumerate(self.fluents)})
        for matrix in self.initial_pairs:
            matrix.setflags(write=False)

    def __len__(self) -> int:
        return len(self.fluents)

    def __iter__(self):
        return iter(self.fluents)

    def __getitem__(self, label: str) -> PairFluent:
        try:
            return self._by_label[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown pair fluent {label!r}") from None

    def __contains__(self, label: object) -> bool:
        return label in self._by_label  # type: ignore[attr-defined]

    def index(self, label: str) -> int:
        return self._index[label]  # type: ignore[attr-defined]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(f.label for f in self.fluents)


EMPTY = PairFluents((), ())


def empty_pair_fluents() -> PairFluents:
    return EMPTY


# ---------------------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------------------


def profile_violations(fluent: PairFluent, matrix: np.ndarray) -> list[str]:
    """Why a pair matrix breaks the pair fluent's profile: each violation as a sentence, none
    when the profile holds. A pair fluent between entities is irreflexive; symmetric: every
    pair with its reverse; asymmetric: no pair with its reverse; functional: at most one pair
    in each row (and in each column of a symmetric one, which symmetry already gives)."""
    found: list[str] = []
    m = np.asarray(matrix, dtype=bool)
    if m.ndim != 2:
        raise WorldError(f"the pairs of {fluent.label} are a 2-D matrix, got {m.shape}")
    # Each check tests the whole matrix with ``any`` first and locates the first violation only
    # when there is one: the runtime checks every profile after every step of every episode.
    if m.shape[0] == m.shape[1]:
        diagonal = np.diagonal(m)
        if diagonal.any():
            found.append(
                f"{fluent.label} holds between entity {int(np.flatnonzero(diagonal)[0])} and itself"
            )
        if fluent.symmetric:
            broken = m & ~m.T
            if broken.any():
                rows, columns = np.nonzero(broken)
                found.append(
                    f"{fluent.label} is symmetric, but ({int(rows[0])}, {int(columns[0])}) "
                    f"holds without ({int(columns[0])}, {int(rows[0])})"
                )
        if fluent.asymmetric:
            broken = m & m.T
            if broken.any():
                rows, columns = np.nonzero(broken)
                found.append(
                    f"{fluent.label} is asymmetric, but ({int(rows[0])}, {int(columns[0])}) "
                    f"and ({int(columns[0])}, {int(rows[0])}) both hold"
                )
    if fluent.functional or fluent.total:
        counts = m.sum(axis=1)
        if fluent.functional and (counts > 1).any():
            over = np.flatnonzero(counts > 1)
            found.append(
                f"{fluent.label} is functional, but the row of entity {int(over[0])} holds "
                f"{int(counts[over[0]])} pairs"
            )
        if fluent.total and (counts == 0).any():
            empty = np.flatnonzero(counts == 0)
            found.append(
                f"{fluent.label} is total, but the row of entity {int(empty[0])} holds no pair"
            )
    return found


# ---------------------------------------------------------------------------------------------
# Initial pairs
# ---------------------------------------------------------------------------------------------


def draw_initial_pairs(fluent: PairFluent, n: int, rng: np.random.Generator) -> np.ndarray:
    """The initial pairs of a pair fluent between ``n`` entities, drawn from ``rng`` at the
    fluent's initial rate by the procedure of "Initial pairs between entities", as a Boolean
    matrix. The draws are made in a fixed order: for a functional pair fluent one draw against
    the rate per entity, in entity order, then the partner uniformly among the entities that
    keep the profile (none when no entity does); otherwise one draw per unordered pair (``i <
    j``, row by row) for a symmetric or an asymmetric pair fluent, with one more draw for the
    direction of an asymmetric pair, and one draw per ordered pair (row by row) for a general
    one. A rate of 0 draws as the others do and selects nothing."""
    matrix = np.zeros((n, n), dtype=bool)
    rate = float(fluent.initial_rate)
    if fluent.functional:
        for x in range(n):
            if rng.random() >= rate:
                continue
            candidates = _partners_keeping_profile(fluent, matrix, x)
            if not candidates:
                continue
            y = candidates[int(rng.integers(len(candidates)))]
            matrix[x, y] = True
            if fluent.symmetric:
                matrix[y, x] = True
        return matrix
    if fluent.symmetry == "general":
        for x in range(n):
            for y in range(n):
                if x != y and rng.random() < rate:
                    matrix[x, y] = True
        return matrix
    for x in range(n):
        for y in range(x + 1, n):
            if rng.random() < rate:
                if fluent.symmetric:
                    matrix[x, y] = matrix[y, x] = True
                elif rng.integers(2) == 0:
                    matrix[x, y] = True
                else:
                    matrix[y, x] = True
    return matrix


def _partners_keeping_profile(fluent: PairFluent, matrix: np.ndarray, x: int) -> list[int]:
    """The entities ``y`` such that adding ``(x, y)`` (and ``(y, x)`` for a symmetric pair
    fluent) keeps the functional profile: the row of ``x`` is empty, ``y`` is not ``x``, for an
    asymmetric pair fluent ``(y, x)`` does not hold, and for a symmetric one the row of ``y`` is
    empty too."""
    n = matrix.shape[0]
    if matrix[x].any():
        return []
    found = []
    for y in range(n):
        if y == x:
            continue
        if fluent.asymmetric and matrix[y, x]:
            continue
        if fluent.symmetric and matrix[y].any():
            continue
        found.append(y)
    return found


def generate_pair_fluents(
    entity_count: int,
    streams,
    explicit: Mapping[str, Mapping[str, Any]] | None = None,
    explicit_source: str | None = None,
    config=None,
) -> PairFluents:
    """The pair fluents of a world and their initial pairs. ``config`` is the ``pair_fluents``
    block (``PairFluentsConfig``; the code's defaults when None): ``count`` profiles and initial
    rates are drawn from ``world:pair_fluents`` in label order, three draws per pair fluent
    (symmetry, functional, initial rate), and an override of ``fluents.PAIRFL.<n>`` or an event
    file's entry (``explicit``: ``{label: {symmetry, functional, initial_rate}}``, labels among
    ``PAIRFL.1`` to ``PAIRFL.<count>``) replaces the drawn values after the draws. The initial
    pairs come from ``world:pair_initial``, fluent by fluent in label order. Without pair
    fluents nothing is drawn at all."""
    count = config.count if config is not None else 0
    explicit = dict(explicit or {})
    source = explicit_source or "<event file>"
    expected = [pair_label(i) for i in range(1, count + 1)]
    labels = sorted(explicit, key=lambda label: int(label.split(".")[1]))
    if explicit and (not count or labels[-1] not in expected):
        raise ConfigError(
            source,
            "pair_fluents",
            f"the event file defines {', '.join(labels)}, but the world configuration's "
            f"pair_fluents.count is {count}; the file's pair fluents must be among "
            f"{pair_label(1)} to {pair_label(count)}"
            if count
            else f"the event file defines {', '.join(labels)}, but the world configuration's "
            f"pair_fluents.count is 0; set it to at least {labels[-1].split('.')[1]}",
        )
    if not count:
        return EMPTY
    rng = streams.pair_fluents
    overrides = config.fluents if config is not None else {}
    fluents: list[PairFluent] = []
    for index, label in enumerate(expected, start=1):
        symmetry = str(_draw(rng, config.symmetry))
        functional = bool(rng.random() < config.functional_rate)
        rate = float(config.initial_rates[int(rng.integers(len(config.initial_rates)))])
        override = overrides.get(label)
        if override is not None:
            if override.symmetry is not None:
                symmetry = override.symmetry
            if override.functional is not None:
                functional = override.functional
            if override.initial_rate is not None:
                rate = float(override.initial_rate)
        entry = explicit.get(label)
        if entry is not None:
            symmetry = str(entry["symmetry"])
            functional = bool(entry["functional"])
            rate = float(entry["initial_rate"])
        fluents.append(PairFluent(label, index, symmetry, functional, rate))
    rng = streams.pair_initial
    initial = tuple(draw_initial_pairs(fluent, entity_count, rng) for fluent in fluents)
    for fluent, matrix in zip(fluents, initial, strict=True):
        violations = profile_violations(fluent, matrix)
        if violations:
            raise WorldError(f"the initial pairs break a profile: {violations[0]}")
    return PairFluents(tuple(fluents), initial)


# ---------------------------------------------------------------------------------------------
# initial_pairs.csv
# ---------------------------------------------------------------------------------------------


def pair_rows(
    fluents: Sequence[PairFluent], matrices: Sequence[np.ndarray], labels: Sequence[str]
) -> list[tuple[str, str, str]]:
    """Every ordered pair that holds, as ``(pair fluent, first, second)`` label triples, by pair
    fluent in label order, then by first and second in entity order; both directions of a
    symmetric pair."""
    rows: list[tuple[str, str, str]] = []
    for fluent, matrix in zip(fluents, matrices, strict=True):
        for x, y in zip(*np.nonzero(np.asarray(matrix, dtype=bool)), strict=True):
            rows.append((fluent.label, labels[int(x)], labels[int(y)]))
    return rows


def initial_pairs_frame(pair_fluents: PairFluents, labels: Sequence[str]) -> pl.DataFrame:
    """``initial_pairs.csv``: one row per ordered pair that holds at ``TIME.1``."""
    rows = pair_rows(pair_fluents.fluents, pair_fluents.initial_pairs, labels)
    return pl.DataFrame(
        {
            "pair_fluent": [r[0] for r in rows],
            "first": [r[1] for r in rows],
            "second": [r[2] for r in rows],
        },
        schema={name: pl.Utf8 for name in INITIAL_PAIRS_COLUMNS},
    )


def matrices_from_rows(
    rows: Sequence[Mapping[str, Any]] | Sequence[Sequence[str]],
    fluents: Sequence[PairFluent],
    entity_index: Mapping[str, int],
    what: str = "the initial pairs",
) -> tuple[np.ndarray, ...]:
    """The pair matrices from rows of ``(pair fluent, first, second)``, given as mappings with
    those keys (the rows of ``initial_pairs.csv``) or as triples (a history's or a fixture's
    ``initial_pairs``). An unknown pair fluent or entity, or a pair of an entity with itself,
    is a :class:`WorldError`."""
    n = len(entity_index)
    index = {f.label: i for i, f in enumerate(fluents)}
    matrices = [np.zeros((n, n), dtype=bool) for _ in fluents]
    for row in rows:
        if isinstance(row, Mapping):
            fluent, first, second = str(row["pair_fluent"]), str(row["first"]), str(row["second"])
        else:
            fluent, first, second = (str(v) for v in row)
        if fluent not in index:
            raise WorldError(f"{what} name {fluent}, which is not a pair fluent of the definition")
        for label in (first, second):
            if label not in entity_index:
                raise WorldError(f"{what} name {label}, which is not an entity of the definition")
        if first == second:
            raise WorldError(f"{what} hold {fluent} between {first} and itself")
        matrices[index[fluent]][entity_index[first], entity_index[second]] = True
    return tuple(matrices)


__all__ = [
    "EMPTY",
    "INITIAL_PAIRS_COLUMNS",
    "INITIAL_PAIRS_FILE",
    "PAIR_FLUENT_KIND",
    "PAIR_PREFIX",
    "SYMMETRIES",
    "PairFluent",
    "PairFluents",
    "draw_initial_pairs",
    "empty_pair_fluents",
    "generate_pair_fluents",
    "initial_pairs_frame",
    "is_pair_label",
    "matrices_from_rows",
    "pair_label",
    "pair_rows",
    "profile_violations",
]
