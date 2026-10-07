"""Boolean functions for the taxonomy generator and the world package: truth tables, the
arity-2 enumeration, the Shepard, Hovland, and Jenkins (SHJ) types, and minimal disjunctive
normal form.

A rule is stored as its input list and a truth table. The truth table lists the output for
input settings ``00..0``, ``00..1``, ..., ``11..1``, with the first input as the most
significant bit. Evaluation is a table lookup, vectorized over all objects with NumPy.

Expressions (the printed and parsed form of rules) and read-once formulas live in
``semantic_world.taxonomy.expressions``, which builds on this module.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

OPERATORS = ("AND", "OR", "XOR")
SHJ_TYPES = ("I", "II", "III", "IV", "V", "VI")

Implicant = tuple[int | None, ...]
"""One product term: for each input, 1 (the input), 0 (its negation), or None (absent)."""


# ---------------------------------------------------------------------------------------------
# Truth tables
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthTable:
    """A Boolean function of ``arity`` inputs as a table of ``2 ** arity`` outputs.

    ``bits[i]`` is the output for the input setting whose binary representation is ``i``, with
    the first input as the most significant bit.
    """

    arity: int
    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.arity < 0:
            raise ValueError("the arity must be non-negative")
        if len(self.bits) != 2**self.arity:
            raise ValueError(
                f"a truth table of arity {self.arity} needs {2**self.arity} outputs, "
                f"got {len(self.bits)}"
            )
        if any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("truth table outputs must be 0 or 1")

    # Construction -----------------------------------------------------------------------------

    @classmethod
    def from_bit_string(cls, text: str) -> TruthTable:
        """The inverse of :meth:`bit_string`."""
        length = len(text)
        if length == 0 or length & (length - 1):
            raise ValueError(
                f"a truth table bit string must have a power-of-two length, got {length}"
            )
        if set(text) - {"0", "1"}:
            raise ValueError("a truth table bit string may only contain 0 and 1")
        return cls(length.bit_length() - 1, tuple(int(c) for c in text))

    @classmethod
    def from_function(cls, arity: int, function: Callable[[tuple[int, ...]], object]) -> TruthTable:
        """Tabulate ``function`` over every input setting, in table order."""
        return cls(arity, tuple(1 if function(setting) else 0 for setting in settings(arity)))

    @classmethod
    def from_minterms(cls, arity: int, minterms: Iterable[int]) -> TruthTable:
        bits = [0] * (2**arity)
        for m in minterms:
            bits[m] = 1
        return cls(arity, tuple(bits))

    @classmethod
    def constant(cls, arity: int, value: bool) -> TruthTable:
        return cls(arity, (1 if value else 0,) * (2**arity))

    # Views ----------------------------------------------------------------------------------------

    def bit_string(self) -> str:
        return "".join(str(bit) for bit in self.bits)

    def minterms(self) -> tuple[int, ...]:
        """The indices of the input settings with output 1."""
        return tuple(i for i, bit in enumerate(self.bits) if bit)

    @property
    def is_constant(self) -> bool:
        return len(set(self.bits)) == 1

    @property
    def is_balanced(self) -> bool:
        """True when exactly half of the input settings give 1."""
        return sum(self.bits) * 2 == len(self.bits)

    def depends_on(self, index: int) -> bool:
        """Whether flipping input ``index`` ever changes the output."""
        stride = 1 << (self.arity - 1 - index)
        return any(
            self.bits[i] != self.bits[i | stride] for i in range(len(self.bits)) if not i & stride
        )

    def relevant_inputs(self) -> tuple[int, ...]:
        """The indices of the inputs the function depends on."""
        return tuple(i for i in range(self.arity) if self.depends_on(i))

    # Evaluation -----------------------------------------------------------------------------------

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Look up the output for every row of ``inputs``, an array of shape ``(n, arity)`` of
        0 and 1 values. Returns an array of shape ``(n,)`` with dtype ``uint8``."""
        values = np.asarray(inputs)
        if values.ndim != 2 or values.shape[1] != self.arity:
            raise ValueError(f"expected an array of shape (n, {self.arity}), got {values.shape}")
        weights = 1 << np.arange(self.arity - 1, -1, -1, dtype=np.int64)
        index = values.astype(np.int64) @ weights
        return np.asarray(self.bits, dtype=np.uint8)[index]

    def __call__(self, *inputs: int) -> int:
        """The output for one input setting, given as separate 0/1 arguments."""
        if len(inputs) != self.arity:
            raise ValueError(f"expected {self.arity} inputs, got {len(inputs)}")
        index = 0
        for value in inputs:
            index = (index << 1) | (1 if value else 0)
        return self.bits[index]

    # Transformations ------------------------------------------------------------------------------

    def permute_inputs(self, permutation: Sequence[int]) -> TruthTable:
        """The function ``g(x) = f(x[p[0]], x[p[1]], ...)``: new input ``p[j]`` feeds old
        position ``j``."""
        if sorted(permutation) != list(range(self.arity)):
            raise ValueError(f"{list(permutation)} is not a permutation of {self.arity} inputs")
        return TruthTable.from_function(self.arity, lambda x: self(*(x[p] for p in permutation)))

    def negate_inputs(self, mask: Sequence[bool]) -> TruthTable:
        """The function ``g(x) = f(x XOR mask)``."""
        if len(mask) != self.arity:
            raise ValueError(f"expected {self.arity} flags, got {len(mask)}")
        return TruthTable.from_function(
            self.arity, lambda x: self(*(v ^ (1 if m else 0) for v, m in zip(x, mask, strict=True)))
        )

    def negate_output(self) -> TruthTable:
        return TruthTable(self.arity, tuple(1 - bit for bit in self.bits))


def settings(arity: int) -> Iterable[tuple[int, ...]]:
    """Every input setting of ``arity`` inputs, in table order (first input most significant)."""
    return itertools.product((0, 1), repeat=arity)


def settings_array(arity: int) -> np.ndarray:
    """The settings as an array of shape ``(2 ** arity, arity)``."""
    return np.array(list(settings(arity)), dtype=np.uint8).reshape(2**arity, arity)


def all_functions(arity: int) -> Iterable[TruthTable]:
    """Every Boolean function of ``arity`` inputs, in bit-string order."""
    for bits in itertools.product((0, 1), repeat=2**arity):
        yield TruthTable(arity, bits)


# ---------------------------------------------------------------------------------------------
# Operators and the arity-2 enumeration
# ---------------------------------------------------------------------------------------------


def apply_operator(operator: str, values: Sequence[int]) -> int:
    """Apply an n-ary operator to 0/1 values. XOR of many inputs is their parity."""
    if operator == "AND":
        return 1 if all(values) else 0
    if operator == "OR":
        return 1 if any(values) else 0
    if operator == "XOR":
        return sum(1 for v in values if v) & 1
    raise ValueError(f"unknown operator {operator!r}")


@dataclass(frozen=True)
class Arity2Function:
    """One of the 10 two-input functions that depend on both inputs: an operator applied to
    each input, possibly negated."""

    operator: str
    negate_first: bool
    negate_second: bool
    table: TruthTable


def _arity_2_functions() -> tuple[Arity2Function, ...]:
    result: list[Arity2Function] = []
    seen: set[TruthTable] = set()
    for operator in OPERATORS:
        for negate_first, negate_second in itertools.product((False, True), repeat=2):
            table = TruthTable.from_function(
                2,
                lambda x, op=operator, na=negate_first, nb=negate_second: apply_operator(
                    op, (x[0] ^ int(na), x[1] ^ int(nb))
                ),
            )
            if table in seen:
                continue  # XOR with one negation equals XNOR, and with two equals XOR
            seen.add(table)
            result.append(Arity2Function(operator, negate_first, negate_second, table))
    return tuple(result)


ARITY_2_FUNCTIONS: tuple[Arity2Function, ...] = _arity_2_functions()
"""The 10 two-input functions that depend on both inputs, in operator order: AND, A AND NOT B,
NOT A AND B, NOR, OR, A OR NOT B, NOT A OR B, NAND, XOR, XNOR."""


# ---------------------------------------------------------------------------------------------
# The Shepard, Hovland, and Jenkins types
# ---------------------------------------------------------------------------------------------

SHJ_TRUE_SETTINGS: dict[str, tuple[str, ...]] = {
    "I": ("000", "001", "010", "011"),
    "II": ("000", "001", "110", "111"),
    "III": ("000", "001", "010", "101"),
    "IV": ("000", "001", "010", "100"),
    "V": ("000", "001", "010", "111"),
    "VI": ("000", "011", "101", "110"),
}
"""The input combinations that make the canonical function of each type true. The numbering
follows Nosofsky, Gluck, Palmeri, McKinley, and Glauthier (1994)."""

SHJ_CANONICAL: dict[str, TruthTable] = {
    shj_type: TruthTable.from_minterms(3, (int(s, 2) for s in true_settings))
    for shj_type, true_settings in SHJ_TRUE_SETTINGS.items()
}

SHJ_CLASS_SIZES: dict[str, int] = {"I": 6, "II": 6, "III": 24, "IV": 8, "V": 24, "VI": 2}


def shj_function(
    shj_type: str, permutation: Sequence[int] = (0, 1, 2), negations: Sequence[bool] = (False,) * 3
) -> TruthTable:
    """The canonical function of a type with its inputs permuted, then each negated as asked."""
    if shj_type not in SHJ_CANONICAL:
        raise ValueError(f"unknown SHJ type {shj_type!r}; the types are {', '.join(SHJ_TYPES)}")
    return SHJ_CANONICAL[shj_type].permute_inputs(permutation).negate_inputs(negations)


def _shj_classes() -> dict[str, frozenset[TruthTable]]:
    classes: dict[str, frozenset[TruthTable]] = {}
    for shj_type in SHJ_TYPES:
        members = {
            shj_function(shj_type, permutation, negations)
            for permutation in itertools.permutations(range(3))
            for negations in itertools.product((False, True), repeat=3)
        }
        classes[shj_type] = frozenset(members)
    return classes


SHJ_CLASSES: dict[str, frozenset[TruthTable]] = _shj_classes()
"""Every three-input function of each type, under permutation and negation of the inputs."""

_SHJ_TYPE_OF: dict[TruthTable, str] = {
    table: shj_type for shj_type, members in SHJ_CLASSES.items() for table in members
}


def shj_type_of(table: TruthTable) -> str | None:
    """The SHJ type of a three-input function, or None when it is not balanced."""
    return _SHJ_TYPE_OF.get(table)


# ---------------------------------------------------------------------------------------------
# Minimal disjunctive normal form (Quine–McCluskey with an exact minimum cover)
# ---------------------------------------------------------------------------------------------


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _to_implicant(arity: int, value: int, mask: int) -> Implicant:
    return tuple(
        (value >> (arity - 1 - k)) & 1 if (mask >> (arity - 1 - k)) & 1 else None
        for k in range(arity)
    )


def prime_implicants(table: TruthTable) -> tuple[Implicant, ...]:
    """Every prime implicant of the function, sorted."""
    arity = table.arity
    minterms = table.minterms()
    if not minterms:
        return ()
    full = (1 << arity) - 1
    current: set[tuple[int, int]] = {(m, full) for m in minterms}
    primes: set[tuple[int, int]] = set()
    while current:
        next_level: set[tuple[int, int]] = set()
        combined: set[tuple[int, int]] = set()
        by_mask: dict[int, list[int]] = defaultdict(list)
        for value, mask in sorted(current):
            by_mask[mask].append(value)
        for mask, values in by_mask.items():
            present = set(values)
            for value in values:
                for k in range(arity):
                    bit = 1 << k
                    if mask & bit and not value & bit and (value | bit) in present:
                        next_level.add((value, mask & ~bit))
                        combined.add((value, mask))
                        combined.add((value | bit, mask))
        primes |= current - combined
        current = next_level
    return tuple(sorted((_to_implicant(arity, v, m) for v, m in primes), key=implicant_key))


def implicant_covers(implicant: Implicant, minterm: int) -> bool:
    arity = len(implicant)
    return all(
        literal is None or ((minterm >> (arity - 1 - k)) & 1) == literal
        for k, literal in enumerate(implicant)
    )


def implicant_literals(implicant: Implicant) -> int:
    return sum(1 for literal in implicant if literal is not None)


def implicant_key(implicant: Implicant) -> tuple[int, ...]:
    """The sort key that orders implicants: 0, then 1, then absent, input by input."""
    return tuple(2 if literal is None else literal for literal in implicant)


def minimal_dnf(table: TruthTable) -> tuple[Implicant, ...]:
    """A disjunctive normal form with the fewest literals, as sorted prime implicants.

    A minimal DNF can always be built from prime implicants, because replacing a term by a
    prime implicant that covers it never adds a literal. The search takes the essential prime
    implicants first, then finds a minimum-cost cover of the rest by branch and bound. Ties are
    broken by the sorted order of the implicants, so the result is deterministic. A constant
    true function gives the single empty term; a constant false function gives no terms.
    """
    primes = prime_implicants(table)
    if not primes:
        return ()
    minterms = table.minterms()
    covers: dict[Implicant, frozenset[int]] = {
        p: frozenset(m for m in minterms if implicant_covers(p, m)) for p in primes
    }
    covering: dict[int, tuple[Implicant, ...]] = {
        m: tuple(p for p in primes if m in covers[p]) for m in minterms
    }
    cost = {p: implicant_literals(p) for p in primes}

    chosen: list[Implicant] = []
    uncovered = set(minterms)
    changed = True
    while changed:
        changed = False
        for m in sorted(uncovered):
            candidates = [p for p in covering[m] if p not in chosen]
            if len(candidates) == 1:
                chosen.append(candidates[0])
                uncovered -= covers[candidates[0]]
                changed = True
                break

    best: tuple[int, list[Implicant]] | None = None

    def search(remaining: set[int], so_far: list[Implicant], so_far_cost: int) -> None:
        nonlocal best
        if best is not None and so_far_cost >= best[0]:
            return
        if not remaining:
            best = (so_far_cost, list(so_far))
            return
        target = min(remaining, key=lambda m: (len(covering[m]), m))
        for p in covering[target]:
            search(remaining - covers[p], so_far + [p], so_far_cost + cost[p])

    search(uncovered, [], 0)
    assert best is not None
    return tuple(sorted(chosen + best[1], key=implicant_key))


def dnf_literal_count(table: TruthTable) -> int:
    """The literal count of a minimal DNF: the complexity score of a rule."""
    return sum(implicant_literals(term) for term in minimal_dnf(table))


def dnf_truth_table(arity: int, terms: Iterable[Implicant]) -> TruthTable:
    """The function that a list of product terms describes."""
    terms = tuple(terms)
    return TruthTable.from_minterms(
        arity, (m for m in range(2**arity) if any(implicant_covers(t, m) for t in terms))
    )
