"""Event types: the one-place and two-place event types of the static side, and the
preconditions and effects that make them dynamic.

The static side comes from ``semantic_world.world.statics`` (``docs/specs/WORLD_AND_LANGUAGE.md``,
"Event types"): the requirement of ``EVENTTYPE1.k`` is its sampled rule over the agent's
features, and an event type's constraints are those of its true event-type features and its own;
an event-type category's requirement is its base relation. Every requirement is a conjunction of
constraints, each a rule over binding literals (``agent.PROPERTY.3``, ``patient.SCALARDIM.1 >
0.4127``, comparisons).

The dynamic side ("Preconditions", "Effects") is drawn here from ``world:effects`` and
``world:preconditions``: effects first, so that enabling literals can be drawn from the values
that effects produce; then preconditions; then the fix-ups (no contradictions, no empty effects,
every literal achievable); then, from ``world:conditions``, the conditions of a share of the
effects ("Conditional effects", stage b1), each satisfiable with its event type's requirement and
precondition. An event file replaces the sampled dynamics of the event types it names.

Since stage b2 ("Numeric fluents") a share of the effects is numeric and a share of the
precondition literals is a numeric threshold literal, when the world has numeric fluents: each
Boolean effect or literal is drawn as before, and then, from ``world:numeric_effects`` or
``world:numeric_preconditions``, replaced by a numeric one at the share, so that with the share
at 0 the Boolean side is drawn exactly as before the stage. A condition literal may be a numeric
threshold too (the kind ``numeric`` of ``condition_kinds``, its symbol and threshold from
``world:numeric_conditions``). A numeric literal is achievable when its fluent's initial
distribution, drift, or some effect can bring a value to its side of the threshold.

Since stage c2b (``docs/specs/WORLD_PHASE_C.md``, "Preconditions", "Effects", "Conditions
(F5)") the same pattern draws pair literals, pair effects, and pair conditions when the world has
pair fluents: a precondition literal is replaced by a pair literal at
``event_types.preconditions.pair_share`` from ``world:pair_preconditions`` (its kind from
``pair_fluents.literal_kinds``, the pair fluent uniformly among those the kind allows, the
direction or role and the polarity uniformly, ``SAME`` only on a functional pair fluent that is
not symmetric, a one-place event type drawing ``SOME`` and ``SOMEOF`` of its agent alone), an
effect of a two-place event type by a pair effect at ``event_types.effects.pair_share`` from
``world:pair_effects`` (the pair fluent, the direction, and the value uniformly), and a
condition literal of the kind ``pair`` is drawn from ``world:pair_conditions``. Every rule that
stage c2a enforces on an explicit entry with an error is a redraw here: an unachievable literal,
a contradiction, an empty effect, an add whose compiled literals contradict the precondition, a
condition that names a pair or row the precondition names (or, under the implications of
``fixed_pair_literals``, whose value the precondition fixes), and a conditional add on an
asymmetric or functional pair fluent, which keeps no condition. The compiled literals of F4 are
appended to every draft after the fix-ups. With the shares at 0 nothing is drawn from the pair
streams, and the Boolean draws are those of the world without pair fluents.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.expressions import Cmp, Const, Expr, Gt, Not, Op, Var
from semantic_world.taxonomy.rules import MAX_TRIES, Rule, _draw, input_key
from semantic_world.world.config import (
    Config,
    EffectsConfig,
    PairFluentsConfig,
    PreconditionsConfig,
)
from semantic_world.world.constraints import Constraint
from semantic_world.world.dynamics import (
    MOVE_ORIGIN,
    PAIR_KIND,
    PAIR_SAME,
    PAIR_SOME,
    PAIR_SOMEOF,
    PROFILE_ORIGIN,
    ROLES,
    AnyLiteral,
    ConditionLiteral,
    Effect,
    Literal,
    PairLiteral,
    precondition_text,
)
from semantic_world.world.errors import WorldError
from semantic_world.world.event_file import EventFile, ExplicitEventType
from semantic_world.world.fluents import Fluents
from semantic_world.world.numeric import NumericFluents, drawn, empty_numeric
from semantic_world.world.pair_fluents import PairFluent, PairFluents, empty_pair_fluents
from semantic_world.world.statics import StaticWorld
from semantic_world.world.streams import WorldStreams

ONE_PLACE = "EVENTTYPE1"
TWO_PLACE = "EVENTTYPE2"
FEATURE_PREFIX = "EVENTFEAT"
CONSTRAINT_PREFIX = "CONSTRAINT"
REQUIREMENT_PREFIX = "REQUIREMENT"
MAX_FIX_ROUNDS = 20


# ---------------------------------------------------------------------------------------------
# Constraints and event types
# ---------------------------------------------------------------------------------------------


def prefixed(expr: Expr, role: str) -> Expr:
    """An entity-level expression as a binding expression of one role: ``PART.4 AND NOT
    PROPERTY.7`` becomes ``agent.PART.4 AND NOT agent.PROPERTY.7``."""
    if isinstance(expr, Var):
        return Var(f"{role}.{expr.name}")
    if isinstance(expr, Gt):
        return Gt(f"{role}.{expr.scalar}", expr.threshold)
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Not):
        return Not(prefixed(expr.operand, role))
    if isinstance(expr, Op):
        return Op(expr.operator, tuple(prefixed(o, role) for o in expr.operands))
    if isinstance(expr, Cmp):
        raise ValueError("a one-place rule holds no comparison")
    raise TypeError(f"unknown expression {expr!r}")


@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint: a rule over binding literals, in the world's vocabulary."""

    label: str
    family: str
    inputs: tuple[str, ...]
    """Binding literal keys, in truth-table order."""
    table: TruthTable
    expression: str
    source: Rule | Constraint
    """The object the constraint came from: a one-place requirement rule, or a constraint."""

    @property
    def arity(self) -> int:
        return len(self.inputs)

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "inputs": list(self.inputs),
            "expression": self.expression,
            "truth_table": self.table.bit_string(),
            "min_dnf_literals": dnf_literal_count(self.table),
        }


def constraint_from_rule(rule: Rule) -> ConstraintSpec:
    """The requirement of a one-place event type: its rule over the agent's features."""
    label = f"{CONSTRAINT_PREFIX}.{rule.output.label}"
    inputs = tuple(f"agent.{input_key(item)}" for item in rule.inputs)
    return ConstraintSpec(
        label, rule.family, inputs, rule.table, str(prefixed(rule.expression, "agent")), rule
    )


def constraint_from_relation(constraint: Constraint) -> ConstraintSpec:
    inputs = tuple(item.key for item in constraint.literals)
    return ConstraintSpec(
        constraint.label,
        constraint.family,
        inputs,
        constraint.table,
        str(constraint.expression),
        constraint,
    )


@dataclass(frozen=True)
class EventType:
    label: str
    arity: int
    category: bool
    """True for an internal node of the event-type tree (a category of two-place event types)."""
    parent: str | None
    level: int
    features: tuple[str, ...]
    """The event-type features true for the event type (defining with value 1, for a
    category), in feature order."""
    constraints: tuple[str, ...]
    """The labels of the constraints whose conjunction is the requirement."""
    precondition: tuple[AnyLiteral, ...] = ()
    """The precondition literals, the compiled literals of stage c2a (origin ``profile``) last."""
    effects: tuple[Effect, ...] = ()
    explicit: bool = False
    explicit_effects: bool = False
    """Whether an event file gave the effects (stage c4a): the Boolean rule of WM.70 is not
    checked on them, as the dead-delete rule of Jon's ruling 1 on stage c2b is not. Not part of
    the record."""

    @property
    def roles(self) -> tuple[str, ...]:
        return ROLES[: self.arity]

    @property
    def kind(self) -> str:
        return "event_type_category" if self.category else "event_type"

    @property
    def requirement_label(self) -> str:
        return f"{REQUIREMENT_PREFIX}.{self.label}"

    @property
    def requirement_expression(self) -> str:
        return " AND ".join(self.constraints) if self.constraints else "TRUE"

    def record(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "arity": self.arity,
            "roles": list(self.roles),
            "parent": self.parent,
            "level": self.level,
            "features": list(self.features),
            "explicit": self.explicit,
            "requirement": {
                "constraints": list(self.constraints),
                "output": self.requirement_label,
                "expression": self.requirement_expression,
            },
            "precondition": {
                "literals": [lit.record() for lit in self.precondition],
                "expression": precondition_text(self.precondition),
            },
            "effects": [e.record() for e in self.effects],
        }


@dataclass(frozen=True)
class EventTypes:
    constraints: tuple[ConstraintSpec, ...]
    event_types: tuple[EventType, ...]
    """One-place event types, then the two-place event-type tree in category order."""
    features: tuple[str, ...]
    """The event-type features, ``EVENTFEAT.1`` and so on."""
    feature_constraints: dict[str, str]
    feature_preconditions: dict[str, Literal | None] = field(default_factory=dict)
    feature_effects: dict[str, Effect | None] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    """Counts of the fix-ups: dropped and redrawn effects and literals."""
    condition_draws: dict[str, int] = field(default_factory=dict)
    """The effects a condition was drawn for (``drawn``) and how many got one
    (``conditional``)."""
    pair_stats: dict[str, int] = field(default_factory=dict)
    """Counts of the pair draws and redraws of stage c2b (``pair_fluents.fix_ups`` of
    ``world_stats.yaml``); empty in a world without pair fluents."""
    same_quota: dict[str, PairLiteral] = field(default_factory=dict)
    """The ``SAME`` literals on a location fluent the quota ``pair_fluents.same_rules`` gave,
    by event type (stage c3b, WM.52): kept through the redraws of never-legal event types, as
    the compiled literals are."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_constraints", {c.label: c for c in self.constraints})
        object.__setattr__(self, "_event_types", {e.label: e for e in self.event_types})

    def constraint(self, label: str) -> ConstraintSpec:
        try:
            return self._constraints[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown constraint {label!r}") from None

    def event_type(self, label: str) -> EventType:
        try:
            return self._event_types[label]  # type: ignore[attr-defined]
        except KeyError:
            raise KeyError(f"unknown event type {label!r}") from None

    def __contains__(self, label: object) -> bool:
        return label in self._event_types  # type: ignore[attr-defined]

    @property
    def unary(self) -> tuple[EventType, ...]:
        return tuple(e for e in self.event_types if e.arity == 1)

    @property
    def binary(self) -> tuple[EventType, ...]:
        """The two-place event types proper (the leaves of the tree)."""
        return tuple(e for e in self.event_types if e.arity == 2 and not e.category)

    @property
    def categories(self) -> tuple[EventType, ...]:
        return tuple(e for e in self.event_types if e.category)

    @property
    def leaves(self) -> tuple[EventType, ...]:
        """Every event type that can occur: one-place, and two-place leaves."""
        return tuple(e for e in self.event_types if not e.category)


def build_event_types(statics: StaticWorld) -> EventTypes:
    """The static side: constraints and event types, without preconditions or effects."""
    constraints: list[ConstraintSpec] = []
    event_types: list[EventType] = []
    for rule in statics.unary.requirement_rules:
        spec = constraint_from_rule(rule)
        constraints.append(spec)
        event_types.append(
            EventType(
                label=rule.output.label,
                arity=1,
                category=False,
                parent=None,
                level=1,
                features=(),
                constraints=(spec.label,),
            )
        )
    features: tuple[str, ...] = ()
    feature_constraints: dict[str, str] = {}
    relations = statics.relations
    if relations is not None:
        tree = relations.event_tree
        features = tuple(tree.features.labels)
        for constraint in relations.constraints:
            constraints.append(constraint_from_relation(constraint))
        for feature, constraint in zip(
            tree.features.labels, relations.feature_constraints, strict=True
        ):
            feature_constraints[feature] = constraint.label
        for category in tree.categories:
            if category.is_leaf:
                mask = category.values == 1
            else:
                mask = category.defining_mask() & (category.free_values == 1)
            true_features = tuple(features[i] for i in np.flatnonzero(mask))
            relation = relations.relation(category)
            event_types.append(
                EventType(
                    label=category.label,
                    arity=2,
                    category=not category.is_leaf,
                    parent=None if category.parent is None else category.parent.label,
                    level=category.level,
                    features=true_features,
                    constraints=tuple(c.label for c in relation.constraints),
                )
            )
    return EventTypes(tuple(constraints), tuple(event_types), features, feature_constraints)


# ---------------------------------------------------------------------------------------------
# Pair literals and effects: what they name (stage c2a)
# ---------------------------------------------------------------------------------------------


def _ordered(fluent: PairFluent, first: str, second: str) -> tuple[str, ...]:
    """The two roles of a pair, in the fluent's canonical order: as given, or sorted for a
    symmetric pair fluent, whose two directions are one pair."""
    return tuple(sorted((first, second))) if fluent.symmetric else (first, second)


def pair_literal_key(literal: PairLiteral, fluent: PairFluent) -> tuple:
    """What a pair literal is about, so that two literals about the same thing compare equal:
    a pair (both directions of a symmetric one), the row of a role (``SOME``; ``SOMEOF`` of a
    symmetric pair fluent names the same set), the column of a role (``SOMEOF``), or the rows of
    two roles (``SAME``, in either order)."""
    if literal.kind == PAIR_KIND:
        assert literal.second is not None
        return ("pair", fluent.label, _ordered(fluent, literal.first, literal.second))
    if literal.kind == PAIR_SOME:
        return ("row", fluent.label, literal.first)
    if literal.kind == PAIR_SOMEOF:
        return ("row" if fluent.symmetric else "column", fluent.label, literal.first)
    assert literal.second is not None
    return ("same", fluent.label, frozenset((literal.first, literal.second)))


def pair_literal_names(literal: PairLiteral, fluent: PairFluent) -> set[tuple]:
    """The pairs, rows, and columns a pair literal names (the rule of F5 that a condition names
    no pair or row the precondition names): ``SAME`` names the rows of both roles."""
    if literal.kind == PAIR_SAME:
        return {("row", fluent.label, literal.first), ("row", fluent.label, literal.second)}
    return {pair_literal_key(literal, fluent)}


def pair_effect_key(effect: Effect, fluent: PairFluent) -> tuple:
    """The pair a pair effect writes, as ``pair_literal_key`` names it."""
    assert effect.second is not None
    return ("pair", fluent.label, _ordered(fluent, effect.role, effect.second))


def compiled_literals(effect: Effect, fluent: PairFluent) -> list[PairLiteral]:
    """The compiled precondition literals (F4) of an add effect: ``NOT PAIRFL.k(second,
    first)`` for an asymmetric pair fluent, ``NOT SOME.PAIRFL.k(first)`` for a functional one,
    and ``NOT SOME.PAIRFL.k(second)`` too for a symmetric functional one; none for a delete or
    for a general pair fluent that is not functional. Each carries the origin ``profile``. An
    add on a location fluent (a move, stage c3a) gets ``NOT PAIRFL.k(first, second)`` with the
    origin ``move`` instead (WM.53): an entity never moves to where it is, and the total
    functional replacement keeps the profile without a ``SOME`` literal."""
    if not effect.pair or not effect.value:
        return []
    assert effect.second is not None
    found: list[PairLiteral] = []
    if fluent.total:
        return [
            PairLiteral(PAIR_KIND, fluent.label, effect.role, effect.second, False, MOVE_ORIGIN)
        ]
    if fluent.asymmetric:
        found.append(
            PairLiteral(PAIR_KIND, fluent.label, effect.second, effect.role, False, PROFILE_ORIGIN)
        )
    if fluent.functional:
        found.append(PairLiteral(PAIR_SOME, fluent.label, effect.role, None, False, PROFILE_ORIGIN))
        if fluent.symmetric:
            found.append(
                PairLiteral(PAIR_SOME, fluent.label, effect.second, None, False, PROFILE_ORIGIN)
            )
    return found


def pair_literal_achievable(
    literal: PairLiteral, fluent: PairFluent, adds: set[str], deletes: set[str]
) -> bool:
    """Whether a pair literal can hold in some state ("Achievable" of the specification): a
    positive literal when its pair fluent has a positive initial rate or some effect adds a
    pair of it; a negative one when some initial state lacks the pair (an initial rate below 1,
    or a functional pair fluent, in which every other pair of a row is absent) or some effect
    deletes a pair of it. ``SAME`` on a symmetric functional pair fluent is never achievable:
    two entities with the same partner would give that partner two partners. On a location
    fluent (stage c3a) every literal but ``SOME`` of an entity is achievable: every entity is
    somewhere and can be at any place (the habitat weights are positive everywhere), so two
    entities can share a place or not, a place can hold someone or no one, and the agent of a
    move or an act can be at the bound place or elsewhere; ``SOME`` of an entity is always
    true, so its negation is never achievable (and the positive literal is never allowed)."""
    if fluent.total:
        return literal.kind != PAIR_SOME or literal.value
    if literal.kind == PAIR_SAME and fluent.symmetric and fluent.functional:
        return not literal.value
    if literal.value:
        return fluent.rate > 0.0 or fluent.label in adds
    return fluent.rate < 1.0 or fluent.functional or fluent.label in deletes


def dead_boolean_values(fluents: Fluents, effects: Iterable[Effect]) -> set[tuple[str, bool]]:
    """The Boolean values ``(fluent, v)`` that an effect ``:= v`` can never change (Jon's
    ruling WM.70, stage c4a): the fluent's initial rate makes every entity ``v`` (a rate of 1
    for true, of 0 for false, in every episode, redrawn or not), and no effect among ``effects``
    (every effect of the world, the place event types' included, WM.66) sets it to ``not v``.
    The Boolean counterpart of ``dead_pair_fluents``: a delete on a pair fluent that can never
    hold."""
    produced = {(e.fluent, bool(e.value)) for e in effects if not e.numeric and not e.pair}
    dead: set[tuple[str, bool]] = set()
    for fluent in fluents.base:
        rate = fluent.initial_rate
        if rate is None:
            continue
        if rate >= 1.0 and (fluent.label, False) not in produced:
            dead.add((fluent.label, True))
        if rate <= 0.0 and (fluent.label, True) not in produced:
            dead.add((fluent.label, False))
    return dead


def add_compiled_literals(
    precondition: Sequence[AnyLiteral], effects: Sequence[Effect], pair_fluents: PairFluents
) -> tuple[AnyLiteral, ...]:
    """The precondition with the compiled literals of every add effect appended, in effect
    order, a literal already present (by what it names, whatever its origin) not added twice."""
    present = {
        pair_literal_key(lit, pair_fluents[lit.pair_fluent]): lit.value
        for lit in precondition
        if isinstance(lit, PairLiteral)
    }
    found = list(precondition)
    for effect in effects:
        if not effect.pair or effect.condition:
            continue
        fluent = pair_fluents[effect.fluent]
        for literal in compiled_literals(effect, fluent):
            key = pair_literal_key(literal, fluent)
            if key in present:
                continue
            present[key] = literal.value
            found.append(literal)
    return tuple(found)


def _other_role(roles: Sequence[str], role: str) -> str | None:
    others = [r for r in roles if r != role]
    return others[0] if others else None


def fixed_pair_literals(
    precondition: Sequence[AnyLiteral], pair_fluents: PairFluents, roles: Sequence[str] = ROLES
) -> dict[tuple, bool]:
    """What the pair literals of a precondition fix, by ``pair_literal_key``, beyond what they
    name (stage c2b, item 6 of the stage prompt): a condition literal with one of these keys
    would be redundant with its value, or never fire with the other, so its effect would be
    unconditional either way. The implications, over the binding's two roles ``x`` and ``y``:

    - ``PAIRFL.k(x, y)`` fixes ``SOME.PAIRFL.k(x)`` and ``SOMEOF.PAIRFL.k(y)`` true; for a
      symmetric pair fluent also ``PAIRFL.k(y, x)``, ``SOME.PAIRFL.k(y)``, and
      ``SOMEOF.PAIRFL.k(x)`` true; for an asymmetric one ``PAIRFL.k(y, x)`` false; and for a
      functional one ``SAME.PAIRFL.k(x, y)`` false (``x``'s one partner is ``y``, which is not
      a partner of itself);
    - ``NOT SOME.PAIRFL.k(x)`` fixes ``PAIRFL.k(x, y)`` and ``SAME.PAIRFL.k(x, y)`` false (and
      for a symmetric pair fluent ``PAIRFL.k(y, x)``, the same pair);
    - ``NOT SOMEOF.PAIRFL.k(y)`` fixes ``PAIRFL.k(x, y)`` false (and for a symmetric pair
      fluent, where the column is the row, ``SAME.PAIRFL.k(x, y)`` false too);
    - ``SAME.PAIRFL.k(x, y)`` fixes ``SOME.PAIRFL.k(x)`` and ``SOME.PAIRFL.k(y)`` true and
      ``PAIRFL.k(x, y)`` and ``PAIRFL.k(y, x)`` false (the shared partner is a third entity,
      and ``k`` is functional).

    A negative pair literal, a positive ``SOME`` or ``SOMEOF``, and a negative ``SAME`` fix
    nothing beyond themselves. The literal's own key is included with its value."""
    fixed: dict[tuple, bool] = {}
    for lit in precondition:
        if not isinstance(lit, PairLiteral) or lit.pair_fluent not in pair_fluents:
            continue
        fluent = pair_fluents[lit.pair_fluent]
        implied: list[tuple[str, str, str | None, bool]] = []
        x, y = lit.first, lit.second
        if lit.kind == PAIR_KIND and lit.value:
            assert y is not None
            implied += [(PAIR_SOME, x, None, True), (PAIR_SOMEOF, y, None, True)]
            if fluent.symmetric:
                implied += [(PAIR_SOME, y, None, True), (PAIR_SOMEOF, x, None, True)]
            if fluent.asymmetric:
                implied.append((PAIR_KIND, y, x, False))
            if fluent.functional:
                implied.append((PAIR_SAME, x, y, False))
        elif lit.kind in (PAIR_SOME, PAIR_SOMEOF) and not lit.value:
            other = _other_role(roles, x)
            if other is not None:
                if lit.kind == PAIR_SOME or fluent.symmetric:
                    implied.append((PAIR_KIND, x, other, False))
                    if fluent.functional:
                        implied.append((PAIR_SAME, x, other, False))
                if lit.kind == PAIR_SOMEOF or fluent.symmetric:
                    implied.append((PAIR_KIND, other, x, False))
        elif lit.kind == PAIR_SAME and lit.value:
            assert y is not None
            implied += [
                (PAIR_SOME, x, None, True),
                (PAIR_SOME, y, None, True),
                (PAIR_KIND, x, y, False),
                (PAIR_KIND, y, x, False),
            ]
        fixed[pair_literal_key(lit, fluent)] = lit.value
        for kind, first, second, value in implied:
            literal = PairLiteral(kind, fluent.label, first, second, value)
            fixed[pair_literal_key(literal, fluent)] = value
    return fixed


def condition_pair_names(effects: Sequence[Effect], pair_fluents: PairFluents) -> set[tuple]:
    """The pairs, rows, and columns the pair literals of the effects' conditions name."""
    return {
        name
        for effect in effects
        for lit in effect.condition
        if isinstance(lit, PairLiteral) and lit.pair_fluent in pair_fluents
        for name in pair_literal_names(lit, pair_fluents[lit.pair_fluent])
    }


def condition_pair_keys(effects: Sequence[Effect], pair_fluents: PairFluents) -> set[tuple]:
    """The keys (``pair_literal_key``) of the pair literals of the effects' conditions."""
    return {
        pair_literal_key(lit, pair_fluents[lit.pair_fluent])
        for effect in effects
        for lit in effect.condition
        if isinstance(lit, PairLiteral) and lit.pair_fluent in pair_fluents
    }


def precondition_literal_fits_conditions(
    literal: PairLiteral, effects: Sequence[Effect], pair_fluents: PairFluents, roles: Sequence[str]
) -> bool:
    """Whether a drawn precondition pair literal leaves the effects' pair conditions as
    conditions: it names no pair or row a condition names (F5), and fixes no condition literal
    under the implications of ``fixed_pair_literals``."""
    fluent = pair_fluents[literal.pair_fluent]
    if pair_literal_names(literal, fluent) & condition_pair_names(effects, pair_fluents):
        return False
    fixed = fixed_pair_literals([literal], pair_fluents, roles)
    return not (set(fixed) & condition_pair_keys(effects, pair_fluents))


def fixed_condition_problem(
    effect: Effect,
    precondition: Sequence[AnyLiteral],
    pair_fluents: PairFluents,
    roles: Sequence[str] = ROLES,
) -> str | None:
    """Why a condition of the effect is refused under Jon's ruling WM.68 (the stage c3b
    follow-up of October 28, 2026), or None: a pair condition literal whose value the
    precondition fixes under the implications of ``fixed_pair_literals`` (the compiled literals
    included; an act's location literal fixes ``SOMEOF`` of the place true) would be redundant
    with its value or never fire with the other, so the effect would be unconditional either
    way. The generator draws such a condition again; an event file's entry with one is an error
    that names the file and the entry, and ``check_dynamics`` and ``check_place_dynamics`` hold
    the rule as an invariant."""
    pair_literals = [lit for lit in precondition if isinstance(lit, PairLiteral)]
    if not pair_literals or not any(isinstance(lit, PairLiteral) for lit in effect.condition):
        return None
    fixed = fixed_pair_literals(pair_literals, pair_fluents, roles)
    for lit in effect.condition:
        if not isinstance(lit, PairLiteral) or lit.pair_fluent not in pair_fluents:
            continue
        key = pair_literal_key(lit, pair_fluents[lit.pair_fluent])
        if key in fixed:
            value = fixed[key]
            return (
                f"the condition of {effect.text} reads {lit.key}, whose value the precondition "
                f"fixes ({'true' if value else 'false'}, by "
                f"{' AND '.join(p.text for p in pair_literals)}), so the effect would be "
                f"{'unconditional' if value == lit.value else 'never applied'}"
            )
    return None


def implied_conflict(
    candidate: PairLiteral,
    kept: Sequence[AnyLiteral],
    pair_fluents: PairFluents,
    roles: Sequence[str],
) -> bool:
    """Whether a drawn pair literal is implied by, contradicted by, or implies another literal
    of the same precondition (Jon's ruling 2 of October 21, 2026, on stage c2b: item 6's
    implications extend to preconditions), under ``fixed_pair_literals``: the candidate's key
    is fixed by the kept literals, or a kept literal's key is fixed by the candidate."""
    others = [lit for lit in kept if isinstance(lit, PairLiteral)]
    if not others:
        return False
    fluent = pair_fluents[candidate.pair_fluent]
    if pair_literal_key(candidate, fluent) in fixed_pair_literals(others, pair_fluents, roles):
        return True
    own = fixed_pair_literals([candidate], pair_fluents, roles)
    return any(pair_literal_key(lit, pair_fluents[lit.pair_fluent]) in own for lit in others)


def random_pair_literal(
    roles: Sequence[str],
    rng: np.random.Generator,
    pair_fluents: PairFluents,
    literal_kinds: Mapping[str, float],
    location_rng: np.random.Generator | None = None,
) -> PairLiteral | None:
    """One drawn pair literal over the roles (stage c2b), unchecked: its kind from
    ``literal_kinds`` (``pair``, ``same``, or ``some``, the last covering ``SOME`` and
    ``SOMEOF``), its pair fluent uniformly among those the kind allows (``SAME`` on a functional
    pair fluent that is not symmetric, on which it could never hold), then its direction
    (``pair``: which role comes first), or its form and role (``some``: ``SOME`` or ``SOMEOF``,
    then the role), and its polarity, each uniformly. A one-place event type draws ``some``
    literals on its agent only. None when no kind with a positive weight can be drawn.

    With ``location_rng`` (stage c3b, a part of ``world:pair_preconditions`` of its own) and
    location fluents in the world, the ``same`` kind covers the location fluents too, for a
    two-place event type: ``location_rng`` draws uniformly over the functional pair fluents
    between entities that are not symmetric and the location fluents, and when the draw falls
    on a location fluent the literal is ``SAME.PAIRFL.k(agent, patient)`` on it, its polarity
    from ``location_rng``; otherwise the pair fluent and the polarity come from ``rng`` as
    before. A world without location fluents never reads ``location_rng``, so its draws are
    those of stage c2b."""
    fluents = list(pair_fluents.between_entities)
    locations = list(pair_fluents.locations) if location_rng is not None else []
    if not fluents and not locations:
        return None
    same_pool = [f for f in fluents if f.functional and not f.symmetric]
    if len(roles) == 1:
        kinds = {k: w for k, w in literal_kinds.items() if k == "some" and w > 0 and fluents}
    else:
        kinds = {
            k: w
            for k, w in literal_kinds.items()
            if w > 0 and (k == "same" and (same_pool or locations) or k != "same" and fluents)
        }
    if not kinds:
        return None
    kind = _draw(rng, kinds)
    if kind == "pair":
        fluent = fluents[int(rng.integers(len(fluents)))]
        first, second = (roles[0], roles[1]) if rng.integers(2) == 0 else (roles[1], roles[0])
        return PairLiteral(PAIR_KIND, fluent.label, first, second, bool(rng.integers(2)))
    if kind == "same":
        if locations:
            assert location_rng is not None
            index = int(location_rng.integers(len(same_pool) + len(locations)))
            if index >= len(same_pool):
                fluent = locations[index - len(same_pool)]
                value = bool(location_rng.integers(2))
                return PairLiteral(PAIR_SAME, fluent.label, roles[0], roles[1], value)
        fluent = same_pool[int(rng.integers(len(same_pool)))]
        return PairLiteral(PAIR_SAME, fluent.label, roles[0], roles[1], bool(rng.integers(2)))
    fluent = fluents[int(rng.integers(len(fluents)))]
    form = PAIR_SOME if rng.integers(2) == 0 else PAIR_SOMEOF
    role = roles[0] if len(roles) == 1 else roles[int(rng.integers(2))]
    return PairLiteral(form, fluent.label, role, None, bool(rng.integers(2)))


def random_pair_effect(
    roles: Sequence[str], rng: np.random.Generator, pair_fluents: PairFluents
) -> Effect:
    """One drawn pair effect between the two roles (stage c2b): the pair fluent (between
    entities; a location fluent is written by moves alone, stage c3a), the direction, and the
    value, each uniformly."""
    fluents = list(pair_fluents.between_entities)
    fluent = fluents[int(rng.integers(len(fluents)))]
    first, second = (roles[0], roles[1]) if rng.integers(2) == 0 else (roles[1], roles[0])
    return Effect(first, fluent.label, bool(rng.integers(2)), (), second)


def reverse_add_conflict(
    effect: Effect, kept: Iterable[Effect], pair_fluents: PairFluents
) -> Effect | None:
    """The kept effect that adds the reverse pair of an add on an asymmetric pair fluent, or
    None: one event type cannot add both directions (each add's compiled literal holds before
    the step, and the profile breaks after it), so the two are a contradiction (found in stage
    c2b's follow-up; stage c2a's rules counted the two directions as two pairs)."""
    if not effect.pair or not effect.value or effect.fluent not in pair_fluents:
        return None
    fluent = pair_fluents[effect.fluent]
    if not fluent.asymmetric:
        return None
    for other in kept:
        if (
            other.pair
            and other.value
            and other.fluent == effect.fluent
            and other.role == effect.second
            and other.second == effect.role
        ):
            return other
    return None


def _effect_slot(effect: Effect, pair_fluents: PairFluents | None = None) -> tuple:
    """The slot an effect fills: ``(role, fluent)`` for a Boolean or numeric effect, and the
    pair it writes (both directions of a symmetric pair are one slot) for a pair effect."""
    if effect.pair and pair_fluents is not None and effect.fluent in pair_fluents:
        return pair_effect_key(effect, pair_fluents[effect.fluent])
    return (effect.role, effect.fluent)


def keeps_no_condition(effect: Effect, pair_fluents: PairFluents) -> bool:
    """An add on an asymmetric or functional pair fluent keeps no condition (F4): its compiled
    literals would not be a plain precondition, so the generator never draws one for it."""
    if not effect.pair or not effect.value or effect.fluent not in pair_fluents:
        return False
    fluent = pair_fluents[effect.fluent]
    return fluent.asymmetric or fluent.functional


# ---------------------------------------------------------------------------------------------
# Preconditions and effects
# ---------------------------------------------------------------------------------------------


@dataclass
class _Draft:
    event_type: EventType
    inherited_effects: list[tuple[int, Effect]] = field(default_factory=list)
    own_effects: list[Effect] = field(default_factory=list)
    inherited_literals: list[tuple[int, AnyLiteral]] = field(default_factory=list)
    own_literals: list[AnyLiteral] = field(default_factory=list)
    explicit: ExplicitEventType | None = None
    effects: tuple[Effect, ...] = ()
    precondition: tuple[AnyLiteral, ...] = ()
    own_effect_keys: set[tuple] = field(default_factory=set)
    """The slots (``_effect_slot``) of the event type's own effects."""
    own_literal_keys: set[tuple] = field(default_factory=set)
    """The slots (``_literal_slot``) of the event type's own precondition literals."""
    quota_literal_keys: set[tuple] = field(default_factory=set)
    """The slots of the ``SAME`` literals the quota ``same_rules`` gave (stage c3b): never
    redrawn or dropped, as the compiled literals are not."""

    @property
    def roles(self) -> tuple[str, ...]:
        return self.event_type.roles


class _Dynamics:
    """Draws and resolves effects and preconditions for every event type."""

    def __init__(
        self,
        event_types: EventTypes,
        fluents: Fluents,
        config: Config,
        streams: WorldStreams,
        event_file: EventFile | None,
        derived_initial: Mapping[str, np.ndarray],
        statics: StaticWorld | None = None,
        numeric: NumericFluents | None = None,
        pair_fluents: PairFluents | None = None,
        extra_effects: Sequence[Effect] = (),
    ) -> None:
        self.static = event_types
        self.statics = statics
        self.extra_effects: tuple[Effect, ...] = tuple(extra_effects)
        """Effects of the world that are not the entity event types' (stage c3b, WM.66: the
        effects of the event file's place entries, fixed before any draw), which the pools of
        achievable values count beside the drafts' effects."""
        self.fluents = fluents
        self.numeric = numeric if numeric is not None else empty_numeric(0)
        self.numeric_labels = self.numeric.labels
        self.pair_fluents = pair_fluents if pair_fluents is not None else empty_pair_fluents()
        self.effects_config: EffectsConfig = config.event_types.effects
        self.preconditions_config: PreconditionsConfig = config.event_types.preconditions
        self.pair_config: PairFluentsConfig = config.pair_fluents
        self.rng_effects = streams.effects
        self.rng_preconditions = streams.preconditions
        self.rng_conditions = streams.conditions
        self.rng_numeric_effects = streams.numeric_effects
        self.rng_numeric_preconditions = streams.numeric_preconditions
        self.rng_numeric_conditions = streams.numeric_conditions
        self.rng_pair_preconditions = streams.pair_preconditions
        self.rng_pair_effects = streams.pair_effects
        self.rng_pair_conditions = streams.pair_conditions
        self.rng_location_same: np.random.Generator | None = (
            streams.part("pair_preconditions", "locations") if self.pair_fluents.locations else None
        )
        """The part of ``world:pair_preconditions`` that decides whether a drawn ``SAME``
        literal is on a location fluent (stage c3b); None without location fluents."""
        self.rng_same_rules = streams.part("pair_preconditions", "same_rules")
        """The part of ``world:pair_preconditions`` the quota ``same_rules`` draws from (stage
        c3b); read only in a world with location fluents."""
        self.same_quota: dict[str, PairLiteral] = {}
        self.event_file = event_file
        self.base_labels = fluents.base_labels
        self.all_labels = fluents.labels
        self.feature_labels: tuple[str, ...] = (
            tuple(f.label for f in statics.taxonomy.features.features)
            if statics is not None
            else ()
        )
        """The static features a condition literal can name: PROPERTY and PART, free or
        determined."""
        self._able: dict[str, np.ndarray] = {}
        self.stats: dict[str, int] = {
            "effects_dropped_contradiction": 0,
            "effects_dropped_duplicate": 0,
            "effects_dropped_empty": 0,
            "effects_dropped_no_slot": 0,
            "effects_redrawn": 0,
            "effects_redrawn_dead": 0,
            "effects_dropped_dead": 0,
            "literals_dropped_contradiction": 0,
            "literals_dropped_duplicate": 0,
            "literals_dropped_unachievable": 0,
            "literals_redrawn": 0,
            "conditions_redrawn": 0,
            "conditions_dropped": 0,
            "numeric_literals_redrawn": 0,
        }
        self.pair_stats: dict[str, int] = (
            {
                "precondition_replacement_draws": 0,
                "precondition_replacements": 0,
                "precondition_replacements_failed": 0,
                "pair_literals_redrawn": 0,
                "effect_replacement_draws": 0,
                "effect_replacements": 0,
                "pair_effects_redrawn_compiled": 0,
                "pair_effects_dropped_compiled": 0,
                "pair_conditions_drawn": 0,
                "pair_conditions_rejected_named": 0,
                "pair_conditions_rejected_fixed": 0,
                "pair_conditions_rejected_other": 0,
                "adds_kept_unconditional": 0,
                "pair_literals_redrawn_implied": 0,
                "pair_literals_dropped_implied": 0,
                "pair_deletes_redrawn_dead": 0,
                "pair_deletes_dropped_dead": 0,
                "pair_effects_redrawn_reverse": 0,
                **(
                    {
                        "same_location_replacements": 0,
                        "same_rules_added": 0,
                        "same_rules_short": 0,
                    }
                    if self.pair_fluents.locations
                    else {}
                ),
            }
            if len(self.pair_fluents)
            else {}
        )
        """The pair draws and redraws of stage c2b (``pair_fluents.fix_ups`` of
        ``world_stats.yaml``): the draws against the share for precondition literals and the
        literals replaced by a pair literal (and the replacements that found no achievable pair
        literal and kept the drawn literal), the pair literal candidates rejected
        (unachievable, a slot already taken, or clashing with a condition), the draws against
        the share for effects and the effects replaced by a pair effect, the pair adds whose
        compiled literals contradicted the precondition (redrawn when own, dropped when
        inherited), the pair condition literals drawn, the conditions rejected for a pair
        literal that names what the precondition names, that the precondition fixes, or that
        breaks another rule, the adds on an asymmetric or functional pair fluent that keep
        no condition, the drawn pair literals drawn again (own) or dropped (inherited) because
        another literal of the precondition, or a compiled literal, implied or contradicted
        them, and the deletes drawn again (own) or dropped (inherited) because their pair
        fluent can never hold (Jon's rulings 1 and 2 of October 21, 2026), and the drawn adds
        rejected because the event type already adds the reverse pair of an asymmetric pair
        fluent; and, only with location fluents (stage c3b), the drawn literals replaced by
        ``SAME`` on a location fluent at the share, the event types the quota ``same_rules``
        gave such a literal, and the quota's shortfall when fewer event types qualified."""
        self.condition_draws: dict[str, int] = {"drawn": 0, "conditional": 0}
        """The effects a condition was drawn for, and how many of them got one (Jon's ruling 2
        on stage b1: the share counts drawn effects; ``world_stats.yaml`` reports it beside the
        share of all effects)."""
        # Values present in the initial state: (fluent, value) pairs.
        self.initial_values: set[tuple[str, bool]] = set()
        for i, fluent in enumerate(fluents.base):
            column = fluents.initial_values[:, i]
            if column.size:
                for value in np.unique(column):
                    self.initial_values.add((fluent.label, bool(value)))
        for label, column in derived_initial.items():
            if column.size:
                for value in np.unique(column):
                    self.initial_values.add((label, bool(value)))
        self.drafts: dict[str, _Draft] = {e.label: _Draft(e) for e in event_types.event_types}
        self.feature_effects: dict[str, Effect | None] = {}
        self.feature_literals: dict[str, Literal | None] = {}

    # Drawing --------------------------------------------------------------------------------------

    def _role(
        self, rng: np.random.Generator, roles: tuple[str, ...], weights: Mapping[str, float]
    ) -> str:
        if len(roles) == 1:
            return roles[0]
        return _draw(rng, weights)

    def _numeric_effect(self, role: str) -> Effect:
        """A numeric effect of a role, from ``world:numeric_effects``: the fluent uniformly, the
        operator uniformly, the constant uniformly from ``numeric_amounts``, rounded to 4
        decimals when drawn (WM.60)."""
        rng = self.rng_numeric_effects
        cfg = self.effects_config
        fluent = self.numeric_labels[int(rng.integers(len(self.numeric_labels)))]
        sign = 1.0 if rng.integers(2) == 0 else -1.0
        amount = drawn(rng.uniform(cfg.numeric_amounts[0], cfg.numeric_amounts[1]))
        return Effect(role, fluent, sign * amount)

    def _maybe_numeric(self, effect: Effect) -> Effect:
        """The Boolean effect, or a numeric one of the same role at ``numeric_share``; nothing
        is drawn when the share is 0 or the world has no numeric fluents."""
        share = self.effects_config.numeric_share
        if not self.numeric_labels or share <= 0:
            return effect
        if self.rng_numeric_effects.random() < share:
            return self._numeric_effect(effect.role)
        return effect

    def _maybe_pair_effect(self, effect: Effect, roles: tuple[str, ...]) -> Effect:
        """The effect, or a pair effect between the two roles at ``pair_share`` (stage c2b),
        from ``world:pair_effects``; nothing is drawn for a one-place event type, with the share
        at 0, or without pair fluents."""
        share = self.effects_config.pair_share
        if len(roles) != 2 or not self.pair_fluents.between_entities or share <= 0:
            return effect
        self.pair_stats["effect_replacement_draws"] += 1
        if self.rng_pair_effects.random() < share:
            self.pair_stats["effect_replacements"] += 1
            return random_pair_effect(roles, self.rng_pair_effects, self.pair_fluents)
        return effect

    def _draw_effect(
        self,
        roles: tuple[str, ...],
        forbidden: set[tuple] = frozenset(),  # type: ignore[assignment]
        kept: Iterable[Effect] = (),
    ) -> Effect | None:
        """An effect over the roles whose slot is not in ``forbidden`` and that adds no reverse
        pair of an add in ``kept`` on an asymmetric pair fluent, or None after ``MAX_TRIES``."""
        rng = self.rng_effects
        kept = tuple(kept)
        for _ in range(MAX_TRIES):
            role = self._role(rng, roles, self.effects_config.roles)
            if self.base_labels:
                fluent = self.base_labels[int(rng.integers(len(self.base_labels)))]
                value = bool(rng.integers(2))
                effect = self._maybe_numeric(Effect(role, fluent, value))
            else:
                effect = self._numeric_effect(role)
            effect = self._maybe_pair_effect(effect, roles)
            if _effect_slot(effect, self.pair_fluents) in forbidden:
                continue
            if reverse_add_conflict(effect, kept, self.pair_fluents) is not None:
                self.pair_stats["pair_effects_redrawn_reverse"] += 1
                continue
            return effect
        return None

    def all_effects(self) -> list[Effect]:
        """Every effect of the world known to the draws: the drafts' and the extra ones."""
        found = [e for draft in self.drafts.values() for e in draft.effects]
        found.extend(self.extra_effects)
        return found

    def produced_numeric(self) -> set[tuple[str, int]]:
        """``(fluent, +1 or -1)`` for every numeric effect of every event type."""
        return {(e.fluent, 1 if e.amount > 0 else -1) for e in self.all_effects() if e.numeric}

    def numeric_achievable(self, fluent: str, above: bool, threshold: float) -> bool:
        return self.numeric.achievable(fluent, above, threshold, self.produced_numeric())

    def _numeric_literal(
        self,
        role: str,
        rng: np.random.Generator,
        forbidden: set[tuple[str, str]],
        produced: set[tuple[str, int]],
    ) -> Literal | None:
        """A numeric threshold literal of a role, from ``rng``: the fluent uniformly, the side
        uniformly, the threshold from ``numeric_thresholds``; drawn again while it is not
        achievable or its fluent is forbidden for the role, up to ``MAX_TRIES``."""
        cfg = self.preconditions_config
        for _ in range(MAX_TRIES):
            fluent = self.numeric_labels[int(rng.integers(len(self.numeric_labels)))]
            above = bool(rng.integers(2) == 0)
            fraction = cfg.numeric_thresholds[int(rng.integers(len(cfg.numeric_thresholds)))]
            threshold = self.numeric[fluent].point(fraction)
            if (role, fluent) in forbidden or not self.numeric.achievable(
                fluent, above, threshold, produced
            ):
                self.stats["numeric_literals_redrawn"] += 1
                continue
            return Literal(role, fluent, above, threshold)
        return None

    def _maybe_numeric_literal(
        self, literal: Literal, forbidden: set[tuple[str, str]], rng: np.random.Generator
    ) -> Literal:
        """The Boolean literal, or a numeric one of the same role at ``numeric_share`` (the
        Boolean one again when no achievable numeric literal was found)."""
        share = self.preconditions_config.numeric_share
        if not self.numeric_labels or share <= 0:
            return literal
        if rng.random() < share:
            candidate = self._numeric_literal(literal.role, rng, forbidden, self.produced_numeric())
            if candidate is not None:
                return candidate
        return literal

    def _pair_literal(
        self,
        roles: tuple[str, ...],
        rng: np.random.Generator,
        forbidden: set[tuple],
        effects: Sequence[Effect],
        kept: Sequence[AnyLiteral] = (),
        location_rng: np.random.Generator | None = None,
    ) -> PairLiteral | None:
        """A pair literal of a precondition (stage c2b), drawn from ``rng`` by
        ``random_pair_literal`` and drawn again while it is not achievable, fills a slot the
        precondition already fills, clashes with a pair condition of the effects, or is implied
        by, contradicted by, or implies a literal of ``kept`` (the precondition's other
        literals and the compiled literals its effects will get), up to ``MAX_TRIES``; None
        when none was found."""
        adds, deletes = self.pair_adds_and_deletes()
        kinds = self.pair_config.literal_kinds
        for _ in range(MAX_TRIES):
            literal = random_pair_literal(roles, rng, self.pair_fluents, kinds, location_rng)
            if literal is None:
                return None
            fluent = self.pair_fluents[literal.pair_fluent]
            if (
                pair_literal_key(literal, fluent) in forbidden
                or not pair_literal_achievable(literal, fluent, adds, deletes)
                or not precondition_literal_fits_conditions(
                    literal, effects, self.pair_fluents, roles
                )
            ):
                self.pair_stats["pair_literals_redrawn"] += 1
                continue
            if implied_conflict(literal, kept, self.pair_fluents, roles):
                self.pair_stats["pair_literals_redrawn"] += 1
                self.pair_stats["pair_literals_redrawn_implied"] += 1
                continue
            if literal.kind == PAIR_SAME and fluent.total:
                self.pair_stats["same_location_replacements"] += 1
            return literal
        return None

    def _maybe_pair_literal(
        self,
        literal: Literal,
        roles: tuple[str, ...],
        forbidden: set[tuple],
        rng: np.random.Generator,
        effects: Sequence[Effect],
        kept: Sequence[AnyLiteral] = (),
    ) -> AnyLiteral:
        """The literal, or a pair literal over the roles at ``pair_share`` (the literal again
        when no achievable pair literal was found); nothing is drawn with the share at 0 or
        without pair fluents."""
        share = self.preconditions_config.pair_share
        if not len(self.pair_fluents) or share <= 0:
            return literal
        self.pair_stats["precondition_replacement_draws"] += 1
        if rng.random() < share:
            self.pair_stats["precondition_replacements"] += 1
            candidate = self._pair_literal(
                roles, rng, forbidden, effects, kept, self.rng_location_same
            )
            if candidate is not None:
                return candidate
            self.pair_stats["precondition_replacements_failed"] += 1
        return literal

    def _draw_literal(
        self,
        roles: tuple[str, ...],
        achievable: set[tuple[str, bool]],
        produced: list[tuple[str, bool]],
        forbidden: set[tuple] = frozenset(),  # type: ignore[assignment]
        numeric_rng: np.random.Generator | None = None,
        pair_rng: np.random.Generator | None = None,
        effects: Sequence[Effect] = (),
        kept: Sequence[AnyLiteral] = (),
    ) -> AnyLiteral | None:
        """A precondition literal over the roles: a Boolean literal from ``world:preconditions``
        (an enabling one at ``enabled_share``), replaced by a numeric literal at
        ``numeric_share`` and then by a pair literal at ``pair_share``, each from its own
        stream; ``forbidden`` holds the slots the precondition already fills, ``effects`` the
        effects whose conditions a pair literal must leave as conditions, and ``kept`` the
        literals a pair literal must neither imply nor be implied by."""
        rng = self.rng_preconditions
        numeric_rng = self.rng_numeric_preconditions if numeric_rng is None else numeric_rng
        pair_rng = self.rng_pair_preconditions if pair_rng is None else pair_rng
        if not achievable and not self.numeric_labels:
            return None
        for _ in range(MAX_TRIES):
            role = self._role(rng, roles, self.preconditions_config.roles)
            if not self.all_labels:
                numeric = self._numeric_literal(
                    role, numeric_rng, set(forbidden), self.produced_numeric()
                )
                if numeric is None:
                    return None
                return self._maybe_pair_literal(
                    numeric, roles, set(forbidden), pair_rng, effects, kept
                )
            if produced and rng.random() < self.preconditions_config.enabled_share:
                fluent, value = produced[int(rng.integers(len(produced)))]
            else:
                fluent = self.all_labels[int(rng.integers(len(self.all_labels)))]
                value = bool(rng.integers(2))
            if (fluent, value) in achievable and (role, fluent) not in forbidden:
                literal = self._maybe_numeric_literal(
                    Literal(role, fluent, value), set(forbidden), numeric_rng
                )
                return self._maybe_pair_literal(
                    literal, roles, set(forbidden), pair_rng, effects, kept
                )
        return None

    def pair_adds_and_deletes(self) -> tuple[set[str], set[str]]:
        """The pair fluents some effect adds a pair of, and those some effect deletes one of."""
        effects = self.all_effects()
        adds = {e.fluent for e in effects if e.pair and e.value}
        deletes = {e.fluent for e in effects if e.pair and not e.value}
        return adds, deletes

    def literal_achievable(self, literal: AnyLiteral, achievable: set[tuple[str, bool]]) -> bool:
        if isinstance(literal, PairLiteral):
            adds, deletes = self.pair_adds_and_deletes()
            return pair_literal_achievable(
                literal, self.pair_fluents[literal.pair_fluent], adds, deletes
            )
        if literal.numeric:
            assert literal.threshold is not None
            return self.numeric_achievable(literal.fluent, literal.value, literal.threshold)
        return (literal.fluent, literal.value) in achievable

    # Effects ----------------------------------------------------------------------------------

    def draw_effects(self) -> None:
        if not self.base_labels and not self.numeric_labels:
            return
        cfg = self.effects_config
        for et in self.static.unary:
            n = _draw(self.rng_effects, cfg.count)
            draft = self.drafts[et.label]
            for _ in range(n):
                effect = self._draw_effect(et.roles)
                if effect is not None:
                    draft.own_effects.append(effect)
        for feature in self.static.features:
            drawn = self.rng_effects.random() < cfg.feature_rate
            self.feature_effects[feature] = self._draw_effect(ROLES) if drawn else None
        for et in self.static.binary:
            n = _draw(self.rng_effects, cfg.count)
            draft = self.drafts[et.label]
            for _ in range(n):
                effect = self._draw_effect(et.roles)
                if effect is not None:
                    draft.own_effects.append(effect)
        for et in self.static.event_types:
            draft = self.drafts[et.label]
            for index, feature in enumerate(self.static.features):
                effect = self.feature_effects.get(feature)
                if effect is not None and feature in et.features:
                    draft.inherited_effects.append((index, effect))

    def resolve_effects(self, draft: _Draft) -> None:
        if draft.explicit is not None and draft.explicit.effects is not None:
            draft.effects = draft.explicit.effects
            return
        kept: dict[tuple, Effect] = {}
        for _, effect in draft.inherited_effects:
            key = _effect_slot(effect, self.pair_fluents)
            if key in kept:
                if kept[key].value != effect.value:
                    self.stats["effects_dropped_contradiction"] += 1
                else:
                    self.stats["effects_dropped_duplicate"] += 1
                continue
            if reverse_add_conflict(effect, kept.values(), self.pair_fluents) is not None:
                self.stats["effects_dropped_contradiction"] += 1
                continue
            kept[key] = effect
        draft.own_effect_keys = set()
        for effect in draft.own_effects:
            candidate: Effect | None = effect
            while candidate is not None and (
                _effect_slot(candidate, self.pair_fluents) in kept
                or reverse_add_conflict(candidate, kept.values(), self.pair_fluents) is not None
            ):
                self.stats["effects_redrawn"] += 1
                candidate = self._draw_effect(draft.roles, set(kept), kept.values())
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[_effect_slot(candidate, self.pair_fluents)] = candidate
            draft.own_effect_keys.add(_effect_slot(candidate, self.pair_fluents))
        draft.effects = tuple(kept.values())

    # Preconditions ----------------------------------------------------------------------------

    def produced(self) -> list[tuple[str, bool]]:
        """``(fluent, value)`` for every Boolean effect of every event type (and of the extra
        effects, WM.66)."""
        values = {(e.fluent, e.value) for e in self.all_effects() if not e.numeric and not e.pair}
        return sorted(values)

    def achievable(self) -> set[tuple[str, bool]]:
        return set(self.produced()) | self.initial_values

    def draw_preconditions(self) -> None:
        if not self.all_labels and not self.numeric_labels:
            return
        cfg = self.preconditions_config
        achievable = self.achievable()
        produced = self.produced()
        for et in self.static.unary:
            n = _draw(self.rng_preconditions, cfg.literals)
            draft = self.drafts[et.label]
            for _ in range(n):
                literal = self._draw_literal(et.roles, achievable, produced, effects=draft.effects)
                if literal is not None:
                    draft.own_literals.append(literal)
        for feature in self.static.features:
            drawn = self.rng_preconditions.random() < cfg.feature_rate
            self.feature_literals[feature] = (
                self._draw_literal(ROLES, achievable, produced) if drawn else None
            )
        for et in self.static.binary:
            n = _draw(self.rng_preconditions, cfg.literals)
            draft = self.drafts[et.label]
            for _ in range(n):
                literal = self._draw_literal(et.roles, achievable, produced, effects=draft.effects)
                if literal is not None:
                    draft.own_literals.append(literal)
        for et in self.static.event_types:
            draft = self.drafts[et.label]
            for index, feature in enumerate(self.static.features):
                literal = self.feature_literals.get(feature)
                if literal is not None and feature in et.features:
                    draft.inherited_literals.append((index, literal))

    def resolve_preconditions(self, draft: _Draft) -> None:
        if draft.explicit is not None and draft.explicit.precondition is not None:
            draft.precondition = draft.explicit.precondition
            return
        achievable = self.achievable()
        produced = self.produced()
        kept: dict[tuple, AnyLiteral] = {}
        compiled = self.virtual_compiled(draft)
        for _, literal in draft.inherited_literals:
            key = _literal_slot(literal, self.pair_fluents)
            if key in kept:
                if kept[key].value != literal.value:
                    self.stats["literals_dropped_contradiction"] += 1
                else:
                    self.stats["literals_dropped_duplicate"] += 1
                continue
            if isinstance(literal, PairLiteral) and implied_conflict(
                literal, list(kept.values()) + compiled, self.pair_fluents, draft.roles
            ):
                self.pair_stats["pair_literals_dropped_implied"] += 1
                continue
            kept[key] = literal
        draft.own_literal_keys = set()
        read = condition_keys(draft.effects)
        for literal in draft.own_literals:
            candidate: AnyLiteral | None = literal
            while candidate is not None and (
                _literal_slot(candidate, self.pair_fluents) in kept
                or _literal_slot(candidate, self.pair_fluents) in read
                or (
                    isinstance(candidate, PairLiteral)
                    and (
                        not precondition_literal_fits_conditions(
                            candidate, draft.effects, self.pair_fluents, draft.roles
                        )
                        or self._implied(candidate, list(kept.values()) + compiled, draft)
                    )
                )
            ):
                self.stats["literals_redrawn"] += 1
                candidate = self._draw_literal(
                    draft.roles,
                    achievable,
                    produced,
                    set(kept) | read,
                    effects=draft.effects,
                    kept=list(kept.values()) + compiled,
                )
            if candidate is None:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            kept[_literal_slot(candidate, self.pair_fluents)] = candidate
            draft.own_literal_keys.add(_literal_slot(candidate, self.pair_fluents))
        draft.precondition = tuple(kept.values())

    # Fix-ups -----------------------------------------------------------------------------------

    def fix_empty_effects(self, draft: _Draft) -> bool:
        """An effect that sets a fluent to the value the precondition already requires changes
        nothing: an own effect is redrawn, an inherited one dropped. Returns whether anything
        changed."""
        required: dict[tuple, bool] = {
            (lit.role, lit.fluent): lit.value
            for lit in draft.precondition
            if not lit.numeric and not lit.pair
        }
        # a pair add whose precondition requires the pair, or a delete whose precondition
        # requires its absence, changes nothing either (stage c2b)
        for lit in draft.precondition:
            if isinstance(lit, PairLiteral) and lit.kind == PAIR_KIND:
                required[pair_literal_key(lit, self.pair_fluents[lit.pair_fluent])] = lit.value
        changed = False
        kept: dict[tuple, Effect] = {_effect_slot(e, self.pair_fluents): e for e in draft.effects}
        for effect in list(draft.effects):
            key = _effect_slot(effect, self.pair_fluents)
            if effect.numeric or required.get(key) != effect.value:
                continue  # a numeric effect is never empty: it can always add or subtract
            changed = True
            del kept[key]
            if key not in draft.own_effect_keys:
                self.stats["effects_dropped_empty"] += 1
                continue
            draft.own_effect_keys.discard(key)
            candidate = None
            for _ in range(MAX_TRIES):
                candidate = self._draw_effect(draft.roles, set(kept), kept.values())
                if (
                    candidate is None
                    or candidate.numeric
                    or required.get(_effect_slot(candidate, self.pair_fluents)) != candidate.value
                ):
                    break
                candidate = None
            self.stats["effects_redrawn"] += 1
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[_effect_slot(candidate, self.pair_fluents)] = candidate
            draft.own_effect_keys.add(_effect_slot(candidate, self.pair_fluents))
        if changed:
            draft.effects = tuple(kept.values())
        return changed

    def fix_compiled_contradictions(self, draft: _Draft) -> bool:
        """A pair add whose compiled literals (F4) contradict the precondition could never
        happen (stage c2b): an own add is redrawn, an inherited one dropped. Returns whether
        anything changed."""
        if not any(e.pair for e in draft.effects):
            return False
        present = {
            pair_literal_key(lit, self.pair_fluents[lit.pair_fluent]): lit.value
            for lit in draft.precondition
            if isinstance(lit, PairLiteral)
        }
        changed = False
        kept: dict[tuple, Effect] = {_effect_slot(e, self.pair_fluents): e for e in draft.effects}
        for effect in list(draft.effects):
            if not effect.pair or effect.condition or not effect.value:
                continue
            fluent = self.pair_fluents[effect.fluent]
            contradicted = any(
                present.get(pair_literal_key(lit, fluent)) == (not lit.value)
                for lit in compiled_literals(effect, fluent)
            )
            if not contradicted:
                continue
            changed = True
            key = _effect_slot(effect, self.pair_fluents)
            del kept[key]
            if key not in draft.own_effect_keys:
                self.pair_stats["pair_effects_dropped_compiled"] += 1
                continue
            draft.own_effect_keys.discard(key)
            self.pair_stats["pair_effects_redrawn_compiled"] += 1
            candidate = self._draw_effect(draft.roles, set(kept), kept.values())
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[_effect_slot(candidate, self.pair_fluents)] = candidate
            draft.own_effect_keys.add(_effect_slot(candidate, self.pair_fluents))
        if changed:
            draft.effects = tuple(kept.values())
        return changed

    def fix_unachievable(
        self, draft: _Draft, achievable: set[tuple[str, bool]], produced: list[tuple[str, bool]]
    ) -> bool:
        changed = False
        kept: dict[tuple, AnyLiteral] = {
            _literal_slot(lit, self.pair_fluents): lit for lit in draft.precondition
        }
        for literal in list(draft.precondition):
            if self.literal_achievable(literal, achievable):
                continue
            key = _literal_slot(literal, self.pair_fluents)
            if key in draft.quota_literal_keys:
                continue  # SAME on a location fluent is always achievable; never redrawn
            changed = True
            del kept[key]
            if key not in draft.own_literal_keys:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            draft.own_literal_keys.discard(key)
            self.stats["literals_redrawn"] += 1
            candidate = self._draw_literal(
                draft.roles,
                achievable,
                produced,
                set(kept) | condition_keys(draft.effects),
                effects=draft.effects,
                kept=list(kept.values()) + self.virtual_compiled(draft),
            )
            if candidate is None:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            kept[_literal_slot(candidate, self.pair_fluents)] = candidate
            draft.own_literal_keys.add(_literal_slot(candidate, self.pair_fluents))
        if changed:
            draft.precondition = tuple(kept.values())
        return changed

    def virtual_compiled(self, draft: _Draft) -> list[PairLiteral]:
        """The compiled literals (F4) the draft's unconditional adds will get, as
        ``compile_profile_literals`` will append them, those already present by what they name
        left out: a drawn literal must neither imply nor be implied by one of them (Jon's
        ruling 2 on stage c2b: compiled literals are never redrawn)."""
        present = {
            pair_literal_key(lit, self.pair_fluents[lit.pair_fluent])
            for lit in draft.precondition
            if isinstance(lit, PairLiteral)
        }
        found: list[PairLiteral] = []
        for effect in draft.effects:
            if not effect.pair or effect.condition or not effect.value:
                continue
            fluent = self.pair_fluents[effect.fluent]
            for literal in compiled_literals(effect, fluent):
                key = pair_literal_key(literal, fluent)
                if key not in present:
                    present.add(key)
                    found.append(literal)
        return found

    def _implied(self, candidate: PairLiteral, kept: Sequence[AnyLiteral], draft: _Draft) -> bool:
        """``implied_conflict`` with its count."""
        if implied_conflict(candidate, kept, self.pair_fluents, draft.roles):
            self.pair_stats["pair_literals_redrawn_implied"] += 1
            return True
        return False

    def fix_implied_literals(
        self, draft: _Draft, achievable: set[tuple[str, bool]], produced: list[tuple[str, bool]]
    ) -> bool:
        """A drawn pair literal implied by, contradicted by, or implying another literal of the
        precondition or a compiled literal the effects will get is drawn again (own) or dropped
        (inherited); compiled literals are never redrawn (Jon's ruling 2 on stage c2b). Run in
        the fix-up rounds, because the effects, and so the compiled literals, can change
        there. Returns whether anything changed."""
        if not any(isinstance(lit, PairLiteral) for lit in draft.precondition):
            return False
        changed = False
        kept: dict[tuple, AnyLiteral] = {
            _literal_slot(lit, self.pair_fluents): lit for lit in draft.precondition
        }
        for literal in list(draft.precondition):
            if not isinstance(literal, PairLiteral):
                continue
            key = _literal_slot(literal, self.pair_fluents)
            if key not in kept or key in draft.quota_literal_keys:
                continue
            others = [lit for lit in kept.values() if lit is not literal]
            others += self.virtual_compiled(_Draft(draft.event_type, effects=draft.effects))
            if not implied_conflict(literal, others, self.pair_fluents, draft.roles):
                continue
            changed = True
            del kept[key]
            if key not in draft.own_literal_keys:
                self.pair_stats["pair_literals_dropped_implied"] += 1
                continue
            draft.own_literal_keys.discard(key)
            self.stats["literals_redrawn"] += 1
            self.pair_stats["pair_literals_redrawn_implied"] += 1
            candidate = self._draw_literal(
                draft.roles,
                achievable,
                produced,
                set(kept) | condition_keys(draft.effects),
                effects=draft.effects,
                kept=others,
            )
            if candidate is None:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            kept[_literal_slot(candidate, self.pair_fluents)] = candidate
            draft.own_literal_keys.add(_literal_slot(candidate, self.pair_fluents))
        if changed:
            draft.precondition = tuple(kept.values())
        return changed

    def dead_pair_fluents(self) -> set[str]:
        """The pair fluents that can never hold: an initial rate of 0 and no add anywhere in
        the world (after the draws); a delete on one can never change anything (Jon's ruling 1
        on stage c2b)."""
        adds, _ = self.pair_adds_and_deletes()
        return {f.label for f in self.pair_fluents if f.initial_rate == 0.0 and f.label not in adds}

    def fix_dead_deletes(self, draft: _Draft) -> bool:
        """A delete on a pair fluent that can never hold is redrawn as an empty effect is: an
        own effect drawn again, an inherited one dropped. Run in the fix-up rounds until
        stable, because a redrawn effect can be an add that brings a pair fluent to life."""
        if not any(e.pair and not e.value for e in draft.effects):
            return False
        dead = self.dead_pair_fluents()
        if not dead:
            return False
        changed = False
        kept: dict[tuple, Effect] = {_effect_slot(e, self.pair_fluents): e for e in draft.effects}
        for effect in list(draft.effects):
            if not effect.pair or effect.value or effect.fluent not in dead:
                continue
            changed = True
            key = _effect_slot(effect, self.pair_fluents)
            del kept[key]
            if key not in draft.own_effect_keys:
                self.pair_stats["pair_deletes_dropped_dead"] += 1
                continue
            draft.own_effect_keys.discard(key)
            self.pair_stats["pair_deletes_redrawn_dead"] += 1
            self.stats["effects_redrawn"] += 1
            candidate = self._draw_effect(draft.roles, set(kept), kept.values())
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[_effect_slot(candidate, self.pair_fluents)] = candidate
            draft.own_effect_keys.add(_effect_slot(candidate, self.pair_fluents))
        if changed:
            draft.effects = tuple(kept.values())
        return changed

    # Explicit entries -----------------------------------------------------------------------

    def fix_dead_booleans(self, draft: _Draft) -> bool:
        """A Boolean effect ``:= v`` on a fluent that can never be ``not v`` (Jon's ruling
        WM.70: its initial rate makes every entity ``v``, and no effect anywhere in the world
        sets it to ``not v``) can never change the state: an own effect is drawn again, an
        inherited one dropped, as ``fix_dead_deletes`` treats a delete on a pair fluent that
        can never hold. Run in the fix-up rounds until stable, because a redrawn effect can set
        the other value and bring the fluent to life."""
        if not any(not e.numeric and not e.pair for e in draft.effects):
            return False
        dead = dead_boolean_values(self.fluents, self.all_effects())
        if not dead:
            return False
        changed = False
        kept: dict[tuple, Effect] = {_effect_slot(e, self.pair_fluents): e for e in draft.effects}
        for effect in list(draft.effects):
            if effect.numeric or effect.pair or (effect.fluent, bool(effect.value)) not in dead:
                continue
            changed = True
            key = _effect_slot(effect, self.pair_fluents)
            del kept[key]
            if key not in draft.own_effect_keys:
                self.stats["effects_dropped_dead"] += 1
                continue
            draft.own_effect_keys.discard(key)
            self.stats["effects_redrawn_dead"] += 1
            self.stats["effects_redrawn"] += 1
            candidate = self._draw_effect(draft.roles, set(kept), kept.values())
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[_effect_slot(candidate, self.pair_fluents)] = candidate
            draft.own_effect_keys.add(_effect_slot(candidate, self.pair_fluents))
        if changed:
            draft.effects = tuple(kept.values())
        return changed

    def apply_event_file(self) -> None:
        if self.event_file is None:
            return
        source = self.event_file.source
        for label, entry in self.event_file.event_types.items():
            if label not in self.drafts:
                raise ConfigError(
                    source, entry.field, "unknown event type; the world has none with this label"
                )
            draft = self.drafts[label]
            et = draft.event_type
            if et.category:
                raise ConfigError(
                    source,
                    entry.field,
                    f"{label} is a category of event types; give the entry to a leaf",
                )
            for lit in entry.precondition or ():
                if isinstance(lit, PairLiteral):
                    self._check_explicit_pair_literal(
                        source, f"{entry.field}.precondition", et, lit
                    )
                    continue
                self._check_role(source, f"{entry.field}.precondition", et, lit.role, lit.key)
                if lit.numeric:
                    if lit.fluent not in self.numeric:
                        raise ConfigError(
                            source,
                            f"{entry.field}.precondition",
                            f"reads unknown numeric fluent {lit.fluent}",
                        )
                    self._check_threshold(
                        source, f"{entry.field}.precondition", lit.fluent, lit.threshold
                    )
                elif lit.fluent not in self.fluents:
                    raise ConfigError(
                        source, f"{entry.field}.precondition", f"reads unknown fluent {lit.fluent}"
                    )
            seen_effects: dict[tuple, bool | float] = {}
            for i, effect in enumerate(entry.effects or ()):
                where = f"{entry.field}.effects[{i}]"
                self._check_role(source, where, et, effect.role, effect.key)
                if effect.pair:
                    self._check_explicit_pair_effect(source, where, et, effect, seen_effects)
                    self._check_explicit_condition(source, where, et, effect)
                    continue
                if effect.numeric:
                    if effect.fluent not in self.numeric:
                        raise ConfigError(
                            source, where, f"adds to unknown numeric fluent {effect.fluent}"
                        )
                else:
                    if effect.fluent not in self.fluents:
                        raise ConfigError(source, where, f"writes unknown fluent {effect.fluent}")
                    if self.fluents[effect.fluent].derived:
                        raise ConfigError(
                            source,
                            where,
                            f"{effect.fluent} is a derived fluent; effects write base fluents only",
                        )
                key = (effect.role, effect.fluent)
                if key in seen_effects and seen_effects[key] != effect.value:
                    raise ConfigError(
                        source, where, f"contradicts an earlier effect on {effect.key}"
                    )
                seen_effects[key] = effect.value
                self._check_explicit_condition(source, where, et, effect)
            seen_literals: dict[tuple, bool] = {}
            for lit in entry.precondition or ():
                key = _literal_slot(lit, self.pair_fluents)
                if key in seen_literals and (lit.numeric or seen_literals[key] != lit.value):
                    raise ConfigError(
                        source,
                        f"{entry.field}.precondition",
                        f"holds {lit.key} twice"
                        if lit.numeric
                        else f"holds {lit.key} and its negation",
                    )
                if key in seen_literals:
                    raise ConfigError(
                        source, f"{entry.field}.precondition", f"holds {lit.key} twice"
                    )
                seen_literals[key] = lit.value
            if entry.precondition is not None and entry.effects is not None:
                for effect in entry.effects:
                    if effect.pair:
                        slot = pair_effect_key(effect, self.pair_fluents[effect.fluent])
                        if seen_literals.get(slot) == effect.value:
                            raise ConfigError(
                                source,
                                entry.field,
                                f"the effect {effect.text} {'adds' if effect.value else 'deletes'} "
                                f"a pair whose {'presence' if effect.value else 'absence'} the "
                                f"precondition already requires, so it changes nothing",
                            )
                    elif (
                        not effect.numeric
                        and seen_literals.get((effect.role, effect.fluent)) == effect.value
                    ):
                        raise ConfigError(
                            source,
                            entry.field,
                            f"the effect {effect.text} sets a fluent to the value the precondition "
                            f"already requires, so it changes nothing",
                        )
                    for lit in effect.condition:
                        if isinstance(lit, PairLiteral):
                            self._check_pair_condition_against(
                                source,
                                entry.field,
                                effect,
                                lit,
                                entry.precondition,
                                "already",
                            )
                        elif lit.kind != "feature" and (lit.role, lit.symbol) in seen_literals:
                            raise ConfigError(
                                source,
                                entry.field,
                                f"the condition of {effect.text} reads {lit.key}, which the "
                                f"precondition already fixes",
                            )
            draft.explicit = entry

    def _check_explicit_condition(
        self, source: str, where: str, et: EventType, effect: Effect
    ) -> None:
        """The rules of a condition that need no other draw: every literal names a role of the
        event type and a known symbol, no symbol twice, and no literal on the effect's own fluent
        with its value (the effect would change nothing)."""
        keys: set[tuple] = set()
        for lit in effect.condition:
            if isinstance(lit, PairLiteral):
                self._check_explicit_pair_literal(source, where, et, lit)
                fluent = self.pair_fluents[lit.pair_fluent]
                key = pair_literal_key(lit, fluent)
                if key in keys:
                    raise ConfigError(source, where, f"the condition reads {lit.key} twice")
                keys.add(key)
                if effect.pair and key == pair_effect_key(effect, self.pair_fluents[effect.fluent]):
                    raise ConfigError(
                        source,
                        where,
                        f"the condition reads {lit.key}, the pair the effect writes: the effect "
                        f"would change the pair, or leave it as it is, either way",
                    )
                continue
            self._check_role(source, where, et, lit.role, lit.key)
            if lit.kind == "numeric":
                if lit.symbol not in self.numeric:
                    raise ConfigError(source, where, f"reads unknown numeric fluent {lit.symbol}")
                self._check_threshold(source, where, lit.symbol, lit.threshold)
            elif lit.kind == "fluent":
                if lit.symbol not in self.fluents:
                    raise ConfigError(source, where, f"reads unknown fluent {lit.symbol}")
            elif lit.symbol not in self.feature_labels:
                raise ConfigError(source, where, f"reads unknown feature {lit.symbol}")
            if (lit.role, lit.symbol) in keys:
                raise ConfigError(source, where, f"the condition reads {lit.key} twice")
            keys.add((lit.role, lit.symbol))
            if lit.kind == "fluent" and (lit.role, lit.symbol) == (effect.role, effect.fluent):
                raise ConfigError(
                    source,
                    where,
                    f"the condition reads {lit.key}, the fluent the effect sets: with the "
                    f"effect's value the effect would change nothing, and with the other value "
                    f"the fluent has the effect's value after the event either way",
                )

    def _check_explicit_pair_literal(
        self, source: str, where: str, et: EventType, lit: PairLiteral
    ) -> None:
        """A pair literal of an explicit entry: a known pair fluent, roles of the event type,
        and ``SAME`` on a functional pair fluent only."""
        if lit.pair_fluent not in self.pair_fluents:
            raise ConfigError(
                source,
                where,
                f"{lit.key} reads unknown pair fluent {lit.pair_fluent}; the world's pair "
                f"fluents are {', '.join(self.pair_fluents.labels) or 'none'} (the event "
                f"file's pair_fluents block and pair_fluents.count of the configuration)",
            )
        for role in lit.roles:
            self._check_role(source, where, et, role, lit.key)
        fluent = self.pair_fluents[lit.pair_fluent]
        if lit.kind == PAIR_SAME and not fluent.functional:
            raise ConfigError(
                source,
                where,
                f"{lit.key}: SAME needs a functional pair fluent, and {fluent.label} is not",
            )
        from semantic_world.world.definition import location_literal_problem

        problem = location_literal_problem(lit, fluent.total, False)
        if problem is not None:
            raise ConfigError(source, where, f"{lit.key} {problem}")

    def _check_explicit_pair_effect(
        self,
        source: str,
        where: str,
        et: EventType,
        effect: Effect,
        seen: dict[tuple, bool | float],
    ) -> None:
        """A pair effect of an explicit entry: a known pair fluent, roles of the event type, no
        delete on a total pair fluent, no conditional add on an asymmetric or functional pair
        fluent (its compiled literals would not be a plain precondition, F4), and no second
        effect on the same pair (both directions of a symmetric one)."""
        if effect.fluent not in self.pair_fluents:
            raise ConfigError(
                source,
                where,
                f"writes unknown pair fluent {effect.fluent}; the world's pair fluents are "
                f"{', '.join(self.pair_fluents.labels) or 'none'} (the event file's "
                f"pair_fluents block and pair_fluents.count of the configuration)",
            )
        assert effect.second is not None
        self._check_role(source, where, et, effect.second, effect.key)
        fluent = self.pair_fluents[effect.fluent]
        if fluent.total:
            raise ConfigError(
                source,
                where,
                f"{effect.text} "
                + (
                    "writes the location fluent"
                    if effect.value
                    else "deletes a pair of the location fluent"
                )
                + f" {fluent.label}; only a move (a place_event_types entry of kind move) "
                f"writes a location fluent, and nothing deletes a pair of it",
            )
        if effect.condition and effect.value and (fluent.asymmetric or fluent.functional):
            raise ConfigError(
                source,
                where,
                f"{effect.text} is a conditional add on the "
                f"{'asymmetric' if fluent.asymmetric else 'functional'} pair fluent "
                f"{fluent.label}; its compiled literals would not be a plain precondition, so a "
                f"conditional add needs a general pair fluent that is not functional",
            )
        key = pair_effect_key(effect, fluent)
        if key in seen:
            raise ConfigError(
                source,
                where,
                f"contradicts an earlier effect on {effect.key}"
                if seen[key] != effect.value
                else f"writes {effect.key} twice"
                + (
                    " (both directions of a symmetric pair are one pair)"
                    if fluent.symmetric
                    else ""
                ),
            )
        reverse = ("pair", fluent.label, (effect.second, effect.role))
        if fluent.asymmetric and effect.value and seen.get(reverse) is True:
            raise ConfigError(
                source,
                where,
                f"{effect.text} adds the reverse of a pair the entry already adds on the "
                f"asymmetric pair fluent {fluent.label}; one event cannot add both directions",
            )
        seen[key] = effect.value

    def _check_pair_condition_against(
        self,
        source: str,
        where: str,
        effect: Effect,
        lit: PairLiteral,
        precondition: Sequence[AnyLiteral],
        word: str,
    ) -> None:
        """The rule of F5: a pair condition literal names no pair or row that the precondition
        names (compiled literals included)."""
        fluent = self.pair_fluents[lit.pair_fluent]
        names = pair_literal_names(lit, fluent)
        for other in precondition:
            if not isinstance(other, PairLiteral):
                continue
            shared = names & pair_literal_names(other, self.pair_fluents[other.pair_fluent])
            if shared:
                raise ConfigError(
                    source,
                    where,
                    f"the condition of {effect.text} reads {lit.key}, which names a "
                    f"{'row' if shared and next(iter(shared))[0] == 'row' else 'pair or column'} "
                    f"the precondition {word} names ({other.key})",
                )

    def compile_profile_literals(self) -> None:
        """Append the compiled literals (F4) to every event type with an add effect on a pair
        fluent; an explicit entry whose compiled literals contradict its own precondition is an
        error naming the file and the entry."""
        for draft in self.drafts.values():
            if not any(e.pair for e in draft.effects):
                continue
            before = draft.precondition
            keys = {
                pair_literal_key(lit, self.pair_fluents[lit.pair_fluent]): lit
                for lit in before
                if isinstance(lit, PairLiteral)
            }
            for effect in draft.effects:
                if not effect.pair or effect.condition or not effect.value:
                    continue
                fluent = self.pair_fluents[effect.fluent]
                for literal in compiled_literals(effect, fluent):
                    other = keys.get(pair_literal_key(literal, fluent))
                    if other is not None and other.value != literal.value:
                        if self.event_file is None or draft.explicit is None:
                            raise WorldError(
                                f"{draft.event_type.label}: the effect {effect.text} needs the "
                                f"compiled literal {literal.text}, which contradicts the "
                                f"precondition literal {other.text} after the fix-ups"
                            )
                        raise ConfigError(
                            self.event_file.source,
                            draft.explicit.field,
                            f"the effect {effect.text} needs the compiled literal {literal.text} "
                            f"to keep the profile of {fluent.label}, which contradicts the "
                            f"precondition literal {other.text}: the event type could never "
                            f"happen",
                        )
            draft.precondition = add_compiled_literals(before, draft.effects, self.pair_fluents)
            if draft.explicit is not None and draft.explicit.effects is not None:
                for effect in draft.effects:
                    for lit in effect.condition:
                        if isinstance(lit, PairLiteral):
                            self._check_pair_condition_against(
                                self.event_file.source,  # type: ignore[union-attr]
                                f"{draft.explicit.field}.effects",
                                effect,
                                lit,
                                draft.precondition,
                                "or its compiled literals",
                            )

    def _check_threshold(
        self, source: str, where: str, fluent: str, threshold: float | None
    ) -> None:
        low, high = self.numeric[fluent].range
        if threshold is None or not low < threshold < high:
            raise ConfigError(
                source,
                where,
                f"the threshold {threshold!r} on {fluent} is not strictly inside its range "
                f"[{low}, {high}]",
            )

    def _check_role(self, source: str, where: str, et: EventType, role: str, key: str) -> None:
        if role not in et.roles:
            raise ConfigError(
                source,
                where,
                f"{key} names the role {role}, but {et.label} has roles {', '.join(et.roles)}",
            )

    def check_explicit_achievable(self) -> None:
        if self.event_file is None:
            return
        achievable = self.achievable()
        for label, entry in self.event_file.event_types.items():
            for lit in entry.precondition or ():
                if not self.literal_achievable(lit, achievable):
                    raise ConfigError(
                        self.event_file.source,
                        f"{entry.field}.precondition",
                        f"{lit.text} is not achievable: no entity starts with that value and no "
                        f"effect or drift produces it",
                    )
            draft = self.drafts[label]
            for i, effect in enumerate(entry.effects or ()):
                where = f"{entry.field}.effects[{i}]"
                # WM.68: over the precondition with its compiled literals
                problem = fixed_condition_problem(
                    effect, draft.precondition, self.pair_fluents, draft.event_type.roles
                )
                if problem is not None:
                    raise ConfigError(self.event_file.source, where, problem)
                for lit in effect.condition:
                    if not self._condition_literal_achievable(lit, achievable):
                        raise ConfigError(
                            self.event_file.source,
                            where,
                            f"the condition literal {lit.text} is not achievable: no entity "
                            f"starts with that value and no effect or drift produces it",
                        )
                    if isinstance(lit, PairLiteral):
                        continue
                    if lit.kind != "feature" and (lit.role, lit.symbol) in {
                        (p.role, p.fluent) for p in draft.precondition if not p.pair
                    }:
                        raise ConfigError(
                            self.event_file.source,
                            where,
                            f"the condition reads {lit.key}, which the precondition fixes",
                        )
                if effect.condition and not self._static_satisfiable(
                    draft.event_type, effect.condition
                ):
                    raise ConfigError(
                        self.event_file.source,
                        where,
                        f"no able binding of {label} satisfies the condition "
                        f"{' AND '.join(lit.text for lit in effect.condition)}",
                    )

    def _condition_literal_achievable(
        self, lit: ConditionLiteral | PairLiteral, achievable: set[tuple[str, bool]]
    ) -> bool:
        if isinstance(lit, PairLiteral):
            return self.literal_achievable(lit, achievable)
        if lit.kind == "feature":
            return True
        if lit.numeric:
            assert lit.threshold is not None
            return self.numeric_achievable(lit.symbol, lit.value, lit.threshold)
        return (lit.symbol, lit.value) in achievable

    # Conditions ------------------------------------------------------------------------------

    def able_bindings(self, et: EventType) -> np.ndarray:
        """The event type's requirement over the entities (a bool vector) or over the ordered
        pairs (a bool matrix with a false diagonal), from the static side."""
        if et.label not in self._able:
            assert self.statics is not None
            if et.arity == 1:
                position = self.statics.features[et.label].position
                self._able[et.label] = self.statics.values[:, position].astype(bool)
            else:
                assert self.statics.relations is not None
                self._able[et.label] = self.statics.relations.matrix(et.label)
        return self._able[et.label]

    def _static_satisfiable(self, et: EventType, condition: tuple[ConditionLiteral, ...]) -> bool:
        """Whether some able binding of the event type satisfies the condition's static
        literals. An event type with no able binding at all (a never-able one) has no event, so
        its conditions are vacuously satisfiable: a feature's effect must not be denied a
        condition because one event type of the branch can never occur."""
        if self.statics is None:
            return True
        able = self.able_bindings(et)
        if not able.any():
            return True
        values = self.statics.values
        n = values.shape[0]
        masks = {role: np.ones(n, dtype=bool) for role in et.roles}
        for lit in condition:
            if lit.kind != "feature":
                continue
            column = values[:, self.statics.features[lit.symbol].position]
            masks[lit.role] &= column == int(lit.value)
        if et.arity == 1:
            return bool((able & masks["agent"]).any())
        return bool(able[np.ix_(masks["agent"], masks["patient"])].any())

    def _condition_ok(
        self,
        effect: Effect,
        condition: tuple[ConditionLiteral, ...],
        holders: Sequence[_Draft],
        achievable: set[tuple[str, bool]],
    ) -> bool:
        """The rules of "Conditional effects" (as built): no symbol twice; a fluent literal is
        achievable, names no fluent of a role that a holder's precondition names (the same value
        would make the condition redundant, the other would make it never fire), and does not
        name the fluent the effect sets (with the effect's value the effect would change
        nothing, and with the other value the fluent has the effect's value after the event
        either way, so the condition would be a condition in name only); and some able binding
        of every holder satisfies the static literals. A pair literal (stage c2b, F5) is
        achievable, names no pair or row a holder's precondition names (its compiled literals
        included), is not fixed by a holder's precondition under ``fixed_pair_literals``, and
        is not the pair the effect itself writes."""
        keys = {
            pair_literal_key(lit, self.pair_fluents[lit.pair_fluent])
            if isinstance(lit, PairLiteral)
            else (lit.role, lit.symbol)
            for lit in condition
        }
        if len(keys) < len(condition):
            self.pair_stats_count("pair_conditions_rejected_other", condition)
            return False
        for lit in condition:
            if isinstance(lit, PairLiteral):
                if not self._pair_condition_ok(effect, lit, holders):
                    return False
                continue
            if lit.kind == "feature":
                continue
            if not self._condition_literal_achievable(lit, achievable):
                return False
            # A Boolean condition on the fluent the effect sets would be a condition in name
            # only; a numeric condition on the effect's own numeric fluent ("when low, add") is
            # a real condition, and allowed.
            if not lit.numeric and (lit.role, lit.symbol) == (effect.role, effect.fluent):
                return False
            for draft in holders:
                if any(
                    not p.pair and (p.role, p.fluent) == (lit.role, lit.symbol)
                    for p in draft.precondition
                ):
                    return False
        return all(self._static_satisfiable(d.event_type, condition) for d in holders)

    def _pair_kind_drawable(self, roles: tuple[str, ...]) -> bool:
        """Whether a pair literal can be drawn over the roles at all (a one-place event type
        needs the ``some`` kind with a positive weight)."""
        kinds = self.pair_config.literal_kinds
        if len(roles) == 1:
            return kinds.get("some", 0) > 0
        return any(w > 0 for w in kinds.values())

    def pair_stats_count(self, key: str, condition: Sequence[AnyLiteral]) -> None:
        """Count a rejected condition under ``key`` when it holds a pair literal."""
        if any(isinstance(lit, PairLiteral) for lit in condition) and self.pair_stats:
            self.pair_stats[key] += 1

    def _pair_condition_ok(
        self, effect: Effect, lit: PairLiteral, holders: Sequence[_Draft]
    ) -> bool:
        """The rules of a drawn pair condition literal (F5, and item 6 of the stage prompt)."""
        fluent = self.pair_fluents[lit.pair_fluent]
        adds, deletes = self.pair_adds_and_deletes()
        key = pair_literal_key(lit, fluent)
        if not pair_literal_achievable(lit, fluent, adds, deletes) or (
            effect.pair and key == pair_effect_key(effect, self.pair_fluents[effect.fluent])
        ):
            self.pair_stats["pair_conditions_rejected_other"] += 1
            return False
        names = pair_literal_names(lit, fluent)
        for draft in holders:
            for other in draft.precondition:
                if not isinstance(other, PairLiteral):
                    continue
                if names & pair_literal_names(other, self.pair_fluents[other.pair_fluent]):
                    self.pair_stats["pair_conditions_rejected_named"] += 1
                    return False
            if key in fixed_pair_literals(draft.precondition, self.pair_fluents, draft.roles):
                self.pair_stats["pair_conditions_rejected_fixed"] += 1
                return False
        return True

    def _draw_condition(
        self,
        effect: Effect,
        roles: tuple[str, ...],
        holders: Sequence[_Draft],
        achievable: set[tuple[str, bool]],
    ) -> tuple[ConditionLiteral, ...] | None:
        """A condition for the effect, satisfiable for every holder, or None after ``MAX_TRIES``
        rejected draws. The number of literals comes from ``condition_literals``; each literal's
        role is drawn uniformly among ``roles``, its kind by the weights ``condition_kinds`` over
        a static feature and a fluent (the pools differ in size, and both kinds of condition are
        wanted; Jon's ruling 1 on stage b1), its symbol uniformly within the kind, and its value
        uniformly."""
        rng = self.rng_conditions
        pools = {
            "feature": self.feature_labels,
            "fluent": self.all_labels,
            "numeric": self.numeric_labels,
            # a pair condition reads a pair fluent between entities (a location fluent is read
            # in preconditions alone, stage c3b)
            "pair": tuple(f.label for f in self.pair_fluents.between_entities),
        }
        weights = {
            kind: weight
            for kind, weight in self.effects_config.condition_kinds.items()
            if weight > 0
            and pools.get(kind)
            and (kind != "pair" or self._pair_kind_drawable(roles))
        }
        # The even split between features and fluents is drawn with rng.integers(2), as stage b1
        # drew it: a weighted draw (rng.choice with probabilities) reads the stream differently,
        # so using it for the even split too would change every condition of the existing
        # worlds. The weighted draw serves the other splits, and the numeric kind (stage b2).
        even = set(weights) == {"feature", "fluent"} and weights["feature"] == weights["fluent"]
        numeric_rng = self.rng_numeric_conditions
        fractions = self.preconditions_config.numeric_thresholds
        for _ in range(MAX_TRIES):
            count = _draw(rng, self.effects_config.condition_literals)
            literals = []
            for _ in range(count):
                role = roles[0] if len(roles) == 1 else roles[int(rng.integers(len(roles)))]
                if not weights:
                    kind = "fluent"
                elif even:
                    kind = ("feature", "fluent")[int(rng.integers(2))]
                elif len(weights) == 1:
                    kind = next(iter(weights))
                else:
                    kind = _draw(rng, weights)
                if kind == "numeric":
                    symbol = self.numeric_labels[
                        int(numeric_rng.integers(len(self.numeric_labels)))
                    ]
                    above = bool(numeric_rng.integers(2) == 0)
                    fraction = fractions[int(numeric_rng.integers(len(fractions)))]
                    literals.append(
                        ConditionLiteral(role, symbol, above, self.numeric[symbol].point(fraction))
                    )
                    continue
                if kind == "pair":
                    # the pair literal draws its own role (stage c2b, "as for preconditions")
                    pair_literal = random_pair_literal(
                        roles,
                        self.rng_pair_conditions,
                        self.pair_fluents,
                        self.pair_config.literal_kinds,
                    )
                    assert pair_literal is not None  # the kind is drawn only when one can be
                    self.pair_stats["pair_conditions_drawn"] += 1
                    literals.append(pair_literal)
                    continue
                pool = pools[kind]
                symbol = pool[int(rng.integers(len(pool)))]
                value = bool(rng.integers(2))
                literals.append(ConditionLiteral(role, symbol, value))
            condition = tuple(literals)
            if self._condition_ok(effect, condition, holders, achievable):  # type: ignore[arg-type]
                return condition  # type: ignore[return-value]
            self.stats["conditions_redrawn"] += 1
        return None

    def draw_conditions(self) -> None:
        """Give a share ``conditional_share`` of the effects a condition, from ``world:conditions``:
        the effects of event-type features first, in feature order (an inherited effect brings
        its condition to every event type that kept it), then each event type's own effects in
        order. An explicit entry's effects are left as the file gives them. With a share of 0
        nothing is drawn, and every other draw is unchanged in any case, because the conditions
        have a stream of their own."""
        cfg = self.effects_config
        # The effects a condition is drawn for: the effects of event-type features, and the own
        # effects of every event type without an explicit entry (an inherited effect counts
        # once, for the feature; the statistics report the share of all effects beside it).
        self.condition_draws["drawn"] = sum(
            1
            for effect in self.feature_effects.values()
            if effect is not None and not keeps_no_condition(effect, self.pair_fluents)
        ) + sum(
            1
            for draft in self.drafts.values()
            if draft.explicit is None or draft.explicit.effects is None
            for effect in draft.effects
            if _effect_slot(effect, self.pair_fluents) in draft.own_effect_keys
            and not keeps_no_condition(effect, self.pair_fluents)
        )
        if self.pair_stats:
            self.pair_stats["adds_kept_unconditional"] = sum(
                1
                for effect in self.feature_effects.values()
                if effect is not None and keeps_no_condition(effect, self.pair_fluents)
            ) + sum(
                1
                for draft in self.drafts.values()
                if draft.explicit is None or draft.explicit.effects is None
                for effect in draft.effects
                if _effect_slot(effect, self.pair_fluents) in draft.own_effect_keys
                and keeps_no_condition(effect, self.pair_fluents)
            )
        if cfg.conditional_share <= 0 or not (self.base_labels or self.numeric_labels):
            return
        rng = self.rng_conditions
        achievable = self.achievable()
        for feature in self.static.features:
            effect = self.feature_effects.get(feature)
            if effect is None or keeps_no_condition(effect, self.pair_fluents):
                continue
            if rng.random() >= cfg.conditional_share:
                continue
            holders = [
                draft
                for draft in self.drafts.values()
                if feature in draft.event_type.features
                and effect in draft.effects
                and (draft.explicit is None or draft.explicit.effects is None)
            ]
            condition = self._draw_condition(effect, ROLES, holders, achievable)
            if condition is None:
                self.stats["conditions_dropped"] += 1
                continue
            conditioned = Effect(effect.role, effect.fluent, effect.value, condition, effect.second)
            self.feature_effects[feature] = conditioned
            self.condition_draws["conditional"] += 1
            for draft in holders:
                draft.effects = tuple(conditioned if e == effect else e for e in draft.effects)
        for draft in self.drafts.values():
            if draft.explicit is not None and draft.explicit.effects is not None:
                continue
            effects = list(draft.effects)
            for i, effect in enumerate(effects):
                if _effect_slot(effect, self.pair_fluents) not in draft.own_effect_keys:
                    continue
                if keeps_no_condition(effect, self.pair_fluents):
                    continue
                if rng.random() >= cfg.conditional_share:
                    continue
                condition = self._draw_condition(effect, draft.roles, [draft], achievable)
                if condition is None:
                    self.stats["conditions_dropped"] += 1
                    continue
                effects[i] = Effect(
                    effect.role, effect.fluent, effect.value, condition, effect.second
                )
                self.condition_draws["conditional"] += 1
            draft.effects = tuple(effects)

    # The quota same_rules (stage c3b) --------------------------------------------------------

    def apply_same_quota(self) -> None:
        """Give ``pair_fluents.same_rules`` two-place event types the precondition literal
        ``SAME.PAIRFL.k(agent, patient)`` on a location fluent (WM.52), drawn from the part
        ``world:pair_preconditions:same_rules``: the event types are drawn uniformly without
        replacement among the two-place leaves without an explicit precondition that have an
        able binding, lack such a literal, and whose other literals and conditions leave it
        free (it names no row a pair condition names, and neither implies nor contradicts
        another literal), in label order; each gets its location fluent drawn uniformly. The
        quota does nothing without location fluents, and records its shortfall when fewer
        event types qualify."""
        quota = self.pair_config.same_rules
        locations = list(self.pair_fluents.locations)
        if quota <= 0 or not locations:
            return
        rng = self.rng_same_rules
        candidates: list[_Draft] = []
        for draft in self.drafts.values():
            et = draft.event_type
            if et.arity != 2 or et.category:
                continue
            if draft.explicit is not None and draft.explicit.precondition is not None:
                continue
            if not self.able_bindings(et).any():
                continue
            if any(
                isinstance(lit, PairLiteral)
                and lit.kind == PAIR_SAME
                and self.pair_fluents[lit.pair_fluent].total
                for lit in draft.precondition
            ):
                continue
            probes = [
                PairLiteral(PAIR_SAME, f.label, et.roles[0], et.roles[1], True) for f in locations
            ]
            kept = list(draft.precondition) + self.virtual_compiled(draft)
            if all(
                precondition_literal_fits_conditions(
                    probe, draft.effects, self.pair_fluents, et.roles
                )
                and not implied_conflict(probe, kept, self.pair_fluents, et.roles)
                for probe in probes
            ):
                candidates.append(draft)
        count = min(quota, len(candidates))
        self.pair_stats["same_rules_short"] = quota - count
        if count == 0:
            return
        chosen = sorted(int(i) for i in rng.choice(len(candidates), size=count, replace=False))
        for index in chosen:
            draft = candidates[index]
            fluent = locations[int(rng.integers(len(locations)))]
            literal = PairLiteral(PAIR_SAME, fluent.label, draft.roles[0], draft.roles[1], True)
            key = _literal_slot(literal, self.pair_fluents)
            draft.precondition = draft.precondition + (literal,)
            draft.quota_literal_keys.add(key)
            self.same_quota[draft.event_type.label] = literal
            self.pair_stats["same_rules_added"] += 1

    # The whole procedure ----------------------------------------------------------------------

    def run(self) -> EventTypes:
        self.apply_event_file()
        self.draw_effects()
        for draft in self.drafts.values():
            self.resolve_effects(draft)
        self.draw_preconditions()
        for draft in self.drafts.values():
            self.resolve_preconditions(draft)
        self.apply_same_quota()
        for _ in range(MAX_FIX_ROUNDS):
            changed = False
            for draft in self.drafts.values():
                if draft.explicit is None or draft.explicit.effects is None:
                    changed |= self.fix_empty_effects(draft)
                    changed |= self.fix_compiled_contradictions(draft)
                    changed |= self.fix_dead_deletes(draft)
                    changed |= self.fix_dead_booleans(draft)
            achievable = self.achievable()
            produced = self.produced()
            for draft in self.drafts.values():
                if draft.explicit is None or draft.explicit.precondition is None:
                    changed |= self.fix_unachievable(draft, achievable, produced)
                    changed |= self.fix_implied_literals(draft, achievable, produced)
            if not changed:
                break
        self.compile_profile_literals()
        self.draw_conditions()
        self.check_explicit_achievable()
        event_types = tuple(
            EventType(
                label=d.event_type.label,
                arity=d.event_type.arity,
                category=d.event_type.category,
                parent=d.event_type.parent,
                level=d.event_type.level,
                features=d.event_type.features,
                constraints=d.event_type.constraints,
                precondition=d.precondition,
                effects=d.effects,
                explicit=d.explicit is not None,
                explicit_effects=d.explicit is not None and d.explicit.effects is not None,
            )
            for d in self.drafts.values()
        )
        return EventTypes(
            self.static.constraints,
            event_types,
            self.static.features,
            dict(self.static.feature_constraints),
            dict(self.feature_literals),
            dict(self.feature_effects),
            dict(self.stats),
            dict(self.condition_draws),
            dict(self.pair_stats),
            dict(self.same_quota),
        )


def generate_event_types(
    statics: StaticWorld,
    fluents: Fluents,
    config: Config,
    streams: WorldStreams,
    event_file: EventFile | None,
    derived_initial: Mapping[str, np.ndarray],
    numeric: NumericFluents | None = None,
    pair_fluents: PairFluents | None = None,
    extra_effects: Sequence[Effect] = (),
) -> EventTypes:
    """Event types with their requirements, preconditions, and effects. ``extra_effects`` are
    effects of the world that are not the entity event types' (stage c3b, WM.66: the event
    file's place entries'), which the pools of achievable values count."""
    static = build_event_types(statics)
    return _Dynamics(
        static,
        fluents,
        config,
        streams,
        event_file,
        derived_initial,
        statics,
        numeric,
        pair_fluents,
        extra_effects,
    ).run()


def _literal_slot(literal: AnyLiteral, pair_fluents: PairFluents | None = None) -> tuple:
    """The slot a precondition literal fills: ``(role, fluent)`` for a Boolean or numeric
    literal, and what the literal names for a pair literal (with the pair fluents, so that both
    directions of a symmetric pair are one slot; without them, the literal's key)."""
    if isinstance(literal, PairLiteral):
        if pair_fluents is not None and literal.pair_fluent in pair_fluents:
            return pair_literal_key(literal, pair_fluents[literal.pair_fluent])
        return ("pair_literal", literal.key)
    return (literal.role, literal.fluent)


def condition_keys(effects: Sequence[Effect]) -> set[tuple[str, str]]:
    """The ``(role, fluent)`` pairs that the conditions of the effects read (Boolean or
    numeric): a precondition literal on one of them would fix what the condition asks, so none
    is drawn there. A pair condition literal (stage c2a) names its pair fluent, which no Boolean
    literal can name."""
    return {
        (lit.role, lit.symbol)
        for effect in effects
        for lit in effect.condition
        if lit.kind != "feature"
    }


def produced_numeric(event_types: EventTypes) -> set[tuple[str, int]]:
    """``(fluent, +1 or -1)`` for every numeric effect of the event types."""
    return {
        (e.fluent, 1 if e.amount > 0 else -1)
        for et in event_types.event_types
        for e in et.effects
        if e.numeric
    }


def redraw_preconditions(
    event_types: EventTypes,
    labels: Sequence[str],
    fluents: Fluents,
    config: Config,
    parts: Mapping[str, np.random.Generator],
    derived_initial: Mapping[str, np.ndarray],
    numeric: NumericFluents | None = None,
    numeric_parts: Mapping[str, np.random.Generator] | None = None,
    pair_fluents: PairFluents | None = None,
    pair_parts: Mapping[str, np.random.Generator] | None = None,
    extra_effects: Sequence[Effect] = (),
) -> EventTypes:
    """The event types with the own precondition literals of the named event types drawn
    again, each from its own generator (the part ``world:preconditions:<label>``,
    ``world:numeric_preconditions:<label>`` for the numeric replacements, and, since stage
    c2b, ``world:pair_preconditions:<label>`` for the pair replacements at ``pair_share``), as
    ``define`` redraws an event type that was never legal in the statistics episodes. The
    inherited literals stay, except one that is no longer achievable (the generator's fix-ups
    dropped it from the event type when the effect that produced its value went), and so do the
    compiled literals; a literal that would make one of the event type's own effects empty,
    that duplicates an inherited or a compiled one, that fixes a fluent one of the event type's
    conditions reads, or (a pair literal) that names what a pair condition names or fixes a
    pair condition's value, is drawn again. No other event type changes. ``extra_effects`` (stage
    c3b, WM.66: the place event types' effects) widen the pool of achievable values; the
    ``SAME`` literal the quota ``same_rules`` gave stays, as the compiled literals do, and the
    part draws ``SAME`` on a location fluent at the share as the first draw did."""
    cfg = config.event_types.preconditions
    numeric = numeric if numeric is not None else empty_numeric(fluents.initial_values.shape[0])
    pair_fluents = pair_fluents if pair_fluents is not None else empty_pair_fluents()
    all_effects = [e for et in event_types.event_types for e in et.effects] + list(extra_effects)
    adds = {e.fluent for e in all_effects if e.pair and e.value}
    deletes = {e.fluent for e in all_effects if e.pair and not e.value}
    achievable: set[tuple[str, bool]] = set()
    for i, fluent in enumerate(fluents.base):
        for value in np.unique(fluents.initial_values[:, i]):
            achievable.add((fluent.label, bool(value)))
    for label, column in derived_initial.items():
        for value in np.unique(column):
            achievable.add((label, bool(value)))
    # the Boolean values effects produce; a pair effect produces no fluent value (stage c2b)
    produced = sorted({(e.fluent, e.value) for e in all_effects if not e.numeric and not e.pair})
    achievable |= set(produced)
    signs = {(e.fluent, 1 if e.amount > 0 else -1) for e in all_effects if e.numeric}

    def ok(literal: Literal) -> bool:
        if literal.numeric:
            assert literal.threshold is not None
            return numeric.achievable(literal.fluent, literal.value, literal.threshold, signs)
        return (literal.fluent, literal.value) in achievable

    feature_literals = event_types.feature_preconditions
    redrawn: dict[str, EventType] = {}
    for label in labels:
        et = event_types.event_type(label)
        rng = parts[label]
        numeric_rng = (numeric_parts or {}).get(label)
        pair_rng = (pair_parts or {}).get(label)
        inherited: dict[tuple, AnyLiteral] = {}
        for feature in event_types.features:
            literal = feature_literals.get(feature)
            if literal is None or feature not in et.features:
                continue
            if isinstance(literal, PairLiteral):
                fluent = pair_fluents[literal.pair_fluent]
                if pair_literal_achievable(literal, fluent, adds, deletes):
                    inherited.setdefault(_literal_slot(literal, pair_fluents), literal)
            elif ok(literal):
                inherited.setdefault((literal.role, literal.fluent), literal)
        empty = {(e.role, e.fluent, e.value) for e in et.effects if not e.numeric and not e.pair}
        empty_pairs = {(_effect_slot(e, pair_fluents), e.value) for e in et.effects if e.pair}
        read = condition_keys(et.effects)
        kept: dict[tuple, AnyLiteral] = dict(inherited)
        # The compiled literals of stage c2a (origin profile) stay with the event type, and so
        # does the quota's SAME literal (stage c3b).
        for lit in et.precondition:
            if isinstance(lit, PairLiteral) and lit.compiled:
                kept[_literal_slot(lit, pair_fluents)] = lit
        quota = event_types.same_quota.get(label)
        if quota is not None:
            kept[_literal_slot(quota, pair_fluents)] = quota

        pair_draw = (pair_fluents, config, et.roles, pair_rng)
        pair_sets = (adds, deletes, empty_pairs)
        n = _draw(rng, cfg.literals)
        for _ in range(n):
            for _ in range(MAX_TRIES):
                role = et.roles[0] if len(et.roles) == 1 else _draw(rng, cfg.roles)
                if not fluents.labels:
                    literal = _redraw_numeric(numeric, cfg, role, numeric_rng, kept, read, signs)
                    if literal is not None:
                        replacement = _redraw_pair(*pair_draw, kept, et.effects, *pair_sets)
                        if replacement is not None:
                            kept[_literal_slot(replacement, pair_fluents)] = replacement
                        else:
                            kept[(role, literal.fluent)] = literal
                        break
                    continue
                if produced and rng.random() < cfg.enabled_share:
                    fluent, value = produced[int(rng.integers(len(produced)))]
                else:
                    fluent = fluents.labels[int(rng.integers(len(fluents.labels)))]
                    value = bool(rng.integers(2))
                if (
                    (fluent, value) in achievable
                    and (role, fluent) not in kept
                    and (role, fluent) not in read
                    and (role, fluent, value) not in empty
                ):
                    literal = Literal(role, fluent, value)
                    if (
                        numeric_rng is not None
                        and len(numeric)
                        and cfg.numeric_share > 0
                        and numeric_rng.random() < cfg.numeric_share
                    ):
                        replacement = _redraw_numeric(
                            numeric, cfg, role, numeric_rng, kept, read, signs
                        )
                        if replacement is not None:
                            literal = replacement
                    pair_literal = _redraw_pair(*pair_draw, kept, et.effects, *pair_sets)
                    if pair_literal is not None:
                        kept[_literal_slot(pair_literal, pair_fluents)] = pair_literal
                    else:
                        kept[(role, literal.fluent)] = literal
                    break
        redrawn[label] = dataclasses.replace(et, precondition=tuple(kept.values()))
    return dataclasses.replace(
        event_types,
        event_types=tuple(redrawn.get(et.label, et) for et in event_types.event_types),
    )


def _redraw_pair(
    pair_fluents: PairFluents,
    config: Config,
    roles: tuple[str, ...],
    rng: np.random.Generator | None,
    kept: Mapping[tuple, AnyLiteral],
    effects: tuple[Effect, ...],
    adds: set[str],
    deletes: set[str],
    empty_pairs: set[tuple],
) -> PairLiteral | None:
    """A pair literal for a redraw at ``pair_share`` from the event type's part of
    ``world:pair_preconditions``, or None: achievable, in a free slot, leaving the conditions
    as conditions, and not making an own pair effect empty."""
    cfg = config.event_types.preconditions
    if rng is None or not len(pair_fluents) or cfg.pair_share <= 0:
        return None
    if rng.random() >= cfg.pair_share:
        return None
    for _ in range(MAX_TRIES):
        candidate = random_pair_literal(
            roles, rng, pair_fluents, config.pair_fluents.literal_kinds, rng
        )
        if candidate is None:
            return None
        fluent = pair_fluents[candidate.pair_fluent]
        slot = pair_literal_key(candidate, fluent)
        if (
            slot not in kept
            and pair_literal_achievable(candidate, fluent, adds, deletes)
            and precondition_literal_fits_conditions(candidate, effects, pair_fluents, roles)
            and (slot, candidate.value) not in empty_pairs
            and not implied_conflict(candidate, list(kept.values()), pair_fluents, roles)
        ):
            return candidate
    return None


def _redraw_numeric(
    numeric: NumericFluents,
    cfg: PreconditionsConfig,
    role: str,
    rng: np.random.Generator | None,
    kept: Mapping[tuple[str, str], Literal],
    read: set[tuple[str, str]],
    signs: set[tuple[str, int]],
) -> Literal | None:
    """A numeric literal for a redraw, achievable and on a fluent the event type's other
    literals and conditions leave free, or None."""
    if rng is None or not len(numeric):
        return None
    for _ in range(MAX_TRIES):
        fluent = numeric.labels[int(rng.integers(len(numeric.labels)))]
        above = bool(rng.integers(2) == 0)
        fraction = cfg.numeric_thresholds[int(rng.integers(len(cfg.numeric_thresholds)))]
        threshold = numeric[fluent].point(fraction)
        if (
            (role, fluent) not in kept
            and (role, fluent) not in read
            and numeric.achievable(fluent, above, threshold, signs)
        ):
            return Literal(role, fluent, above, threshold)
    return None


def check_dynamics(
    event_types: EventTypes,
    fluents: Fluents,
    initial_present: set[tuple[str, bool]],
    features: Sequence[str] | None = None,
    numeric: NumericFluents | None = None,
    pair_fluents: PairFluents | None = None,
    extra_effects: Sequence[Effect] = (),
) -> None:
    """The invariants of "Preconditions", "Effects", and "Conditional effects": no effect on a
    derived fluent, no contradiction (whatever the conditions), no empty effect, every literal
    achievable, every role valid; every condition literal a known fluent or static feature (when
    ``features`` lists them) of a role of the event type, no symbol twice in a condition, no
    fluent literal on a fluent the precondition fixes or on the fluent the effect sets, no pair
    condition literal whose value the precondition fixes under ``fixed_pair_literals`` (WM.68),
    every fluent literal achievable, and (Jon's ruling WM.70, stage c4a) no Boolean effect
    ``:= v`` on a fluent that can never be ``not v``, among the event types whose effects the
    generator drew (an event file's effects are what the file says, as the dead-delete rule
    leaves them). Raises :class:`ValueError` naming the event type otherwise. Tests and the
    generator both call it. Whether some able binding satisfies the
    static literals needs the static side; the generator checks it when it draws. A literal's
    achievability counts every effect of the world (stage c3b, WM.66): ``extra_effects`` are
    the effects of event types outside ``event_types``, the place event types'."""
    all_effects = [e for et in event_types.event_types for e in et.effects] + list(extra_effects)
    produced = {(e.fluent, e.value) for e in all_effects if not e.numeric and not e.pair}
    achievable = produced | initial_present
    signs = {(e.fluent, 1 if e.amount > 0 else -1) for e in all_effects if e.numeric}
    numeric = numeric if numeric is not None else empty_numeric(0)
    pair_fluents = pair_fluents if pair_fluents is not None else empty_pair_fluents()
    adds = {e.fluent for e in all_effects if e.pair and e.value}
    deletes = {e.fluent for e in all_effects if e.pair and not e.value}
    dead = dead_boolean_values(fluents, all_effects)

    def numeric_ok(fluent: str, above: bool, threshold: float | None) -> bool:
        return (
            fluent in numeric
            and threshold is not None
            and numeric.achievable(fluent, above, threshold, signs)
        )

    def pair_ok(lit: PairLiteral) -> bool:
        return lit.pair_fluent in pair_fluents and pair_literal_achievable(
            lit, pair_fluents[lit.pair_fluent], adds, deletes
        )

    for et in event_types.event_types:
        effect_keys: dict[tuple, bool | float] = {}
        precondition_keys = {(lit.role, lit.fluent) for lit in et.precondition if not lit.pair}
        pair_precondition: dict[tuple, PairLiteral] = {}
        pair_names: set[tuple] = set()
        for lit in et.precondition:
            if not isinstance(lit, PairLiteral):
                continue
            if lit.pair_fluent not in pair_fluents:
                raise ValueError(f"{et.label}: the literal {lit.text} reads an unknown pair fluent")
            fluent = pair_fluents[lit.pair_fluent]
            for role in lit.roles:
                if role not in et.roles:
                    raise ValueError(f"{et.label}: the literal {lit.text} uses a role it has not")
            if lit.kind == PAIR_SAME and not fluent.functional:
                raise ValueError(
                    f"{et.label}: the literal {lit.text} reads SAME on a pair fluent that is not "
                    f"functional"
                )
            key = pair_literal_key(lit, fluent)
            if key in pair_precondition:
                raise ValueError(
                    f"{et.label}: the precondition reads {lit.key} twice"
                    if pair_precondition[key].value == lit.value
                    else f"{et.label}: the precondition holds {lit.key} and its negation"
                )
            pair_precondition[key] = lit
            pair_names |= pair_literal_names(lit, fluent)
            if not pair_ok(lit):
                raise ValueError(f"{et.label}: the literal {lit.text} is not achievable")
        for effect in et.effects:
            if effect.role not in et.roles:
                raise ValueError(f"{et.label}: the effect {effect.text} uses a role it has not")
            if effect.pair:
                _check_pair_effect(
                    et, effect, pair_fluents, effect_keys, pair_precondition, pair_names, pair_ok
                )
                continue
            if effect.numeric:
                if effect.fluent not in numeric:
                    raise ValueError(
                        f"{et.label}: the effect {effect.text} writes an unknown numeric fluent"
                    )
                if effect.amount == 0:
                    raise ValueError(f"{et.label}: the effect {effect.text} adds nothing")
            elif fluents[effect.fluent].derived:
                raise ValueError(f"{et.label}: the effect {effect.text} writes a derived fluent")
            elif not et.explicit_effects and (effect.fluent, bool(effect.value)) in dead:
                raise ValueError(
                    f"{et.label}: the effect {effect.text} can never change the state: the "
                    f"initial rate of {effect.fluent} makes every entity "
                    f"{'true' if effect.value else 'false'}, and no effect sets it "
                    f"{'false' if effect.value else 'true'} (WM.70)"
                )
            key = (effect.role, effect.fluent)
            if key in effect_keys:
                raise ValueError(f"{et.label}: two effects write {effect.key}")
            effect_keys[key] = effect.value
            seen: set[tuple] = set()
            for lit in effect.condition:
                if isinstance(lit, PairLiteral):
                    _check_pair_condition_literal(
                        et, effect, lit, pair_fluents, pair_names, seen, pair_ok
                    )
                    continue
                if lit.role not in et.roles:
                    raise ValueError(
                        f"{et.label}: the condition of {effect.text} uses a role it has not"
                    )
                if lit.numeric:
                    if not numeric_ok(lit.symbol, lit.value, lit.threshold):
                        raise ValueError(
                            f"{et.label}: the condition literal {lit.text} is unknown or not "
                            f"achievable"
                        )
                    if (lit.role, lit.symbol) in precondition_keys:
                        raise ValueError(
                            f"{et.label}: the condition of {effect.text} reads {lit.key}, which "
                            f"the precondition fixes"
                        )
                elif lit.kind == "fluent":
                    if lit.symbol not in fluents:
                        raise ValueError(
                            f"{et.label}: the condition of {effect.text} reads an unknown fluent"
                        )
                    if (lit.symbol, lit.value) not in achievable:
                        raise ValueError(
                            f"{et.label}: the condition literal {lit.text} is not achievable"
                        )
                    if (lit.role, lit.symbol) in precondition_keys:
                        raise ValueError(
                            f"{et.label}: the condition of {effect.text} reads {lit.key}, which "
                            f"the precondition fixes"
                        )
                    if (lit.role, lit.symbol) == key:
                        raise ValueError(
                            f"{et.label}: the condition of {effect.text} reads the fluent the "
                            f"effect sets"
                        )
                elif features is not None and lit.symbol not in features:
                    raise ValueError(
                        f"{et.label}: the condition of {effect.text} reads an unknown feature"
                    )
                if (lit.role, lit.symbol) in seen:
                    raise ValueError(
                        f"{et.label}: the condition of {effect.text} reads {lit.key} twice"
                    )
                seen.add((lit.role, lit.symbol))
            problem = fixed_condition_problem(effect, et.precondition, pair_fluents, et.roles)
            if problem is not None:
                raise ValueError(f"{et.label}: {problem}")
        literal_keys: dict[tuple[str, str], bool] = {}
        for lit in et.precondition:
            if lit.pair:
                continue
            if lit.role not in et.roles:
                raise ValueError(f"{et.label}: the literal {lit.text} uses a role it has not")
            key = (lit.role, lit.fluent)
            if key in literal_keys:
                raise ValueError(f"{et.label}: the precondition reads {lit.key} twice")
            if lit.numeric:
                if not numeric_ok(lit.fluent, lit.value, lit.threshold):
                    raise ValueError(
                        f"{et.label}: the literal {lit.text} is unknown or not achievable"
                    )
                continue
            literal_keys[key] = lit.value
            if (lit.fluent, lit.value) not in achievable:
                raise ValueError(f"{et.label}: the literal {lit.text} is not achievable")
        for key, value in effect_keys.items():
            if key[0] == "pair":
                continue
            if not key[1].startswith("NUMFL.") and literal_keys.get(key) == value:
                raise ValueError(f"{et.label}: the effect on {key[0]}.{key[1]} changes nothing")


def _check_pair_effect(
    et: EventType,
    effect: Effect,
    pair_fluents: PairFluents,
    effect_keys: dict[tuple, bool | float],
    pair_precondition: Mapping[tuple, PairLiteral],
    pair_names: set[tuple],
    pair_ok,
) -> None:
    """The invariants of a pair effect (stage c2a): a known pair fluent, roles of the event
    type, no delete on a total pair fluent, no second effect on the same pair (both directions
    of a symmetric one), no add whose precondition requires the pair and no delete whose
    precondition requires its absence, no conditional add on an asymmetric or functional pair
    fluent, the compiled literals of an add present in the precondition, and every pair
    condition literal well formed."""
    if effect.fluent not in pair_fluents:
        raise ValueError(f"{et.label}: the effect {effect.text} writes an unknown pair fluent")
    fluent = pair_fluents[effect.fluent]
    assert effect.second is not None
    if effect.second not in et.roles:
        raise ValueError(f"{et.label}: the effect {effect.text} uses a role it has not")
    if fluent.total:
        raise ValueError(
            f"{et.label}: the effect {effect.text} writes a location fluent, which moves alone "
            f"write"
            if effect.value
            else f"{et.label}: the effect {effect.text} deletes a pair of a location fluent"
        )
    key = pair_effect_key(effect, fluent)
    if key in effect_keys:
        raise ValueError(f"{et.label}: two effects write {effect.key}")
    reverse = ("pair", fluent.label, (effect.second, effect.role))
    if fluent.asymmetric and effect.value and effect_keys.get(reverse) is True:
        raise ValueError(
            f"{et.label}: the effect {effect.text} adds the reverse of a pair the event type "
            f"adds on the asymmetric pair fluent {fluent.label}"
        )
    effect_keys[key] = effect.value
    required = pair_precondition.get(key)
    if required is not None and required.value == effect.value:
        raise ValueError(f"{et.label}: the effect {effect.text} changes nothing")
    if effect.condition and effect.value and (fluent.asymmetric or fluent.functional):
        raise ValueError(
            f"{et.label}: the effect {effect.text} is a conditional add on an asymmetric or "
            f"functional pair fluent"
        )
    if not effect.condition:
        for literal in compiled_literals(effect, fluent):
            present = pair_precondition.get(pair_literal_key(literal, fluent))
            if present is None or present.value != literal.value:
                raise ValueError(
                    f"{et.label}: the effect {effect.text} lacks its compiled literal "
                    f"{literal.text}"
                )
    seen: set[tuple] = set()
    for lit in effect.condition:
        if isinstance(lit, PairLiteral):
            _check_pair_condition_literal(et, effect, lit, pair_fluents, pair_names, seen, pair_ok)
        elif lit.role not in et.roles:
            raise ValueError(f"{et.label}: the condition of {effect.text} uses a role it has not")


def _check_pair_condition_literal(
    et: EventType,
    effect: Effect,
    lit: PairLiteral,
    pair_fluents: PairFluents,
    pair_names: set[tuple],
    seen: set[tuple],
    pair_ok,
) -> None:
    """The invariants of a pair condition literal (F5): a known pair fluent, roles of the event
    type, ``SAME`` on a functional pair fluent, achievable, no literal twice, naming no pair or
    row the precondition names, and not the pair the effect itself writes."""
    if lit.pair_fluent not in pair_fluents:
        raise ValueError(f"{et.label}: the condition of {effect.text} reads an unknown pair fluent")
    fluent = pair_fluents[lit.pair_fluent]
    for role in lit.roles:
        if role not in et.roles:
            raise ValueError(f"{et.label}: the condition of {effect.text} uses a role it has not")
    if lit.kind == PAIR_SAME and not fluent.functional:
        raise ValueError(
            f"{et.label}: the condition literal {lit.text} reads SAME on a pair fluent that is "
            f"not functional"
        )
    if not pair_ok(lit):
        raise ValueError(f"{et.label}: the condition literal {lit.text} is not achievable")
    key = pair_literal_key(lit, fluent)
    if key in seen:
        raise ValueError(f"{et.label}: the condition of {effect.text} reads {lit.key} twice")
    seen.add(key)
    if pair_literal_names(lit, fluent) & pair_names:
        raise ValueError(
            f"{et.label}: the condition of {effect.text} reads {lit.key}, which names a pair or "
            f"a row the precondition names"
        )
    if effect.pair and key == pair_effect_key(effect, pair_fluents[effect.fluent]):
        raise ValueError(
            f"{et.label}: the condition of {effect.text} reads the pair the effect writes"
        )


__all__ = [
    "CONSTRAINT_PREFIX",
    "add_compiled_literals",
    "compiled_literals",
    "condition_pair_keys",
    "condition_pair_names",
    "dead_boolean_values",
    "fixed_condition_problem",
    "fixed_pair_literals",
    "implied_conflict",
    "keeps_no_condition",
    "precondition_literal_fits_conditions",
    "random_pair_effect",
    "random_pair_literal",
    "reverse_add_conflict",
    "pair_effect_key",
    "pair_literal_achievable",
    "pair_literal_key",
    "pair_literal_names",
    "FEATURE_PREFIX",
    "ONE_PLACE",
    "REQUIREMENT_PREFIX",
    "TWO_PLACE",
    "ConstraintSpec",
    "EventType",
    "EventTypes",
    "build_event_types",
    "check_dynamics",
    "condition_keys",
    "constraint_from_relation",
    "constraint_from_rule",
    "generate_event_types",
    "prefixed",
    "produced_numeric",
    "redraw_preconditions",
]
