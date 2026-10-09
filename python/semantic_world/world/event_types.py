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
every literal achievable). An event file replaces the sampled dynamics of the event types it
names.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semantic_world.common.boolean import TruthTable, dnf_literal_count
from semantic_world.taxonomy.config import ConfigError
from semantic_world.taxonomy.expressions import Cmp, Const, Expr, Gt, Not, Op, Var
from semantic_world.taxonomy.rules import MAX_TRIES, Rule, _draw, input_key
from semantic_world.world.config import Config, EffectsConfig, PreconditionsConfig
from semantic_world.world.constraints import Constraint
from semantic_world.world.dynamics import ROLES, Effect, Literal, precondition_text
from semantic_world.world.event_file import EventFile, ExplicitEventType
from semantic_world.world.fluents import Fluents
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
    precondition: tuple[Literal, ...] = ()
    effects: tuple[Effect, ...] = ()
    explicit: bool = False

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
# Preconditions and effects
# ---------------------------------------------------------------------------------------------


@dataclass
class _Draft:
    event_type: EventType
    inherited_effects: list[tuple[int, Effect]] = field(default_factory=list)
    own_effects: list[Effect] = field(default_factory=list)
    inherited_literals: list[tuple[int, Literal]] = field(default_factory=list)
    own_literals: list[Literal] = field(default_factory=list)
    explicit: ExplicitEventType | None = None
    effects: tuple[Effect, ...] = ()
    precondition: tuple[Literal, ...] = ()
    own_effect_keys: set[tuple[str, str]] = field(default_factory=set)
    own_literal_keys: set[tuple[str, str]] = field(default_factory=set)

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
    ) -> None:
        self.static = event_types
        self.fluents = fluents
        self.effects_config: EffectsConfig = config.event_types.effects
        self.preconditions_config: PreconditionsConfig = config.event_types.preconditions
        self.rng_effects = streams.effects
        self.rng_preconditions = streams.preconditions
        self.event_file = event_file
        self.base_labels = fluents.base_labels
        self.all_labels = fluents.labels
        self.stats: dict[str, int] = {
            "effects_dropped_contradiction": 0,
            "effects_dropped_duplicate": 0,
            "effects_dropped_empty": 0,
            "effects_dropped_no_slot": 0,
            "effects_redrawn": 0,
            "literals_dropped_contradiction": 0,
            "literals_dropped_duplicate": 0,
            "literals_dropped_unachievable": 0,
            "literals_redrawn": 0,
        }
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

    def _draw_effect(
        self, roles: tuple[str, ...], forbidden: set[tuple[str, str]] = frozenset()
    ) -> Effect | None:  # type: ignore[assignment]
        rng = self.rng_effects
        for _ in range(MAX_TRIES):
            role = self._role(rng, roles, self.effects_config.roles)
            fluent = self.base_labels[int(rng.integers(len(self.base_labels)))]
            value = bool(rng.integers(2))
            if (role, fluent) not in forbidden:
                return Effect(role, fluent, value)
        return None

    def _draw_literal(
        self,
        roles: tuple[str, ...],
        achievable: set[tuple[str, bool]],
        produced: list[tuple[str, bool]],
        forbidden: set[tuple[str, str]] = frozenset(),  # type: ignore[assignment]
    ) -> Literal | None:
        rng = self.rng_preconditions
        if not achievable:
            return None
        for _ in range(MAX_TRIES):
            role = self._role(rng, roles, self.preconditions_config.roles)
            if produced and rng.random() < self.preconditions_config.enabled_share:
                fluent, value = produced[int(rng.integers(len(produced)))]
            else:
                fluent = self.all_labels[int(rng.integers(len(self.all_labels)))]
                value = bool(rng.integers(2))
            if (fluent, value) in achievable and (role, fluent) not in forbidden:
                return Literal(role, fluent, value)
        return None

    # Effects ----------------------------------------------------------------------------------

    def draw_effects(self) -> None:
        if not self.base_labels:
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
        kept: dict[tuple[str, str], Effect] = {}
        for _, effect in draft.inherited_effects:
            key = (effect.role, effect.fluent)
            if key in kept:
                if kept[key].value != effect.value:
                    self.stats["effects_dropped_contradiction"] += 1
                else:
                    self.stats["effects_dropped_duplicate"] += 1
                continue
            kept[key] = effect
        draft.own_effect_keys = set()
        for effect in draft.own_effects:
            candidate: Effect | None = effect
            while candidate is not None and (candidate.role, candidate.fluent) in kept:
                self.stats["effects_redrawn"] += 1
                candidate = self._draw_effect(draft.roles, set(kept))
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[(candidate.role, candidate.fluent)] = candidate
            draft.own_effect_keys.add((candidate.role, candidate.fluent))
        draft.effects = tuple(kept.values())

    # Preconditions ----------------------------------------------------------------------------

    def produced(self) -> list[tuple[str, bool]]:
        values = {(e.fluent, e.value) for draft in self.drafts.values() for e in draft.effects}
        return sorted(values)

    def achievable(self) -> set[tuple[str, bool]]:
        return set(self.produced()) | self.initial_values

    def draw_preconditions(self) -> None:
        if not self.all_labels:
            return
        cfg = self.preconditions_config
        achievable = self.achievable()
        produced = self.produced()
        for et in self.static.unary:
            n = _draw(self.rng_preconditions, cfg.literals)
            draft = self.drafts[et.label]
            for _ in range(n):
                literal = self._draw_literal(et.roles, achievable, produced)
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
                literal = self._draw_literal(et.roles, achievable, produced)
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
        kept: dict[tuple[str, str], Literal] = {}
        for _, literal in draft.inherited_literals:
            key = (literal.role, literal.fluent)
            if key in kept:
                if kept[key].value != literal.value:
                    self.stats["literals_dropped_contradiction"] += 1
                else:
                    self.stats["literals_dropped_duplicate"] += 1
                continue
            kept[key] = literal
        draft.own_literal_keys = set()
        for literal in draft.own_literals:
            candidate: Literal | None = literal
            while candidate is not None and (candidate.role, candidate.fluent) in kept:
                self.stats["literals_redrawn"] += 1
                candidate = self._draw_literal(draft.roles, achievable, produced, set(kept))
            if candidate is None:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            kept[(candidate.role, candidate.fluent)] = candidate
            draft.own_literal_keys.add((candidate.role, candidate.fluent))
        draft.precondition = tuple(kept.values())

    # Fix-ups -----------------------------------------------------------------------------------

    def fix_empty_effects(self, draft: _Draft) -> bool:
        """An effect that sets a fluent to the value the precondition already requires changes
        nothing: an own effect is redrawn, an inherited one dropped. Returns whether anything
        changed."""
        required = {(lit.role, lit.fluent): lit.value for lit in draft.precondition}
        changed = False
        kept: dict[tuple[str, str], Effect] = {(e.role, e.fluent): e for e in draft.effects}
        for effect in list(draft.effects):
            key = (effect.role, effect.fluent)
            if required.get(key) != effect.value:
                continue
            changed = True
            del kept[key]
            if key not in draft.own_effect_keys:
                self.stats["effects_dropped_empty"] += 1
                continue
            draft.own_effect_keys.discard(key)
            candidate = None
            for _ in range(MAX_TRIES):
                candidate = self._draw_effect(draft.roles, set(kept))
                if (
                    candidate is None
                    or required.get((candidate.role, candidate.fluent)) != candidate.value
                ):
                    break
                candidate = None
            self.stats["effects_redrawn"] += 1
            if candidate is None:
                self.stats["effects_dropped_no_slot"] += 1
                continue
            kept[(candidate.role, candidate.fluent)] = candidate
            draft.own_effect_keys.add((candidate.role, candidate.fluent))
        if changed:
            draft.effects = tuple(kept.values())
        return changed

    def fix_unachievable(
        self, draft: _Draft, achievable: set[tuple[str, bool]], produced: list[tuple[str, bool]]
    ) -> bool:
        changed = False
        kept: dict[tuple[str, str], Literal] = {
            (lit.role, lit.fluent): lit for lit in draft.precondition
        }
        for literal in list(draft.precondition):
            if (literal.fluent, literal.value) in achievable:
                continue
            changed = True
            key = (literal.role, literal.fluent)
            del kept[key]
            if key not in draft.own_literal_keys:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            draft.own_literal_keys.discard(key)
            self.stats["literals_redrawn"] += 1
            candidate = self._draw_literal(draft.roles, achievable, produced, set(kept))
            if candidate is None:
                self.stats["literals_dropped_unachievable"] += 1
                continue
            kept[(candidate.role, candidate.fluent)] = candidate
            draft.own_literal_keys.add((candidate.role, candidate.fluent))
        if changed:
            draft.precondition = tuple(kept.values())
        return changed

    # Explicit entries -----------------------------------------------------------------------

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
                self._check_role(source, f"{entry.field}.precondition", et, lit.role, lit.key)
                if lit.fluent not in self.fluents:
                    raise ConfigError(
                        source, f"{entry.field}.precondition", f"reads unknown fluent {lit.fluent}"
                    )
            seen_effects: dict[tuple[str, str], bool] = {}
            for i, effect in enumerate(entry.effects or ()):
                where = f"{entry.field}.effects[{i}]"
                self._check_role(source, where, et, effect.role, effect.key)
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
            seen_literals: dict[tuple[str, str], bool] = {}
            for lit in entry.precondition or ():
                key = (lit.role, lit.fluent)
                if key in seen_literals and seen_literals[key] != lit.value:
                    raise ConfigError(
                        source, f"{entry.field}.precondition", f"holds {lit.key} and its negation"
                    )
                seen_literals[key] = lit.value
            if entry.precondition is not None and entry.effects is not None:
                for effect in entry.effects:
                    if seen_literals.get((effect.role, effect.fluent)) == effect.value:
                        raise ConfigError(
                            source,
                            entry.field,
                            f"the effect {effect.text} sets a fluent to the value the precondition "
                            f"already requires, so it changes nothing",
                        )
            draft.explicit = entry

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
        for entry in self.event_file.event_types.values():
            for lit in entry.precondition or ():
                if (lit.fluent, lit.value) not in achievable:
                    raise ConfigError(
                        self.event_file.source,
                        f"{entry.field}.precondition",
                        f"{lit.text} is not achievable: no entity starts with that value and no "
                        f"effect produces it",
                    )

    # The whole procedure ----------------------------------------------------------------------

    def run(self) -> EventTypes:
        self.apply_event_file()
        self.draw_effects()
        for draft in self.drafts.values():
            self.resolve_effects(draft)
        self.draw_preconditions()
        for draft in self.drafts.values():
            self.resolve_preconditions(draft)
        for _ in range(MAX_FIX_ROUNDS):
            changed = False
            for draft in self.drafts.values():
                if draft.explicit is None or draft.explicit.effects is None:
                    changed |= self.fix_empty_effects(draft)
            achievable = self.achievable()
            produced = self.produced()
            for draft in self.drafts.values():
                if draft.explicit is None or draft.explicit.precondition is None:
                    changed |= self.fix_unachievable(draft, achievable, produced)
            if not changed:
                break
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
        )


def generate_event_types(
    statics: StaticWorld,
    fluents: Fluents,
    config: Config,
    streams: WorldStreams,
    event_file: EventFile | None,
    derived_initial: Mapping[str, np.ndarray],
) -> EventTypes:
    """Event types with their requirements, preconditions, and effects."""
    static = build_event_types(statics)
    return _Dynamics(static, fluents, config, streams, event_file, derived_initial).run()


def redraw_preconditions(
    event_types: EventTypes,
    labels: Sequence[str],
    fluents: Fluents,
    config: Config,
    parts: Mapping[str, np.random.Generator],
    derived_initial: Mapping[str, np.ndarray],
) -> EventTypes:
    """The event types with the own precondition literals of the named event types drawn
    again, each from its own generator (the part ``world:preconditions:<label>``), as
    ``define`` redraws an event type that was never legal in the statistics episodes. The
    inherited literals stay; a literal that would make one of the event type's own effects
    empty, or that duplicates an inherited one, is drawn again. No other event type changes."""
    cfg = config.event_types.preconditions
    achievable: set[tuple[str, bool]] = set()
    for i, fluent in enumerate(fluents.base):
        for value in np.unique(fluents.initial_values[:, i]):
            achievable.add((fluent.label, bool(value)))
    for label, column in derived_initial.items():
        for value in np.unique(column):
            achievable.add((label, bool(value)))
    produced = sorted({(e.fluent, e.value) for et in event_types.event_types for e in et.effects})
    achievable |= set(produced)
    feature_literals = event_types.feature_preconditions
    redrawn: dict[str, EventType] = {}
    for label in labels:
        et = event_types.event_type(label)
        rng = parts[label]
        inherited: dict[tuple[str, str], Literal] = {}
        for feature in event_types.features:
            literal = feature_literals.get(feature)
            if literal is not None and feature in et.features:
                inherited.setdefault((literal.role, literal.fluent), literal)
        empty = {(e.role, e.fluent, e.value) for e in et.effects}
        kept = dict(inherited)
        n = _draw(rng, cfg.literals)
        for _ in range(n):
            for _ in range(MAX_TRIES):
                role = et.roles[0] if len(et.roles) == 1 else _draw(rng, cfg.roles)
                if produced and rng.random() < cfg.enabled_share:
                    fluent, value = produced[int(rng.integers(len(produced)))]
                else:
                    fluent = fluents.labels[int(rng.integers(len(fluents.labels)))]
                    value = bool(rng.integers(2))
                if (
                    (fluent, value) in achievable
                    and (role, fluent) not in kept
                    and (role, fluent, value) not in empty
                ):
                    kept[(role, fluent)] = Literal(role, fluent, value)
                    break
        redrawn[label] = dataclasses.replace(et, precondition=tuple(kept.values()))
    return dataclasses.replace(
        event_types,
        event_types=tuple(redrawn.get(et.label, et) for et in event_types.event_types),
    )


def check_dynamics(
    event_types: EventTypes, fluents: Fluents, initial_present: set[tuple[str, bool]]
) -> None:
    """The invariants of "Preconditions" and "Effects": no effect on a derived fluent, no
    contradiction, no empty effect, every literal achievable, every role valid. Raises
    :class:`ValueError` naming the event type otherwise. Tests and the generator both call it."""
    produced = {(e.fluent, e.value) for et in event_types.event_types for e in et.effects}
    achievable = produced | initial_present
    for et in event_types.event_types:
        effect_keys: dict[tuple[str, str], bool] = {}
        for effect in et.effects:
            if effect.role not in et.roles:
                raise ValueError(f"{et.label}: the effect {effect.text} uses a role it has not")
            if fluents[effect.fluent].derived:
                raise ValueError(f"{et.label}: the effect {effect.text} writes a derived fluent")
            key = (effect.role, effect.fluent)
            if key in effect_keys:
                raise ValueError(f"{et.label}: two effects write {effect.key}")
            effect_keys[key] = effect.value
        literal_keys: dict[tuple[str, str], bool] = {}
        for lit in et.precondition:
            if lit.role not in et.roles:
                raise ValueError(f"{et.label}: the literal {lit.text} uses a role it has not")
            key = (lit.role, lit.fluent)
            if key in literal_keys:
                raise ValueError(f"{et.label}: the precondition reads {lit.key} twice")
            literal_keys[key] = lit.value
            if (lit.fluent, lit.value) not in achievable:
                raise ValueError(f"{et.label}: the literal {lit.text} is not achievable")
        for key, value in effect_keys.items():
            if literal_keys.get(key) == value:
                raise ValueError(f"{et.label}: the effect on {key[0]}.{key[1]} changes nothing")


__all__ = [
    "CONSTRAINT_PREFIX",
    "FEATURE_PREFIX",
    "ONE_PLACE",
    "REQUIREMENT_PREFIX",
    "TWO_PLACE",
    "ConstraintSpec",
    "EventType",
    "EventTypes",
    "build_event_types",
    "check_dynamics",
    "constraint_from_relation",
    "constraint_from_rule",
    "generate_event_types",
    "prefixed",
    "redraw_preconditions",
]
