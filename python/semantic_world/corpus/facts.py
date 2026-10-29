"""True propositions about a world: the facts a document can state.

:class:`Facts` enumerates the true class-level and instance-level propositions that the lexicon
can express, each with its grounding, and the rule statements of the world. The discourse
planner chooses among them.

For a class-level subject and a predicate, the *strongest* true proposition that the language
can state is stated: ``nec_all`` before ``all`` before ``most`` before ``some``, and for a
negative fact ``nec_no`` before ``no`` before ``most ... not`` before ``some ... not``, among the
quantifiers the language has words for (``quantifiers.universal_words``) or can say with a bare
plural, and that its usage rules allow (``quantifiers.most.usage_min``,
``quantifiers.some.exclude_all``). A fact whose strongest true quantifier the language cannot
say falls to the next quantifier down. Whether a fact is then said with a quantifier word or
with a bare plural is the planner's choice, and never changes the proposition.

A rule statement takes one term of the minimal DNF of a determined feature's rule: the generic
noun as its head, the term's literals as its restriction, and the determined feature as its
predicate ("things with wings and with feathers can fly"). All of a term's negated PROPERTY
literals go into one relative clause, joined with "and". A term is skipped, and counted, when it
reads a scalar threshold, when it has more literals than
``propositions.rule_statements.max_literals``, when one of its concepts has no word, or when no
instance satisfies it. A rule statement is always ``nec_all`` (CG.61).

A class-level subject can be restricted, in the proposition layer, where the truth of the
sentence is grounded. A restriction is one literal ("red penguins"), and a relative clause is
restrictive ("penguins that can swim", "owls that eat mice", "mice that owls eat"). Both are drawn
among those that some members of the category satisfy, and not all, so the restriction does work.

A causal statement (``docs/specs/WORLD_AND_LANGUAGE.md``, "Causal statements") states one entry
of the definition: an effect or a precondition literal of an event type with a word, about a base
fluent with a word, as ``nec_all`` ("things that things catch become caught"). A category of
event types with a word states the entries that every event type below it has. The statements
are enumerated from the definition (:meth:`Facts.causal_statements`) and grounded by the truth
tests, in the order of the event types, then the roles, the fluents, and the values.

An event is reported by an event-level proposition. Its event type is named at a level of the
event-type tree, as a noun names a category at a level of the category tree: the event type
itself, or a category above it ("chase" or "hunt"), drawn by ``mention.event_level_weights``.
Each report chooses its aspect (CG.64).

**Comparisons** (stage b3a). A scalar pole is a comparison with its class: at the instance level
the category the noun names, at the class level the ancestor ``scalar_adjectives.class_levels_up``
levels above the subject's category, or ``ENTITY`` when the tree runs out (CG.3). A comparison
fact (CG.66, decision 24) compares two categories' means on one scalar, ``GREATER`` one way or
the other, never when they are equal; a category document compares its topic with one of its
*comparison partners*: by default its siblings with a word (``documents.comparison_partners:
siblings``, Jon's ruling 2 on stage b3a); ``thematic`` gives as many thematic partners, the
other named categories of its level whose leaves are most related to the topic's leaves by the
world's thematic relatedness, and ``both`` gives both. An instance comparison compares two
instances' values the same way.

**Numeric fluents** (stage b3b). A numeric fluent with words (``numeric_fluents``) gives a
comparative between two participants at a time point of a scene (``instance_comparison_at``),
and two numeric poles, stated at a time point as a Boolean fluent is. The causal statements
gain the numeric effect statements (``RAISES`` and ``LOWERS``, one for each unconditional
numeric effect) and the conditional statements, one for each conditional effect, Boolean or
numeric, whose condition the grammar can realize; ``causal_report`` counts the conditional
effects that make no statement, and why.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.grammar import realizable_condition
from semantic_world.corpus.histories import SceneEvent
from semantic_world.corpus.lexicon import Lexicon, world_concepts
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CAUSAL_KINDS,
    CLASS,
    COMPARATIVE,
    COMPARISON,
    EFFECT,
    EFFECT_DIRECTIONS,
    EVENT,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    NEC_ALL,
    NEC_NO,
    NEGATIVE_ORDER,
    NO,
    POSITIVE_ORDER,
    PRECONDITION,
    PROJECTION,
    RAISES,
    ROLES,
    SIMPLE,
    STATE,
    STATE_KIND,
    VERB,
    CategoryTerm,
    Clause,
    Condition,
    EventTerm,
    Literal,
    Predicate,
    Proposition,
    Truth,
    pole_dimension,
)
from semantic_world.corpus.world import ENTITY, PROPERTY_PREFIX, THING, World
from semantic_world.world.runtime import RANGE

SKIP_THRESHOLD = "threshold"
"""The term reads a scalar threshold, which no pole adjective states."""
SKIP_MAX_LITERALS = "max_literals"
SKIP_NO_WORD = "no_word"
SKIP_NO_INSTANCE = "no_instance"
"""No instance satisfies the term, so the statement would be vacuous."""
SKIP_UNCONFIRMED = "unconfirmed"
"""The cone is too large to enumerate, and the local test does not confirm the statement."""
SKIP_REASONS = (
    SKIP_THRESHOLD,
    SKIP_MAX_LITERALS,
    SKIP_NO_WORD,
    SKIP_NO_INSTANCE,
    SKIP_UNCONFIRMED,
)

_POSITIVE_OPTIONS = tuple((q, True) for q in POSITIVE_ORDER)
_NEGATIVE_OPTIONS = tuple((q, q in (NEC_NO, NO)) for q in NEGATIVE_ORDER)
CONCEPT_KIND = {
    "property": IS,
    "part": HAS,
    "state": STATE_KIND,
    "event_unary": CAN,
    "event": VERB,
    "event_category": VERB,
}


class Facts:
    """The true, expressible propositions of one world. With a lexicon, only concepts that have
    a word take part. Without one, every concept of the world does."""

    def __init__(
        self,
        config: Config,
        world: World,
        lexicon: Lexicon | None = None,
        truth: Truth | None = None,
    ) -> None:
        self.config = config
        self.world = world
        self.truth = truth or Truth(config, world)
        concepts = world_concepts(world)
        if lexicon is not None:
            concepts = tuple(c for c in concepts if lexicon.is_named(c.label))
        self.named = frozenset(c.label for c in concepts)

        def of_type(*types: str) -> tuple[str, ...]:
            return tuple(c.label for c in concepts if c.type in types)

        self.categories = of_type("category")
        self.features = {
            IS: of_type("property"),
            HAS: of_type("part"),
            CAN: of_type("event_unary"),
        }
        self.projections = of_type("patient_projection")
        self.poles = of_type("scalar")
        self.scalars: tuple[str, ...] = tuple(dict.fromkeys(pole_dimension(p) for p in self.poles))
        """The scalar dimensions whose poles have words: what a comparative can be about (the
        two poles of a dimension are named together)."""
        self.fluents = of_type("state")
        """The Boolean fluents that have a state adjective, base then derived."""
        self.numeric_poles = of_type("numeric")
        """The poles of the numeric fluents that have words (stage b3b), the two of a fluent
        together: ``NUMFL.1.HIGH``, ``NUMFL.1.LOW``."""
        self.numeric_fluents: tuple[str, ...] = tuple(
            dict.fromkeys(pole_dimension(p) for p in self.numeric_poles)
        )
        """The numeric fluents that have words: what a numeric comparative, a numeric change,
        and a numeric effect statement can be about."""
        self.base_fluents = tuple(f for f in self.fluents if f in world.base_fluents)
        """The base fluents that have a state adjective: what a causal statement is about."""
        self.verbs = of_type("event", "event_category")
        """The two-place event types and categories of them that have a word, in tree order."""
        self.level = {label: info.level for label, info in world.category.items()}
        self._class_facts: dict[tuple, tuple[Proposition, ...]] = {}
        self._restrictions: dict[CategoryTerm, tuple[Literal, ...]] = {}
        self._clauses: dict[tuple[CategoryTerm, bool], tuple[Clause, ...]] = {}
        self._rules: tuple[Proposition, ...] | None = None
        self.rule_report: dict[str, Any] = {}
        self._causal: tuple[Proposition, ...] | None = None
        self.causal_report: dict[str, Any] = {}
        """The conditional effects of the world's event types, and how many make no conditional
        statement, by reason (stage b3b): ``numeric_condition`` (a numeric threshold literal,
        decision 23), ``patient_relative`` (a fluent or a negated PROPERTY literal of the
        patient's role in a statement about the patient, which an object relative cannot hold),
        ``no_word`` (a literal whose concept has no word), and, since stage c4a,
        ``pair_condition`` (a pair literal in the condition, WM.59) and ``pair_effect`` (a
        conditional pair effect, whose statement is stage c5a's)."""
        self._partners: dict[str, tuple[str, ...]] = {}

    def expressible(self, proposition: Proposition) -> bool:
        """Whether every concept the proposition needs has a word."""
        return all(label in self.named for label in proposition.concepts())

    # Class level -----------------------------------------------------------------------------

    def class_predicates(
        self, category: str, patients: tuple[str, ...] | None = None
    ) -> list[Predicate]:
        """Every predicate that a sentence about a category could have: features, one-place
        capacities, patient capacities, scalar poles, membership in a category at the same level
        or above, and a two-place event type with a patient category (every named category, or
        ``patients``)."""
        predicates = [Predicate(kind, f) for kind in (IS, HAS, CAN) for f in self.features[kind]]
        predicates += [Predicate(PROJECTION, p) for p in self.projections]
        if category != THING:
            comparison = self.comparison_class(category)
            predicates += [Predicate(COMPARISON, p, comparison=comparison) for p in self.poles]
            level = self.level[category]
            predicates += [
                Predicate(MEMBER, c)
                for c in self.categories
                if c != category and self.level[c] <= level
            ]
        for verb in self.verbs:
            for patient in self.categories if patients is None else patients:
                predicates.append(Predicate(VERB, verb, CategoryTerm(patient)))
        return predicates

    def class_fact(
        self, subject: CategoryTerm, predicate: Predicate, negative: bool = False
    ) -> Proposition | None:
        """The strongest true proposition that the language can state, asserting a predicate of
        a subject, or denying it with ``negative``; None when there is none. A scalar pole takes
        no quantifier; membership takes a universal."""
        if predicate.kind == COMPARISON:
            options: tuple[tuple[str | None, bool], ...] = ((None, not negative),)
        elif predicate.kind == COMPARATIVE:
            if negative:
                return None  # a comparative is never negated: it is said the other way round
            options = ((None, True),)
        elif predicate.kind == MEMBER:
            options = ((NEC_NO, True), (NO, True)) if negative else ((NEC_ALL, True), (ALL, True))
        else:
            options = _NEGATIVE_OPTIONS if negative else _POSITIVE_OPTIONS
        for quantifier, polarity in options:
            candidate = Proposition(CLASS, subject, predicate, polarity, quantifier)
            if not self.expressible(candidate):
                return None
            grounded = self.truth.grounded(candidate)
            if grounded is not None:
                return grounded
        return None

    def class_facts(
        self,
        subject: CategoryTerm | str,
        negative: bool = False,
        patients: tuple[str, ...] | None = None,
    ) -> tuple[Proposition, ...]:
        """The strongest true proposition for every predicate, about a category or a category
        term, in the order of :meth:`class_predicates`."""
        term = CategoryTerm(subject) if isinstance(subject, str) else subject
        key = (term, negative, patients)
        if key not in self._class_facts:
            facts = [
                self.class_fact(term, predicate, negative)
                for predicate in self.class_predicates(term.category, patients)
            ]
            self._class_facts[key] = tuple(f for f in facts if f is not None)
        return self._class_facts[key]

    def patient_facts(
        self, category: str, negative: bool = False, agents: tuple[str, ...] | None = None
    ) -> tuple[Proposition, ...]:
        """The strongest true relation facts with a category as the patient, for every two-place
        event type and every agent category (every named category, or ``agents``)."""
        patient = CategoryTerm(category)
        facts = []
        for verb in self.verbs:
            for agent in self.categories if agents is None else agents:
                fact = self.class_fact(
                    CategoryTerm(agent), Predicate(VERB, verb, patient), negative
                )
                if fact is not None:
                    facts.append(fact)
        return tuple(facts)

    # Instance level --------------------------------------------------------------------------

    def instance_predicates(
        self, instance: str, patients: tuple[str, ...] | None = None
    ) -> list[Predicate]:
        """Every predicate that a sentence about an instance could have: features, one-place
        capacities, patient capacities, scalar poles against every named category the instance
        is below, membership in a category, and a two-place event type with a patient instance
        (every other instance, or ``patients``)."""
        index = self.truth.instance_index[instance]
        predicates = [Predicate(kind, f) for kind in (IS, HAS, CAN) for f in self.features[kind]]
        predicates += [Predicate(PROJECTION, p) for p in self.projections]
        path = [c for c in self.truth.paths[index] if c in self.named]
        predicates += [Predicate(COMPARISON, p, comparison=c) for p in self.poles for c in path]
        predicates += [Predicate(MEMBER, c) for c in self.categories]
        others = self.world.instances if patients is None else patients
        for verb in self.verbs:
            predicates += [Predicate(VERB, verb, other) for other in others if other != instance]
        return predicates

    def instance_fact(
        self, instance: str, predicate: Predicate, negative: bool = False
    ) -> Proposition | None:
        """The true proposition that asserts a predicate of an instance, or denies it with
        ``negative``; None when it is false."""
        candidate = Proposition(INSTANCE, instance, predicate, not negative)
        if not self.expressible(candidate):
            return None
        return self.truth.grounded(candidate)

    def instance_facts(
        self, instance: str, negative: bool = False, patients: tuple[str, ...] | None = None
    ) -> tuple[Proposition, ...]:
        """Every true proposition about an instance, in the order of
        :meth:`instance_predicates`."""
        facts = [
            self.instance_fact(instance, predicate, negative)
            for predicate in self.instance_predicates(instance, patients)
        ]
        return tuple(f for f in facts if f is not None)

    # Restricted subjects ---------------------------------------------------------------------

    def restriction_options(self, term: CategoryTerm) -> tuple[Literal, ...]:
        """The literals that can restrict a category term further: the PROPERTY and PART
        literals, positive and negative, and the scalar poles, that some members satisfy and not
        all."""
        if term not in self._restrictions:
            truth, world = self.truth, self.world
            members = truth.members(term)
            stated = {literal.feature for literal in term.restriction}
            options: list[Literal] = []
            for feature in self.features[IS] + self.features[HAS]:
                if feature in stated:
                    continue
                count = int(world.column(feature)[members].sum())
                if 0 < count < len(members):
                    options += [Literal(feature), Literal(feature, False)]
            for pole in self.poles:
                if pole in stated:
                    continue
                mask = truth.pole_mask(pole, term.category)[0]
                if 0 < int(mask[members].sum()) < len(members):
                    options.append(Literal(pole))
            self._restrictions[term] = tuple(options)
        return self._restrictions[term]

    def draw_restriction(self, rng: np.random.Generator, term: CategoryTerm) -> CategoryTerm | None:
        """The term with one more literal in its restriction ("red penguins"): negative at the
        class-level negation rate, and drawn among the literals that some members satisfy and
        not all. None when there is none."""
        options = self.restriction_options(term)
        if not options:
            return None
        negative = rng.random() < self.config.propositions.negation_rate[CLASS]
        pool = [x for x in options if x.positive != negative] or list(options)
        literal = pool[int(rng.integers(len(pool)))]
        return CategoryTerm(term.category, term.restriction + (literal,), term.clauses)

    def clause_options(self, term: CategoryTerm, object_relative: bool) -> tuple[Clause, ...]:
        """The relative clauses that can restrict a category term: those that some members
        satisfy and not all. A subject relative holds a one-place event type ("that can swim")
        or a two-place event type with a patient category ("that eat mice"). An object relative
        holds a two-place event type with an agent category ("that owls eat")."""
        key = (term, object_relative)
        if key not in self._clauses:
            truth, world = self.truth, self.world
            members = truth.members(term)
            total = len(members)
            options: list[Clause] = []
            if not object_relative:
                for feature in self.features[CAN]:
                    count = int(world.column(feature)[members].sum())
                    if 0 < count < total:
                        options.append(Clause(CAN, feature))
            for verb in self.verbs:
                matrix = truth.matrix(verb)
                block = matrix[:, members].T if object_relative else matrix[members]
                for category in self.categories:
                    others = truth.members(CategoryTerm(category))
                    count = int(block[:, others].any(axis=1).sum())
                    if 0 < count < total:
                        other = CategoryTerm(category)
                        if object_relative:
                            options.append(Clause(VERB, verb, agent=other))
                        else:
                            options.append(Clause(VERB, verb, patient=other))
            self._clauses[key] = tuple(options)
        return self._clauses[key]

    def draw_clause(
        self, rng: np.random.Generator, term: CategoryTerm, depth: int = 1
    ) -> CategoryTerm | None:
        """The term with a restrictive relative clause. The clause is an object relative with
        probability ``mention.relative_clauses.object_share``, and a subject relative otherwise:
        a one-place event type or a two-place one, each kind with the same chance. When the
        drawn kind has no clause, the other kind is used. The other category of a two-place
        event type can take a relative clause of its own, at the rate, up to ``max_depth``.
        None when the term has a clause already, or when no clause restricts it."""
        settings = self.config.mention.relative_clauses
        if term.clauses or depth > settings.max_depth:
            return None
        negated = any(
            not x.positive and x.feature.startswith(PROPERTY_PREFIX) for x in term.restriction
        )
        as_object = rng.random() < settings.object_share and not negated
        total = len(self.truth.members(term))
        for object_relative in (as_object, not as_object):
            if object_relative and negated:
                continue  # the negated PROPERTY literals need a subject relative to join
            options = self.clause_options(term, object_relative)
            if not options:
                continue
            groups = [
                [c for c in options if c.kind == CAN],
                [c for c in options if c.kind == VERB],
            ]
            groups = [group for group in groups if group]
            group = groups[int(rng.integers(len(groups)))]
            clause = group[int(rng.integers(len(group)))]
            if clause.other is not None and rng.random() < settings.rate:
                inner = self.draw_clause(rng, clause.other, depth + 1)
                if inner is not None:
                    deeper = (
                        Clause(VERB, clause.label, agent=inner)
                        if object_relative
                        else Clause(VERB, clause.label, patient=inner)
                    )
                    candidate = CategoryTerm(term.category, term.restriction, (deeper,))
                    if 0 < len(self.truth.members(candidate)) < total:
                        clause = deeper
            return CategoryTerm(term.category, term.restriction, (clause,))
        return None

    # Comparisons -----------------------------------------------------------------------------

    def comparison_class(self, category: str) -> str:
        """The comparison class of a class-level pole of a category (CG.3): the ancestor
        ``scalar_adjectives.class_levels_up`` levels above it, or ``ENTITY`` when the tree runs
        out (a top-level category, at the default of 1)."""
        path = self.world.category[category].path
        up = self.config.class_levels_up
        return path[-1 - up] if len(path) > up else ENTITY

    def comparison_fact(self, a: str, b: str, dimension: str) -> Proposition | None:
        """The true comparison fact between two categories on a scalar: ``GREATER`` of the
        mean of ``a`` over the mean of ``b``, or the other way round; None when the means are
        equal, or when the language cannot state it."""
        for first, second in ((a, b), (b, a)):
            candidate = Proposition(
                CLASS, CategoryTerm(first), Predicate(COMPARATIVE, dimension, CategoryTerm(second))
            )
            if not self.expressible(candidate):
                return None
            grounded = self.truth.grounded(candidate)
            if grounded is not None:
                return grounded
        return None

    def instance_comparison(self, x: str, y: str, dimension: str) -> Proposition | None:
        """The true comparative between two instances on a scalar: ``GREATER`` of ``x`` over
        ``y``, or the other way round; None when the values are equal, or when the language
        cannot state it."""
        for first, second in ((x, y), (y, x)):
            candidate = Proposition(INSTANCE, first, Predicate(COMPARATIVE, dimension, second))
            if not self.expressible(candidate):
                return None
            grounded = self.truth.grounded(candidate)
            if grounded is not None:
                return grounded
        return None

    def instance_comparison_at(
        self, x: str, y: str, fluent: str, scene: str, time: str
    ) -> Proposition | None:
        """The true comparative between two participants of a scene on a numeric fluent at a
        time point (stage b3b): ``GREATER`` of ``x`` over ``y``, or the other way round, under
        ``HOLDS``; None when the values are equal, or when the language cannot state it."""
        tense = self.config.propositions.event_tense
        for first, second in ((x, y), (y, x)):
            candidate = Proposition(
                STATE,
                first,
                Predicate(COMPARATIVE, fluent, second),
                scene=scene,
                time=time,
                tense=tense,
            )
            if not self.expressible(candidate):
                return None
            grounded = self.truth.grounded(candidate)
            if grounded is not None:
                return grounded
        return None

    def numeric_pole_predicates(self) -> list[Predicate]:
        """The predicates of the numeric poles with words, against the range (stage b3b)."""
        return [Predicate(COMPARISON, pole, comparison=RANGE) for pole in self.numeric_poles]

    def comparison_partners(self, topic: str) -> tuple[str, ...]:
        """The categories a document about ``topic`` compares it with
        (``documents.comparison_partners``): its siblings with a word (``siblings``, the
        default), as many thematic partners, at least one (``thematic``: the other named
        categories of the topic's level that are not its siblings, in order of the mean thematic
        relatedness of their leaves to the topic's leaves, ties in tree order), or both
        (``both``, the pool of stage b3a). Empty when no other category of the level has a
        word."""
        if topic not in self._partners:
            world = self.world
            info = world.category[topic]
            siblings = [
                c
                for c in self.categories
                if c != topic
                and info.parent is not None
                and world.category[c].parent == info.parent
            ]
            others = [
                c
                for c in self.categories
                if c != topic and c not in siblings and self.level[c] == info.level
            ]
            leaf_index = {leaf: i for i, leaf in enumerate(world.leaves)}

            def leaves_below(category: str) -> list[int]:
                return [
                    leaf_index[leaf]
                    for leaf in world.leaves
                    if category in world.category[leaf].path
                ]

            own = leaves_below(topic)

            def relatedness(category: str) -> float:
                rows = leaves_below(category)
                if not own or not rows:
                    return 0.0
                return float(world.thematic[np.ix_(own, rows)].mean())

            ranked = sorted(others, key=lambda c: -relatedness(c))
            wanted = max(1, len(siblings))
            setting = self.config.documents.comparison_partners
            partners = siblings if setting != "thematic" else []
            if setting != "siblings":
                partners = partners + ranked[:wanted]
            self._partners[topic] = tuple(partners)
        return self._partners[topic]

    # Rule statements -------------------------------------------------------------------------

    def rule_statements(self, feature: str | None = None) -> tuple[Proposition, ...]:
        """The world's rule statements, in rule order and then term order, or those whose
        predicate is ``feature``: the sufficient conditions for it."""
        if self._rules is None:
            self._rules = self._build_rule_statements()
        if feature is None:
            return self._rules
        return tuple(p for p in self._rules if p.predicate.label == feature)

    def rule_statements_reading(self, feature: str) -> tuple[Proposition, ...]:
        """The rule statements in which a feature appears in the restriction: what the feature
        makes possible."""
        return tuple(
            p
            for p in self.rule_statements()
            if isinstance(p.subject, CategoryTerm)
            and any(literal.feature == feature for literal in p.subject.restriction)
        )

    def _build_rule_statements(self) -> tuple[Proposition, ...]:
        cap = self.config.propositions.rule_max_literals
        skipped = dict.fromkeys(SKIP_REASONS, 0)
        statements: list[Proposition] = []
        terms = self.world.rule_terms()
        for term in terms:
            if term.threshold:
                skipped[SKIP_THRESHOLD] += 1
                continue
            literals = tuple(Literal(label, value) for label, value in term.literals)
            if cap is not None and len(literals) > cap:
                skipped[SKIP_MAX_LITERALS] += 1
                continue
            statement = Proposition(
                CLASS,
                CategoryTerm(THING, literals),
                Predicate(term.kind, term.output),
                True,
                NEC_ALL,
                rule=(term.output, term.number),
            )
            if not self.expressible(statement):
                skipped[SKIP_NO_WORD] += 1
                continue
            if len(self.truth.members(statement.subject)) == 0:
                skipped[SKIP_NO_INSTANCE] += 1
                continue
            grounded = self.truth.grounded(statement)
            if grounded is None:
                # The term is a sufficient condition, so the statement is true. The truth
                # test fails to confirm it only when the cone is too large to enumerate.
                skipped[SKIP_UNCONFIRMED] += 1
                continue
            statements.append(grounded)
        self.rule_report = {
            "rules": self.world.rule_count,
            "terms": len(terms),
            "stated": len(statements),
            "skipped": skipped,
        }
        return tuple(statements)

    # Causal statements -----------------------------------------------------------------------

    @property
    def event_type_labels(self) -> tuple[str, ...]:
        """The event types with a word that a causal statement can be about: the one-place
        event types, then the two-place event types and categories in tree order."""
        return self.features[CAN] + self.verbs

    def causal_statements(
        self, event_type: str | None = None, fluent: str | None = None, kind: str | None = None
    ) -> tuple[Proposition, ...]:
        """Every true causal statement that the language can state, or those about an event
        type (or category), about a fluent, or of one kind (``effect`` or ``precondition``):
        for each event type with a word, each role, each base fluent with a word, and each
        value, the effect statement and the precondition statement that the definition makes
        true."""
        if self._causal is None:
            self._causal = self._build_causal_statements()
        return tuple(
            p
            for p in self._causal
            if (event_type is None or p.subject.event_type == event_type)
            and (fluent is None or p.predicate.label == fluent)
            and (kind is None or p.predicate.kind == kind)
        )

    def _conditions(
        self, records: list[Any], role: str, fluent: str, value: bool | None, direction: str | None
    ) -> list[tuple[Condition, ...]]:
        """The distinct conditions under which the effects of the given event-type records set
        the fluent of the role to the value (or move it in the direction), the empty condition
        first when some effect is unconditional, then in one order. An effect under a numeric
        threshold condition gives none (decision 23), unless the threshold is the language's
        cut, when the condition is the numeric pole (WM.35)."""
        found: list[tuple[Condition, ...]] = []
        for record in records:
            for effect in record.effects:
                if effect.role != role or effect.fluent != fluent:
                    continue
                if effect.numeric:
                    if direction is None or (effect.amount >= 0) != (direction == RAISES):
                        continue
                elif direction is not None or effect.value != value:
                    continue
                condition = self.truth.condition_of(effect)
                if condition is not None and condition not in found:
                    found.append(condition)
        return sorted(found, key=lambda c: (len(c), [x.sort_key for x in c]))

    def _build_causal_statements(self) -> tuple[Proposition, ...]:
        truth, world = self.truth, self.world
        definition = world.definition
        statements: list[Proposition] = []
        for label in self.event_type_labels:
            records = [definition.event_type(leaf) for leaf in world.event_types_below(label)]
            for role in ROLES[: world.event_types[label].arity]:
                entries: list[tuple[str, str, bool | None, str | None]] = []
                for fluent in self.base_fluents:
                    for value in (True, False):
                        entries += [(kind, fluent, value, None) for kind in CAUSAL_KINDS]
                for fluent in self.numeric_fluents:
                    entries += [(EFFECT, fluent, None, d) for d in EFFECT_DIRECTIONS]
                for kind, fluent, value, direction in entries:
                    if kind == PRECONDITION:
                        conditions: list[tuple[Condition, ...]] = [()]
                    else:
                        conditions = self._conditions(records, role, fluent, value, direction)
                    for condition in conditions:
                        term = EventTerm(label, role, condition)
                        if not realizable_condition(term):
                            continue
                        if not truth.has_entry(
                            label, kind, role, fluent, value, direction, condition
                        ):
                            continue
                        candidate = Proposition(
                            CLASS,
                            term,
                            Predicate(kind, fluent, value=value, direction=direction),
                            True,
                            NEC_ALL,
                        )
                        if not self.expressible(candidate):
                            continue
                        grounded = truth.grounded(candidate)
                        if grounded is not None:
                            statements.append(grounded)
        self.causal_report = self._conditional_report(statements)
        return tuple(statements)

    def _conditional_report(self, statements: list[Proposition]) -> dict[str, Any]:
        """The conditional effects of the event types that have events, and those that make no
        conditional statement about their own event type, by reason."""
        world = self.world
        definition = world.definition
        stated = {
            (p.subject.event_type, p.subject.role, p.predicate.label, p.subject.condition)
            for p in statements
            if isinstance(p.subject, EventTerm) and p.subject.condition
        }
        reasons = {
            "numeric_condition": 0,
            "patient_relative": 0,
            "no_word": 0,
            "pair_condition": 0,
            "pair_effect": 0,
        }
        total = made = 0
        for label in world.binary_leaves + world.unary:
            for effect in definition.event_type(label).effects:
                if not effect.condition:
                    continue
                total += 1
                if effect.pair:
                    reasons["pair_effect"] += 1  # a conditional pair effect (stage c5a's)
                    continue
                if any(lit.pair for lit in effect.condition):
                    reasons["pair_condition"] += 1  # a pair literal in the condition (WM.59)
                    continue
                condition = self.truth.condition_of(effect)
                if condition is None:
                    reasons["numeric_condition"] += 1  # a threshold not at a cut (WM.35)
                    continue
                term = EventTerm(label, effect.role, condition)
                if not realizable_condition(term):
                    reasons["patient_relative"] += 1
                    continue
                if (label, effect.role, effect.fluent, condition) in stated:
                    made += 1
                else:
                    reasons["no_word"] += 1
        return {"conditional_effects": total, "stated": made, "unstated": reasons}

    def draw_causal_statement(
        self, rng: np.random.Generator, pool: tuple[Proposition, ...] | None = None
    ) -> Proposition | None:
        """One causal statement, from every one of the world or from ``pool``."""
        pool = self.causal_statements() if pool is None else pool
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    # Events ----------------------------------------------------------------------------------

    def event_names(self, event: SceneEvent) -> tuple[str, ...]:
        """The labels that have a word and can name an event: its event type, and the
        categories above it, from the event type upward."""
        return tuple(label for label in self.truth.verb_names(event.type) if label in self.named)

    def event_fact(
        self, event: SceneEvent, verb: str | None = None, aspect: str = SIMPLE
    ) -> Proposition | None:
        """The event-level proposition that reports an event, named by its own event type or by
        ``verb``, a category above it, with the report's aspect. None when the name has no
        word, or does not name the event."""
        kind = VERB if event.transitive else CAN
        predicate = Predicate(kind, event.type if verb is None else verb, event.patient)
        candidate = Proposition(
            EVENT,
            event.agent,
            predicate,
            scene=event.scene,
            event=event.label,
            tense=self.config.propositions.event_tense,
            aspect=aspect,
        )
        if not self.expressible(candidate):
            return None
        return self.truth.grounded(candidate)

    def draw_event(
        self, rng: np.random.Generator, event: SceneEvent, aspect: str = SIMPLE
    ) -> Proposition | None:
        """The proposition that reports an event, with its event type named at a level drawn by
        ``mention.event_level_weights`` among the levels that have a word, and the given
        aspect. A level with the weight 0 is never used. None when no level can name the
        event."""
        names = self.event_names(event)
        if not event.transitive:
            return self.event_fact(event, aspect=aspect) if names else None
        levels = self.config.mention.event_level_weights
        weights = np.array(
            [levels[self.world.event_types[label].level - 1] for label in names], dtype=float
        )
        if weights.sum() == 0:
            return None
        name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        return self.event_fact(event, name, aspect)

    # Drawing ---------------------------------------------------------------------------------

    def draw_class(
        self,
        rng: np.random.Generator,
        subject: CategoryTerm | str,
        patients: tuple[str, ...] | None = None,
    ) -> Proposition | None:
        """One true class-level proposition about a subject: negative at the class-level
        negation rate. When the subject has no fact of the drawn polarity, the other polarity is
        used."""
        negative = rng.random() < self.config.propositions.negation_rate[CLASS]
        pool = self.class_facts(subject, negative, patients) or self.class_facts(
            subject, not negative, patients
        )
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    def draw_rule_statement(
        self, rng: np.random.Generator, pool: tuple[Proposition, ...] | None = None
    ) -> Proposition | None:
        """One rule statement, from every rule statement of the world or from ``pool``."""
        pool = self.rule_statements() if pool is None else pool
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    def draw_instance(
        self, rng: np.random.Generator, instance: str, patients: tuple[str, ...] | None = None
    ) -> Proposition | None:
        """One true instance-level proposition about an instance, negative at the
        instance-level negation rate."""
        negative = rng.random() < self.config.propositions.negation_rate[INSTANCE]
        pool = self.instance_facts(instance, negative, patients) or self.instance_facts(
            instance, not negative, patients
        )
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]
