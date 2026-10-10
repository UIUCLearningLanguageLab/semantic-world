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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from semantic_world.corpus.config import Config
from semantic_world.corpus.histories import SceneEvent
from semantic_world.corpus.lexicon import Lexicon, world_concepts
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CAUSAL_KINDS,
    CLASS,
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
    PROJECTION,
    ROLES,
    SCALAR,
    SIMPLE,
    STATE_KIND,
    VERB,
    CategoryTerm,
    Clause,
    EventTerm,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.corpus.world import PROPERTY_PREFIX, THING, World

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
        self.fluents = of_type("state")
        """The fluents that have a state adjective, base then derived."""
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
        predicates += [Predicate(SCALAR, p) for p in self.poles]
        if category != THING:
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
        if predicate.kind == SCALAR:
            options: tuple[tuple[str | None, bool], ...] = ((None, not negative),)
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
        predicates += [Predicate(SCALAR, p, comparison=c) for p in self.poles for c in path]
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
                mask = truth.pole_mask(pole, world.below(term.category))[0]
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

    def _build_causal_statements(self) -> tuple[Proposition, ...]:
        truth, world = self.truth, self.world
        statements: list[Proposition] = []
        for label in self.event_type_labels:
            for role in ROLES[: world.event_types[label].arity]:
                for fluent in self.base_fluents:
                    for value in (True, False):
                        for kind in CAUSAL_KINDS:
                            if not truth.has_entry(label, kind, role, fluent, value):
                                continue
                            candidate = Proposition(
                                CLASS,
                                EventTerm(label, role),
                                Predicate(kind, fluent, value=value),
                                True,
                                NEC_ALL,
                            )
                            grounded = truth.grounded(candidate)
                            if grounded is not None:
                                statements.append(grounded)
        return tuple(statements)

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
