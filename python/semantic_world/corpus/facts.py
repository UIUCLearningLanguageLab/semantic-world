"""True propositions about a world: the facts a document can state.

:class:`Facts` enumerates the true class-level and instance-level propositions that the lexicon
can express, each with its grounding, and the rule statements of the world. The discourse
planner chooses among them.

For a class-level subject and a predicate, the *strongest* true proposition is stated: ``all``
before ``most`` before ``some``, and for a negative fact ``no`` before ``most ... not`` before
``some ... not``. The bare generic can stand in for any of them when it is true.

A rule statement takes one term of the minimal DNF of a determined feature's rule: the generic
noun as its head, the term's literals as its restriction, and the determined feature as its
predicate ("things with wings and with feathers can fly"). All of a term's negated IS literals
go into one relative clause, joined with "and" ("things with wings that are not red and not big
can fly"). A term is skipped, and counted, when it reads a scalar threshold, when it has more
literals than ``propositions.rule_statements.max_literals``, when one of its concepts has no
word, or when no instance satisfies it. A rule statement is true with ``all`` and as a bare
generic, and its quantifier is drawn like that of any class-level proposition.

A class-level subject can be restricted, in the proposition layer, where the truth of the
sentence is grounded. A restriction is one literal ("red penguins"), and a relative clause is
restrictive ("penguins that can swim", "owls that eat mice", "mice that owls eat"). Both are drawn
among those that some members of the category satisfy, and not all, so the restriction does work.

An event is reported by an event-level proposition. Its verb is named at a level of the verb
tree, as a noun names a category at a level of the noun tree: the verb itself, or a verb category
above it ("chase" or "hunt"), drawn by ``mention.verb_level_weights``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from semantic_world.common.boolean import minimal_dnf
from semantic_world.corpus.config import Config
from semantic_world.corpus.lexicon import THING, Lexicon, world_concepts
from semantic_world.corpus.propositions import (
    ALL,
    CAN,
    CLASS,
    GENERIC,
    HAS,
    INSTANCE,
    IS,
    MEMBER,
    MOST,
    NO,
    PROJECTION,
    SCALAR,
    SOME,
    VERB,
    CategoryTerm,
    Clause,
    Literal,
    Predicate,
    Proposition,
    Truth,
)
from semantic_world.taxonomy.generate import TaxonomyResult
from semantic_world.taxonomy.rules import Threshold

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

_POSITIVE_ORDER = ((ALL, True), (MOST, True), (SOME, True))
_NEGATIVE_ORDER = ((NO, True), (MOST, False), (SOME, False))


class Facts:
    """The true, expressible propositions of one world. With a lexicon, only concepts that have
    a word take part. Without one, every concept of the world does."""

    def __init__(
        self,
        config: Config,
        result: TaxonomyResult,
        lexicon: Lexicon | None = None,
        truth: Truth | None = None,
    ) -> None:
        self.config = config
        self.result = result
        self.truth = truth or Truth(config, result)
        concepts, _ = world_concepts(result)
        if lexicon is not None:
            concepts = tuple(c for c in concepts if lexicon.is_named(c.label))
        self.named = frozenset(c.label for c in concepts)

        def of_type(*types: str) -> tuple[str, ...]:
            return tuple(c.label for c in concepts if c.type in types)

        self.categories = of_type("category")
        self.features = {kind: of_type(kind) for kind in (IS, HAS, CAN)}
        self.projections = of_type("patient_projection")
        self.poles = of_type("scalar")
        self.verbs = of_type("verb", "verb_category")
        self.level = {c.label: c.level for c in result.tree.categories}
        self._class_facts: dict[tuple, tuple[Proposition, ...]] = {}
        self._restrictions: dict[CategoryTerm, tuple[Literal, ...]] = {}
        self._clauses: dict[tuple[CategoryTerm, bool], tuple[Clause, ...]] = {}
        self._rules: tuple[Proposition, ...] | None = None
        self.rule_report: dict[str, Any] = {}

    def expressible(self, proposition: Proposition) -> bool:
        """Whether every concept the proposition needs has a word."""
        return all(label in self.named for label in proposition.concepts())

    # Class level -----------------------------------------------------------------------------

    def class_predicates(
        self, category: str, patients: tuple[str, ...] | None = None
    ) -> list[Predicate]:
        """Every predicate that a sentence about a category could have: features, exposed patient
        projections, scalar poles, membership in a category at the same level or above, and a
        verb with a patient category (every named category, or ``patients``)."""
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
        """The strongest true proposition that asserts a predicate of a subject, or denies it
        with ``negative``; None when there is none. A scalar pole takes the generic only."""
        if predicate.kind == SCALAR:
            options: tuple[tuple[str, bool], ...] = ((GENERIC, not negative),)
        elif predicate.kind == MEMBER:
            options = ((NO, True),) if negative else ((ALL, True),)
        else:
            options = _NEGATIVE_ORDER if negative else _POSITIVE_ORDER
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
        """The strongest true relation facts with a category as the patient, for every verb and
        every agent category (every named category, or ``agents``)."""
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

    def generic(self, proposition: Proposition) -> Proposition | None:
        """The bare generic that says the same as a class-level proposition, when it is true:
        "penguins swim" for "most penguins swim", and "penguins can not fly" for "no penguins
        can fly"."""
        candidate = dataclasses.replace(
            proposition, quantifier=GENERIC, polarity=not proposition.negative, grounding=None
        )
        return self.truth.grounded(candidate)

    # Instance level --------------------------------------------------------------------------

    def instance_predicates(
        self, instance: str, patients: tuple[str, ...] | None = None
    ) -> list[Predicate]:
        """Every predicate that a sentence about an instance could have: features, exposed
        patient projections, scalar poles against every named category the instance is below,
        membership in a category, and a verb with a patient instance (every other instance, or
        ``patients``)."""
        index = self.truth.instance_index[instance]
        predicates = [Predicate(kind, f) for kind in (IS, HAS, CAN) for f in self.features[kind]]
        predicates += [Predicate(PROJECTION, p) for p in self.projections]
        path = [c for c in self.truth.paths[index] if c in self.named]
        predicates += [Predicate(SCALAR, p, comparison=c) for p in self.poles for c in path]
        predicates += [Predicate(MEMBER, c) for c in self.categories]
        others = self.result.instances.labels if patients is None else patients
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
        """The literals that can restrict a category term further: the IS and HAS literals,
        positive and negative, and the scalar poles, that some members satisfy and not all."""
        if term not in self._restrictions:
            truth = self.truth
            members = truth.members(term)
            stated = {literal.feature for literal in term.restriction}
            options: list[Literal] = []
            for feature in self.features[IS] + self.features[HAS]:
                if feature in stated:
                    continue
                count = int(truth.values[members, truth.features[feature].position].sum())
                if 0 < count < len(members):
                    options += [Literal(feature), Literal(feature, False)]
            for pole in self.poles:
                if pole in stated:
                    continue
                mask = truth.pole_mask(pole, truth._below[term.category])[0]
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
        satisfy and not all. A subject relative holds a CAN feature ("that can swim") or a verb
        with a patient category ("that eat mice"). An object relative holds a verb with an agent
        category ("that owls eat")."""
        key = (term, object_relative)
        if key not in self._clauses:
            truth = self.truth
            members = truth.members(term)
            total = len(members)
            options: list[Clause] = []
            if not object_relative:
                for feature in self.features[CAN]:
                    count = int(truth.values[members, truth.features[feature].position].sum())
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
        a CAN feature or a verb, each kind with the same chance. When the drawn kind has no
        clause, the other kind is used. The other category of a verb can take a relative clause
        of its own, at the rate, up to ``max_depth``. None when the term has a clause already,
        or when no clause restricts it."""
        settings = self.config.mention.relative_clauses
        if term.clauses or depth > settings.max_depth:
            return None
        negated = any(not x.positive and x.feature.startswith("IS.") for x in term.restriction)
        as_object = rng.random() < settings.object_share and not negated
        total = len(self.truth.members(term))
        for object_relative in (as_object, not as_object):
            if object_relative and negated:
                continue  # the negated IS literals need a subject relative to join
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
        terms = 0
        for rule in self.result.rules.rules:
            output = rule.output
            for number, term in enumerate(minimal_dnf(rule.table), start=1):
                terms += 1
                inputs = [(item, value) for item, value in zip(rule.inputs, term, strict=True)]
                used = [(item, value) for item, value in inputs if value is not None]
                if any(isinstance(item, Threshold) for item, _ in used):
                    skipped[SKIP_THRESHOLD] += 1
                    continue
                literals = tuple(Literal(item.label, bool(value)) for item, value in used)
                if cap is not None and len(literals) > cap:
                    skipped[SKIP_MAX_LITERALS] += 1
                    continue
                statement = Proposition(
                    CLASS,
                    CategoryTerm(THING, literals),
                    Predicate(output.type, output.label),
                    True,
                    ALL,
                    rule=(output.label, number),
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
            "rules": len(self.result.rules.rules),
            "terms": terms,
            "stated": len(statements),
            "skipped": skipped,
        }
        return tuple(statements)

    # Events ----------------------------------------------------------------------------------

    def event_names(self, event: Any) -> tuple[str, ...]:
        """The labels that have a word and can name an event: its CAN feature, or its verb and
        the verb categories above it, from the verb upward."""
        return tuple(label for label in self.truth.verb_names(event.verb) if label in self.named)

    def event_fact(self, event: Any, verb: str | None = None) -> Proposition | None:
        """The event-level proposition that reports an event, named by its own verb or by
        ``verb``, a verb category above it. None when the name has no word, or does not name
        the event."""
        candidate = event.proposition(verb, self.config.propositions.event_tense)
        if not self.expressible(candidate):
            return None
        return self.truth.grounded(candidate)

    def draw_event(self, rng: np.random.Generator, event: Any) -> Proposition | None:
        """The proposition that reports an event, with its verb named at a level drawn by
        ``mention.verb_level_weights`` among the levels that have a word. A level with the
        weight 0 is never used. None when no level can name the event."""
        names = self.event_names(event)
        if not event.transitive:
            return self.event_fact(event) if names else None
        assert self.result.verbs is not None
        levels = self.config.mention.verb_level_weights
        weights = np.array(
            [levels[self.result.verbs.tree[label].level - 1] for label in names], dtype=float
        )
        if weights.sum() == 0:
            return None
        name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        return self.event_fact(event, name)

    # Drawing ---------------------------------------------------------------------------------

    def draw_class(
        self,
        rng: np.random.Generator,
        subject: CategoryTerm | str,
        patients: tuple[str, ...] | None = None,
    ) -> Proposition | None:
        """One true class-level proposition about a subject: negative at the class-level
        negation rate, and a bare generic at the generic rate when the generic is true. When the
        subject has no fact of the drawn polarity, the other polarity is used."""
        negative = rng.random() < self.config.propositions.negation_rate[CLASS]
        pool = self.class_facts(subject, negative, patients) or self.class_facts(
            subject, not negative, patients
        )
        if not pool:
            return None
        fact = pool[int(rng.integers(len(pool)))]
        if rng.random() < self.config.quantifiers.generic_rate:
            fact = self.generic(fact) or fact
        return fact

    def draw_rule_statement(
        self, rng: np.random.Generator, pool: tuple[Proposition, ...] | None = None
    ) -> Proposition | None:
        """One rule statement, from every rule statement of the world or from ``pool``. Its
        quantifier is drawn like that of any class-level proposition: ``all``, or the bare
        generic at the generic rate. A rule statement is true under both."""
        pool = self.rule_statements() if pool is None else pool
        if not pool:
            return None
        statement = pool[int(rng.integers(len(pool)))]
        if rng.random() < self.config.quantifiers.generic_rate:
            statement = self.generic(statement) or statement
        return statement

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
