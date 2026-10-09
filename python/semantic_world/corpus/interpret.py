"""Layer 4, backwards: a parse tree becomes the sentence plan it realizes.

``interpret`` reads a tree by its labels, so it does not need the word-order settings: a subject
is ``NP-SBJ`` wherever it stands. The words give the concepts. What the words cannot give comes
from the sentence's record: which instance each noun phrase refers to (``referents``), and which
event each verb phrase reports, with its tense and aspect (``events``), both in the order of the
tree, a node before its children.

The reading undoes the grammar's fixed choices:

- an adjective is a positive PROPERTY literal or a scalar pole, a with-phrase a positive PART
  literal, and a without-phrase a negative one;
- a verb phrase "is not A" in a relative clause is a negated PROPERTY literal of the
  restriction;
- a noun phrase with ``a`` or ``the``, or a pronoun, names an instance, and any other noun
  phrase names a category;
- a verb states a capacity when its subject is a category, with ``can`` or without. With an
  instance as its subject, a verb with ``can`` states a capacity, and a verb without ``can``
  reports an event or states a capacity. The words cannot tell the two apart when the tense and
  the aspect are not marked, so the record decides: the verb phrase reports the event that the
  record gives, and states a capacity when the record gives none. A verb with ``can`` (or the
  word ``can_now``) whose record gives a time point says what was possible then (``able_now``);
- a state adjective (the word of a fluent) states the fluent at the time point the record
  gives: a state with the copula, and a change with ``become``. With a category as the subject
  it makes a causal statement, an effect with ``become`` and a precondition with the copula and
  ``before``, and the subject's relative clause then names the event: its verb is bound to the
  event variable. Nothing of the record is needed: the words say it;
- a verb phrase without an auxiliary in a joined relative clause takes the auxiliary of the
  verb phrase before it, when that auxiliary fits its predicate word.

The quantifier of a class-level sentence is not in the words either: what "all" expresses is a
setting of the language, and a bare plural says nothing of it. The record gives it
(``quantifier``), as it gives the events.

The inflections, the adjective order, and the choice between synonyms carry no part of the plan,
and are ignored.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from semantic_world.corpus.grammar import (
    CLASS_NP,
    INSTANCE_DETERMINERS,
    INSTANCE_NP,
    GrammarError,
    NounPhrase,
    Predication,
    RelativeClause,
    SentencePlan,
)
from semantic_world.corpus.histories import is_time_key
from semantic_world.corpus.lexicon import ABLE_NOW_WORD, BEFORE_WORD, Lexicon
from semantic_world.corpus.propositions import (
    CAN,
    CAUSAL_KINDS,
    EFFECT,
    EVENT_VARIABLE,
    HAS,
    IS,
    MEMBER,
    NEC_ALL,
    NEC_NO,
    NO,
    PRECONDITION,
    PRESENT,
    PROJECTION,
    SCALAR,
    STATE_KIND,
    VERB,
    Literal,
)
from semantic_world.corpus.realize import NP_LABELS, Tree, preorder, token_parts
from semantic_world.corpus.world import (
    FLUENT_PREFIX,
    ONE_PLACE_PREFIX,
    PATIENT_CAPACITY_PREFIX,
    SCALAR_PREFIX,
)

ABLE_NOW_AUXILIARIES = ("can", ABLE_NOW_WORD)


def interpret(
    tree: Tree,
    lexicon: Lexicon,
    referents: Sequence[tuple[str, str | None]] | Sequence[str],
    events: Sequence[tuple[str, str, str | None] | None] = (),
    quantifier: str | None = None,
) -> SentencePlan:
    """The sentence plan that a tree realizes. ``referents`` gives the referent of every noun
    phrase, ``events`` the event of every verb phrase, as its label, its tense, and its aspect,
    or the time point a verb phrase is about (``SCENE.8.TIME.2``), its tense, and None, or
    None, in the order of the tree, and ``quantifier`` the quantifier of a class-level
    sentence."""
    return _Reader(tree, lexicon, referents, events, quantifier).sentence()


class _Reader:
    def __init__(
        self,
        tree: Tree,
        lexicon: Lexicon,
        referents: Sequence,
        events: Sequence,
        quantifier: str | None,
    ) -> None:
        self.tree = tree
        self.lexicon = lexicon
        self.quantifier = quantifier
        nodes = preorder(tree)
        phrases = [n for n in nodes if n[0] in NP_LABELS]
        verb_phrases = [n for n in nodes if n[0] == "VP"]
        if len(referents) != len(phrases):
            raise GrammarError(
                f"the tree has {len(phrases)} noun phrases, and {len(referents)} referents "
                "were given"
            )
        events = list(events) or [None] * len(verb_phrases)
        if len(events) != len(verb_phrases):
            raise GrammarError(
                f"the tree has {len(verb_phrases)} verb phrases, and {len(events)} events "
                "were given"
            )
        self.referent = {
            id(node): (r if isinstance(r, str) else r[0])
            for node, r in zip(phrases, referents, strict=True)
        }
        self.event = {id(node): e for node, e in zip(verb_phrases, events, strict=True)}

    # Words -----------------------------------------------------------------------------------

    def token(self, node: Tree) -> str:
        """The token of a word node, past any inflection realized as a separate word."""
        for child in node[1:]:
            if isinstance(child, str):
                return child
            if child[0] == node[0]:
                return self.token(child)
        raise GrammarError(f"the node {node[0]} holds no word")

    def marks(self, node: Tree) -> set[str]:
        """The inflections of a word node: its affix, and the markers that stand beside it."""
        found: set[str] = set()
        for child in node[1:]:
            if isinstance(child, str):
                affix = token_parts(child)[1]
            elif child[0] == node[0]:
                found |= self.marks(child)
                continue
            else:
                affix = child[0]
            if affix is not None:
                found.add(affix)
        return found

    def concept(self, node: Tree) -> str:
        return self.lexicon.lexeme(token_parts(self.token(node))[0]).concept

    def gloss(self, node: Tree) -> str:
        return self.lexicon.lexeme(token_parts(self.token(node))[0]).gloss

    @staticmethod
    def children(node: Tree, *labels: str) -> list[Tree]:
        return [c for c in node[1:] if not isinstance(c, str) and c[0] in labels]

    def child(self, node: Tree, *labels: str) -> Tree | None:
        found = self.children(node, *labels)
        if len(found) > 1:
            raise GrammarError(f"{node[0]} has more than one {' or '.join(labels)}")
        return found[0] if found else None

    # The sentence ----------------------------------------------------------------------------

    def sentence(self) -> SentencePlan:
        if self.tree[0] != "S":
            raise GrammarError(f"a sentence is an S, not {self.tree[0]}")
        subject_node = self.child(self.tree, "NP-SBJ")
        verb_phrase = self.child(self.tree, "VP")
        if subject_node is None or verb_phrase is None:
            raise GrammarError("a sentence has a subject and a verb phrase")
        subject = self.noun_phrase(subject_node)
        target = self.child(self.tree, "NP-OBJ") or self.child(verb_phrase, "NP-OBJ")
        predication, _ = self.predication(verb_phrase, target, subject.kind, None)
        if predication.kind in CAUSAL_KINDS:
            # a causal statement: the subject's relative clause names the event
            return SentencePlan(self.bound_subject(subject), predication, NEC_ALL)
        quantifier = self.quantifier if subject.kind == CLASS_NP else None
        if (
            subject.kind == CLASS_NP
            and subject.determiner is None
            and quantifier in (NO, NEC_NO)
            and not predication.polarity
        ):
            # the negation of a bare plural states the quantifier ("penguins are not fish")
            predication = dataclasses.replace(predication, polarity=True)
        return SentencePlan(subject, predication, quantifier)

    @staticmethod
    def bound_subject(subject: NounPhrase) -> NounPhrase:
        """The subject of a causal statement, with the verb of its relative clause bound to the
        event variable: the clause names the events of the verb's event type."""
        clause = subject.clause
        if subject.kind != CLASS_NP or clause is None or len(clause.predications) != 1:
            raise GrammarError(
                "the subject of a causal statement is a category with a relative clause that "
                "names the event"
            )
        bound = dataclasses.replace(
            clause.predications[0], event=EVENT_VARIABLE, tense=PRESENT, aspect=None
        )
        return dataclasses.replace(
            subject, clause=dataclasses.replace(clause, predications=(bound,))
        )

    def noun_phrase(self, node: Tree) -> NounPhrase:
        referent = self.referent[id(node)]
        if self.child(node, "Pro") is not None:
            return NounPhrase(INSTANCE_NP, referent)
        noun_node = self.child(node, "N")
        if noun_node is None:
            raise GrammarError("a noun phrase has a noun or a pronoun")
        noun = self.concept(noun_node)
        determiner_node = self.child(node, "Det")
        determiner = None if determiner_node is None else self.gloss(determiner_node)
        kind = INSTANCE_NP if determiner in INSTANCE_DETERMINERS else CLASS_NP
        if kind == CLASS_NP and referent != noun:
            raise GrammarError(
                f"the noun names the category {noun}, and the referent given is {referent}"
            )
        literals: list[Literal] = []
        adjectives = self.child(node, "AP")
        if adjectives is not None:
            literals += [Literal(self.concept(a)) for a in self.children(adjectives, "A")]
        for phrase in self.children(node, "PP"):
            adposition = self.child(phrase, "P")
            part = self.child(phrase, "N")
            if adposition is None or part is None:
                raise GrammarError("a with-phrase has an adposition and a noun")
            literals.append(Literal(self.concept(part), self.gloss(adposition) == "with"))
        clause: RelativeClause | None = None
        clause_node = self.child(node, "RC")
        if clause_node is not None:
            agent_node = self.child(clause_node, "NP-SBJ")
            verb_phrases = self.children(clause_node, "VP")
            if agent_node is not None:
                # an object relative: its own subject, and one verb whose patient is the head
                agent = self.noun_phrase(agent_node)
                if len(verb_phrases) != 1:
                    raise GrammarError("an object relative has one verb phrase")
                predication, _ = self.predication(verb_phrases[0], None, agent.kind, None)
                clause = RelativeClause((predication,), agent)
            else:
                predications = []
                previous: str | None = None
                for verb_phrase in verb_phrases:
                    target = self.child(verb_phrase, "NP-OBJ")
                    predication, previous = self.predication(verb_phrase, target, kind, previous)
                    if predication.kind == IS and not predication.polarity:
                        literals.append(Literal(predication.label, False))
                    else:
                        predications.append(predication)
                if predications:
                    clause = RelativeClause(tuple(predications))
        return NounPhrase(kind, referent, noun, determiner, tuple(literals), clause)

    def predication(
        self, node: Tree, target: Tree | None, subject_kind: str, previous: str | None
    ) -> tuple[Predication, str | None]:
        """The predication of a verb phrase, and the gloss of its auxiliary. ``previous`` is the
        auxiliary of the verb phrase before it in a joined relative clause."""
        auxiliary_node = self.child(node, "AUX")
        auxiliary = None if auxiliary_node is None else self.gloss(auxiliary_node)
        polarity = self.child(node, "Neg") is None
        verb = self.child(node, "V")
        adjective = self.child(node, "A")
        nominal = self.child(node, "NP-PRD")
        if auxiliary is None and previous is not None:
            # an auxiliary that is not said again, when it fits the predicate word
            fits = (
                (previous in ABLE_NOW_AUXILIARIES and verb is not None)
                or (previous in ("is", "are") and (adjective is not None or nominal is not None))
                or (previous in ("has", "have") and nominal is not None)
            )
            if fits:
                auxiliary = previous
        if verb is not None:
            concept = self.concept(verb)
            kind = CAN if concept.startswith(ONE_PLACE_PREFIX) else VERB
            patient = None if target is None else self.noun_phrase(target)
            report = self.event[id(node)]
            if report is not None and is_time_key(report[0]):
                if auxiliary not in ABLE_NOW_AUXILIARIES or subject_kind != INSTANCE_NP:
                    raise GrammarError(
                        "what was possible at a time point is said with can (or can_now) and an "
                        "instance as its subject"
                    )
                time, tense, _ = report
                return (
                    Predication(kind, concept, polarity, patient, tense=tense, time=time),
                    auxiliary,
                )
            if report is None:
                marks = self.marks(verb) & {"PAST", "PROGRESSIVE"}
                if marks:
                    raise GrammarError(
                        f"the verb is marked {sorted(marks)}, so it reports an event, and no "
                        "event was given for it"
                    )
                return Predication(kind, concept, polarity, patient), auxiliary
            if auxiliary is not None or subject_kind != INSTANCE_NP:
                raise GrammarError(
                    "an event is reported by a verb with no auxiliary and an instance as its "
                    "subject"
                )
            event, tense, aspect = report
            return Predication(kind, concept, polarity, patient, event, tense, aspect), auxiliary
        if adjective is not None:
            concept = self.concept(adjective)
            if concept.startswith(FLUENT_PREFIX) and subject_kind == CLASS_NP:
                # a causal statement: an effect with become, a precondition with the copula
                # and "before"
                adverb = self.child(node, "Adv")
                before = adverb is not None and self.gloss(adverb) == BEFORE_WORD
                if auxiliary == "become" and not before:
                    return Predication(EFFECT, concept, polarity), auxiliary
                if auxiliary in ("is", "are") and before:
                    return Predication(PRECONDITION, concept, polarity), auxiliary
                raise GrammarError(
                    f"the state adjective {concept} with a category as the subject makes a "
                    "causal statement: an effect with become, or a precondition with the copula "
                    f"and {BEFORE_WORD}"
                )
            if concept.startswith(FLUENT_PREFIX):
                report = self.event[id(node)]
                if report is None or not is_time_key(report[0]):
                    raise GrammarError(
                        f"the state adjective {concept} states a fluent at a time point, and no "
                        "time point was given for its verb phrase"
                    )
                if auxiliary not in ("is", "are", "become"):
                    raise GrammarError("a fluent is stated with the copula or with become")
                time, tense, _ = report
                become = auxiliary == "become"
                return (
                    Predication(
                        STATE_KIND, concept, polarity, tense=tense, time=time, become=become
                    ),
                    auxiliary,
                )
            if self.event[id(node)] is not None:
                raise GrammarError(
                    f"the adjective {concept} is static, and a time point was given for its "
                    "verb phrase"
                )
            if concept.startswith(SCALAR_PREFIX):
                kind = SCALAR
            elif concept.startswith(PATIENT_CAPACITY_PREFIX):
                kind = PROJECTION
            else:
                kind = IS
            return Predication(kind, concept, polarity), auxiliary
        if nominal is None:
            raise GrammarError("a verb phrase has a verb, an adjective, or a predicate noun")
        noun = self.child(nominal, "N")
        if noun is None:
            raise GrammarError("a predicate noun phrase has a noun")
        concept = self.concept(noun)
        if auxiliary in ("has", "have"):
            determiner = self.child(nominal, "Det")
            negative = determiner is not None and self.gloss(determiner) == "no"
            return Predication(HAS, concept, not negative), auxiliary
        return Predication(MEMBER, concept, polarity), auxiliary
