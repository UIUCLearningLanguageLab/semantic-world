"""Layer 4, backwards: a parse tree becomes the sentence plan it realizes.

``interpret`` reads a tree by its labels, so it does not need the word-order settings: a subject
is ``NP-SBJ`` wherever it stands. The words give the concepts. What the words cannot give comes
from the sentence's record: which instance each noun phrase refers to (``referents``), and which
event each verb phrase reports (``events``), both in the order of the tree, a node before its
children.

The reading undoes the grammar's fixed choices:

- an adjective is a positive IS literal or a scalar pole, a with-phrase a positive HAS literal,
  and a without-phrase a negative one;
- a verb phrase "is not A" in a relative clause is a negated IS literal of the restriction;
- a noun phrase with ``a`` or ``the``, or a pronoun, names an instance, and any other noun
  phrase names a category;
- a verb with no auxiliary reports an event when its subject is an instance, and states a
  capacity when its subject is a category, with ``can`` or without;
- a verb phrase without an auxiliary in a joined relative clause takes the auxiliary of the
  verb phrase before it, when that auxiliary fits its predicate word.

The inflections, the adjective order, and the choice between synonyms carry no part of the plan,
and are ignored.
"""

from __future__ import annotations

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
from semantic_world.corpus.lexicon import Lexicon
from semantic_world.corpus.propositions import (
    CAN,
    HAS,
    IS,
    MEMBER,
    PROJECTION,
    SCALAR,
    VERB,
    Literal,
)
from semantic_world.corpus.realize import NP_LABELS, Tree, preorder, token_parts


def interpret(
    tree: Tree,
    lexicon: Lexicon,
    referents: Sequence[tuple[str, str | None]] | Sequence[str],
    events: Sequence[str | None] = (),
) -> SentencePlan:
    """The sentence plan that a tree realizes. ``referents`` gives the referent of every noun
    phrase, and ``events`` the event of every verb phrase, in the order of the tree."""
    return _Reader(tree, lexicon, referents, events).sentence()


class _Reader:
    def __init__(self, tree: Tree, lexicon: Lexicon, referents: Sequence, events: Sequence) -> None:
        self.tree = tree
        self.lexicon = lexicon
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
        return SentencePlan(subject, predication)

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
                (previous == "can" and verb is not None)
                or (previous in ("is", "are") and (adjective is not None or nominal is not None))
                or (previous in ("has", "have") and nominal is not None)
            )
            if fits:
                auxiliary = previous
        if verb is not None:
            concept = self.concept(verb)
            kind = CAN if concept.startswith("CAN.") else VERB
            patient = None if target is None else self.noun_phrase(target)
            event = None
            if auxiliary is None and subject_kind == INSTANCE_NP:
                event = self.event[id(node)]
                if event is None:
                    raise GrammarError(
                        "a verb with no auxiliary and an instance as its subject reports an "
                        "event, and no event was given for it"
                    )
            return Predication(kind, concept, polarity, patient, event), auxiliary
        if adjective is not None:
            concept = self.concept(adjective)
            if concept.startswith("SC."):
                kind = SCALAR
            elif concept.startswith("CANBE."):
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
