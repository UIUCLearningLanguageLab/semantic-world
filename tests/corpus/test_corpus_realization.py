"""Stage 4 acceptance tests: realization of generated sentences.

For sentences made from the facts, the rule statements, and the events of several worlds, with
relative clauses drawn, and under several grammars: every tree's leaves equal its tokens;
``interpret(tree)`` recovers the plan exactly; with agreement on, every verb and auxiliary agrees
with its subject, including across relative clauses; and the relative-clause depth never exceeds
the limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_world.corpus import Streams, scene_events
from semantic_world.corpus.grammar import CLASS_NP, check_plan
from semantic_world.corpus.interpret import interpret
from semantic_world.corpus.mentions import RelativeClauses, plan_for
from semantic_world.corpus.propositions import HAS, IS, VERB, CategoryTerm, Literal, Predicate
from semantic_world.corpus.realize import leaves, token_parts, tree_depth

WORLDS = ("tiny", "default", "deep")
ORDERS = ("SVO", "SOV", "VSO", "VOS", "OVS", "OSV")
FLIPPED = {
    "determiner": "after",
    "adjective": "after",
    "with_phrase": "before",
    "relative_clause": "before",
    "adposition": "postposition",
    "auxiliary": "after",
    "negation": "before_auxiliary",
}
GRAMMARS = {
    "default": {},
    "english": {
        "morphology": {
            "number": {"enabled": True, "verb_marks": "singular"},
            "tense": {"enabled": True},
        }
    },
    "plural_verbs": {
        "morphology": {"number": {"enabled": True}, "aspect": {"enabled": True}},
        "can_rate": {"class": 0.2},
    },
    "marker_words": {
        "morphology": {
            "number": {"enabled": True, "realization": "word", "position": "before"},
            "tense": {"enabled": True, "realization": "word"},
            "aspect": {"enabled": True},
        }
    },
    "bare": {"can_rate": {"class": 0.5, "instance": 0.3}},
    "bare_english": {
        "can_rate": {"class": 0.0, "instance": 0.0},
        "morphology": {"number": {"enabled": True, "verb_marks": "singular"}},
    },
    "no_agreement": {"morphology": {"number": {"enabled": True, "agreement": False}}},
    "flipped": {
        "word_order": {"clause": "VSO", **FLIPPED},
        "morphology": {"number": {"enabled": True}, "tense": {"enabled": True}},
        "adjective_order": {"fixed": False},
    },
    **{f"clause_{order}": {"word_order": {"clause": order}} for order in ORDERS},
}
MENTION = {"relative_clauses": {"rate": 0.5, "max_depth": 2, "object_share": 0.4}}
LEXICON = {"synonym_rate": 0.3}


def plans_of(case, limit: int = 2):
    """Sentence plans of every kind for a world: class-level facts with and without
    restrictions, rule statements, instance-level facts, and events, with relative clauses."""
    settings = {
        "mention": {"relative_clauses": {**MENTION["relative_clauses"], "max_depth": limit}}
    }
    facts = case.facts(lexicon=LEXICON, **settings)
    clauses = RelativeClauses(case.config(**settings), facts)
    rng = np.random.default_rng(8)
    propositions = []
    bare = []  # class-level propositions said with a bare plural
    features = facts.features[IS] + facts.features[HAS]
    for category in facts.categories[:: max(1, len(facts.categories) // 6)]:
        for negative in (False, True):
            stated = facts.class_facts(category, negative, patients=facts.categories[:4])
            propositions += stated[:: max(1, len(stated) // 12)]
            bare += stated[::9]
        # subjects with a restriction that one member satisfies, negated IS literals included
        members = facts.truth.members(CategoryTerm(category))
        for _ in range(3):
            row = int(rng.choice(members))
            chosen = [features[int(i)] for i in rng.choice(len(features), size=3, replace=False)]
            literals = [Literal(f, bool(case.world.column(f)[row])) for f in chosen]
            if facts.poles:
                literals.append(Literal(facts.poles[0]))
            term = CategoryTerm(category, tuple(literals))
            propositions += facts.class_facts(term, patients=facts.categories[:2])[::7]
        # subjects and patients with a restrictive relative clause, of the three kinds
        for _ in range(3):
            term = facts.draw_clause(rng, CategoryTerm(category))
            if term is not None:
                propositions += facts.class_facts(term, patients=facts.categories[:2])[::7]
                propositions += facts.patient_facts(category, agents=facts.categories[:2])[:1]
                reversed_role = [
                    facts.class_fact(CategoryTerm(agent), Predicate(VERB, verb, term))
                    for agent in facts.categories[:2]
                    for verb in facts.verbs[:2]
                ]
                propositions += [fact for fact in reversed_role if fact is not None]
    propositions += list(facts.rule_statements())
    bare += facts.rule_statements()[::2]
    plans = [plan_for(facts, p) for p in propositions]
    plans += [plan_for(facts, p, bare=True) for p in bare]

    labels = case.world.instances
    others = labels[:: max(1, len(labels) // 6)][:6]
    for instance in others[:4]:
        for negative in (False, True):
            stated = facts.instance_facts(instance, negative, patients=others)
            for fact in stated[:: max(1, len(stated) // 12)]:
                plans.append(clauses.attach(rng, plan_for(facts, fact), others))
    scenes = case.scenes(lexicon=LEXICON, **settings)
    streams = Streams(1)
    for number in range(1, 26):
        scene = scenes.scene(streams, number, labels[(number * 5) % len(labels)])
        facts.truth.add_scene(scene)
        for event in scene_events(scene):
            report = facts.draw_event(rng, event, "progressive" if rng.random() < 0.3 else "simple")
            if report is not None:
                plans.append(clauses.attach(rng, plan_for(facts, report)))
    for plan in plans:
        check_plan(plan)
    return plans


@pytest.fixture(scope="module")
def plans(cases):
    made = {}

    def of(name: str):
        if name not in made:
            made[name] = plans_of(cases(name))
        return made[name]

    return of


# ---------------------------------------------------------------------------------------------
# An independent check of agreement
# ---------------------------------------------------------------------------------------------


class Agreement:
    """Checks every verb phrase of a tree against the number of its subject, from the tree and
    the lexicon alone."""

    def __init__(self, lexicon, morphology) -> None:
        self.lexicon = lexicon
        self.morphology = morphology
        self.checked = 0
        self.across = 0

    def gloss(self, node) -> str:
        token = next(
            c if isinstance(c, str) else c[1]
            for c in node[1:]
            if isinstance(c, str) or c[0] == node[0]
        )
        return self.lexicon.lexeme(token_parts(token)[0]).gloss

    @staticmethod
    def marks(node) -> set[str]:
        """The inflections of a word: its affix, and the markers that stand beside it."""
        found = set()
        for child in node[1:]:
            if isinstance(child, str):
                affix = token_parts(child)[1]
                found |= {affix} if affix else set()
            elif child[0] == node[0]:
                affix = token_parts(child[1])[1]
                found |= {affix} if affix else set()
            else:
                found.add(child[0])
        return found

    @staticmethod
    def child(node, label):
        return next((c for c in node[1:] if not isinstance(c, str) and c[0] == label), None)

    def plural(self, phrase) -> bool:
        noun = self.child(phrase, "N")
        return noun is not None and "PLURAL" in self.marks(noun)

    def sentence(self, tree) -> None:
        subject = self.child(tree, "NP-SBJ")
        self.phrase(subject)
        for node in tree[1:]:
            if node[0] == "VP":
                self.verb_phrase(
                    node, self.plural(subject), None, self.child(subject, "RC") is not None
                )
            elif node[0] == "NP-OBJ":
                self.phrase(node)

    def phrase(self, phrase) -> None:
        clause = self.child(phrase, "RC")
        if clause is None:
            return
        agent = self.child(clause, "NP-SBJ")
        if agent is not None:
            self.phrase(agent)
        # the verbs of a subject relative agree with the head, and those of an object relative
        # with the clause's own subject
        plural = self.plural(agent if agent is not None else phrase)
        previous = None
        for node in clause[1:]:
            if node[0] == "VP":
                previous = self.verb_phrase(node, plural, previous, False)

    def verb_phrase(self, node, plural: bool, previous: str | None, across: bool) -> str | None:
        number = self.morphology.number
        agreement = number.enabled and number.agreement
        target = self.child(node, "NP-OBJ")
        if target is not None:
            self.phrase(target)
        auxiliary_node = self.child(node, "AUX")
        auxiliary = None if auxiliary_node is None else self.gloss(auxiliary_node)
        verb = self.child(node, "V")
        nominal = self.child(node, "NP-PRD")
        self.checked += 1
        self.across += across
        if auxiliary in ("is", "are"):
            assert (auxiliary == "are") == (plural and agreement), node
        if auxiliary in ("has", "have"):
            assert (auxiliary == "have") == (plural and agreement), node
        if nominal is not None and auxiliary in ("is", "are", None):
            noun = self.child(nominal, "N")
            if auxiliary is not None or previous in ("is", "are"):
                # a predicate noun is plural with a plural subject, and takes "a" otherwise
                assert ("PLURAL" in self.marks(noun)) == plural, node
                assert (self.child(nominal, "Det") is None) == plural, node
        if verb is not None:
            marks = self.marks(verb)
            under_can = auxiliary == "can" or (auxiliary is None and previous == "can")
            if under_can or marks & {"PAST", "PROGRESSIVE"} or not agreement:
                assert "PLURAL" not in marks, node
            else:
                marked = number.verb_marks == ("plural" if plural else "singular")
                assert ("PLURAL" in marks) == marked, node
        return auxiliary or previous


# ---------------------------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", WORLDS)
@pytest.mark.parametrize("grammar", list(GRAMMARS))
def test_generated_sentences(cases, plans, name, grammar) -> None:
    case = cases(name)
    sections = {"grammar": GRAMMARS[grammar], "lexicon": LEXICON}
    realizer = case.realizer(**sections)
    lexicon = case.lexicon(**sections)
    agreement = Agreement(lexicon, case.config(**sections).grammar.morphology)
    rng = Streams(3).grammar
    made = plans(name)
    assert len(made) > 300
    levels = set()
    for plan in made:
        sentence = realizer.realize(plan, rng)
        # every tree's leaves equal its tokens
        assert tuple(leaves(sentence.tree)) == sentence.tokens
        assert all(
            token_parts(t)[0] in {x.label for x in lexicon.lexemes} for t in sentence.tokens[:3]
        )
        # interpret(tree) recovers the plan exactly
        read = interpret(
            sentence.tree, lexicon, sentence.referents, sentence.events, plan.quantifier
        )
        assert read == plan
        # every verb and auxiliary agrees with its subject
        agreement.sentence(sentence.tree)
        # the tree is as deep as the plan, and no deeper than the limit
        assert tree_depth(sentence.tree) == plan.depth() <= 2
        assert len(sentence.referents) == len(plan.noun_phrases())
        assert [r for r, _ in sentence.referents if r] and len(sentence.conceptual.split()) == len(
            sentence.tokens
        )
        levels.add(plan.level)
    assert levels == {"class", "instance", "event"}
    assert agreement.checked > len(made)
    assert agreement.across > 20  # subjects with a relative clause between the noun and the verb


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_relative_clause_depth_never_exceeds_the_limit(cases, limit) -> None:
    case = cases("default")
    made = plans_of(case, limit)
    realizer = case.realizer(lexicon=LEXICON)
    rng = np.random.default_rng(0)
    deepest = 0
    for plan in made:
        sentence = realizer.realize(plan, rng)
        depth = tree_depth(sentence.tree)
        # a class-level restriction with negated IS literals is a relative clause of depth 1,
        # which is what the sentence says, and not a drawn clause
        drawn_limit = limit if plan.subject.kind != CLASS_NP else max(limit, 1)
        assert depth == plan.depth() <= drawn_limit
        if plan.subject.kind != CLASS_NP:
            deepest = max(deepest, depth)
    assert deepest == limit  # and the limit is reached


def test_the_generated_sentences_cover_the_grammar(cases, plans) -> None:
    case = cases("default")
    realizer = case.realizer(grammar=GRAMMARS["english"], lexicon=LEXICON)
    rng = np.random.default_rng(1)
    labels: set[str] = set()
    function_words: set[str] = set()
    lexicon = case.lexicon(grammar=GRAMMARS["english"], lexicon=LEXICON)

    def visit(node) -> None:
        labels.add(node[0])
        for child in node[1:]:
            if not isinstance(child, str):
                visit(child)

    affixes = set()
    for plan in plans("default"):
        sentence = realizer.realize(plan, rng)
        visit(sentence.tree)
        for token in sentence.tokens:
            label, affix = token_parts(token)
            affixes.add(affix)
            lexeme = lexicon.lexeme(label)
            if not lexeme.content:
                function_words.add(lexeme.gloss)
    assert labels == {
        "S", "NP-SBJ", "NP-OBJ", "NP-PRD", "VP", "AP", "PP", "RC",
        "N", "V", "A", "Det", "P", "Conj", "Rel", "AUX", "Neg",
    }  # fmt: skip
    # every function word of the language is used, apart from the pronoun, which the planner
    # of the documents brings in
    # "it" needs a document, "become" a change of a fluent (test_corpus_states.py), and
    # "before" a causal statement (test_corpus_causal.py)
    assert function_words == {x.gloss for x in lexicon.function_lexemes} - {
        "it",
        "become",
        "before",
    }
    assert affixes == {None, "PLURAL", "PAST"}
