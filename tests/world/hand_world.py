"""The hand world and the hand-written conformance fixtures (``tests/fixtures/world/hand_*``).

The hand world has three entities (``INSTANCE.1.1.1``, ``INSTANCE.1.1.2``, ``INSTANCE.1.2.1``),
a free feature ``PROPERTY.1`` and a derived one ``PROPERTY.2 = NOT PROPERTY.1``, a scalar
``SCALARDIM.1``, the base fluents ``BOOLFL.1`` and ``BOOLFL.2`` and the derived ``BOOLFL.3 =
BOOLFL.1 AND BOOLFL.2`` (the base fluents carry the initial rate 0.5 in their symbols, which no
hand case uses: every case gives its initial state), and four event types: ``EVENTTYPE1.1``
(requires ``agent.PROPERTY.1``, needs ``NOT agent.BOOLFL.1``, sets ``agent.BOOLFL.1``),
``EVENTTYPE1.2`` (requires
``agent.PROPERTY.2``, needs ``agent.BOOLFL.3``, clears ``agent.BOOLFL.2``), ``EVENTTYPE1.3``
(always able and legal, clears ``agent.BOOLFL.1``), ``EVENTTYPE2.1`` (requires
``agent.PROPERTY.1 AND NOT patient.PROPERTY.1`` and the agent's scalar above the patient's, needs
``agent.BOOLFL.1``, sets ``patient.BOOLFL.2``), and, since stage b1, two event types with
conditional effects, always able and legal: ``EVENTTYPE1.4`` (``when agent.PROPERTY.1 then
agent.BOOLFL.2 := 1``, a static condition) and ``EVENTTYPE1.5`` (``when agent.BOOLFL.3 then
agent.BOOLFL.2 := 0``, a condition on the derived fluent, whose cone is ``BOOLFL.1`` and
``BOOLFL.2``).

Since stage b2 the hand world has two numeric fluents with the range ``[0, 1]``: ``NUMFL.1``,
which does not drift, and ``NUMFL.2``, which drifts by 0.25 at every step (both with the fixed
initial distribution 0, which no case uses: every case gives its initial values); and three more
event types, always able: ``EVENTTYPE1.6`` (always legal, ``agent.NUMFL.1 += 0.5``),
``EVENTTYPE1.7`` (needs ``agent.NUMFL.1 > 0.5``, ``agent.NUMFL.2 -= 0.5``), and ``EVENTTYPE1.8``
(always legal, ``when agent.NUMFL.1 > 0.5 then agent.BOOLFL.2 := 1``, a numeric condition). The
amounts and the drift are multiples of a quarter, so every value a case computes is exact in
float64 and can be typed by hand. In the cases of phases (a) and (b1), every entity starts with
both numeric fluents at 0, so ``EVENTTYPE1.7`` is never legal there, ``NUMFL.1`` stays 0, and
``NUMFL.2`` is 0.25, 0.5, and 0.75 after the first, second, and third step (the drift).
``hand_18`` to ``hand_20`` are the comparison fixtures (ties, ``ENTITY``, a category mean), with
their arithmetic in their descriptions; ``hand_21`` to ``hand_27`` are the numeric fixtures, and
``hand_28`` the ``RANGE`` comparisons at the cut itself (``ABOVE`` strict, ``BELOW`` inclusive).

Every expected value in ``CASES`` (the legal bindings before each step, and the state, the
derived fluents, and the numeric values after it) is typed in by hand, never computed by the
runtime or the brute-force evaluator. This module only assembles the JSON files and fills in the
rule-set identity, which cannot be computed by hand. Run it as ``python tests/world/hand_world.py``
to rewrite the files; ``test_world_fixtures.py`` checks that the committed files are what it
writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_world.world.fixtures import fixture_record, write_fixture

FOLDER = Path("tests/fixtures/world")

A, B, C = "INSTANCE.1.1.1", "INSTANCE.1.1.2", "INSTANCE.1.2.1"


def sym(label, kind, derived=False, fluent=False, arity=1, rate=None):
    record = {"label": label, "kind": kind, "derived": derived, "fluent": fluent, "arity": arity}
    if fluent:
        record["initial_rate"] = rate
    return record


def numsym(label, drift):
    """A numeric fluent's symbol: the range [0, 1], the fixed initial value 0, and a drift."""
    return {
        "label": label,
        "kind": "numeric_fluent",
        "derived": False,
        "fluent": True,
        "arity": 1,
        "range": [0.0, 1.0],
        "initial": {"kind": "fixed", "value": 0.0},
        "drift": drift,
    }


def feat(i, key, role, feature):
    return {"index": i, "key": key, "kind": "feature", "role": role, "feature": feature}


def con(i, label):
    return {"index": i, "key": label, "kind": "constraint", "role": "binding", "constraint": label}


def rule(output, scope, family, inputs, table, expression):
    return {
        "output": output,
        "scope": scope,
        "family": family,
        "inputs": inputs,
        "truth_table": table,
        "expression": expression,
    }


def term(literals, complemented):
    return {"literals": literals, "complemented": complemented, "threshold": len(literals)}


def out(output, terms):
    return {"output": output, "terms": terms, "threshold": 1}


def lit(role, fluent, value):
    return {"role": role, "fluent": fluent, "value": value}


def _condition_literal(item):
    """A condition literal's record and text, from ``(role, kind, symbol, value)`` or, for a
    numeric literal, ``(role, "numeric", symbol, operator, threshold)``."""
    if item[1] == "numeric":
        r, _, sym, op, threshold = item
        return {
            "role": r,
            "kind": "numeric",
            "symbol": sym,
            "operator": op,
            "threshold": threshold,
        }, f"{r}.{sym} {op} {threshold!r}"
    r, k, sym, v = item
    return {"role": r, "kind": k, "symbol": sym, "value": v}, (
        f"{r}.{sym}" if v else f"NOT {r}.{sym}"
    )


def eff(role, fluent, value, condition=None):
    """An effect's record; ``condition`` is a list of ``(role, kind, symbol, value)`` tuples, or
    ``(role, "numeric", symbol, operator, threshold)`` for a numeric literal."""
    record = {
        "role": role,
        "fluent": fluent,
        "value": value,
        "expression": f"{role}.{fluent} := {int(value)}",
    }
    return _with_condition(record, condition)


def numeff(role, fluent, operator, amount, condition=None):
    """A numeric effect's record: ``agent.NUMFL.1 += 0.5``."""
    record = {
        "role": role,
        "fluent": fluent,
        "operator": operator,
        "amount": amount,
        "expression": f"{role}.{fluent} {operator} {amount!r}",
    }
    return _with_condition(record, condition)


def _with_condition(record, condition):
    if condition:
        literals, texts = zip(*(_condition_literal(item) for item in condition), strict=True)
        text = " AND ".join(texts)
        record["expression"] = f"when {text} then {record['expression']}"
        record["condition"] = {"literals": list(literals), "expression": text}
    return record


def numlit(role, fluent, operator, threshold):
    """A numeric precondition literal's record."""
    return {"role": role, "fluent": fluent, "operator": operator, "threshold": threshold}


def et(label, arity, constraints, pre_literals, pre_text, effects):
    return {
        "label": label,
        "kind": "event_type",
        "arity": arity,
        "roles": ["agent", "patient"][:arity],
        "parent": None,
        "level": 1,
        "features": [],
        "explicit": True,
        "requirement": {
            "constraints": constraints,
            "output": f"REQUIREMENT.{label}",
            "expression": " AND ".join(constraints) or "TRUE",
        },
        "precondition": {"literals": pre_literals, "expression": pre_text},
        "effects": effects,
    }


DEFINITION = {
    "version": 1,
    "rule_set_id": "",
    "symbols": [
        sym("PROPERTY.1", "property"),
        sym("PROPERTY.2", "property", derived=True),
        sym("SCALARDIM.1", "scalar"),
        sym("BOOLFL.1", "fluent", fluent=True, rate=0.5),
        sym("BOOLFL.2", "fluent", fluent=True, rate=0.5),
        sym("BOOLFL.3", "fluent", derived=True, fluent=True),
        sym("EVENTTYPE1.1", "event_type"),
        sym("EVENTTYPE1.2", "event_type"),
        sym("EVENTTYPE1.3", "event_type"),
        sym("EVENTTYPE2.1", "event_type", arity=2),
        sym("EVENTTYPE1.4", "event_type"),
        sym("EVENTTYPE1.5", "event_type"),
        numsym("NUMFL.1", 0.0),
        numsym("NUMFL.2", 0.25),
        sym("EVENTTYPE1.6", "event_type"),
        sym("EVENTTYPE1.7", "event_type"),
        sym("EVENTTYPE1.8", "event_type"),
    ],
    "literals": [
        feat(0, "PROPERTY.1", None, "PROPERTY.1"),
        {"index": 1, "key": "BOOLFL.1", "kind": "fluent", "role": None, "fluent": "BOOLFL.1"},
        {"index": 2, "key": "BOOLFL.2", "kind": "fluent", "role": None, "fluent": "BOOLFL.2"},
        feat(3, "agent.PROPERTY.1", "agent", "PROPERTY.1"),
        feat(4, "agent.PROPERTY.2", "agent", "PROPERTY.2"),
        feat(5, "patient.PROPERTY.1", "patient", "PROPERTY.1"),
        {
            "index": 6,
            "key": "agent.SCALARDIM.1-patient.SCALARDIM.1>0.0000",
            "kind": "comparison",
            "role": "binding",
            "agent_scalar": "SCALARDIM.1",
            "patient_scalar": "SCALARDIM.1",
            "operator": ">",
            "low": 0.0,
            "high": None,
        },
        con(7, "CONSTRAINT.EVENTTYPE1.1"),
        con(8, "CONSTRAINT.EVENTTYPE1.2"),
        con(9, "CONSTRAINT.EVENTTYPE2.1"),
    ],
    "rules": [
        rule("PROPERTY.2", "entity", "hand", [0], "10", "NOT PROPERTY.1"),
        rule("BOOLFL.3", "entity", "hand", [1, 2], "0001", "BOOLFL.1 AND BOOLFL.2"),
        rule("CONSTRAINT.EVENTTYPE1.1", "binding", "hand", [3], "01", "agent.PROPERTY.1"),
        rule("CONSTRAINT.EVENTTYPE1.2", "binding", "hand", [4], "01", "agent.PROPERTY.2"),
        rule(
            "CONSTRAINT.EVENTTYPE2.1",
            "binding",
            "hand",
            [3, 5, 6],
            "00000100",
            "agent.PROPERTY.1 AND NOT patient.PROPERTY.1 AND agent.SCALARDIM.1 - "
            "patient.SCALARDIM.1 > 0.0",
        ),
        rule(
            "REQUIREMENT.EVENTTYPE1.1",
            "binding",
            "requirement",
            [7],
            "01",
            "CONSTRAINT.EVENTTYPE1.1",
        ),
        rule(
            "REQUIREMENT.EVENTTYPE1.2",
            "binding",
            "requirement",
            [8],
            "01",
            "CONSTRAINT.EVENTTYPE1.2",
        ),
        rule("REQUIREMENT.EVENTTYPE1.3", "binding", "requirement", [], "1", "TRUE"),
        rule(
            "REQUIREMENT.EVENTTYPE2.1",
            "binding",
            "requirement",
            [9],
            "01",
            "CONSTRAINT.EVENTTYPE2.1",
        ),
        rule("REQUIREMENT.EVENTTYPE1.4", "binding", "requirement", [], "1", "TRUE"),
        rule("REQUIREMENT.EVENTTYPE1.5", "binding", "requirement", [], "1", "TRUE"),
        rule("REQUIREMENT.EVENTTYPE1.6", "binding", "requirement", [], "1", "TRUE"),
        rule("REQUIREMENT.EVENTTYPE1.7", "binding", "requirement", [], "1", "TRUE"),
        rule("REQUIREMENT.EVENTTYPE1.8", "binding", "requirement", [], "1", "TRUE"),
    ],
    "layers": [
        {
            "scope": "entity",
            "layer": 1,
            "terms": [term([0], [True]), term([1, 2], [False, False])],
            "outputs": [out("PROPERTY.2", [0]), out("BOOLFL.3", [1])],
        },
        {
            "scope": "binding",
            "layer": 1,
            "terms": [
                term([3], [False]),
                term([4], [False]),
                term([3, 5, 6], [False, True, False]),
                term([], []),
                term([], []),
                term([], []),
                term([], []),
                term([], []),
                term([], []),
            ],
            "outputs": [
                out("CONSTRAINT.EVENTTYPE1.1", [0]),
                out("CONSTRAINT.EVENTTYPE1.2", [1]),
                out("CONSTRAINT.EVENTTYPE2.1", [2]),
                out("REQUIREMENT.EVENTTYPE1.3", [3]),
                out("REQUIREMENT.EVENTTYPE1.4", [4]),
                out("REQUIREMENT.EVENTTYPE1.5", [5]),
                out("REQUIREMENT.EVENTTYPE1.6", [6]),
                out("REQUIREMENT.EVENTTYPE1.7", [7]),
                out("REQUIREMENT.EVENTTYPE1.8", [8]),
            ],
        },
        {
            "scope": "binding",
            "layer": 2,
            "terms": [term([7], [False]), term([8], [False]), term([9], [False])],
            "outputs": [
                out("REQUIREMENT.EVENTTYPE1.1", [0]),
                out("REQUIREMENT.EVENTTYPE1.2", [1]),
                out("REQUIREMENT.EVENTTYPE2.1", [2]),
            ],
        },
    ],
    "event_types": [
        et(
            "EVENTTYPE1.1",
            1,
            ["CONSTRAINT.EVENTTYPE1.1"],
            [lit("agent", "BOOLFL.1", False)],
            "NOT agent.BOOLFL.1",
            [eff("agent", "BOOLFL.1", True)],
        ),
        et(
            "EVENTTYPE1.2",
            1,
            ["CONSTRAINT.EVENTTYPE1.2"],
            [lit("agent", "BOOLFL.3", True)],
            "agent.BOOLFL.3",
            [eff("agent", "BOOLFL.2", False)],
        ),
        et("EVENTTYPE1.3", 1, [], [], "TRUE", [eff("agent", "BOOLFL.1", False)]),
        et(
            "EVENTTYPE2.1",
            2,
            ["CONSTRAINT.EVENTTYPE2.1"],
            [lit("agent", "BOOLFL.1", True)],
            "agent.BOOLFL.1",
            [eff("patient", "BOOLFL.2", True)],
        ),
        et(
            "EVENTTYPE1.4",
            1,
            [],
            [],
            "TRUE",
            [eff("agent", "BOOLFL.2", True, [("agent", "feature", "PROPERTY.1", True)])],
        ),
        et(
            "EVENTTYPE1.5",
            1,
            [],
            [],
            "TRUE",
            [eff("agent", "BOOLFL.2", False, [("agent", "fluent", "BOOLFL.3", True)])],
        ),
        et("EVENTTYPE1.6", 1, [], [], "TRUE", [numeff("agent", "NUMFL.1", "+=", 0.5)]),
        et(
            "EVENTTYPE1.7",
            1,
            [],
            [numlit("agent", "NUMFL.1", ">", 0.5)],
            "agent.NUMFL.1 > 0.5",
            [numeff("agent", "NUMFL.2", "-=", 0.5)],
        ),
        et(
            "EVENTTYPE1.8",
            1,
            [],
            [],
            "TRUE",
            [eff("agent", "BOOLFL.2", True, [("agent", "numeric", "NUMFL.1", ">", 0.5)])],
        ),
    ],
}

ENTITIES = [
    {"label": A, "leaf": "CATEGORY.1.1", "PROPERTY.1": 1, "SCALARDIM.1": 1.0},
    {"label": B, "leaf": "CATEGORY.1.1", "PROPERTY.1": 0, "SCALARDIM.1": 0.5},
    {"label": C, "leaf": "CATEGORY.1.2", "PROPERTY.1": 1, "SCALARDIM.1": 0.25},
]

B1, B2, B3 = "BOOLFL.1", "BOOLFL.2", "BOOLFL.3"
N1, N2 = "NUMFL.1", "NUMFL.2"
ALL = [{"agent": A}, {"agent": B}, {"agent": C}]
AB = [{"agent": A, "patient": B}]


def ev(label, agent, patient=None):
    binding = {"agent": agent} if patient is None else {"agent": agent, "patient": patient}
    return {"event_type": label, "binding": binding}


def legal(e11, e12, e21, e17=()):
    """The legal bindings before a step. EVENTTYPE1.3, EVENTTYPE1.4, EVENTTYPE1.5, EVENTTYPE1.6,
    and EVENTTYPE1.8 are always able and have no precondition, so every entity is a legal agent
    of each; a condition never bears on legality. EVENTTYPE1.7 is legal for the entities whose
    NUMFL.1 is above 0.5, strictly (``e17``)."""
    return {
        "EVENTTYPE1.1": [{"agent": x} for x in e11],
        "EVENTTYPE1.2": [{"agent": x} for x in e12],
        "EVENTTYPE1.3": ALL,
        "EVENTTYPE2.1": AB if e21 else [],
        "EVENTTYPE1.4": ALL,
        "EVENTTYPE1.5": ALL,
        "EVENTTYPE1.6": ALL,
        "EVENTTYPE1.7": [{"agent": x} for x in e17],
        "EVENTTYPE1.8": ALL,
    }


def state(a, b, c):
    return {A: a, B: b, C: c}


def numeric(a, b, c):
    """The numeric values of the three entities, each ``(NUMFL.1, NUMFL.2)``."""
    return {A: {N1: a[0], N2: a[1]}, B: {N1: b[0], N2: b[1]}, C: {N1: c[0], N2: c[1]}}


ZERO = numeric((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))


def drifted(k):
    """The numeric values after step k of a case of phases (a) and (b1): every entity started
    at 0 on both fluents, NUMFL.1 does not drift, and NUMFL.2 rose by 0.25 at each step."""
    return numeric((0.0, 0.25 * k), (0.0, 0.25 * k), (0.0, 0.25 * k))


CASES = [
    (
        "hand_01_no_event",
        "A fluent with no event: a step with no events leaves every base fluent as it was; "
        "derived fluents are recomputed and nothing changes.",
        state([B2], [B1, B2], [B1]),
        [
            {
                "events": [],
                "legal": legal([A], [B], False),
                "derived": state([], [B3], []),
                "state": state([B2], [B1, B2], [B1]),
            }
        ],
        None,
    ),
    (
        "hand_02_effect_sets",
        "An effect sets a fluent: EVENTTYPE1.1 sets BOOLFL.1 of INSTANCE.1.1.1 from 0 to 1.",
        state([], [B1], [B1]),
        [
            {
                "events": [ev("EVENTTYPE1.1", A)],
                "legal": legal([A], [], False),
                "derived": state([], [], []),
                "state": state([B1], [B1], [B1]),
            }
        ],
        None,
    ),
    (
        "hand_03_effect_clears",
        "An effect clears a fluent: EVENTTYPE1.2 sets BOOLFL.2 of INSTANCE.1.1.2 from 1 to 0, so "
        "its derived BOOLFL.3 goes false too.",
        state([], [B1, B2], []),
        [
            {
                "events": [ev("EVENTTYPE1.2", B)],
                "legal": legal([A, C], [B], False),
                "derived": state([], [], []),
                "state": state([], [B1], []),
            }
        ],
        None,
    ),
    (
        "hand_04_effect_no_change",
        "An effect sets a fluent that is already true: EVENTTYPE2.1 sets BOOLFL.2 of "
        "INSTANCE.1.1.2, which is 1 already; the state does not change.",
        state([B1], [B2], []),
        [
            {
                "events": [ev("EVENTTYPE2.1", A, B)],
                "legal": legal([C], [], True),
                "derived": state([], [], []),
                "state": state([B1], [B2], []),
            }
        ],
        None,
    ),
    (
        "hand_05_derived_changes",
        "A derived fluent changes because a base fluent changed: BOOLFL.3 is BOOLFL.1 AND "
        "BOOLFL.2. Step 1 sets BOOLFL.1 of INSTANCE.1.1.1, so its BOOLFL.3 becomes true; step 2 "
        "clears BOOLFL.2 of INSTANCE.1.1.2, so its BOOLFL.3 becomes false.",
        state([B2], [B1, B2], []),
        [
            {
                "events": [ev("EVENTTYPE1.1", A)],
                "legal": legal([A, C], [B], False),
                "derived": state([B3], [B3], []),
                "state": state([B1, B2], [B1, B2], []),
            },
            {
                "events": [ev("EVENTTYPE1.2", B)],
                "legal": legal([C], [B], True),
                "derived": state([B3], [], []),
                "state": state([B1, B2], [B1], []),
            },
        ],
        None,
    ),
    (
        "hand_06_precondition_blocks",
        "A precondition blocks an event: EVENTTYPE1.1 needs NOT agent.BOOLFL.1, but BOOLFL.1 of "
        "INSTANCE.1.1.1 is 1. The binding is not legal, and applying the event is an "
        "illegal-event error.",
        state([B1], [], []),
        [{"events": [ev("EVENTTYPE1.1", A)], "legal": legal([C], [], True)}],
        {"step": 1, "kind": "illegal"},
    ),
    (
        "hand_07_two_events_independent",
        "Three events in one step that do not interfere: EVENTTYPE2.1 (INSTANCE.1.1.1, "
        "INSTANCE.1.1.2) writes BOOLFL.2 of the patient and reads BOOLFL.1 of the agent; "
        "EVENTTYPE1.3 (INSTANCE.1.1.2) writes BOOLFL.1 of the same patient entity, a different "
        "fluent, and reads nothing; EVENTTYPE1.1 (INSTANCE.1.2.1) touches a third entity.",
        state([B1], [], []),
        [
            {
                "events": [ev("EVENTTYPE2.1", A, B), ev("EVENTTYPE1.3", B), ev("EVENTTYPE1.1", C)],
                "legal": legal([C], [], True),
                "derived": state([], [], []),
                "state": state([B1], [B2], [B1]),
            }
        ],
        None,
    ),
    (
        "hand_08_two_events_interfere",
        "Two events in one step that interfere: EVENTTYPE1.3 (INSTANCE.1.1.1) writes BOOLFL.1 of "
        "INSTANCE.1.1.1, which the precondition of EVENTTYPE2.1 (INSTANCE.1.1.1, INSTANCE.1.1.2) "
        "reads. Both are legal; applying them together is an interference error.",
        state([B1], [], []),
        [
            {
                "events": [ev("EVENTTYPE2.1", A, B), ev("EVENTTYPE1.3", A)],
                "legal": legal([C], [], True),
            }
        ],
        {"step": 1, "kind": "interference"},
    ),
    (
        "hand_09_interfere_same_fluent",
        "Two events that both write the same fluent of the same entity: EVENTTYPE1.1 sets and "
        "EVENTTYPE1.3 clears BOOLFL.1 of INSTANCE.1.1.1.",
        state([], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.1", A), ev("EVENTTYPE1.3", A)],
                "legal": legal([A, C], [], False),
            }
        ],
        {"step": 1, "kind": "interference"},
    ),
    (
        "hand_10_interfere_through_cone",
        "Interference through the cone of a derived fluent: EVENTTYPE1.2 (INSTANCE.1.1.2) reads "
        "the derived BOOLFL.3, whose cone is BOOLFL.1 and BOOLFL.2, and EVENTTYPE1.3 "
        "(INSTANCE.1.1.2) writes BOOLFL.1. The two write different fluents, so only the cone "
        "makes them interfere.",
        state([], [B1, B2], []),
        [
            {
                "events": [ev("EVENTTYPE1.2", B), ev("EVENTTYPE1.3", B)],
                "legal": legal([A, C], [B], False),
            }
        ],
        {"step": 1, "kind": "interference"},
    ),
    (
        "hand_11_requirement_blocks",
        "A requirement blocks an event: EVENTTYPE1.1 needs agent.PROPERTY.1, and INSTANCE.1.1.2 "
        "lacks it. The entity is not able, so the event is illegal in every state.",
        state([], [], []),
        [{"events": [ev("EVENTTYPE1.1", B)], "legal": legal([A, C], [], False)}],
        {"step": 1, "kind": "illegal"},
    ),
    (
        "hand_12_duplicate_event",
        "The same event twice in one step: EVENTTYPE1.3 (INSTANCE.1.1.1) listed twice is an "
        "interference error.",
        state([], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.3", A), ev("EVENTTYPE1.3", A)],
                "legal": legal([A, C], [], False),
            }
        ],
        {"step": 1, "kind": "interference"},
    ),
    (
        "hand_13_condition_holds",
        "A static condition that holds (stage b1): EVENTTYPE1.4 sets agent.BOOLFL.2 when "
        "agent.PROPERTY.1, and INSTANCE.1.1.1 has PROPERTY.1, so BOOLFL.2 of INSTANCE.1.1.1 goes "
        "from 0 to 1. The event is legal whatever the condition.",
        state([], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.4", A)],
                "legal": legal([A, C], [], False),
                "derived": state([], [], []),
                "state": state([B2], [], []),
            }
        ],
        None,
    ),
    (
        "hand_14_condition_fails",
        "A static condition that fails (stage b1): EVENTTYPE1.4 on INSTANCE.1.1.2, which lacks "
        "PROPERTY.1, is legal but its effect does nothing, so the state does not change and a "
        "history would record no change.",
        state([], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.4", B)],
                "legal": legal([A, C], [], False),
                "derived": state([], [], []),
                "state": state([], [], []),
            }
        ],
        None,
    ),
    (
        "hand_15_fluent_condition_holds",
        "A fluent condition that holds (stage b1): EVENTTYPE1.5 clears agent.BOOLFL.2 when the "
        "derived agent.BOOLFL.3 (BOOLFL.1 AND BOOLFL.2) holds. INSTANCE.1.1.1 has both base "
        "fluents, so BOOLFL.2 is cleared and BOOLFL.3 goes false with it. In the same step "
        "EVENTTYPE1.1 on INSTANCE.1.2.1 touches another entity and does not interfere.",
        state([B1, B2], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.5", A), ev("EVENTTYPE1.1", C)],
                "legal": legal([C], [], True),
                "derived": state([], [], []),
                "state": state([B1], [], [B1]),
            }
        ],
        None,
    ),
    (
        "hand_16_fluent_condition_fails",
        "A fluent condition that fails, judged in the step's starting state (stage b1): "
        "EVENTTYPE1.5 on INSTANCE.1.1.1, which has BOOLFL.2 but not BOOLFL.1, so BOOLFL.3 is "
        "false and the effect does nothing; in the same step EVENTTYPE1.1 on INSTANCE.1.2.1 sets "
        "BOOLFL.1 of another entity. In step 2 EVENTTYPE1.1 sets BOOLFL.1 of INSTANCE.1.1.1, so "
        "BOOLFL.3 becomes true; in step 3 EVENTTYPE1.5 fires and clears BOOLFL.2.",
        state([B2], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.5", A), ev("EVENTTYPE1.1", C)],
                "legal": legal([A, C], [], False),
                "derived": state([], [], []),
                "state": state([B2], [], [B1]),
            },
            {
                "events": [ev("EVENTTYPE1.1", A)],
                "legal": legal([A], [], False),
                "derived": state([B3], [], []),
                "state": state([B1, B2], [], [B1]),
            },
            {
                "events": [ev("EVENTTYPE1.5", A)],
                "legal": legal([], [], True),
                "derived": state([], [], []),
                "state": state([B1], [], [B1]),
            },
        ],
        None,
    ),
    (
        "hand_17_condition_read_by_another_event",
        "Interference through a condition (stage b1): EVENTTYPE1.5 (INSTANCE.1.1.1) reads the "
        "derived BOOLFL.3, whose cone is BOOLFL.1 and BOOLFL.2, and EVENTTYPE1.3 (INSTANCE.1.1.1) "
        "writes BOOLFL.1. Both are legal, and EVENTTYPE1.5 has no precondition: only its "
        "condition makes the two interfere.",
        state([B1, B2], [], []),
        [
            {
                "events": [ev("EVENTTYPE1.5", A), ev("EVENTTYPE1.3", A)],
                "legal": legal([C], [], True),
            }
        ],
        {"step": 1, "kind": "interference"},
    ),
]
"""The cases of phases (a) and (b1): every entity starts at 0 on both numeric fluents, and each
step's numeric values are the drift alone (``drifted``), added by ``hand_fixtures``."""

S = "SCALARDIM.1"
C11, C12, C1 = "CATEGORY.1.1", "CATEGORY.1.2", "CATEGORY.1"


def mean(category):
    return {"mean": category}


def greater(dimension, x, y, expected, time=None):
    entry = {"kind": "greater", "dimension": dimension, "x": x, "y": y, "expected": expected}
    if time is not None:
        entry["time"] = time
    return entry


def against(kind, dimension, x, klass, z, expected, time=None):
    entry = {
        "kind": kind,
        "dimension": dimension,
        "x": x,
        "class": klass,
        "z": z,
        "expected": expected,
    }
    if time is not None:
        entry["time"] = time
    return entry


def change(kind, fluent, x, time, expected):
    return {"kind": kind, "dimension": fluent, "x": x, "time": time, "expected": expected}


def quiet_step(k, e17=()):
    """A step with no events in a world at rest: nothing but the drift changes."""
    return {
        "events": [],
        "legal": legal([A, C], [], False, e17),
        "derived": state([], [], []),
        "state": state([], [], []),
        "numeric": drifted(k),
    }


NUMERIC_CASES = [
    # (name, description, initial Boolean state, initial numeric values, steps, error, comparisons)
    (
        "hand_18_comparison_ties",
        "Ties (stage b2): a comparison is strict, so two equal values give false both ways. The "
        "scalar of INSTANCE.1.2.1 is 0.25, and the mean of CATEGORY.1.2 (its only instance) is "
        "0.25: neither GREATER holds. At TIME.1 every entity has NUMFL.1 at 0, so no entity is "
        "GREATER than another on it either way.",
        state([], [], []),
        ZERO,
        [quiet_step(1)],
        None,
        [
            greater(S, C, mean(C12), False),
            greater(S, mean(C12), C, False),
            greater(N1, A, B, False, time=1),
            greater(N1, B, A, False, time=1),
            greater(S, A, B, True),
            greater(S, B, A, False),
        ],
    ),
    (
        "hand_19_comparison_entity",
        "ENTITY as the comparison class (stage b2), the class of every entity: the scalars are "
        "1.0, 0.5, and 0.25, with the mean 0.583333... and the population standard deviation "
        "0.311805... (the squared deviations 0.173611, 0.006944, and 0.111111 sum to 0.291667, "
        "a third of which is 0.097222). ABOVE at z = 1 needs at least 0.895138: INSTANCE.1.1.1 "
        "(1.0) is above, INSTANCE.1.1.2 (0.5) is not. BELOW at z = 1 needs at most 0.271528: "
        "INSTANCE.1.2.1 (0.25) is below; at z = 1.5 the bound is 0.115625, and it is not; at z = "
        "0.5 the bound is 0.427431, and INSTANCE.1.1.2 (0.5) is not below. Within "
        "CATEGORY.1.1 (1.0 and 0.5: mean 0.75, standard deviation 0.25), INSTANCE.1.1.1 is exactly "
        "one standard deviation above the mean, which counts (at least z), and INSTANCE.1.1.2 "
        "exactly one below. CATEGORY.1.2 has one instance and no spread, so nothing is above or "
        "below in it.",
        state([], [], []),
        ZERO,
        [quiet_step(1)],
        None,
        [
            against("above", S, A, "ENTITY", 1.0, True),
            against("above", S, B, "ENTITY", 1.0, False),
            against("below", S, C, "ENTITY", 1.0, True),
            against("below", S, C, "ENTITY", 1.5, False),
            against("below", S, B, "ENTITY", 0.5, False),
            against("above", S, A, C11, 1.0, True),
            against("below", S, B, C11, 1.0, True),
            against("above", S, C, C12, 1.0, False),
            against("below", S, C, C12, 1.0, False),
        ],
    ),
    (
        "hand_20_comparison_category_mean",
        "A category mean as an argument (stage b2): CATEGORY.1.1 holds INSTANCE.1.1.1 (1.0) and "
        "INSTANCE.1.1.2 (0.5), so its mean is 0.75; CATEGORY.1 holds all three entities, mean "
        "0.583333...; CATEGORY.1.2 holds INSTANCE.1.2.1 alone, mean 0.25. The mean of CATEGORY.1.1 "
        "is greater than INSTANCE.1.2.1 and than the mean of CATEGORY.1.2, and the mean of "
        "CATEGORY.1 is greater than the mean of CATEGORY.1.2. Against ENTITY (mean 0.583333..., "
        "standard deviation 0.311805...), the mean of CATEGORY.1.1 is above at z = 0.5 (bound "
        "0.739236) and not at z = 1 (bound 0.895138).",
        state([], [], []),
        ZERO,
        [quiet_step(1)],
        None,
        [
            greater(S, mean(C11), C, True),
            greater(S, C, mean(C11), False),
            greater(S, mean(C11), mean(C12), True),
            greater(S, mean(C1), mean(C12), True),
            greater(S, A, mean("ENTITY"), True),
            against("above", S, mean(C11), "ENTITY", 0.5, True),
            against("above", S, mean(C11), "ENTITY", 1.0, False),
        ],
    ),
    (
        "hand_21_numeric_effect",
        "A numeric effect (stage b2): EVENTTYPE1.6 adds 0.5 to NUMFL.1 of INSTANCE.1.1.1, which "
        "starts at 0.25, so it is 0.75 after the step; NUMFL.1 does not drift. NUMFL.2 of every "
        "entity rises from 0 to 0.25 by the drift. EVENTTYPE1.7 is legal for nobody before the "
        "step (no NUMFL.1 is above 0.5) and for INSTANCE.1.1.1 after it. The history would "
        "record the effect's change (0.25 to 0.75, caused by the event) and three drift changes. "
        "INSTANCE.1.1.1 is GREATER than INSTANCE.1.1.2 on NUMFL.1 at both time points (0.25 and "
        "0.75 against 0), and INSTANCE.1.1.2 and INSTANCE.1.2.1 tie at 0.",
        state([], [], []),
        numeric((0.25, 0.0), (0.0, 0.0), (0.0, 0.0)),
        [
            {
                "events": [ev("EVENTTYPE1.6", A)],
                "legal": legal([A, C], [], False),
                "derived": state([], [], []),
                "state": state([], [], []),
                "numeric": numeric((0.75, 0.25), (0.0, 0.25), (0.0, 0.25)),
            },
            quiet_step(2, e17=[A]) | {"numeric": numeric((0.75, 0.5), (0.0, 0.5), (0.0, 0.5))},
        ],
        None,
        [
            change("increase", N1, A, 1, True),
            change("increase", N1, B, 1, False),
            change("decrease", N1, A, 1, False),
            change("increase", N2, B, 1, True),
            greater(N1, A, B, True, time=2),
            greater(N1, A, B, True, time=1),
            greater(N1, B, C, False, time=1),
            greater(N1, C, B, False, time=1),
        ],
    ),
    (
        "hand_22_numeric_drift",
        "Drift (stage b2): with no event at all, NUMFL.2 of every entity rises by 0.25 at each "
        "step, 0.25 after the first and 0.5 after the second, and NUMFL.1 stays at 0. Against "
        "RANGE, a value is ABOVE at a fraction q when it is strictly more than q of the way "
        "through [0, 1], and BELOW when at most (WM.35): after the first step NUMFL.2 (0.25) is "
        "below at 0.25 and not above at 0.5; after the second (0.5) it is not above at 0.5 either, "
        "being exactly there, and no longer below at 0.25.",
        state([], [], []),
        ZERO,
        [quiet_step(1), quiet_step(2)],
        None,
        [
            against("below", N2, A, "RANGE", 0.25, True, time=2),
            against("above", N2, A, "RANGE", 0.5, False, time=2),
            against("above", N2, A, "RANGE", 0.5, False, time=3),
            against("below", N2, A, "RANGE", 0.25, False, time=3),
            change("increase", N2, C, 1, True),
            change("increase", N1, C, 1, False),
        ],
    ),
    (
        "hand_28_comparison_at_the_cut",
        "A RANGE comparison at its own point (WM.35, October 15, 2026): ABOVE at a fraction q is "
        "strict (the value is more than q of the way through the range) and BELOW inclusive (at "
        "most), as the numeric literals > t and <= t are. With no event, the drift takes NUMFL.2 "
        "of every entity to 0.25 after the first step, 0.5 after the second, and 0.75 after the "
        "third, in exact float64 arithmetic. At 0.25 the value is not ABOVE at 0.25 and is BELOW "
        "at 0.25; at 0.75 it is not ABOVE at 0.75 and is BELOW at 0.75; at 0.5 it is ABOVE at "
        "0.25 and not BELOW at 0.25, and BELOW at 0.75 and not ABOVE at 0.75.",
        state([], [], []),
        ZERO,
        [quiet_step(1), quiet_step(2), quiet_step(3)],
        None,
        [
            against("above", N2, A, "RANGE", 0.25, False, time=2),
            against("below", N2, A, "RANGE", 0.25, True, time=2),
            against("above", N2, B, "RANGE", 0.25, True, time=3),
            against("below", N2, B, "RANGE", 0.25, False, time=3),
            against("above", N2, B, "RANGE", 0.75, False, time=3),
            against("below", N2, B, "RANGE", 0.75, True, time=3),
            against("above", N2, C, "RANGE", 0.75, False, time=4),
            against("below", N2, C, "RANGE", 0.75, True, time=4),
            against("above", N2, C, "RANGE", 0.5, True, time=4),
        ],
    ),
    (
        "hand_23_numeric_clamp_both_ends",
        "Clamping at both ends of the range (stage b2): EVENTTYPE1.6 adds 0.5 to NUMFL.1 of "
        "INSTANCE.1.1.1, which starts at 0.75, so the sum 1.25 is clamped to 1.0; EVENTTYPE1.7 "
        "subtracts 0.5 from NUMFL.2 of INSTANCE.1.1.2, which starts at 0.25, so the difference "
        "-0.25 is clamped to 0.0, and the drift then brings it to 0.25. NUMFL.2 of INSTANCE.1.2.1 "
        "starts at 1.0, the top of the range, and the drift leaves it there (a history records no "
        "change for it). Both EVENTTYPE1.7 agents have NUMFL.1 at 0.75, above 0.5.",
        state([], [], []),
        numeric((0.75, 0.0), (0.75, 0.25), (0.0, 1.0)),
        [
            {
                "events": [ev("EVENTTYPE1.6", A), ev("EVENTTYPE1.7", B)],
                "legal": legal([A, C], [], False, e17=[A, B]),
                "derived": state([], [], []),
                "state": state([], [], []),
                "numeric": numeric((1.0, 0.25), (0.75, 0.25), (0.0, 1.0)),
            }
        ],
        None,
        [
            change("increase", N2, C, 1, False),
            change("decrease", N2, B, 1, False),
            change("increase", N1, A, 1, True),
        ],
    ),
    (
        "hand_24_numeric_effect_and_drift",
        "An effect and the drift in one step, in that order (stage b2): NUMFL.2 of INSTANCE.1.1.1 "
        "starts at 1.0; EVENTTYPE1.7 subtracts 0.5, which gives 0.5, and the drift then adds 0.25, "
        "which gives 0.75. With the drift first, the value would have been clamped at 1.0 and "
        "ended at 0.5. In the same step EVENTTYPE1.6 adds 0.5 to NUMFL.1 of INSTANCE.1.1.2 (0 to "
        "0.5), whose NUMFL.2 drifts from 0 to 0.25. Only INSTANCE.1.1.1 (NUMFL.1 at 1.0) can be "
        "the agent of EVENTTYPE1.7 before the step; after it INSTANCE.1.1.2 (0.5, not above 0.5) "
        "still cannot.",
        state([], [], []),
        numeric((1.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        [
            {
                "events": [ev("EVENTTYPE1.7", A), ev("EVENTTYPE1.6", B)],
                "legal": legal([A, C], [], False, e17=[A]),
                "derived": state([], [], []),
                "state": state([], [], []),
                "numeric": numeric((1.0, 0.75), (0.5, 0.25), (0.0, 0.25)),
            },
            {
                "events": [],
                "legal": legal([A, C], [], False, e17=[A]),
                "derived": state([], [], []),
                "state": state([], [], []),
                "numeric": numeric((1.0, 1.0), (0.5, 0.5), (0.0, 0.5)),
            },
        ],
        None,
        [change("decrease", N2, A, 1, True), change("increase", N2, A, 2, True)],
    ),
    (
        "hand_25_numeric_precondition_at_threshold",
        "A numeric precondition at its threshold (stage b2): EVENTTYPE1.7 needs agent.NUMFL.1 > "
        "0.5, strictly. INSTANCE.1.1.1 has exactly 0.5, so it is not a legal agent, and applying "
        "the event is an illegal-event error; INSTANCE.1.1.2 has 0.75 and is.",
        state([], [], []),
        numeric((0.5, 0.0), (0.75, 0.0), (0.0, 0.0)),
        [{"events": [ev("EVENTTYPE1.7", A)], "legal": legal([A, C], [], False, e17=[B])}],
        {"step": 1, "kind": "illegal"},
        [],
    ),
    (
        "hand_26_numeric_condition",
        "A numeric condition (stage b2, deferred from stage b1): EVENTTYPE1.8 sets agent.BOOLFL.2 "
        "when agent.NUMFL.1 > 0.5. INSTANCE.1.1.1 has 0.75, so its BOOLFL.2 is set; INSTANCE.1.1.2 "
        "has exactly 0.5, so its effect does nothing. Both events are legal, and both are in the "
        "same step on different entities.",
        state([], [], []),
        numeric((0.75, 0.0), (0.5, 0.0), (0.0, 0.0)),
        [
            {
                "events": [ev("EVENTTYPE1.8", A), ev("EVENTTYPE1.8", B)],
                "legal": legal([A, C], [], False, e17=[A]),
                "derived": state([], [], []),
                "state": state([B2], [], []),
                "numeric": numeric((0.75, 0.25), (0.5, 0.25), (0.0, 0.25)),
            }
        ],
        None,
        [],
    ),
    (
        "hand_27_numeric_read_by_another_event",
        "Interference through a numeric fluent (stage b2): EVENTTYPE1.6 (INSTANCE.1.1.1) writes "
        "NUMFL.1 of INSTANCE.1.1.1, which the condition of EVENTTYPE1.8 (INSTANCE.1.1.1) reads. "
        "Both are legal, and applying them together is an interference error.",
        state([], [], []),
        numeric((0.75, 0.0), (0.0, 0.0), (0.0, 0.0)),
        [
            {
                "events": [ev("EVENTTYPE1.6", A), ev("EVENTTYPE1.8", A)],
                "legal": legal([A, C], [], False, e17=[A]),
            }
        ],
        {"step": 1, "kind": "interference"},
        [],
    ),
]


def hand_fixtures() -> list[dict[str, Any]]:
    """Every hand-written fixture, assembled from the hand world and the cases."""
    fixtures = []
    for name, description, initial, steps, error in CASES:
        with_numeric = [
            dict(step, numeric=drifted(k)) if "state" in step else dict(step)
            for k, step in enumerate(steps, start=1)
        ]
        fixtures.append(
            fixture_record(
                name, description, DEFINITION, ENTITIES, initial, with_numeric, error, ZERO
            )
        )
    for name, description, initial, numbers, steps, error, comparisons in NUMERIC_CASES:
        fixtures.append(
            fixture_record(
                name, description, DEFINITION, ENTITIES, initial, steps, error, numbers, comparisons
            )
        )
    return fixtures


def main(folder: Path = FOLDER) -> None:
    for fixture in hand_fixtures():
        print(write_fixture(folder / f"{fixture['name']}.json", fixture))


if __name__ == "__main__":
    main()
