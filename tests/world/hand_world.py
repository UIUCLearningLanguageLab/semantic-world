"""The hand world and the hand-written conformance fixtures (``tests/fixtures/world/hand_*``).

The hand world has three entities (``INSTANCE.1.1.1``, ``INSTANCE.1.1.2``, ``INSTANCE.1.2.1``),
a free feature ``PROPERTY.1`` and a derived one ``PROPERTY.2 = NOT PROPERTY.1``, a scalar
``SCALARDIM.1``, the base fluents ``BOOLFL.1`` and ``BOOLFL.2`` and the derived ``BOOLFL.3 =
BOOLFL.1 AND BOOLFL.2`` (the base fluents carry the initial rate 0.5 in their symbols, which no
hand case uses: every case gives its initial state), and four event types: ``EVENTTYPE1.1``
(requires ``agent.PROPERTY.1``, needs ``NOT agent.BOOLFL.1``, sets ``agent.BOOLFL.1``),
``EVENTTYPE1.2`` (requires
``agent.PROPERTY.2``, needs ``agent.BOOLFL.3``, clears ``agent.BOOLFL.2``), ``EVENTTYPE1.3``
(always able and legal, clears ``agent.BOOLFL.1``), and ``EVENTTYPE2.1`` (requires
``agent.PROPERTY.1 AND NOT patient.PROPERTY.1`` and the agent's scalar above the patient's, needs
``agent.BOOLFL.1``, sets ``patient.BOOLFL.2``).

Every expected value in ``CASES`` (the legal bindings before each step, and the state and the
derived fluents after it) is typed in by hand, never computed by the runtime or the brute-force
evaluator. This module only assembles the JSON files and fills in the rule-set identity, which
cannot be computed by hand. Run it as ``python tests/world/hand_world.py`` to rewrite the files;
``test_world_fixtures.py`` checks that the committed files are what it writes.
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


def eff(role, fluent, value):
    return {
        "role": role,
        "fluent": fluent,
        "value": value,
        "expression": f"{role}.{fluent} := {int(value)}",
    }


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
            ],
            "outputs": [
                out("CONSTRAINT.EVENTTYPE1.1", [0]),
                out("CONSTRAINT.EVENTTYPE1.2", [1]),
                out("CONSTRAINT.EVENTTYPE2.1", [2]),
                out("REQUIREMENT.EVENTTYPE1.3", [3]),
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
    ],
}

ENTITIES = [
    {"label": A, "leaf": "CATEGORY.1.1", "PROPERTY.1": 1, "SCALARDIM.1": 1.0},
    {"label": B, "leaf": "CATEGORY.1.1", "PROPERTY.1": 0, "SCALARDIM.1": 0.5},
    {"label": C, "leaf": "CATEGORY.1.2", "PROPERTY.1": 1, "SCALARDIM.1": 0.25},
]

B1, B2, B3 = "BOOLFL.1", "BOOLFL.2", "BOOLFL.3"
ALL = [{"agent": A}, {"agent": B}, {"agent": C}]
AB = [{"agent": A, "patient": B}]


def ev(label, agent, patient=None):
    binding = {"agent": agent} if patient is None else {"agent": agent, "patient": patient}
    return {"event_type": label, "binding": binding}


def legal(e11, e12, e21):
    return {
        "EVENTTYPE1.1": [{"agent": x} for x in e11],
        "EVENTTYPE1.2": [{"agent": x} for x in e12],
        "EVENTTYPE1.3": ALL,
        "EVENTTYPE2.1": AB if e21 else [],
    }


def state(a, b, c):
    return {A: a, B: b, C: c}


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
]


def hand_fixtures() -> list[dict[str, Any]]:
    """Every hand-written fixture, assembled from the hand world and the cases."""
    return [
        fixture_record(name, description, DEFINITION, ENTITIES, initial, steps, error)
        for name, description, initial, steps, error in CASES
    ]


def main(folder: Path = FOLDER) -> None:
    for fixture in hand_fixtures():
        print(write_fixture(folder / f"{fixture['name']}.json", fixture))


if __name__ == "__main__":
    main()
