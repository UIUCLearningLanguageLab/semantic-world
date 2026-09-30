# Proposal: add `awake` to a need's `rise_when`

Draft, September 29, 2026. Raised while building stage 2 (the schema and the data files).

## The question

Addition A1 in `docs/CONTRACTS.md` lists two values for a need's `rise_when`: `always` and `night_outside_shelter`. It says the passive `fall_per_day` applies "when the rise condition does not hold". The milestone 1 world says fatigue rises "1.0 per day while awake" and falls "4.0 per day while asleep". Fatigue therefore rises under a condition that A1 has no word for. With `rise_when: always`, the rise condition always holds, and fatigue could never fall. A rise multiplier of 0 for `asleep` would stop the rise, but would not let the fall rate apply, because the rise condition would still hold.

## Options

1. **Add `awake` to `rise_when`.** Fatigue is declared as `rise_when: awake`, `fall_per_day: 4.0`, `fall_multipliers: {in_shelter: 2.0}`. One word is added to the enumerated set. Nothing else in A1 changes.
2. **Treat a rise rate of 0 as the rise condition not holding.** Fatigue would use `rise_when: always` with `rise_multipliers: {asleep: 0}`. No new word, but the meaning of a multiplier changes, and `in_shelter: 2.0` in `fall_multipliers` would have no way to apply only while asleep.
3. **Make fatigue a rule module instead of a need.** Fatigue would leave the generic needs mechanism. This contradicts the goal that a need is a generic mechanism and hunger exists only in data.

## Recommendation

Option 1. The schema, the data file `data/types/human.yaml`, and the stage 3 needs mechanism are built with `awake` as the third value. If Jon approves, the one-line change to A1 in `docs/CONTRACTS.md` and in `docs/specs/MILESTONE_1.md` is: `rise_when: always # always | night_outside_shelter | awake`. If Jon prefers another option, the data file and the enum change in one place each.
