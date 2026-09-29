//! The generic needs and health mechanism (addition A1), driven entirely by the rates in
//! the data files. "Hunger" exists only in `data/`.

use sw_schema::entity::{AgentState, AtMax, NeedSpec, RiseWhen};

use crate::components::{Agent, AtMaxRule, Health, NeedRates, NeedState, Needs};

/// The conditions a need's rates depend on, at one tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conditions {
    pub asleep: bool,
    pub running: bool,
    pub in_shelter: bool,
    pub night: bool,
}

impl Conditions {
    fn holds(&self, state: AgentState) -> bool {
        match state {
            AgentState::Running => self.running,
            AgentState::Asleep => self.asleep,
            AgentState::InShelter => self.in_shelter,
        }
    }

    fn rise_condition(&self, when: RiseWhen) -> bool {
        match when {
            RiseWhen::Always => true,
            RiseWhen::NightOutsideShelter => self.night && !self.in_shelter,
            RiseWhen::Awake => !self.asleep,
        }
    }
}

impl NeedRates {
    /// Convert a need's per-day and per-hour traits to per-second rates.
    pub fn from_spec(spec: &NeedSpec, seconds_per_day: f64, seconds_per_hour: f64) -> Self {
        let per_day = |v: f64| v / seconds_per_day;
        NeedRates {
            rise_per_s: per_day(*spec.rise_per_day.value()),
            rise_when: *spec.rise_when.value(),
            rise_multipliers: spec
                .rise_multipliers
                .iter()
                .map(|(s, t)| (*s, *t.value()))
                .collect(),
            fall_per_s: per_day(*spec.fall_per_day.value()),
            fall_multipliers: spec
                .fall_multipliers
                .iter()
                .map(|(s, t)| (*s, *t.value()))
                .collect(),
            insulation_reduces_rise: *spec.insulation_reduces_rise.value(),
            at_max: match &spec.at_max {
                AtMax::HealthDrainPerHour(t) => {
                    AtMaxRule::DrainPerSecond(*t.value() / seconds_per_hour)
                }
                AtMax::Collapse { duration_hours } => AtMaxRule::Collapse {
                    duration_s: *duration_hours.value() * seconds_per_hour,
                },
            },
        }
    }
}

/// What happened to the agent during one update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Outcome {
    pub died: bool,
    pub collapsed: bool,
}

fn product(multipliers: &[(AgentState, f64)], cond: &Conditions) -> f64 {
    multipliers
        .iter()
        .filter(|(state, _)| cond.holds(*state))
        .map(|(_, m)| *m)
        .product()
}

/// Advance every need by `dt` simulated seconds, then health. Sets the collapse and death
/// flags on the agent.
pub fn update(
    needs: &mut Needs,
    health: &mut Health,
    agent: &mut Agent,
    cond: Conditions,
    dt: f64,
    tick: u64,
    tick_s: f64,
) -> Outcome {
    let mut outcome = Outcome::default();
    let mut drain_per_s = 0.0;
    let mut any_at_max = false;
    for need in &mut needs.needs {
        step_need(need, needs.insulation, &cond, dt);
        if need.at_max() {
            any_at_max = true;
            match need.rates.at_max {
                AtMaxRule::DrainPerSecond(rate) => drain_per_s += rate,
                AtMaxRule::Collapse { duration_s } => {
                    if !agent.collapsed() {
                        let ticks = (duration_s / tick_s).ceil() as u64;
                        agent.collapsed_until_tick = Some(tick + ticks);
                        agent.asleep = true;
                        outcome.collapsed = true;
                    }
                }
            }
        }
    }
    if any_at_max {
        health.value -= drain_per_s * dt;
    } else {
        health.value += health.recover_per_s * dt;
    }
    health.value = health.value.clamp(0.0, 1.0);
    if health.value <= 0.0 && agent.alive {
        agent.alive = false;
        agent.died_at_tick = Some(tick);
        outcome.died = true;
    }
    outcome
}

fn step_need(need: &mut NeedState, insulation: f64, cond: &Conditions, dt: f64) {
    let rates = &need.rates;
    if cond.rise_condition(rates.rise_when) {
        let mut rate = rates.rise_per_s * product(&rates.rise_multipliers, cond);
        if rates.insulation_reduces_rise {
            rate *= 1.0 - insulation;
        }
        need.value += rate * dt;
    } else {
        need.value -= rates.fall_per_s * product(&rates.fall_multipliers, cond) * dt;
    }
    need.value = need.value.clamp(0.0, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need(name: &str, rates: NeedRates) -> NeedState {
        NeedState {
            name: name.into(),
            value: 0.0,
            rates,
        }
    }

    fn rates() -> NeedRates {
        NeedRates {
            rise_per_s: 1.0 / 1440.0,
            rise_when: RiseWhen::Always,
            rise_multipliers: vec![],
            fall_per_s: 0.0,
            fall_multipliers: vec![],
            insulation_reduces_rise: false,
            at_max: AtMaxRule::DrainPerSecond(1.0 / 60.0),
        }
    }

    fn agent() -> Agent {
        Agent {
            alive: true,
            died_at_tick: None,
            asleep: false,
            collapsed_until_tick: None,
            in_shelter: false,
            last_action_failed: false,
        }
    }

    const DAY: Conditions = Conditions {
        asleep: false,
        running: false,
        in_shelter: false,
        night: false,
    };

    #[test]
    fn a_need_rises_at_its_rate_and_drains_health_at_max() {
        let mut needs = Needs {
            needs: vec![need("thirst", rates())],
            insulation: 0.0,
        };
        let mut health = Health {
            value: 1.0,
            recover_per_s: 0.05 / 60.0,
        };
        let mut a = agent();
        // Half a day: thirst 0.5, health untouched.
        for t in 0..7200 {
            update(&mut needs, &mut health, &mut a, DAY, 0.1, t, 0.1);
        }
        assert!((needs.needs[0].value - 0.5).abs() < 1e-6);
        assert_eq!(health.value, 1.0);
        // Another half day plus one hour: thirst at max, health drained 1.0 per hour.
        for t in 7200..(14400 + 600) {
            update(&mut needs, &mut health, &mut a, DAY, 0.1, t, 0.1);
        }
        assert_eq!(needs.needs[0].value, 1.0);
        assert!(health.value < 1e-6, "{}", health.value);
        assert!(!a.alive);
        let died = a.died_at_tick.unwrap();
        assert!(
            died.abs_diff(14999) <= 1,
            "died at {died}: floating-point accumulation is within a tick"
        );
    }

    #[test]
    fn multipliers_conditions_and_insulation() {
        let mut r = rates();
        r.rise_when = RiseWhen::NightOutsideShelter;
        r.rise_multipliers = vec![(AgentState::Running, 2.0)];
        r.fall_per_s = 2.0 / 1440.0;
        r.fall_multipliers = vec![(AgentState::InShelter, 2.0)];
        r.insulation_reduces_rise = true;
        let mut n = need("cold", r);
        let night_running = Conditions {
            night: true,
            running: true,
            ..DAY
        };
        step_need(&mut n, 0.2, &night_running, 1.0);
        assert!((n.value - (1.0 / 1440.0) * 2.0 * 0.8).abs() < 1e-12);
        step_need(&mut n, 0.2, &DAY, 1.0);
        assert!(
            (n.value - 0.0).abs() < 1e-12,
            "falls at 2/day by day, clamped at 0"
        );
        n.value = 0.5;
        let night_in_shelter = Conditions {
            night: true,
            in_shelter: true,
            ..DAY
        };
        step_need(&mut n, 0.2, &night_in_shelter, 1.0);
        assert!(
            (n.value - (0.5 - 4.0 / 1440.0)).abs() < 1e-12,
            "inside a shelter at night, cold falls ×2"
        );
    }

    #[test]
    fn collapse_forces_sleep_and_blocks_recovery() {
        let mut r = rates();
        r.rise_when = RiseWhen::Awake;
        r.fall_per_s = 4.0 / 1440.0;
        r.at_max = AtMaxRule::Collapse { duration_s: 120.0 };
        let mut needs = Needs {
            needs: vec![need("fatigue", r)],
            insulation: 0.0,
        };
        needs.needs[0].value = 0.9999;
        let mut health = Health {
            value: 0.5,
            recover_per_s: 0.05 / 60.0,
        };
        let mut a = agent();
        let out = update(&mut needs, &mut health, &mut a, DAY, 1.0, 100, 0.1);
        assert!(out.collapsed && a.asleep && a.collapsed_until_tick == Some(100 + 1200));
        assert_eq!(health.value, 0.5, "no recovery while a need is at max");
        let asleep = Conditions {
            asleep: true,
            ..DAY
        };
        let out = update(&mut needs, &mut health, &mut a, asleep, 1.0, 101, 0.1);
        assert!(!out.collapsed, "a collapse does not restart while it lasts");
        assert!(needs.needs[0].value < 1.0, "fatigue falls while asleep");
        assert!(health.value > 0.5, "health recovers once no need is at max");
    }
}
