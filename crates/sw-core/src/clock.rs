//! The clock (contract 6): ticks, decision points, the calendar, and the light level.
//!
//! The day starts at dawn. Tick 0 is the dawn boundary, the middle of dawn's transition, so
//! the light level is 0.5 and rising. Daylight holds for `daylight_fraction` of the day. The
//! light level rises and falls linearly over `twilight_hours`, centered on the dawn and dusk
//! boundaries. It is night whenever the light level is below `night_below`.

use bevy_ecs::prelude::Resource;
use sw_schema::world::Calendar;

#[derive(Resource, Debug, Clone, PartialEq)]
pub struct Clock {
    pub tick: u64,
    /// Simulated seconds per tick.
    pub tick_s: f64,
    /// Agents are asked to act every this many ticks.
    pub decision_every_ticks: u32,
    pub calendar: Calendar,
    /// The run ends after this many calendar days.
    pub lifetime_days: f64,
}

impl Clock {
    pub fn new(
        tick_s: f64,
        decision_every_ticks: u32,
        calendar: Calendar,
        lifetime_days: f64,
    ) -> Self {
        Clock {
            tick: 0,
            tick_s,
            decision_every_ticks,
            calendar,
            lifetime_days,
        }
    }

    /// Simulated seconds since the start of the run.
    pub fn time_s(&self) -> f64 {
        self.tick as f64 * self.tick_s
    }

    pub fn seconds_per_day(&self) -> f64 {
        self.calendar.seconds_per_day()
    }

    pub fn seconds_per_hour(&self) -> f64 {
        self.calendar.seconds_per_hour()
    }

    /// Calendar days since the start of the run, fractional.
    pub fn days(&self) -> f64 {
        self.time_s() / self.seconds_per_day()
    }

    /// Calendar hours since the start of the run, fractional.
    pub fn hours(&self) -> f64 {
        self.time_s() / self.seconds_per_hour()
    }

    /// The whole calendar day, counting from 0.
    pub fn day(&self) -> u64 {
        self.days().floor() as u64
    }

    /// The hour within the current day, 0 to 24.
    pub fn hour_of_day(&self) -> f64 {
        self.hours() - (self.day() as f64) * 24.0
    }

    /// The light level, 0 (full night) to 1 (full day).
    pub fn light_level(&self) -> f64 {
        light_level(
            self.hour_of_day(),
            self.calendar.daylight_fraction * 24.0,
            self.calendar.twilight_hours,
        )
    }

    pub fn is_night(&self) -> bool {
        self.light_level() < self.calendar.night_below
    }

    /// Whether agents are asked to act at this tick.
    pub fn is_decision_point(&self) -> bool {
        self.tick % u64::from(self.decision_every_ticks) == 0
    }

    /// The tick at which the lifetime ends.
    pub fn lifetime_ticks(&self) -> u64 {
        (self.lifetime_days * self.seconds_per_day() / self.tick_s).round() as u64
    }

    /// Whether the run's lifetime has ended.
    pub fn lifetime_over(&self) -> bool {
        self.tick >= self.lifetime_ticks()
    }

    /// The number of ticks in one calendar hour, as a float.
    pub fn ticks_per_hour(&self) -> f64 {
        self.seconds_per_hour() / self.tick_s
    }
}

/// The light level at `hour` (0 to 24) for a day whose daylight lasts `daylight_hours` from
/// dawn at hour 0, with linear transitions of `twilight_hours` centered on dawn and dusk.
pub fn light_level(hour: f64, daylight_hours: f64, twilight_hours: f64) -> f64 {
    let half = twilight_hours / 2.0;
    let ramp = |from_boundary: f64| -> f64 {
        // -half at the start of a transition, +half at its end, mapped to 0..1.
        if twilight_hours <= 0.0 {
            if from_boundary < 0.0 { 0.0 } else { 1.0 }
        } else {
            ((from_boundary + half) / twilight_hours).clamp(0.0, 1.0)
        }
    };
    // Distance from the nearest dawn boundary (hour 0, or hour 24 of the previous day).
    let from_dawn = if hour > 12.0 { hour - 24.0 } else { hour };
    let from_dusk = hour - daylight_hours;
    if from_dawn.abs() <= half.max(0.0) && from_dawn.abs() <= from_dusk.abs() {
        ramp(from_dawn)
    } else if from_dusk.abs() <= half.max(0.0) {
        1.0 - ramp(from_dusk)
    } else if hour < daylight_hours {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> Clock {
        Clock::new(
            0.1,
            2,
            Calendar {
                minutes_per_day: 24.0,
                daylight_fraction: 0.6,
                twilight_hours: 1.0,
                night_below: 0.5,
            },
            10.0,
        )
    }

    #[test]
    fn light_follows_the_specification() {
        let l = |h: f64| light_level(h, 14.4, 1.0);
        assert_eq!(l(0.0), 0.5);
        assert!((l(0.25) - 0.75).abs() < 1e-9);
        assert_eq!(l(0.5), 1.0);
        assert_eq!(l(7.0), 1.0);
        assert_eq!(l(13.9), 1.0);
        assert!((l(14.4) - 0.5).abs() < 1e-9);
        assert!((l(14.65) - 0.25).abs() < 1e-9);
        assert_eq!(l(14.9), 0.0);
        assert_eq!(l(20.0), 0.0);
        assert_eq!(l(23.5), 0.0);
        assert!((l(23.75) - 0.25).abs() < 1e-9);
        // Without twilight the light is a step.
        assert_eq!(light_level(0.0, 14.4, 0.0), 1.0);
        assert_eq!(light_level(14.39, 14.4, 0.0), 1.0);
        assert_eq!(light_level(14.4, 14.4, 0.0), 0.0);
    }

    #[test]
    fn night_starts_at_the_dusk_boundary() {
        let mut c = clock();
        assert!(!c.is_night(), "tick 0 is the dawn boundary, not night");
        c.tick = (14.4 * 600.0) as u64 - 1; // 14.4 hours × 60 s × 10 ticks/s, minus one
        assert!(!c.is_night(), "just before the dusk boundary");
        c.tick += 2;
        assert!(c.is_night(), "just after the dusk boundary");
        c.tick = (24.0 * 600.0) as u64;
        assert_eq!(c.day(), 1);
        assert!((c.hour_of_day()).abs() < 1e-9);
        assert!(!c.is_night());
    }

    #[test]
    fn decision_points_and_lifetime() {
        let mut c = clock();
        assert!(c.is_decision_point());
        c.tick = 1;
        assert!(!c.is_decision_point());
        c.tick = 2;
        assert!(c.is_decision_point());
        assert_eq!(c.lifetime_ticks(), 144_000);
        assert_eq!(c.ticks_per_hour(), 600.0);
    }
}
