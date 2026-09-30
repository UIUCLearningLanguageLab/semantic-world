//! Geometry on the ground plane: overlap tests, blocking, and the shelter footprint.

use glam::DVec2;

use crate::components::{Blocking, Facing, Footprint, Position};

/// One solid thing a mover can hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    pub position: DVec2,
    pub yaw: f64,
    pub footprint: Footprint,
}

impl Obstacle {
    pub fn new(position: &Position, facing: &Facing, footprint: &Footprint) -> Self {
        Obstacle {
            position: position.0,
            yaw: facing.yaw,
            footprint: *footprint,
        }
    }

    /// Whether a point is within this entity's rectangular footprint (walls only). Circles
    /// have no interior in this sense.
    pub fn contains(&self, point: DVec2) -> bool {
        match self.footprint.blocking {
            Blocking::Circle => false,
            Blocking::Walls {
                half_forward,
                half_side,
                ..
            } => {
                let local = self.local_of(point);
                local.x.abs() <= half_forward && local.y.abs() <= half_side
            }
        }
    }

    /// World point to the entity's frame: `x` along the facing, `y` to the left of it.
    fn local_of(&self, point: DVec2) -> DVec2 {
        let d = point - self.position;
        let forward = DVec2::new(self.yaw.cos(), self.yaw.sin());
        let left = DVec2::new(-forward.y, forward.x);
        DVec2::new(d.dot(forward), d.dot(left))
    }

    fn world_of(&self, local: DVec2) -> DVec2 {
        let forward = DVec2::new(self.yaw.cos(), self.yaw.sin());
        let left = DVec2::new(-forward.y, forward.x);
        self.position + forward * local.x + left * local.y
    }

    /// The nearest point of this obstacle's solid surface region to `point`, in world
    /// coordinates, and the gap between a circle of `radius` at `point` and that region
    /// (negative when overlapping).
    pub fn nearest(&self, point: DVec2, radius: f64) -> (DVec2, f64) {
        match self.footprint.blocking {
            Blocking::Circle => {
                let d = point - self.position;
                let distance = d.length();
                let gap = distance - radius - self.footprint.radius;
                let nearest = if distance > 0.0 {
                    self.position + d / distance * self.footprint.radius
                } else {
                    self.position
                };
                (nearest, gap)
            }
            Blocking::Walls {
                half_forward,
                half_side,
                thickness,
            } => {
                let local = self.local_of(point);
                // Three wall rectangles in the local frame: back, left, right. The open side
                // is the front (+x).
                let walls = [
                    (
                        DVec2::new(-half_forward, -half_side),
                        DVec2::new(-half_forward + thickness, half_side),
                    ),
                    (
                        DVec2::new(-half_forward, -half_side),
                        DVec2::new(half_forward, -half_side + thickness),
                    ),
                    (
                        DVec2::new(-half_forward, half_side - thickness),
                        DVec2::new(half_forward, half_side),
                    ),
                ];
                let mut best: Option<(DVec2, f64)> = None;
                for (lo, hi) in walls {
                    let clamped = local.clamp(lo, hi);
                    let gap = (local - clamped).length() - radius;
                    if best.is_none_or(|(_, g)| gap < g) {
                        best = Some((clamped, gap));
                    }
                }
                let (local_nearest, gap) = best.expect("three walls");
                (self.world_of(local_nearest), gap)
            }
        }
    }

    pub fn overlaps(&self, point: DVec2, radius: f64) -> bool {
        self.nearest(point, radius).1 < 0.0
    }

    /// Whether the segment from `a` to `b` passes through this obstacle's solid region.
    pub fn blocks_segment(&self, a: DVec2, b: DVec2) -> bool {
        match self.footprint.blocking {
            Blocking::Circle => segment_point_distance(a, b, self.position) < self.footprint.radius,
            Blocking::Walls {
                half_forward,
                half_side,
                thickness,
            } => {
                let la = self.local_of(a);
                let lb = self.local_of(b);
                let walls = [
                    (
                        DVec2::new(-half_forward, -half_side),
                        DVec2::new(-half_forward + thickness, half_side),
                    ),
                    (
                        DVec2::new(-half_forward, -half_side),
                        DVec2::new(half_forward, -half_side + thickness),
                    ),
                    (
                        DVec2::new(-half_forward, half_side - thickness),
                        DVec2::new(half_forward, half_side),
                    ),
                ];
                walls
                    .iter()
                    .any(|(lo, hi)| segment_intersects_box(la, lb, *lo, *hi))
            }
        }
    }
}

/// Distance from `p` to the segment `a`–`b`.
pub fn segment_point_distance(a: DVec2, b: DVec2, p: DVec2) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + ab * t - p).length()
}

/// Whether the segment `a`–`b` intersects the axis-aligned box `lo`–`hi` (slab method).
pub fn segment_intersects_box(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> bool {
    let d = b - a;
    let mut t_min = 0.0_f64;
    let mut t_max = 1.0_f64;
    for axis in 0..2 {
        let (o, dir, l, h) = if axis == 0 {
            (a.x, d.x, lo.x, hi.x)
        } else {
            (a.y, d.y, lo.y, hi.y)
        };
        if dir.abs() < 1e-12 {
            if o < l || o > h {
                return false;
            }
        } else {
            let mut t1 = (l - o) / dir;
            let mut t2 = (h - o) / dir;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return false;
            }
        }
    }
    true
}

/// The walls at the meadow's edges: a square of `half` meters from the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapBounds {
    pub half_x: f64,
    pub half_z: f64,
}

impl MapBounds {
    /// Whether a circle at `point` is fully inside the map.
    pub fn contains(&self, point: DVec2, radius: f64) -> bool {
        point.x.abs() + radius <= self.half_x && point.y.abs() + radius <= self.half_z
    }

    /// The nearest wall point and the gap to it.
    pub fn nearest(&self, point: DVec2, radius: f64) -> (DVec2, f64) {
        let gap_x = self.half_x - point.x.abs() - radius;
        let gap_z = self.half_z - point.y.abs() - radius;
        if gap_x <= gap_z {
            (DVec2::new(self.half_x.copysign(point.x), point.y), gap_x)
        } else {
            (DVec2::new(point.x, self.half_z.copysign(point.y)), gap_z)
        }
    }
}

/// The outcome of a blocked or unblocked move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveResult {
    pub position: DVec2,
    pub blocked: bool,
    /// Unit vector from the mover's center toward the blocking surface, or zero.
    pub contact_direction: DVec2,
}

/// Whether a circle at `point` is free of every obstacle and inside the map.
pub fn is_free(point: DVec2, radius: f64, obstacles: &[Obstacle], bounds: &MapBounds) -> bool {
    bounds.contains(point, radius) && !obstacles.iter().any(|o| o.overlaps(point, radius))
}

/// The sweep checks the path every this many meters, so a step cannot pass through a solid
/// thinner than this.
const SWEEP_STEP_M: f64 = 0.05;

/// Move a circle from `from` by `delta`, stopping at contact with the first solid it would
/// overlap along the way. The contact position is found by bisection to well under a
/// millimeter.
pub fn try_move(
    from: DVec2,
    delta: DVec2,
    radius: f64,
    obstacles: &[Obstacle],
    bounds: &MapBounds,
) -> MoveResult {
    // Sweep the path: find the first sample that is blocked.
    let length = delta.length();
    let samples = (length / SWEEP_STEP_M).ceil().max(1.0) as u32;
    let mut lo = 0.0_f64;
    let mut hi = None;
    for i in 1..=samples {
        let t = f64::from(i) / f64::from(samples);
        if is_free(from + delta * t, radius, obstacles, bounds) {
            lo = t;
        } else {
            hi = Some(t);
            break;
        }
    }
    let Some(mut hi) = hi else {
        return MoveResult {
            position: from + delta,
            blocked: false,
            contact_direction: DVec2::ZERO,
        };
    };
    // Bisect on the fraction of the step that stays free. `lo` is always free (the start
    // position is assumed free), `hi` always blocked.
    for _ in 0..20 {
        let mid = 0.5 * (lo + hi);
        if is_free(from + delta * mid, radius, obstacles, bounds) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let position = from + delta * lo;
    // The contact direction points at the nearest surface from the stopped position.
    let mut nearest: Option<(DVec2, f64)> = None;
    for obstacle in obstacles {
        let (point, gap) = obstacle.nearest(position, radius);
        if nearest.is_none_or(|(_, g)| gap < g) {
            nearest = Some((point, gap));
        }
    }
    let (wall_point, wall_gap) = bounds.nearest(position, radius);
    if nearest.is_none_or(|(_, g)| wall_gap < g) {
        nearest = Some((wall_point, wall_gap));
    }
    let contact_direction = nearest
        .map(|(point, _)| (point - position).normalize_or_zero())
        .unwrap_or(DVec2::ZERO);
    MoveResult {
        position,
        blocked: true,
        contact_direction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(x: f64, z: f64, radius: f64) -> Obstacle {
        Obstacle {
            position: DVec2::new(x, z),
            yaw: 0.0,
            footprint: Footprint {
                radius,
                solid: true,
                blocking: Blocking::Circle,
                height_m: 3.0,
            },
        }
    }

    fn shelter(x: f64, z: f64, yaw: f64) -> Obstacle {
        Obstacle {
            position: DVec2::new(x, z),
            yaw,
            footprint: Footprint {
                radius: 1.5,
                solid: true,
                blocking: Blocking::Walls {
                    half_forward: 1.5,
                    half_side: 1.5,
                    thickness: 0.2,
                },
                height_m: 2.5,
            },
        }
    }

    const BOUNDS: MapBounds = MapBounds {
        half_x: 50.0,
        half_z: 50.0,
    };

    #[test]
    fn walking_into_a_tree_stops_at_contact() {
        let tree = circle(5.0, 0.0, 0.3);
        let r = try_move(DVec2::ZERO, DVec2::new(10.0, 0.0), 0.25, &[tree], &BOUNDS);
        assert!(r.blocked);
        assert!(
            (r.position.x - (5.0 - 0.3 - 0.25)).abs() < 1e-4,
            "{:?}",
            r.position
        );
        assert!((r.contact_direction - DVec2::X).length() < 1e-6);
        let r = try_move(DVec2::ZERO, DVec2::new(1.0, 0.0), 0.25, &[tree], &BOUNDS);
        assert!(!r.blocked);
        assert_eq!(r.position, DVec2::new(1.0, 0.0));
    }

    #[test]
    fn segments_are_blocked_by_trunks_and_walls() {
        let tree = circle(5.0, 0.0, 0.3);
        assert!(tree.blocks_segment(DVec2::ZERO, DVec2::new(10.0, 0.0)));
        assert!(!tree.blocks_segment(DVec2::ZERO, DVec2::new(10.0, 1.0)));
        assert!(!tree.blocks_segment(DVec2::ZERO, DVec2::new(4.0, 0.0)));
        let s = shelter(0.0, 0.0, 0.0);
        // Through the back wall.
        assert!(s.blocks_segment(DVec2::new(-5.0, 0.0), DVec2::new(5.0, 0.0)));
        // In through the open side, stopping inside: nothing blocks.
        assert!(!s.blocks_segment(DVec2::new(5.0, 0.0), DVec2::new(0.0, 0.0)));
        // Across a side wall.
        assert!(s.blocks_segment(DVec2::new(0.0, -5.0), DVec2::new(0.0, 0.0)));
        assert!(segment_intersects_box(
            DVec2::new(-1.0, 0.5),
            DVec2::new(1.0, 0.5),
            DVec2::ZERO,
            DVec2::ONE
        ));
        assert!(!segment_intersects_box(
            DVec2::new(-1.0, 2.0),
            DVec2::new(1.0, 2.0),
            DVec2::ZERO,
            DVec2::ONE
        ));
    }

    #[test]
    fn the_map_edge_blocks() {
        let r = try_move(
            DVec2::new(49.0, 0.0),
            DVec2::new(5.0, 0.0),
            0.25,
            &[],
            &BOUNDS,
        );
        assert!(r.blocked);
        assert!((r.position.x - 49.75).abs() < 1e-4);
        assert!((r.contact_direction - DVec2::X).length() < 1e-6);
    }

    #[test]
    fn shelter_walls_block_and_the_open_side_admits() {
        // A shelter at the origin facing +x: open toward +x, back wall at x = -1.5.
        let s = shelter(0.0, 0.0, 0.0);
        assert!(s.contains(DVec2::ZERO));
        assert!(s.contains(DVec2::new(1.4, 1.4)));
        assert!(!s.contains(DVec2::new(1.6, 0.0)));
        // Walking in through the open side reaches the interior.
        let r = try_move(
            DVec2::new(3.0, 0.0),
            DVec2::new(-3.0, 0.0),
            0.25,
            &[s],
            &BOUNDS,
        );
        assert!(!r.blocked, "{r:?}");
        // Walking further hits the back wall, whose inner face is at x = -1.3.
        let r = try_move(DVec2::ZERO, DVec2::new(-3.0, 0.0), 0.25, &[s], &BOUNDS);
        assert!(r.blocked);
        assert!(
            (r.position.x - (-1.3 + 0.25)).abs() < 1e-4,
            "{:?}",
            r.position
        );
        assert!((r.contact_direction + DVec2::X).length() < 1e-6);
        // Walking at the side wall from outside stops at its outer face, z = 1.5.
        let r = try_move(
            DVec2::new(0.0, 3.0),
            DVec2::new(0.0, -3.0),
            0.25,
            &[s],
            &BOUNDS,
        );
        assert!(r.blocked);
        assert!((r.position.y - 1.75).abs() < 1e-4, "{:?}", r.position);
        // A rotated shelter: facing +z, so open toward +z.
        let s = shelter(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let r = try_move(
            DVec2::new(0.0, 3.0),
            DVec2::new(0.0, -3.0),
            0.25,
            &[s],
            &BOUNDS,
        );
        assert!(!r.blocked, "{r:?}");
        let r = try_move(
            DVec2::new(3.0, 0.0),
            DVec2::new(-3.0, 0.0),
            0.25,
            &[s],
            &BOUNDS,
        );
        assert!(r.blocked);
    }
}
