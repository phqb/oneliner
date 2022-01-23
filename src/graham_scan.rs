use std::collections::BinaryHeap;

use ordered_float::NotNan;

use crate::squared_norm;

fn ccw(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
}

pub fn graham_scan(points: &[(f64, f64)]) -> Vec<usize> {
    let n = points.len();

    if n <= 3 {
        return (0..n).collect();
    }

    let mut p0 = 0;
    for i in 1..n {
        if points[i].1 > points[p0].1 {
        } else if points[i].1 < points[p0].1 || points[i].0 < points[p0].0 {
            p0 = i;
        }
    }

    let (x0, y0) = points[p0];

    let mut ps = BinaryHeap::new();
    for i in 0..n {
        if i != p0 {
            let (x, y) = points[i];
            let d = squared_norm(x - x0, y - y0).sqrt();
            // cos(angle between p0 -> p[i] and Ox)
            let cos = (x - x0) / d;
            ps.push((NotNan::new(cos).unwrap(), NotNan::new(d).unwrap(), i));
        }
    }

    let mut hull = vec![p0];

    while let Some((_, _, i)) = ps.pop() {
        let (x, y) = points[i];

        while hull.len() > 1 {
            let (prev_x, prev_y) = points[hull[hull.len() - 2]];
            let (top_x, top_y) = points[hull[hull.len() - 1]];

            if ccw(prev_x, prev_y, top_x, top_y, x, y) < 0.0 {
                hull.pop();
            } else {
                break;
            }
        }

        hull.push(i);
    }

    hull
}

#[cfg(test)]
mod test {
    use super::graham_scan;

    #[test]
    fn test_graham_scan_0() {
        let points = vec![
            (2.06, -5.22),
            (2.26, 1.6),
            (-3.3, 1.4),
            (-3.24, -5.48),
            (-0.5, -2.1),
        ];

        let hull = graham_scan(&points);
        assert_eq!(hull, vec![3, 0, 1, 2]);
    }

    #[test]
    fn test_graham_scan_1() {
        let points = vec![(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)];

        let hull = graham_scan(&points);
        assert_eq!(hull, vec![0, 3, 2, 1]);
    }
}
