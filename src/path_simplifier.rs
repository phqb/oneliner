use ordered_float::NotNan;

/// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
fn squared_perpendicular_dist(x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let x2_minus_x1 = x2 - x1;
    let y2_minus_y1 = y2 - y1;

    let a = x2_minus_x1 * (y1 - y0) - (x1 - x0) * y2_minus_y1;
    let a = a * a;
    let b = x2_minus_x1 * x2_minus_x1 + y2_minus_y1 * y2_minus_y1;

    if b != 0.0 {
        a / b
    } else {
        0.0
    }
}

/// https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
pub fn ramer_douglas_peucker(path: &[(f64, f64)], epsilon_squared: f64, kept: &mut [bool]) {
    let n = path.len();
    if n < 2 {
        return;
    }

    kept[0] = true;
    kept[n - 1] = true;

    let (x1, y1) = path[0];
    let (x2, y2) = path[n - 1];

    let max = path
        .iter()
        .enumerate()
        .skip(1)
        .take(n - 2)
        .map(|(i, &(x0, y0))| {
            (
                i,
                NotNan::new(squared_perpendicular_dist(x0, y0, x1, y1, x2, y2)).unwrap(),
            )
        })
        .max_by_key(|(_, d)| *d);

    if let Some((max_i, max_d)) = max {
        if max_d.into_inner() > epsilon_squared {
            kept[max_i] = true;
            ramer_douglas_peucker(&path[..=max_i], epsilon_squared, &mut kept[..=max_i]);
            ramer_douglas_peucker(&path[max_i..], epsilon_squared, &mut kept[max_i..]);
        }
    }
}

#[cfg(test)]
mod test {
    use super::ramer_douglas_peucker;

    #[test]
    fn test_rdp() {
        let path = vec![
            (38.0, 38.0),
            (53.0, 76.0),
            (117.0, 111.0),
            (174.0, 76.0),
            (245.0, 50.0),
            (276.0, 58.0),
            (313.0, 85.0),
            (434.0, 30.0),
        ];

        let epsilon = 18.0;

        let mut kept = vec![false; path.len()];
        ramer_douglas_peucker(&path, epsilon * epsilon, &mut kept);

        assert_eq!(
            kept,
            vec![true, false, true, false, true, false, true, true]
        );
    }
}
