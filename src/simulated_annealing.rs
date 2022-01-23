use rand::Rng;

fn tsp_weight(cities: &[(f64, f64)], state: &[usize]) -> f64 {
    let n = state.len();
    let mut w = 0.0;

    for i in 0..state.len() {
        let (x1, y1) = cities[state[i]];
        let (x2, y2) = cities[state[(i + 1) % n]];
        w += ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt();
    }

    w
}

pub fn tsp_simulated_annealing(cities: &[(f64, f64)], num_iterations: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let n = cities.len();
    let mut s = (0..n).collect::<Vec<_>>();
    let mut w = tsp_weight(cities, &s);
    let mut min_s = s.clone();
    let mut min_w = w;
    let mut t = 100.0;
    let alpha = 0.9995;

    for _ in 0..num_iterations {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n);
        while j == i {
            j = rng.gen_range(0..n);
        }

        s.swap(i, j);
        let w_new = tsp_weight(cities, &s);

        if w_new <= w {
            w = w_new;
        } else {
            let r = rng.gen::<f64>();
            let p = (-(w_new - w) / t).exp();

            if p >= r {
                w = w_new;
            } else {
                s.swap(i, j);
            }
        }

        if w < min_w {
            min_w = w;
            min_s = s.clone();
        }

        t *= alpha;
    }

    min_s
}

#[cfg(test)]
mod test {
    use std::io::Write;

    use super::tsp_simulated_annealing;

    #[test]
    fn test_tsp() {
        let cities = vec![
            (14.843435, 64.155902),
            (276.895963, 4.798338),
            (127.779724, 409.131953),
            (205.897745, 206.113209),
            (450.317844, 463.120257),
            (269.845155, 146.064847),
            (144.35725, 48.303781),
            (481.429614, 125.520198),
            (12.59621, 241.08094),
            (324.583947, 356.489695),
            (268.760392, 196.065698),
            (396.925564, 171.485993),
            (70.148814, 102.961104),
            (92.067259, 420.967975),
            (173.391911, 104.672534),
            (42.057297, 414.01046),
            (216.875209, 260.844929),
            (270.270021, 177.836912),
            (469.339253, 336.190592),
            (405.818279, 228.16992),
            (247.803323, 457.268561),
            (23.55585, 95.458963),
            (499.454448, 196.360554),
            (228.542659, 123.29338),
            (262.303273, 421.230876),
            (108.504974, 460.106655),
            (499.726018, 282.106906),
            (369.829464, 394.798983),
            (229.398167, 340.463102),
            (317.805933, 97.267981),
            (96.365578, 282.627837),
            (207.073666, 227.996349),
            (245.123898, 248.28607),
            (339.892912, 408.707086),
            (103.248022, 381.131207),
        ];

        let s = tsp_simulated_annealing(&cities, 50000);

        let output_file = std::fs::File::create("tsp_output.svg").unwrap();
        let mut buf_writer = std::io::BufWriter::new(output_file);

        writeln!(
            buf_writer,
            r#"<?xml version="1.0" standalone="no"?>
        <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
        "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
        <svg width="100%" height="100%" version="1.0" 
        xmlns="http://www.w3.org/2000/svg">"#
        )
        .unwrap();

        for (cx, cy) in cities.iter() {
            writeln!(
                buf_writer,
                r#"<circle cx="{:.4}" cy="{:.4}" r="5" stroke="black" stroke-width="2" fill="red"/>"#,
                cx, cy
            ).unwrap();
        }

        for i in 0..s.len() {
            let (from_x, from_y) = cities[s[i]];
            let (to_x, to_y) = cities[s[(i + 1) % s.len()]];
            writeln!(
                buf_writer,
                r#"<line x1="{:.4}" y1="{:.4}" x2="{:.4}" y2="{:.4}" style="stroke:black;stroke-width:2"/>"#,
                from_x, from_y, to_x, to_y
            ).unwrap();
        }

        writeln!(buf_writer, "</svg>").unwrap();
    }
}
