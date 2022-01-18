pub fn rgb_to_grayscale(r: u8, g: u8, b: u8) -> u8 {
    (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8
}

fn squared_norm(x: f64, y: f64) -> f64 {
    x * x + y * y
}

fn coord_to_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

fn index_to_coord(index: usize, width: usize) -> (usize, usize) {
    let y = index / width;
    let x = index % width;
    (x, y)
}

fn dot(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    x1 * x2 + y1 * y2
}

fn rotate_90_deg(x: f64, y: f64) -> (f64, f64) {
    (-y, x)
}

const NONE: usize = usize::MAX;

// https://stackoverflow.com/a/8204880
pub fn gaussian_kernel(height: usize, width: usize, s: f64) -> Vec<f64> {
    let half_height = height >> 1;
    let half_width = width >> 1;

    let mut kernel = vec![];

    let c = 2.0 * s * s;

    for y in 0..height {
        let a = (y as f64 - half_height as f64) * (y as f64 - half_height as f64);

        for x in 0..width {
            let b = (x as f64 - half_width as f64) * (x as f64 - half_width as f64);
            kernel.push((-(b + a) / c).exp());
        }
    }

    let sum = kernel.iter().sum::<f64>();
    kernel.iter_mut().for_each(|v| *v /= sum);

    kernel
}

pub fn conv(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    kernel: &[f64],
    kernel_height: usize,
    kernel_width: usize,
) -> Vec<f64> {
    assert!(kernel_height % 2 != 0);
    assert!(kernel_width % 2 != 0);

    assert_eq!(kernel.len(), kernel_height * kernel_width);
    assert_eq!(input.len(), input_height * input_width);

    let half_width = (kernel_width >> 1) as i64;
    let half_height = (kernel_height >> 1) as i64;

    let mut output = vec![];

    for iy in 0..input_height {
        for ix in 0..input_width {
            let mut v = 0f64;

            for ky in 0..kernel_height {
                let mut y = (iy as i64 + ky as i64 - half_height) % input_height as i64;
                if y < 0 {
                    y += input_height as i64;
                }

                let y = y as usize;

                for kx in 0..kernel_width {
                    let mut x = (ix as i64 + kx as i64 - half_width) % input_width as i64;
                    if x < 0 {
                        x += input_width as i64;
                    }

                    let x = x as usize;

                    v += input[coord_to_index(x, y, input_width)] as f64
                        * kernel[coord_to_index(kx, ky, kernel_width)];
                }
            }

            output.push(v);
        }
    }

    output
}

fn neighbors_5x5(x: usize, y: usize, width: usize, height: usize) -> Vec<usize> {
    let mut ns = vec![];

    for yi in 0..5 {
        if y + yi >= 2 && y + yi < height + 2 {
            for xi in 0..5 {
                if x + xi >= 2 && x + xi < width + 2 {
                    let xn = x + xi - 2;
                    let yn = y + yi - 2;
                    if !(xn == x && yn == y) {
                        ns.push(coord_to_index(xn, yn, width));
                    }
                }
            }
        }
    }

    ns
}

fn unlink(prev: &mut [usize], next: &mut [usize], a: usize, b: usize) {
    if next[a] == b && prev[b] == a {
        next[a] = NONE;
        prev[b] = NONE;
    }
}

fn link(prev: &mut [usize], next: &mut [usize], a: usize, b: usize) {
    next[a] = b;
    prev[b] = a;
}

fn argmin_dist(e_x: f64, e_y: f64, e_xs: &[f64], e_ys: &[f64], ns: &[usize]) -> Option<usize> {
    if ns.is_empty() {
        return None;
    }

    let mut i = ns[0];
    let mut min_d = squared_norm(e_xs[i] - e_x, e_ys[i] - e_y);

    for &n in ns.iter().skip(1) {
        let d = squared_norm(e_xs[n] - e_x, e_ys[n] - e_y);
        if d < min_d {
            min_d = d;
            i = n;
        }
    }

    Some(i)
}

pub fn image_gradient(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    s: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(input_height > 2);
    assert!(input_width > 2);

    let (kernel_height, kernel_width) = (11, 11);
    let kernel = gaussian_kernel(kernel_height, kernel_width, s);

    // I_S <- I * K_S convolution with a Gaussian kernel
    let input = conv(
        input,
        input_height,
        input_width,
        &kernel,
        kernel_height,
        kernel_width,
    );

    let mut g_xs = vec![];
    let mut g_ys = vec![];

    // for (x, y) in I_S do
    for y in 1..input_height - 1 {
        for x in 1..input_width - 1 {
            // g_x(x, y) <- I_S(x + 1, y) - I_S(x - 1, y)
            g_xs.push(
                input[coord_to_index(x + 1, y, input_width)]
                    - input[coord_to_index(x - 1, y, input_width)],
            );
            // g_y(x, y) <- I_S(x, y + 1) - I_S(x, y - 1)
            g_ys.push(
                input[coord_to_index(x, y + 1, input_width)]
                    - input[coord_to_index(x, y - 1, input_width)],
            );
        }
    }

    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(g_ys.len(), (input_height - 2) * (input_width - 2));

    // g <- (g_x, g_y)
    (g_xs, g_ys)
}

pub fn compute_edge_points(
    g_xs: &[f64],
    g_ys: &[f64],
    input_height: usize,
    input_width: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert!(input_height > 4);
    assert!(input_width > 4);

    let g_height = input_height - 2;
    let g_width = input_width - 2;

    assert_eq!(g_xs.len(), g_height * g_width);
    assert_eq!(g_ys.len(), g_height * g_width);

    let mut e_xs = vec![];
    let mut e_ys = vec![];

    // for (x, y) in g do
    for y in 1..g_height - 1 {
        for x in 1..g_width - 1 {
            // θ_x <- 0
            let mut theta_x = 0usize;
            // θ_y <- 0
            let mut theta_y = 0usize;

            let norm_xy = squared_norm(
                g_xs[coord_to_index(x, y, g_width)],
                g_ys[coord_to_index(x, y, g_width)],
            );
            let norm_x_minus_1_y = squared_norm(
                g_xs[coord_to_index(x - 1, y, g_width)],
                g_ys[coord_to_index(x - 1, y, g_width)],
            );
            let norm_x_plus_1_y = squared_norm(
                g_xs[coord_to_index(x + 1, y, g_width)],
                g_ys[coord_to_index(x + 1, y, g_width)],
            );
            let norm_x_y_minus_1 = squared_norm(
                g_xs[coord_to_index(x, y - 1, g_width)],
                g_ys[coord_to_index(x, y - 1, g_width)],
            );
            let norm_x_y_plus_1 = squared_norm(
                g_xs[coord_to_index(x, y + 1, g_width)],
                g_ys[coord_to_index(x, y + 1, g_width)],
            );

            // if ||g(x - 1, y)|| < ||g(x, y)| >= ||g(x + 1, y)|| and |g_x(x, y)| >= |g_y(x, y)| then
            if norm_x_minus_1_y < norm_xy
                && norm_xy >= norm_x_plus_1_y
                && g_xs[coord_to_index(x, y, g_width)].abs()
                    >= g_ys[coord_to_index(x, y, g_width)].abs()
            {
                // θ_x <- 1
                theta_x = 1;
            } else if norm_x_y_minus_1 < norm_xy
                && norm_xy >= norm_x_y_plus_1
                && g_xs[coord_to_index(x, y, g_width)].abs()
                    <= g_ys[coord_to_index(x, y, g_width)].abs()
            {
                // θ_y <- 1
                theta_y = 1;
            }

            // if θ_x =/= 0 or θ_y =/= 0 then
            if theta_x != 0 || theta_y != 0 {
                // a <- ||g(x - θ_x, y - θ_y)||
                let a = squared_norm(
                    g_xs[coord_to_index(x - theta_x, y - theta_y, g_width)],
                    g_ys[coord_to_index(x - theta_x, y - theta_y, g_width)],
                )
                .sqrt();
                // b <- ||g(x, y)||
                let b = norm_xy.sqrt();
                // c <- ||g(x + θ_x, y + θ_y)||
                let c = squared_norm(
                    g_xs[coord_to_index(x + theta_x, y + theta_y, g_width)],
                    g_ys[coord_to_index(x + theta_x, y + theta_y, g_width)],
                )
                .sqrt();
                // λ <- (a - c) / (2*(a - 2b + c))
                let lambda = (a - c) / (2.0 * (a - 2.0 * b + c));
                // e_x <- x + λθ_x
                e_xs.push(x as f64 + lambda * (theta_x as f64));
                // e_y <- y + λθ_y
                e_ys.push(y as f64 + lambda * (theta_y as f64));
            } else {
                e_xs.push(f64::NAN);
                e_ys.push(f64::NAN);
            }
        }
    }

    assert_eq!(e_xs.len(), (g_height - 2) * (g_width - 2));
    assert_eq!(e_ys.len(), (g_height - 2) * (g_width - 2));

    (e_xs, e_ys)
}

pub fn chain_edge_points(
    g_xs: &[f64],
    g_ys: &[f64],
    e_xs: &[f64],
    e_ys: &[f64],
    input_height: usize,
    input_width: usize,
) -> (Vec<usize>, Vec<usize>) {
    assert!(input_height > 4);
    assert!(input_width > 4);

    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(e_xs.len(), (input_height - 4) * (input_width - 4));
    assert_eq!(e_xs.len(), (input_height - 4) * (input_width - 4));

    let mut prev = vec![NONE; (input_height - 4) * (input_width - 4)];
    let mut next = vec![NONE; (input_height - 4) * (input_width - 4)];

    // foreach e in E do
    for (e, (&e_x, &e_y)) in e_xs.iter().zip(e_ys).enumerate() {
        if e_x.is_nan() || e_y.is_nan() {
            // skip non edge points
            continue;
        }

        let (x_e, y_e) = index_to_coord(e, input_width - 4);
        let ns = neighbors_5x5(x_e, y_e, input_width - 4, input_height - 4);

        let mut n_f = vec![];
        let mut n_b = vec![];

        // N_F <- { n in N_E(e, 2): g(e) · g(n) > 0 and (vector from e to n) · g(e)^⊥ > 0 }
        // N_B <- { n in N_E(e, 2): g(e) · g(n) > 0 and (vector from e to n) · g(e)^⊥ < 0 }
        for n in ns {
            if e_xs[n].is_nan() || e_ys[n].is_nan() {
                // skip non edge points
                continue;
            }

            let (x_n, y_n) = index_to_coord(n, input_width - 4);

            // g(e) · g(n)
            let d1 = dot(
                g_xs[coord_to_index(x_e + 1, y_e + 1, input_width - 2)],
                g_ys[coord_to_index(x_e + 1, y_e + 1, input_width - 2)],
                g_xs[coord_to_index(x_n + 1, y_n + 1, input_width - 2)],
                g_ys[coord_to_index(x_n + 1, y_n + 1, input_width - 2)],
            );

            if d1 > 0.0 {
                // g(e)^⊥
                let (tx, ty) = rotate_90_deg(
                    g_xs[coord_to_index(x_e + 1, y_e + 1, input_width - 2)],
                    g_ys[coord_to_index(x_e + 1, y_e + 1, input_width - 2)],
                );
                // (vector from e to n) · g(e)^⊥
                let d2 = dot(e_xs[n] - e_x, e_ys[n] - e_y, tx, ty);

                if d2 > 0.0 {
                    // forward neighbors
                    n_f.push(n);
                } else if d2 < 0.0 {
                    // backward neighbors
                    n_b.push(n);
                }
            }
        }

        // f <- argmin_{n in N_F} dist(e, n)
        let f = argmin_dist(e_x, e_y, e_xs, e_ys, &n_f);

        // b <- argmin_{n in N_B} dist(e, n)
        let b = argmin_dist(e_x, e_y, e_xs, e_ys, &n_b);

        if let Some(f) = f {
            // if ∅ -> f or (a -> f and dist(e, f) < dist(a, f)) then
            if prev[f] == NONE
                || squared_norm(e_xs[f] - e_x, e_ys[f] - e_y)
                    < squared_norm(e_xs[f] - e_xs[prev[f]], e_ys[f] - e_ys[prev[f]])
            {
                // unlink * -> f, if linked
                if prev[f] != NONE {
                    let a = prev[f];
                    unlink(&mut prev, &mut next, a, f);
                }
                // unlink e -> *, if linked
                if next[e] != NONE {
                    let a = next[e];
                    unlink(&mut prev, &mut next, e, a);
                }
                // link e -> f
                link(&mut prev, &mut next, e, f);
            }
        }

        if let Some(b) = b {
            // if b -> ∅ or (b -> a and dist(b, e) < dist(b, a))
            if next[b] == NONE
                || squared_norm(e_x - e_xs[b], e_y - e_ys[b])
                    < squared_norm(e_xs[next[b]] - e_xs[b], e_ys[next[b]] - e_ys[b])
            {
                // unlink b -> *, if linked
                if next[b] != NONE {
                    let a = next[b];
                    unlink(&mut prev, &mut next, b, a);
                }
                // unlink * -> e, if linked
                if prev[e] != NONE {
                    let a = prev[e];
                    unlink(&mut prev, &mut next, a, e);
                }
                // link b -> e
                link(&mut prev, &mut next, b, e);
            }
        }
    }

    (prev, next)
}

pub fn thresholds_with_hysteresis(
    prev: &mut [usize],
    next: &mut [usize],
    g_xs: &[f64],
    g_ys: &[f64],
    h: f64,
    l: f64,
    input_height: usize,
    input_width: usize,
) {
    assert!(input_height > 4);
    assert!(input_width > 4);

    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(prev.len(), (input_height - 4) * (input_width - 4));
    assert_eq!(next.len(), (input_height - 4) * (input_width - 4));

    // foreach e in E do set e as not valid
    let mut valid = vec![false; (input_height - 4) * (input_width - 4)];

    let hh = h * h;
    let ll = l * l;

    // foreach e in E
    for e in 0..(input_height - 4) * (input_width - 4) {
        if next[e] != NONE && prev[e] != NONE {
            // and e is not valid and ||g(e)|| >= H do
            if !valid[e] && squared_norm(g_xs[e], g_ys[e]) >= hh {
                // set e as valid
                valid[e] = true;

                // f <- e
                let mut f = e;

                // while f -> n
                while next[f] != NONE {
                    let n = next[f];
                    // and n is not valid and ||g(n)|| >= L do
                    if !valid[n] && squared_norm(g_xs[n], g_ys[n]) >= ll {
                        // set n as valid
                        valid[n] = true;
                        // f <- n
                        f = n;
                    } else {
                        break;
                    }
                }

                // b <- e
                let mut b = e;

                // while n -> b
                while prev[b] != NONE {
                    let n = prev[b];
                    // and n is not valid and ||g(n)|| >= L do
                    if !valid[n] && squared_norm(g_xs[n], g_ys[n]) >= ll {
                        // set n as valid
                        valid[n] = true;
                        // b <- n
                        b = n;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // foreach e in E
    for e in 0..(input_height - 4) * (input_width - 4) {
        if next[e] != NONE && prev[e] != NONE {
            // and e is not valid do
            if !valid[e] {
                // unlink e -> *, if linked
                if next[e] != NONE {
                    unlink(prev, next, e, next[e]);
                }

                // unlink * -> e, if linked
                if prev[e] != NONE {
                    unlink(prev, next, prev[e], e);
                }
            }
        }
    }
}

fn chained_edge_points_to_pathes(
    prev: &[usize],
    next: &[usize],
    e_xs: &[f64],
    e_ys: &[f64],
    input_height: usize,
    input_width: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut marked = vec![false; (input_height - 4) * (input_width - 4)];
    let mut pathes = vec![];

    for i in 0..(input_height - 4) * (input_width - 4) {
        if marked[i] {
            continue;
        }

        let mut start = i;
        while prev[start] != NONE {
            if prev[start] == i {
                break;
            }

            start = prev[start];
        }

        let mut path = vec![];

        let mut end = start;
        while next[end] != NONE {
            if marked[end] {
                break;
            }

            path.push((e_xs[end], e_ys[end]));
            marked[end] = true;
            end = next[end];
        }

        if end != start {
            marked[end] = true;
            path.push((e_xs[end], e_ys[end]));

            pathes.push(path);
        }
    }

    pathes
}

pub fn canny_devernay(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    s: f64,
    h: f64,
    l: f64,
) -> Vec<Vec<(f64, f64)>> {
    let t = std::time::Instant::now();
    let (g_xs, g_ys) = image_gradient(input, input_height, input_width, s);
    eprintln!("image_gradient() took {:?}", t.elapsed());

    let t = std::time::Instant::now();
    let (e_xs, e_ys) = compute_edge_points(&g_xs, &g_ys, input_height, input_width);
    eprintln!("compute_edge_points() took {:?}", t.elapsed());

    let t = std::time::Instant::now();
    let (mut prev, mut next) =
        chain_edge_points(&g_xs, &g_ys, &e_xs, &e_ys, input_height, input_width);
    eprintln!("chain_edge_points() took {:?}", t.elapsed());

    let t = std::time::Instant::now();
    thresholds_with_hysteresis(
        &mut prev,
        &mut next,
        &g_xs,
        &g_ys,
        h,
        l,
        input_height,
        input_width,
    );
    eprintln!("thresholds_with_hysteresis() took {:?}", t.elapsed());

    let t = std::time::Instant::now();
    let pathes =
        chained_edge_points_to_pathes(&prev, &next, &e_xs, &e_ys, input_height, input_width);
    eprintln!("converting to pathes took {:?}", t.elapsed());

    pathes
}

pub fn write_pathes_as_svg<W: std::io::Write>(
    mut output: W,
    pathes: &[Vec<(f64, f64)>],
    input_height: usize,
    input_width: usize,
) -> std::io::Result<()> {
    writeln!(
        output,
        r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>"#
    )?;
    writeln!(
        output,
        r#"<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">"#
    )?;
    writeln!(
        output,
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
        input_width, input_height
    )?;

    for path in pathes {
        write!(
            output,
            r#"<polyline stroke-width="1" fill="none" stroke="black" points=""#
        )?;

        for (x, y) in path {
            write!(output, "{:.4},{:.4} ", *x, *y)?;
        }

        writeln!(output, "\"/>")?;
    }

    writeln!(output, "</svg>")?;

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{conv, coord_to_index, gaussian_kernel, index_to_coord, neighbors_5x5};

    #[test]
    fn test_gaussian_kernel() {
        let k = gaussian_kernel(5, 5, 0.84089642);
        println!("{:?}", k);
    }

    #[test]
    fn test_conv() {
        let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kernel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let output = conv(&input, 4, 4, &kernel, 3, 3);
        println!("{:?}", output);
    }

    #[test]
    fn test_coord_index_conversion() {
        let width = 1000;
        for y in 0..width {
            for x in 0..width {
                let index = coord_to_index(x, y, width);
                let (x1, y1) = index_to_coord(index, width);
                assert_eq!(x, x1);
                assert_eq!(y, y1);
            }
        }
    }

    #[test]
    fn test_neighbors_5x5() {
        let n = neighbors_5x5(2, 2, 5, 5);
        println!("{}", n.len());
        println!(
            "{:?}",
            n.iter().map(|i| index_to_coord(*i, 5)).collect::<Vec<_>>()
        );
    }
}
