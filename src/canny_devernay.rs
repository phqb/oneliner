use crate::utils::{c2i, dot, i2c, rotate_90_deg, squared_norm};

const NONE: usize = usize::MAX;

/// compute a Gaussian kernel of length n, standard deviation sigma,
/// and centered at value mean.
///
/// for example, if mean=0.5, the Gaussian will be centered in the middle point
/// between values kernel\[0\] and kernel\[1\].
fn gaussian_kernel(n: usize, sigma: f64, mean: f64) -> Vec<f64> {
    assert!(sigma > 0.0, "sigma must be positive");

    let mut kernel = vec![];
    let mut sum = 0.0;

    // compute Gaussian kernel
    for i in 0..n {
        let val = (i as f64 - mean) / sigma;
        let k = (-0.5 * val * val).exp();
        kernel.push(k);
        sum += k;
    }

    // normalization
    if sum > 0.0 {
        kernel.iter_mut().for_each(|k| *k /= sum);
    }

    kernel
}

fn gaussian_filter(input: &[u8], input_height: usize, input_width: usize, sigma: f64) -> Vec<f64> {
    assert!(sigma > 0.0, "sigma must be positive");

    // The size of the kernel is selected to guarantee that the first discarded
    // term is at least 10^prec times smaller than the central value. For that,
    // the half size of the kernel must be larger than x, with
    //   e^(-x^2/2sigma^2) = 1/10^prec
    // Then,
    //   x = sigma * sqrt( 2 * prec * ln(10) )
    let prec = 3.0;
    let offset = (sigma * (2.0 * prec * (10.0f64).ln()).sqrt()).ceil() as usize;
    let n = 1 + 2 + offset;
    let kernel = gaussian_kernel(n, sigma, offset as f64);

    let w2 = (2 * input_width) as isize;
    let h2 = (2 * input_height) as isize;

    let mut temp = vec![];
    let mut output = vec![];

    // x axis convolution
    for y in 0..input_height {
        for x in 0..input_width {
            let mut val = 0.0;

            for i in 0..n {
                let mut j = x as isize - offset as isize + i as isize;

                while j < 0 {
                    j += w2;
                }

                while j >= w2 {
                    j -= w2;
                }

                if j >= input_width as isize {
                    j = w2 - 1 - j;
                }

                val += input[y * input_width + j as usize] as f64 * kernel[i];
            }

            temp.push(val);
        }
    }

    // y axis convolution
    for y in 0..input_height {
        for x in 0..input_width {
            let mut val = 0.0;

            for i in 0..n {
                let mut j = y as isize - offset as isize + i as isize;

                while j < 0 {
                    j += h2;
                }

                while j >= h2 {
                    j -= h2;
                }

                if j >= input_height as isize {
                    j = h2 - 1 - j;
                }

                val += temp[(j as usize) * input_width + x] as f64 * kernel[i];
            }

            output.push(val);
        }
    }

    output
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

fn neighbors_5x5(x: usize, y: usize, width: usize, height: usize) -> Vec<usize> {
    let mut ns = vec![];

    for yi in 0..5 {
        if y + yi >= 2 && y + yi < height + 2 {
            for xi in 0..5 {
                if x + xi >= 2 && x + xi < width + 2 {
                    let xn = x + xi - 2;
                    let yn = y + yi - 2;
                    if !(xn == x && yn == y) {
                        ns.push(c2i(xn, yn, width));
                    }
                }
            }
        }
    }

    ns
}

fn image_gradient(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert!(input_height > 2);
    assert!(input_width > 2);

    let input = if sigma == 0.0 {
        input.iter().map(|e| *e as f64).collect()
    } else {
        gaussian_filter(input, input_height, input_width, sigma)
    };

    let mut g_xs = vec![];
    let mut g_ys = vec![];

    // for (x, y) in I_S do
    for y in 1..input_height - 1 {
        for x in 1..input_width - 1 {
            // g_x(x, y) <- I_S(x + 1, y) - I_S(x - 1, y)
            g_xs.push(input[c2i(x + 1, y, input_width)] - input[c2i(x - 1, y, input_width)]);
            // g_y(x, y) <- I_S(x, y + 1) - I_S(x, y - 1)
            g_ys.push(input[c2i(x, y + 1, input_width)] - input[c2i(x, y - 1, input_width)]);
        }
    }

    assert_eq!(g_xs.len(), (input_height - 2) * (input_width - 2));
    assert_eq!(g_ys.len(), (input_height - 2) * (input_width - 2));

    // g <- (g_x, g_y)
    (g_xs, g_ys)
}

fn compute_edge_points(
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

            let norm_xy = squared_norm(g_xs[c2i(x, y, g_width)], g_ys[c2i(x, y, g_width)]);
            let norm_x_minus_1_y =
                squared_norm(g_xs[c2i(x - 1, y, g_width)], g_ys[c2i(x - 1, y, g_width)]);
            let norm_x_plus_1_y =
                squared_norm(g_xs[c2i(x + 1, y, g_width)], g_ys[c2i(x + 1, y, g_width)]);
            let norm_x_y_minus_1 =
                squared_norm(g_xs[c2i(x, y - 1, g_width)], g_ys[c2i(x, y - 1, g_width)]);
            let norm_x_y_plus_1 =
                squared_norm(g_xs[c2i(x, y + 1, g_width)], g_ys[c2i(x, y + 1, g_width)]);

            // if ||g(x - 1, y)|| < ||g(x, y)| >= ||g(x + 1, y)|| and |g_x(x, y)| >= |g_y(x, y)| then
            if norm_x_minus_1_y < norm_xy
                && norm_xy >= norm_x_plus_1_y
                && g_xs[c2i(x, y, g_width)].abs() >= g_ys[c2i(x, y, g_width)].abs()
            {
                // θ_x <- 1
                theta_x = 1;
            } else if norm_x_y_minus_1 < norm_xy
                && norm_xy >= norm_x_y_plus_1
                && g_xs[c2i(x, y, g_width)].abs() <= g_ys[c2i(x, y, g_width)].abs()
            {
                // θ_y <- 1
                theta_y = 1;
            }

            // if θ_x =/= 0 or θ_y =/= 0 then
            if theta_x != 0 || theta_y != 0 {
                // a <- ||g(x - θ_x, y - θ_y)||
                let a = squared_norm(
                    g_xs[c2i(x - theta_x, y - theta_y, g_width)],
                    g_ys[c2i(x - theta_x, y - theta_y, g_width)],
                )
                .sqrt();
                // b <- ||g(x, y)||
                let b = norm_xy.sqrt();
                // c <- ||g(x + θ_x, y + θ_y)||
                let c = squared_norm(
                    g_xs[c2i(x + theta_x, y + theta_y, g_width)],
                    g_ys[c2i(x + theta_x, y + theta_y, g_width)],
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

fn chain_edge_points(
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

        let (x_e, y_e) = i2c(e, input_width - 4);
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

            let (x_n, y_n) = i2c(n, input_width - 4);

            // g(e) · g(n)
            let d1 = dot(
                g_xs[c2i(x_e + 1, y_e + 1, input_width - 2)],
                g_ys[c2i(x_e + 1, y_e + 1, input_width - 2)],
                g_xs[c2i(x_n + 1, y_n + 1, input_width - 2)],
                g_ys[c2i(x_n + 1, y_n + 1, input_width - 2)],
            );

            if d1 > 0.0 {
                // g(e)^⊥
                let (tx, ty) = rotate_90_deg(
                    g_xs[c2i(x_e + 1, y_e + 1, input_width - 2)],
                    g_ys[c2i(x_e + 1, y_e + 1, input_width - 2)],
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

fn thresholds_with_hysteresis(
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

    let hh = h * h;
    let ll = l * l;

    // foreach e in E do set e as not valid
    let mut valid = vec![false; (input_height - 4) * (input_width - 4)];

    for e in 0..(input_height - 4) * (input_width - 4) {
        // foreach e in E
        if next[e] != NONE || prev[e] != NONE {
            let (x, y) = i2c(e, input_width - 4);
            let e_g = c2i(x + 1, y + 1, input_width - 2);

            // and e is not valid and ||g(e)|| >= H do
            if !valid[e] && squared_norm(g_xs[e_g], g_ys[e_g]) >= hh {
                // set e as valid
                valid[e] = true;

                // f <- e
                let mut f = e;

                // while f -> n
                while next[f] != NONE {
                    let n = next[f];
                    let (x, y) = i2c(n, input_width - 4);
                    let n_g = c2i(x + 1, y + 1, input_width - 2);

                    // and n is not valid and ||g(n)|| >= L do
                    if !valid[n] && squared_norm(g_xs[n_g], g_ys[n_g]) >= ll {
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
                    let (x, y) = i2c(n, input_width - 4);
                    let n_g = c2i(x + 1, y + 1, input_width - 2);

                    // and n is not valid and ||g(n)|| >= L do
                    if !valid[n] && squared_norm(g_xs[n_g], g_ys[n_g]) >= ll {
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

    for e in 0..(input_height - 4) * (input_width - 4) {
        // foreach e in E
        if next[e] != NONE || prev[e] != NONE {
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
                path.push((e_xs[end], e_ys[end]));
                break;
            }

            path.push((e_xs[end], e_ys[end]));
            marked[end] = true;
            end = next[end];
        }

        if end != start {
            marked[end] = true;
            path.push((e_xs[end], e_ys[end]));
        }

        if !path.is_empty() {
            pathes.push(path);
        }
    }

    pathes
}

pub struct Params {
    pub s: f64,
    pub l: f64,
    pub h: f64,
}

/// Edges detection. Return a list of edge pathes.
pub fn canny_devernay(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    params: Params,
) -> Vec<Vec<(f64, f64)>> {
    assert!(input.len() >= input_height * input_width);

    let s = params.s;
    let h = params.h;
    let l = params.l;

    let (g_xs, g_ys) = image_gradient(input, input_height, input_width, s);

    let (e_xs, e_ys) = compute_edge_points(&g_xs, &g_ys, input_height, input_width);

    let (mut prev, mut next) =
        chain_edge_points(&g_xs, &g_ys, &e_xs, &e_ys, input_height, input_width);

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

    let pathes =
        chained_edge_points_to_pathes(&prev, &next, &e_xs, &e_ys, input_height, input_width);

    pathes
}
