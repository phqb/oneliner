use wasm_bindgen::prelude::*;

use canny_devernay::Params;
use convex_hull::graham_scan;
use euler_cycle::euler_cycle;
use path_simplifier::ramer_douglas_peucker;
use utils::{c2i, squared_norm};

pub mod canny_devernay;
pub mod convex_hull;
pub mod csr_graph;
pub mod euler_cycle;
pub mod kruskal;
pub mod path_simplifier;
pub mod utils;

pub fn write_pathes_as_svg<W: std::io::Write>(
    mut output: W,
    pathes: &[&Vec<(f64, f64)>],
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

        for (x, y) in path.iter() {
            write!(output, "{:.4},{:.4} ", *x, *y)?;
        }

        writeln!(output, "\"/>")?;
    }

    writeln!(output, "</svg>")?;

    Ok(())
}

pub fn write_pathes_as_json<W: std::io::Write>(
    output: &mut W,
    pathes: &[&[(f64, f64)]],
    input_height: usize,
    input_width: usize,
) -> std::io::Result<()> {
    writeln!(output, "{{")?;
    writeln!(output, r#"  "width": {},"#, input_width)?;
    writeln!(output, r#"  "height": {},"#, input_height)?;
    writeln!(output, r#"  "pathes": ["#)?;

    let n = pathes.len();

    for (i, path) in pathes.iter().enumerate() {
        write!(output, "    [")?;

        let m = path.len();
        for (j, (x, y)) in path.iter().enumerate() {
            write!(output, "[{:.4}, {:.4}]", x, y)?;
            if j + 1 < m {
                write!(output, ", ")?;
            }
        }

        write!(output, "]")?;

        if i + 1 < n {
            writeln!(output, ",")?;
        } else {
            writeln!(output)?;
        }
    }

    writeln!(output, "  ]")?;
    writeln!(output, "}}")?;

    Ok(())
}

pub fn write_html<W: std::io::Write>(
    output: &mut W,
    path: &[(f64, f64)],
    input_height: usize,
    input_width: usize,
) -> std::io::Result<()> {
    writeln!(
        output,
        r#"<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
    </head>
    <body>"#
    )?;
    writeln!(output, "<script>")?;
    write!(output, "const DATA = ")?;

    write_pathes_as_json(output, &[path], input_height, input_width)?;

    writeln!(
        output,
        r#"
    </script>

    <canvas id="drawing" width="{}" height="{}"></canvas>
  
    <script>
    function startDrawing() {{
      let start;

      const points = DATA.pathes[0];
      const n = points.length;
      const ctx = document.getElementById('drawing').getContext('2d'); 
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
      ctx.lineWidth = 4;
  
      let i = 0;

      function drawStep(timestamp) {{
        ctx.beginPath();
        ctx.moveTo(points[i][0], points[i][1]);
        ctx.lineTo(points[i + 1][0], points[i + 1][1]);
        ctx.closePath(); 
        ctx.stroke(); 

        i += 1;

        if (i + 1 == n) return;

        window.requestAnimationFrame(drawStep);
      }}

      window.requestAnimationFrame(drawStep);
    }}

    startDrawing();
  </script>
  </body>
  </html>
    "#,
        input_width, input_height
    )?;

    Ok(())
}

fn path_length(path: &[(f64, f64)]) -> f64 {
    let mut len = 0.0;

    for ((u_x, u_y), (v_x, v_y)) in path.iter().zip(path.iter().skip(1)) {
        len += squared_norm(v_x - u_x, v_y - u_y).sqrt();
    }

    len
}

pub fn n_longest(mut pathes: Vec<Vec<(f64, f64)>>, n: usize) -> Vec<Vec<(f64, f64)>> {
    pathes.sort_by(|a, b| path_length(b).partial_cmp(&path_length(a)).unwrap());
    pathes.truncate(n);
    pathes
}

pub struct ShortestDists {
    /// A 2-dimensional array containing shortest pairs of points of every pair of pathes.
    ///
    /// For the pair of pathes (i, j), `indices[i][j]` is the index of the point of path `i`,
    /// and `indices[j][i]` is the index of the point of path `j`
    indices: Vec<usize>,
    /// A 2-dimensional array containing the shortest distance of every pair of pathes.
    ///
    /// `lengths[i][j]` is the shortest distance between path i and path j
    dists: Vec<f64>,
}

pub fn shortest_dists(pathes: &[Vec<(f64, f64)>]) -> ShortestDists {
    let n = pathes.len();
    let mut dist_indices = vec![0usize; n * n];
    let mut dist_lens = vec![0.0; n * n];

    for (i, from) in pathes.iter().enumerate() {
        for (j, to) in pathes.iter().enumerate() {
            if i <= j {
                continue;
            }

            let mut min_d = f64::MAX;
            let mut min_from = usize::MAX;
            let mut min_to = usize::MAX;

            for (k, (from_x, from_y)) in from.iter().enumerate() {
                for (l, (to_x, to_y)) in to.iter().enumerate() {
                    let d = squared_norm(to_x - from_x, to_y - from_y);
                    if d < min_d {
                        min_d = d;
                        min_from = k;
                        min_to = l;
                    }
                }
            }

            if min_from != usize::MAX && min_to != usize::MAX {
                dist_indices[c2i(i, j, n)] = min_from;
                dist_indices[c2i(j, i, n)] = min_to;

                dist_lens[c2i(i, j, n)] = min_d.sqrt();
            }
        }
    }

    ShortestDists {
        indices: dist_indices,
        dists: dist_lens,
    }
}

pub fn shortest_dists_by_path_indexes(
    pathes: &[Vec<(f64, f64)>],
    indexes: &[Vec<usize>],
) -> ShortestDists {
    let n = pathes.len();
    let mut dist_indices = vec![0usize; n * n];
    let mut dist_lens = vec![0.0; n * n];

    for (i, from) in pathes.iter().enumerate() {
        let from_hull = &indexes[i];

        for (j, to) in pathes.iter().enumerate() {
            if i <= j {
                continue;
            }

            let to_hull = &indexes[j];

            let mut min_d = f64::MAX;
            let mut min_from = usize::MAX;
            let mut min_to = usize::MAX;

            for &k in from_hull.iter() {
                let (from_x, from_y) = from[k];
                for &l in to_hull.iter() {
                    let (to_x, to_y) = to[l];
                    let d = squared_norm(to_x - from_x, to_y - from_y);
                    if d < min_d {
                        min_d = d;
                        min_from = k;
                        min_to = l;
                    }
                }
            }

            if min_from != usize::MAX && min_to != usize::MAX {
                dist_indices[c2i(i, j, n)] = min_from;
                dist_indices[c2i(j, i, n)] = min_to;

                dist_lens[c2i(i, j, n)] = min_d.sqrt();
            }
        }
    }

    ShortestDists {
        indices: dist_indices,
        dists: dist_lens,
    }
}

pub struct Connector {
    /// The index of source path
    pub from: usize,
    /// The index of the point in the source path
    pub from_point: usize,
    /// The index of destination path
    pub to: usize,
    /// The index of the point in the destination path
    pub to_point: usize,
}

pub fn connect_pathes(pathes: &[Vec<(f64, f64)>], indexes: &[Vec<usize>]) -> Vec<Connector> {
    let n = pathes.len();

    let shortest_dists = shortest_dists_by_path_indexes(pathes, &indexes);
    let dist_indices = shortest_dists.indices;
    let dists = shortest_dists.dists;

    let mst = kruskal::kruskal(n, &dists);

    let points = mst
        .iter()
        .map(|&(u, v)| {
            let from = dist_indices[c2i(u, v, n)];
            let to = dist_indices[c2i(v, u, n)];
            Connector {
                from: u,
                from_point: from,
                to: v,
                to_point: to,
            }
        })
        .collect();

    points
}

pub fn convex_hulls(pathes: &[Vec<(f64, f64)>]) -> Vec<Vec<usize>> {
    pathes
        .iter()
        .map(|path| graham_scan(path))
        .collect::<Vec<_>>()
}

pub fn simplify_pathes(pathes: &mut [Vec<(f64, f64)>], epsilon_squared: f64) {
    let max_len = pathes.iter().map(|p| p.len()).max().unwrap_or_default();
    let mut kept = vec![false; max_len];

    for path in pathes {
        for i in 0..path.len() {
            kept[i] = false;
        }

        ramer_douglas_peucker(path, epsilon_squared, &mut kept);
        for i in (0..path.len()).rev() {
            if !kept[i] {
                path.remove(i);
            }
        }
    }
}

pub fn make_cycle(pathes: &[Vec<(f64, f64)>], connectors: &[Connector]) -> Vec<(f64, f64)> {
    let mut path_starts = pathes.iter().map(|path| path.len()).collect::<Vec<_>>();
    path_starts.insert(0, 0);
    for i in 1..path_starts.len() {
        path_starts[i] += path_starts[i - 1];
    }

    let mut u_s = vec![];
    let mut v_s = vec![];

    for (i, path) in pathes.iter().enumerate() {
        for j in 0..path.len() - 1 {
            let u = path_starts[i] + j;
            let v = path_starts[i] + j + 1;

            for _ in 0..2 {
                u_s.push(u);
                v_s.push(v);

                u_s.push(v);
                v_s.push(u);
            }
        }
    }

    for c in connectors.iter() {
        let u = c.from;
        let from = c.from_point;
        let v = c.to;
        let to = c.to_point;
        let u = path_starts[u] + from;
        let v = path_starts[v] + to;

        for _ in 0..2 {
            u_s.push(u);
            v_s.push(v);

            u_s.push(v);
            v_s.push(u);
        }
    }

    let g = csr_graph::from_edges(path_starts[path_starts.len() - 1], &u_s, &v_s);
    let cycle = euler_cycle(&g.adjs, &g.adj_starts, u_s[0]);

    assert_eq!(cycle[0], cycle[cycle.len() - 1]);

    cycle
        .into_iter()
        .map(|u| {
            let p = path_starts.partition_point(|&i| i <= u) - 1;
            let i = u - path_starts[p];
            pathes[p][i]
        })
        .collect::<Vec<_>>()
}

pub fn image_to_cycle(
    image: &[u8],
    height: usize,
    width: usize,
    params: Params,
    max_num_pathes: usize,
) -> Vec<(f64, f64)> {
    assert!(image.len() >= height * width);

    let pathes = canny_devernay::canny_devernay(&image, height, width, params);

    let mut pathes = n_longest(pathes, max_num_pathes);

    simplify_pathes(&mut pathes, 0.004 * width as f64);

    let hulls = convex_hulls(&pathes);

    let connectors = connect_pathes(&pathes, &hulls);

    make_cycle(&pathes, &connectors)
}

#[wasm_bindgen]
pub fn wasm_image_to_cycle(
    image: &[u8],
    height: usize,
    width: usize,
    s: f64,
    l: f64,
    h: f64,
    max_num_pathes: usize,
) -> Vec<f64> {
    console_error_panic_hook::set_once();

    image_to_cycle(image, height, width, Params { s, l, h }, max_num_pathes)
        .into_iter()
        .flat_map(|(x, y)| [x, y])
        .collect()
}
