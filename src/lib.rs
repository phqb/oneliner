use ramer_douglas_peucker::ramer_douglas_peucker;
use utils::{c2i, squared_norm};

pub mod canny_devernay;
pub mod csr_graph;
pub mod euler_cycle;
pub mod graham_scan;
pub mod kruskal;
pub mod ramer_douglas_peucker;
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

pub fn path_length(path: &[(f64, f64)]) -> f64 {
    let mut len = 0.0;

    for ((u_x, u_y), (v_x, v_y)) in path.iter().zip(path.iter().skip(1)) {
        len += squared_norm(v_x - u_x, v_y - u_y).sqrt();
    }

    len
}

pub fn shortest_dists(pathes: &[Vec<(f64, f64)>]) -> (Vec<usize>, Vec<f64>) {
    let n = pathes.len();
    let mut dist_indexes = vec![0usize; n * n];
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
                dist_indexes[c2i(i, j, n)] = min_from;
                dist_indexes[c2i(j, i, n)] = min_to;

                dist_lens[c2i(i, j, n)] = min_d.sqrt();
            }
        }
    }

    (dist_indexes, dist_lens)
}

pub fn shortest_dists_by_path_indexes(
    pathes: &[Vec<(f64, f64)>],
    indexes: &[Vec<usize>],
) -> (Vec<usize>, Vec<f64>) {
    let n = pathes.len();
    let mut dist_indexes = vec![0usize; n * n];
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
                dist_indexes[c2i(i, j, n)] = min_from;
                dist_indexes[c2i(j, i, n)] = min_to;

                dist_lens[c2i(i, j, n)] = min_d.sqrt();
            }
        }
    }

    (dist_indexes, dist_lens)
}

pub fn connect_pathes(
    pathes: &[Vec<(f64, f64)>],
    indexes: &[Vec<usize>],
) -> Vec<((usize, usize), (usize, usize))> {
    let n = pathes.len();

    let t = std::time::Instant::now();
    let (dist_indexes, dists) = shortest_dists_by_path_indexes(pathes, &indexes);
    println!("shortest_dists took {:?}", t.elapsed());

    let t = std::time::Instant::now();
    let mst = kruskal::kruskal(n, &dists);
    println!("kruskal took {:?}", t.elapsed());

    let points = mst
        .iter()
        .map(|&(u, v)| {
            let from = dist_indexes[c2i(u, v, n)];
            let to = dist_indexes[c2i(v, u, n)];
            ((u, from), (v, to))
        })
        .collect();

    points
}

pub fn simplify_pathes(
    pathes: &mut [Vec<(f64, f64)>],
    kept_indexes: &mut [Vec<usize>],
    epsilon_squared: f64,
) {
    let mut kepts = pathes
        .iter()
        .zip(kept_indexes.iter())
        .map(|(p, hull)| {
            let mut kept = vec![false; p.len()];
            for &h in hull {
                kept[h] = true;
            }
            kept
        })
        .collect::<Vec<_>>();

    for (path, kept) in pathes.iter().zip(kepts.iter_mut()) {
        ramer_douglas_peucker(path, epsilon_squared, kept);
    }

    for ((path, kept), hull) in pathes.iter_mut().zip(kepts).zip(kept_indexes.iter_mut()) {
        for (i, k) in kept.into_iter().enumerate().rev() {
            if !k {
                path.remove(i);
                for h in hull.iter_mut() {
                    if *h > i {
                        *h -= 1;
                    }
                }
            }
        }
    }
}
