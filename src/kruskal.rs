use std::{cmp::Reverse, collections::BinaryHeap};

use ordered_float::NotNan;

fn find(parent: &mut [usize], u: usize) -> usize {
    if parent[u] == u {
        return u;
    }

    let p = find(parent, parent[u]);
    parent[u] = p;

    p
}

fn union(parent: &mut [usize], u: usize, v: usize) {
    let pu = find(parent, u);
    let pv = find(parent, v);
    parent[pu] = pv;
}

pub fn kruskal(num_vertices: usize, weight_matrix: &[f64]) -> Vec<(usize, usize)> {
    assert!(num_vertices > 0);
    assert!(weight_matrix.len() >= num_vertices * num_vertices);

    let mut edges = BinaryHeap::new();

    for u in 0..num_vertices - 1 {
        for v in u + 1..num_vertices {
            if !weight_matrix[u * num_vertices + v].is_nan() {
                edges.push(Reverse((
                    NotNan::new(weight_matrix[u * num_vertices + v]).unwrap(),
                    u,
                    v,
                )));
            }
        }
    }

    let mut parent = (0..num_vertices).collect::<Vec<_>>();
    let mut mst = vec![];

    while let Some(Reverse((_, u, v))) = edges.pop() {
        let pu = find(&mut parent, u);
        let pv = find(&mut parent, v);

        if pu != pv {
            mst.push((u, v));
            union(&mut parent, pu, pv);
        }
    }

    mst
}

#[cfg(test)]
mod test {
    use super::kruskal;

    #[test]
    fn test_kruskal() {
        // https://en.wikipedia.org/wiki/Kruskal's_algorithm#Example
        let num_vertices = 7;
        let (a, b, c, d, e, f, g) = (0, 1, 2, 3, 4, 5, 6);

        let labels = vec!["a", "b", "c", "d", "e", "f", "g"];

        let mut weight_matrix = vec![f64::NAN; num_vertices * num_vertices];
        weight_matrix[a * num_vertices + b] = 7.0;
        weight_matrix[a * num_vertices + d] = 5.0;
        weight_matrix[b * num_vertices + c] = 8.0;
        weight_matrix[b * num_vertices + d] = 9.0;
        weight_matrix[b * num_vertices + e] = 7.0;
        weight_matrix[c * num_vertices + e] = 5.0;
        weight_matrix[d * num_vertices + e] = 15.0;
        weight_matrix[d * num_vertices + f] = 6.0;
        weight_matrix[e * num_vertices + f] = 8.0;
        weight_matrix[e * num_vertices + g] = 9.0;
        weight_matrix[f * num_vertices + g] = 11.0;

        let mst = kruskal(num_vertices, &weight_matrix);
        for (u, v) in mst.iter() {
            println!("{} - {}", labels[*u], labels[*v]);
        }

        let mst_weight = mst
            .iter()
            .map(|(u, v)| weight_matrix[u * num_vertices + v])
            .sum::<f64>();

        assert_eq!(mst_weight, (5 + 7 + 7 + 5 + 6 + 9) as f64);
    }
}
