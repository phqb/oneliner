pub fn from_edges(
    num_nodes: usize,
    u_s: &[usize],
    v_s: &[usize],
    w_s: &[f64],
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    assert!(u_s.iter().all(|u| *u < num_nodes));
    assert!(v_s.iter().all(|v| *v < num_nodes));

    let mut adjs = vec![0usize; u_s.len()];
    let mut adj_starts = vec![0usize; num_nodes + 1];
    let mut adj_weights = vec![0.0; w_s.len()];

    let mut adj_counts = vec![0usize; num_nodes];

    for &u in u_s.iter() {
        adj_starts[u + 1] += 1;
    }

    for i in 1..adj_starts.len() {
        adj_starts[i] += adj_starts[i - 1];
    }

    if w_s.is_empty() {
        for (&u, &v) in u_s.iter().zip(v_s) {
            let i = adj_starts[u] + adj_counts[u];
            adjs[i] = v;
            adj_counts[u] += 1;
        }
    } else {
        for ((&u, &v), &w) in u_s.iter().zip(v_s).zip(w_s) {
            let i = adj_starts[u] + adj_counts[u];
            adjs[i] = v;
            adj_weights[i] = w;
            adj_counts[u] += 1;
        }
    }

    (adj_weights, adjs, adj_starts)
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::from_edges;

    fn csr_to_matrix(
        num_nodes: usize,
        adj_weights: &[f64],
        adjs: &[usize],
        adj_starts: &[usize],
    ) -> Vec<f64> {
        let mut matrix = vec![0.0; num_nodes * num_nodes];

        for u in 0..num_nodes {
            let vs = &adjs[adj_starts[u]..adj_starts[u + 1]];
            let ws = &adj_weights[adj_starts[u]..adj_starts[u + 1]];

            for (&v, &w) in vs.iter().zip(ws) {
                matrix[u * num_nodes + v] = w;
            }
        }

        matrix
    }

    fn edges_to_matrix(num_nodes: usize, u_s: &[usize], v_s: &[usize], w_s: &[f64]) -> Vec<f64> {
        let mut matrix = vec![0.0; num_nodes * num_nodes];

        for ((&u, &v), &w) in u_s.iter().zip(v_s).zip(w_s) {
            matrix[u * num_nodes + v] = w;
        }

        matrix
    }

    #[test]
    fn test_csr_from_edges() {
        let mut rng = rand::thread_rng();

        for _ in 0..100000 {
            let n = rng.gen_range(1..100);
            let m = rng.gen_range(0..1000);

            let u_s = (0..m).map(|_| rng.gen_range(0..n)).collect::<Vec<_>>();
            let v_s = (0..m).map(|_| rng.gen_range(0..n)).collect::<Vec<_>>();
            let w_s = (0..m).map(|_| rng.gen::<f64>()).collect::<Vec<_>>();

            let (weights, adjs, starts) = from_edges(n, &u_s, &v_s, &w_s);

            let expected_matrix = edges_to_matrix(n, &u_s, &v_s, &w_s);
            let actual_matrix = csr_to_matrix(n, &weights, &adjs, &starts);

            assert_eq!(expected_matrix, actual_matrix);
        }
    }
}
