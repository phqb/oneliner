pub struct CSRGraph {
    pub adjs: Vec<usize>,
    pub adj_starts: Vec<usize>,
}

pub fn from_edges(num_nodes: usize, u_s: &[usize], v_s: &[usize]) -> CSRGraph {
    assert!(u_s.iter().all(|u| *u < num_nodes));
    assert!(v_s.iter().all(|v| *v < num_nodes));

    let mut adjs = vec![0usize; u_s.len()];
    let mut adj_starts = vec![0usize; num_nodes + 1];

    let mut adj_counts = vec![0usize; num_nodes];

    for &u in u_s.iter() {
        adj_starts[u + 1] += 1;
    }

    for i in 1..adj_starts.len() {
        adj_starts[i] += adj_starts[i - 1];
    }

    for (&u, &v) in u_s.iter().zip(v_s) {
        let i = adj_starts[u] + adj_counts[u];
        adjs[i] = v;
        adj_counts[u] += 1;
    }

    CSRGraph { adjs, adj_starts }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::from_edges;

    fn csr_to_matrix(num_nodes: usize, adjs: &[usize], adj_starts: &[usize]) -> Vec<bool> {
        let mut matrix = vec![false; num_nodes * num_nodes];

        for u in 0..num_nodes {
            let vs = &adjs[adj_starts[u]..adj_starts[u + 1]];

            for v in vs.iter() {
                matrix[u * num_nodes + v] = true;
            }
        }

        matrix
    }

    fn edges_to_matrix(num_nodes: usize, u_s: &[usize], v_s: &[usize]) -> Vec<bool> {
        let mut matrix = vec![false; num_nodes * num_nodes];

        for (&u, &v) in u_s.iter().zip(v_s) {
            matrix[u * num_nodes + v] = true;
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

            let g = from_edges(n, &u_s, &v_s);

            let expected_matrix = edges_to_matrix(n, &u_s, &v_s);
            let actual_matrix = csr_to_matrix(n, &g.adjs, &g.adj_starts);

            assert_eq!(expected_matrix, actual_matrix);
        }
    }
}
