pub fn euler_cycle(adjs: &[usize], adj_starts: &[usize], start_node: usize) -> Vec<usize> {
    let mut removed = vec![false; *adj_starts.last().unwrap()];
    let mut cycle = vec![];
    let mut st = vec![start_node];

    while !st.is_empty() {
        let v = *st.last().unwrap();

        let mut is_bridge = true;

        for ti in adj_starts[v]..adj_starts[v + 1] {
            if !removed[ti] {
                let t = adjs[ti];
                removed[ti] = true;

                for vi in adj_starts[t]..adj_starts[t + 1] {
                    if !removed[vi] && adjs[vi] == v {
                        removed[vi] = true;
                        break;
                    }
                }

                st.push(t);

                is_bridge = false;

                break;
            }
        }

        if is_bridge {
            cycle.push(v);
            st.pop();
        }
    }

    cycle
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use rand::Rng;

    use crate::csr_graph;

    fn generate_eulerian_graph(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut rng = rand::thread_rng();

        let mut u_s = vec![];
        let mut v_s = vec![];
        let mut degs = vec![0; n];
        let mut odds = vec![];

        for _ in 0..4 * n {
            let u = rng.gen_range(0..n);
            let v = if u_s.is_empty() {
                rng.gen_range(0..n)
            } else {
                u_s[rng.gen_range(0..u_s.len())]
            };

            degs[v] += 1;
            degs[u] += 1;
            u_s.push(u);
            v_s.push(v);
            u_s.push(v);
            v_s.push(u);
        }

        for u in 0..n {
            if degs[u] % 2 != 0 {
                odds.push(u);
            }
        }

        assert!(odds.len() % 2 == 0);

        for uv in odds.chunks(2) {
            let (u, v) = (uv[0], uv[1]);
            u_s.push(u);
            v_s.push(v);
            u_s.push(v);
            v_s.push(u);
            degs[v] += 1;
            degs[u] += 1;
        }

        assert!(degs.iter().all(|d| *d % 2 == 0));

        (u_s, v_s)
    }

    #[test]
    fn test_euler_cycle_0() {
        let num_nodes = 6;

        let (u_s, v_s): (Vec<_>, Vec<_>) = vec![
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 0),
            (0, 3),
            (3, 0),
            (0, 4),
            (4, 0),
            (1, 2),
            (2, 1),
            (1, 3),
            (3, 1),
            (1, 4),
            (4, 1),
            (2, 3),
            (3, 2),
            (2, 4),
            (4, 2),
            (3, 5),
            (5, 3),
            (4, 5),
            (5, 4),
        ]
        .into_iter()
        .unzip();

        let (_, adjs, adj_starts) = csr_graph::from_edges(num_nodes, &u_s, &v_s, &[]);

        let cycle = super::euler_cycle(&adjs, &adj_starts, 0);
        assert_eq!(cycle.len() - 1, u_s.len() / 2);
        assert_eq!(cycle, vec![0, 4, 5, 3, 2, 4, 1, 3, 0, 2, 1, 0]);
    }

    #[test]
    fn random_test_euler_cycle() {
        let mut rng = rand::thread_rng();

        for _ in 0..100000 {
            let num_nodes = rng.gen_range(1..=100);
            let (u_s, v_s) = generate_eulerian_graph(num_nodes);
            let (_, adjs, adj_starts) = csr_graph::from_edges(num_nodes, &u_s, &v_s, &[]);

            let mut expected_edges = HashMap::new();
            for (&u, &v) in u_s.iter().zip(v_s.iter()) {
                let uv = if u < v { (u, v) } else { (v, u) };
                if !expected_edges.contains_key(&uv) {
                    expected_edges.insert(uv, 1);
                } else {
                    *expected_edges.get_mut(&uv).unwrap() += 1;
                }
            }
            assert!(expected_edges.iter().all(|(_, c)| *c % 2 == 0));
            expected_edges.iter_mut().for_each(|(_, c)| *c >>= 1);

            let cycle = super::euler_cycle(&adjs, &adj_starts, u_s[0]);

            let mut actual_edges = HashMap::new();
            for (&u, &v) in cycle.iter().zip(cycle.iter().skip(1)) {
                let uv = if u < v { (u, v) } else { (v, u) };
                if !actual_edges.contains_key(&uv) {
                    actual_edges.insert(uv, 1);
                } else {
                    *actual_edges.get_mut(&uv).unwrap() += 1;
                }
            }

            assert_eq!(expected_edges, actual_edges);
        }
    }
}
