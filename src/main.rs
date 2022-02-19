use std::fs::File;

use circles_drawing::{
    canny_devernay::canny_devernay, connect_pathes, convex_hulls, csr_graph,
    euler_cycle::euler_cycle, path_length, simplify_pathes, utils::*, write_pathes_as_svg,
};
use image::{codecs::pnm::PnmDecoder, ColorType, ImageDecoder};

fn main() {
    let args = std::env::args().into_iter().take(3).collect::<Vec<_>>();
    let (input_path, output_prefix) = (&args[1], &args[2]);
    let input_path = std::path::Path::new(input_path);
    let input_file_name = input_path.file_name().unwrap().to_str().unwrap();

    let image_file = File::open(input_path).unwrap();
    let decoder = PnmDecoder::new(image_file).unwrap();
    let (width, height) = decoder.dimensions();
    let color_type = decoder.color_type();
    let mut image_buf = vec![0; decoder.total_bytes() as usize];
    decoder.read_image(image_buf.as_mut()).unwrap();

    let image_gray = match color_type {
        ColorType::Rgb8 => image_buf
            .chunks(3)
            .map(|rgb| rgb_to_grayscale(rgb[0], rgb[1], rgb[2]))
            .collect::<Vec<_>>(),
        ColorType::L8 => image_buf,
        _ => panic!("unsupported color type {:?}", color_type),
    };

    let params = vec![
        // (0, 0, 0),
        // (1, 0, 0),
        // (2, 0, 0),
        // (1, 5, 5),
        // (1, 10, 10),
        // (1, 15, 15),
        // (1, 20, 20),
        // (1, 5, 10),
        // (1, 5, 15),
        (1, 5, 20),
        // (1, 0, 15),
        // (1, 1, 15),
        // (1, 3, 15),
        // (1, 10, 15),
    ];

    let num_pathes = 300;

    for (s, l, h) in params {
        println!("S = {}, L = {}, H = {}", s, l, h);

        let pathes = canny_devernay(
            &image_gray,
            height as usize,
            width as usize,
            s as f64,
            h as f64,
            l as f64,
        );

        let mut pathes = pathes;
        pathes.sort_by(|a, b| path_length(b).partial_cmp(&path_length(a)).unwrap());
        pathes.truncate(num_pathes);

        simplify_pathes(&mut pathes, 0.004 * width as f64);

        let hulls = convex_hulls(&pathes);

        let connectors = connect_pathes(&pathes, &hulls);

        let mut path_starts = pathes.iter().map(|path| path.len()).collect::<Vec<_>>();
        path_starts.insert(0, 0);
        for i in 1..path_starts.len() {
            path_starts[i] += path_starts[i - 1];
        }

        let final_path = {
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

            for &((u, from), (v, to)) in connectors.iter() {
                let u = path_starts[u] + from;
                let v = path_starts[v] + to;

                for _ in 0..2 {
                    u_s.push(u);
                    v_s.push(v);

                    u_s.push(v);
                    v_s.push(u);
                }
            }

            let (_, adjs, adj_starts) =
                csr_graph::from_edges(path_starts[path_starts.len() - 1], &u_s, &v_s, &[]);
            let cycle = euler_cycle(&adjs, &adj_starts, u_s[0]);

            assert_eq!(cycle[0], cycle[cycle.len() - 1]);

            cycle
                .into_iter()
                .map(|u| {
                    let p = path_starts.partition_point(|&i| i <= u) - 1;
                    let i = u - path_starts[p];
                    pathes[p][i]
                })
                .collect::<Vec<_>>()
        };

        println!("num points = {}", final_path.len());

        let svg_output_file = std::path::Path::new(output_prefix).join(format!(
            "{}_{}_{}_{}_{}_hull_simplified_connected_eulerian_2.svg",
            input_file_name, s, l, h, num_pathes
        ));

        write_pathes_as_svg(
            std::io::BufWriter::new(File::create(svg_output_file).unwrap()),
            &[&final_path],
            height as usize,
            width as usize,
        )
        .unwrap();
    }
}
