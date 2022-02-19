use std::fs::File;

use circles_drawing::{canny_devernay::Params, image_to_cycle, utils::*, write_pathes_as_svg};
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

        let final_path = image_to_cycle(
            &image_gray,
            height as usize,
            width as usize,
            Params {
                s: s as f64,
                h: h as f64,
                l: l as f64,
            },
            num_pathes,
        );

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
