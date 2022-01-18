use std::fs::File;

use canny_devernay::{canny_devernay, rgb_to_grayscale, write_pathes_as_svg};
use image::{codecs::pnm::PnmDecoder, ColorType, ImageDecoder};

fn main() {
    let image_file = File::open("image.pgm").unwrap();
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

    const S: f64 = 2.0;
    const H: f64 = 5.0;
    const L: f64 = 5.0;

    let pathes = canny_devernay(&image_gray, height as usize, width as usize, S, H, L);
    write_pathes_as_svg(std::io::stdout(), &pathes, height as usize, width as usize).unwrap();
}
