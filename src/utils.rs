pub fn rgb_to_grayscale(r: u8, g: u8, b: u8) -> u8 {
    (0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) as u8
}

pub fn squared_norm(x: f64, y: f64) -> f64 {
    x * x + y * y
}

pub fn c2i(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

pub fn i2c(index: usize, width: usize) -> (usize, usize) {
    let y = index / width;
    let x = index % width;
    (x, y)
}

pub fn dot(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    x1 * x2 + y1 * y2
}

pub fn rotate_90_deg(x: f64, y: f64) -> (f64, f64) {
    (-y, x)
}

#[cfg(test)]
mod test {
    use super::{c2i, i2c};

    #[test]
    fn test_coord_index_conversion() {
        let width = 1000;
        for y in 0..width {
            for x in 0..width {
                let index = c2i(x, y, width);
                let (x1, y1) = i2c(index, width);
                assert_eq!(x, x1);
                assert_eq!(y, y1);
            }
        }
    }
}
