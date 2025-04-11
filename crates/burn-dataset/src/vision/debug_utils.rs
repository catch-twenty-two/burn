use burn_tensor::{Tensor, backend::Backend};
/// place holder
pub fn print_tensor_img<B: Backend>(img_tensor: &Tensor<B, 3>) {
    let [width, height, _ch] = img_tensor.dims();

    let tensor_vec: Vec<u8> = img_tensor
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&p| p as u8)
        .collect();

    println!("Padded Tensor contains:");

    for y in 0..tensor_vec.len() {
        if y != 0 && (y % (height as usize * 3)) == 0 {
            println!("");
        }
        print!("{} ", tensor_vec[y]);
    }
}
/// placeholder
pub fn save_tensor_as_image<B: Backend>(t: &Tensor<B, 3>) {
    let t = t.clone().permute([1, 2, 0]);

    let buf: Vec<u8> = t
        .to_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&p| p as u8)
        .collect();

    let height = t.shape().dims::<3>()[0] as u32;
    let width = t.shape().dims::<3>()[1] as u32;
    let img = image::RgbImage::from_vec(width, height, buf).unwrap();
    img.save("/home/jimmy/Development/Images/test.png").unwrap();
}
