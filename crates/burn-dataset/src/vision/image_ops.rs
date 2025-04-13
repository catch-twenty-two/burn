use burn_tensor::{Tensor, TensorData, backend::Backend};
use image::RgbImage;

use super::BoundingBox;

/// Vertically flips an image and optionally its associated bounding boxes.
///
/// # Type Parameters
/// - `B`: The backend used by the tensor, implementing the `Backend` trait.
///
/// # Parameters
/// - `img_data`: A tuple containing:
///     - A 3D tensor representing the image, in shape `[3, H, W]` (channel-first format).
///     - An `Option` containing a list of 1D tensors representing bounding boxes in `[x, y, w, h]` format,
///       where `(x, y)` is the top-left corner of the box, and `w` and `h` are the width and height.
///
/// # Returns
/// Returns a new tuple:
/// - The vertically flipped image tensor.
/// - The vertically flipped bounding box tensors (if provided), adjusted for the new vertical orientation,
///   still in `[x, y, w, h]` format.
pub fn vertical_flip<B: Backend>(
    img_data: (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>),
) -> (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>) {
    let (mut img_tnsr, mut maybe_bbox_t_list) = img_data;
    let [_ch, height, _width] = img_tnsr.dims();

    // Flip image vertically

    img_tnsr = img_tnsr.flip([1]);

    let mut new_bbox_tnsr_list = None;

    // Flip bounding boxes vertically

    if let Some(cur_bb_tnsr_lst) = maybe_bbox_t_list.as_mut() {
        let device = B::Device::default();
        let mut tmp_bb_tnsr_list = Vec::<Tensor<B, 1>>::new();
        for bbox in cur_bb_tnsr_lst.iter() {
            let mut trans = bbox.clone().into_data().to_vec::<f32>().unwrap();
            trans[1] = height as f32 - trans[1] - trans[3];
            tmp_bb_tnsr_list.push(Tensor::<B, 1>::from_data(
                TensorData::new(trans, [4]),
                &device,
            ));
        }

        new_bbox_tnsr_list = Some(tmp_bb_tnsr_list);
    }

    (img_tnsr, new_bbox_tnsr_list)
}

/// Adjusts the contrast of an image tensor by a specified percentage.
///
/// # Type Parameters
/// - `B`: The backend used by the tensor, implementing the `Backend` trait.
///
/// # Parameters
/// - `tensor_img`: A 3D tensor representing the image in `[3, H, W]` format (channel-first).
/// - `contrast`: A `f32` value representing the percentage of contrast adjustment:
///     - `0.0` leaves the image unchanged.
///     - Positive values increase contrast (e.g., `20.0` increases by 20%).
///     - Negative values decrease contrast (e.g., `-20.0` decreases by 20%).
///
/// # Returns
/// A new tensor representing the image with adjusted contrast.
pub fn contrast<B: Backend>(tensor_img: Tensor<B, 3>, contrast: f32) -> Tensor<B, 3> {
    let max = 255.0;
    let percent = ((100.0 + contrast) / 100.0).powi(2);
    tensor_img
        .div_scalar(max)
        .sub_scalar(0.5)
        .mul_scalar(percent)
        .add_scalar(0.5)
        .mul_scalar(max)
        .clamp(0.0, 255.0)
}

/// Rotates the hue of an RGB image tensor in RGB color space.
///
/// Applies a hue shift using a rotation matrix derived from the input angle.
/// Operates directly in RGB space (no HSV conversion).
///
/// # Parameters
/// - `tensor_img`: A 3D tensor representing the image in `[3, H, W]` format (channel-first).
/// - `angle`: Hue rotation angle in degrees. Positive rotates clockwise, negative counter-clockwise.
///
/// # Returns
/// A new tensor representing the image with adjusted hue.
pub fn hue_rotate<B: Backend>(img_tnsrensor: Tensor<B, 3>, angle: f32) -> Tensor<B, 3> {
    let cosv = angle.to_radians().cos();
    let sinv = angle.to_radians().sin();

    let coeffs: [f32; 9] = [
        // Reds
        0.213 + cosv * 0.787 - sinv * 0.213,
        0.715 - cosv * 0.715 - sinv * 0.715,
        0.072 - cosv * 0.072 + sinv * 0.928,
        // Greens
        0.213 - cosv * 0.213 + sinv * 0.143,
        0.715 + cosv * 0.285 + sinv * 0.140,
        0.072 - cosv * 0.072 - sinv * 0.283,
        // Blues
        0.213 - cosv * 0.213 - sinv * 0.787,
        0.715 - cosv * 0.715 + sinv * 0.715,
        0.072 + cosv * 0.928 + sinv * 0.072,
    ];

    let chunks = img_tnsrensor.split(1, 0);

    let red = chunks[0]
        .clone()
        .mul_scalar(coeffs[0])
        .add(chunks[1].clone().mul_scalar(coeffs[1]))
        .add(chunks[2].clone().mul_scalar(coeffs[2]));

    let green = chunks[0]
        .clone()
        .mul_scalar(coeffs[3])
        .add(chunks[1].clone().mul_scalar(coeffs[4]))
        .add(chunks[2].clone().mul_scalar(coeffs[5]));

    let blue = chunks[0]
        .clone()
        .mul_scalar(coeffs[6])
        .add(chunks[1].clone().mul_scalar(coeffs[7]))
        .add(chunks[2].clone().mul_scalar(coeffs[8]));

    Tensor::cat(vec![red, green, blue], 0).clamp(0.0, 255.0)
}

/// Brightens an RGB image tensor by adding a scalar value to all pixels.
///
/// Increases the brightness of the image by adding the specified `value` to each pixel.
///
/// # Parameters
/// - `tensor_img`: A 3D tensor representing the image in `[3, H, W]` format (channel-first).
/// - `value`: The brightness adjustment value. Positive values brighten, negative values darken.
///
/// /// # Returns
/// A new tensor representing the image with adjusted brigthness.
pub fn brighten<B: Backend>(tensor_img: Tensor<B, 3>, value: i32) -> Tensor<B, 3> {
    tensor_img.add_scalar(value as f32).clamp(0.0, 255.0)
}

/// Converts an RGB image to a tensor.
///
/// Takes an `RgbImage` and converts it into a `[3, H, W]` tensor, where the channels
/// correspond to Red, Green, and Blue, respectively.
///
/// # Parameters
/// - `rgb_image`: The input RGB image to be converted.
///
/// # Returns
/// A `[3, H, W]` tensor representing the RGB image.
pub fn rgb_img_as_tensor<B: Backend>(rgb_img: image::RgbImage) -> Tensor<B, 3> {
    let width = rgb_img.width() as usize;
    let height = rgb_img.height() as usize;
    let img_vec = rgb_img.into_raw().iter().map(|&p| p as f32).collect();

    let device = B::Device::default();

    // [H, W, C] -> [C, H, W]
    Tensor::<B, 3>::from_data(
        TensorData::new(img_vec, [height, width, 3]).convert::<B::FloatElem>(),
        &device,
    )
    .permute([2, 0, 1])
}

/// Converts a bounding box into a tensor.
///
/// Takes a `BoundingBox` and converts it into a 1D tensor.
///
/// # Parameters
/// - `bbox`: The bounding box to be converted.
///
/// # Returns
/// An `Option` containing the `[x, y, w, h]` tensor representing the bounding box,
/// or `None` if the bounding box is invalid.
pub fn bbox_as_tensor<B: Backend>(bbox: BoundingBox) -> Tensor<B, 1> {
    let device = B::Device::default();
    Tensor::<B, 1>::from_data(bbox.coords, &device)
}

/// Creates an RGB test image with a specified pattern.
///
/// Generates a new image of the given width and height, filling it with the specified
/// RGB pattern.
///
/// # Arguments
///
/// * `width` – The width of the image in pixels.
/// * `height` – The height of the image in pixels.
/// * `pattern` – A 3-element array representing the RGB pattern to fill the image with.
///
/// # Returns
///
/// An `RgbImage` with the specified width, height, and pattern applied to all pixels.
pub fn create_test_image(width: u32, height: u32, pattern: [u8; 3]) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    let img_pattern: image::Rgb<u8> = image::Rgb(pattern);

    for px in img.pixels_mut() {
        *px = img_pattern;
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::BoundingBox;
    use burn_ndarray::NdArray;
    use std::hash::{DefaultHasher, Hash, Hasher};

    #[test]
    fn vertical_flip_test() {
        let img = create_test_image(12, 12, [127, 128, 255]);
        let mut bbox_tnsr_list = Vec::<Tensor<NdArray<f32>, 1>>::new();

        let bb = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };

        bbox_tnsr_list.push(bbox_as_tensor::<NdArray<f32>>(bb));

        let bb = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 1,
        };

        bbox_tnsr_list.push(bbox_as_tensor::<NdArray<f32>>(bb));

        let img_tnsr = rgb_img_as_tensor::<NdArray<f32>>(img);

        let (img_tnsr, bbox_tnsr_list) = vertical_flip((img_tnsr, Some(bbox_tnsr_list)));

        // Test hash of image
        let test_success_hash: u64 = 10732386221966926898;
        let mut h = DefaultHasher::new();
        img_tnsr.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations
        let bb_vec = bbox_tnsr_list.unwrap();

        let eq_test_t = bb_vec[0].to_data();
        let eq_test_t = eq_test_t.as_slice::<f32>().unwrap();

        assert_eq!(eq_test_t == [1.0, 5.0, 6.0, 6.0], true);

        let eq_test_t = bb_vec[1].to_data();
        let eq_test_t = eq_test_t.as_slice::<f32>().unwrap();

        assert_eq!(eq_test_t == [1.0, 6.0, 3.0, 4.0], true);
    }

    #[test]
    fn brighten_test() {
        let img = create_test_image(12, 12, [127, 128, 255]);
        let img_tnsr = rgb_img_as_tensor::<NdArray<f32>>(img);

        let img_tnsr = brighten::<NdArray>(img_tnsr, 4);

        // Test hash of image
        let test_success_hash: u64 = 10243697479348201339;
        let mut h = DefaultHasher::new();
        img_tnsr.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn contrast_test() {
        let img = create_test_image(12, 12, [127, 128, 100]);
        let img_tnsr = rgb_img_as_tensor::<NdArray<f32>>(img);

        let img_tnsr = contrast::<NdArray>(img_tnsr, 70.0);

        // Test hash of image
        let test_success_hash: u64 = 15073504107166547824;
        let mut h = DefaultHasher::new();
        img_tnsr.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn hue_rotate_test() {
        let img = create_test_image(12, 12, [127, 128, 255]);
        let img_tnsr = rgb_img_as_tensor::<NdArray<f32>>(img);

        let img_tnsr = hue_rotate::<NdArray>(img_tnsr, 180.0);

        // Test hash of image
        let test_success_hash: u64 = 13006992659458449744;
        let mut h = DefaultHasher::new();
        img_tnsr.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }
}
