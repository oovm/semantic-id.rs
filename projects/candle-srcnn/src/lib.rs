#![deny(missing_debug_implementations, missing_copy_implementations)]
#![warn(missing_docs, rustdoc::missing_crate_level_docs)]
#![doc = include_str!("../readme.md")]
#![doc(html_logo_url = "https://raw.githubusercontent.com/oovm/shape-rs/dev/projects/images/Trapezohedron.svg")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/oovm/shape-rs/dev/projects/images/Trapezohedron.svg")]

mod errors;

pub use crate::errors::{ExampleErrorKind, Result, ExampleError};


fn build_srcnn_model() -> impl ModuleT {
    Sequential::new(vec![
        Conv2d::new(1, 64, (9, 9), (1, 1), (4, 4), None),
        nn::Activation::new("relu"),
        Conv2d::new(64, 32, (1, 1), (1, 1), (0, 0), None),
        nn::Activation::new("relu"),
        Conv2d::new(32, 1, (5, 5), (1, 1), (2, 2), None),
    ])
}

fn main() {
    let device = candle::device();
    let mut model = build_srcnn_model().to_device(device);
    let mut optimizer = Adam::new(model.parameters(), 1e-3);

    // 加载数据集并训练模型
    let (x_train, y_train) = load_data();
    for epoch in 0..100 {
        let loss = model.loss(&x_train, &y_train);
        optimizer.backward_step(&loss);
        println!("Epoch {}, Loss: {}", epoch, loss.item());
    }
}