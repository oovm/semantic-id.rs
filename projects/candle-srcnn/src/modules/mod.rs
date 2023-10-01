use std::fmt::{Debug, Formatter};
use std::path::Path;
use candle_core::{Device, DType, Module, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, VarMap};

/// The Super Resolution Convolutional Neural Network
///
/// ### References
/// - [SRCNN: Single Image Super Resolution Using Convolutional Neural Networks](https://arxiv.org/abs/1501.00092)
pub struct SRCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    vars: VarMap,
}

impl Debug for SRCNN {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SRCNN")
            .field("upscale", &2)
            .finish()
    }
}

impl Module for SRCNN {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv1.forward(xs)?;
        let xs = Activation::Relu.forward(&xs)?;
        let xs = self.conv2.forward(&xs)?;
        let xs = Activation::Relu.forward(&xs)?;
        let xs = self.conv3.forward(&xs)?;
        Ok(xs)
    }
}

impl Default for SRCNN {
    fn default() -> Self {
        Self::empty().unwrap()
    }
}

impl SRCNN {
    fn empty() -> candle_core::Result<Self> {
        let conv1 = Conv2d::new(
            Tensor::zeros(vec![64, 1, 9, 9], DType::F32, &Device::Cpu)?,
            Some(Tensor::zeros(vec![64], DType::F32, &Device::Cpu)?),
            Conv2dConfig {
                padding: 4,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
        );

        let conv2 = Conv2d::new(
            Tensor::zeros(vec![32, 64, 5, 5], DType::F32, &Device::Cpu)?,
            Some(Tensor::zeros(vec![32], DType::F32, &Device::Cpu)?),
            Conv2dConfig {
                padding: 2,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
        );

        let conv3 = Conv2d::new(
            Tensor::zeros(vec![1, 32, 5, 5], DType::F32, &Device::Cpu)?,
            Some(Tensor::zeros(vec![1], DType::F32, &Device::Cpu)?),
            Conv2dConfig {
                padding: 2,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
        );

        Ok(Self {
            conv1,
            conv2,
            conv3,
            vars: VarMap::new(),
        })
    }

    /// Save the model to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> candle_core::Result<()> {
        self.vars.save(path)
    }

    /// Load the model from a file.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> candle_core::Result<()> {
        self.vars.load(path)
    }
}