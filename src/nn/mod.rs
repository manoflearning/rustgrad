use crate::tensor::Tensor;
use rayon::prelude::*;
use rand::Rng;
use std::sync::{Arc, RwLock};
use ndarray::{s, ArrayD, IxDyn};

pub trait Layer {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn zerograd(&self) {
        self.parameters().par_iter().for_each(|tensor| {
            tensor.data.par_iter().for_each(|value| {
                value.0.write().unwrap().grad = 0.0;
            });
        });
    }
    fn requires_grad(&self, requires_grad: bool) {
        self.parameters().par_iter().for_each(|tensor| {
            tensor.data.par_iter().for_each(|value| {
                value.0.write().unwrap().requires_grad = requires_grad;
            });
        });
    }
}

pub struct ReLU;
impl Layer for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor { x.relu() }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}
pub struct Softmax;
impl Layer for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor { x.softmax() }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}

pub struct Neuron {
    pub w: Tensor,
    pub b: Tensor,
}
impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng(); // TODO: fixed random seed
        let _w = vec![rng.gen_range(-0.01..0.01); nin];
        let w: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[nin]), _w).unwrap());
        let _b = vec![rng.gen_range(-0.01..0.01)];
        let b: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1]), _b).unwrap());
        Neuron { w, b }
    }
}
impl Layer for Neuron {
    fn forward(&self, x: &Tensor) -> Tensor { x.dot(&self.w) + self.b.clone() }
    fn parameters(&self) -> Vec<Tensor> { vec![self.w.clone(), self.b.clone()] }
}

pub struct Linear {
    pub neurons: Vec<Neuron>,
}
impl Linear {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Linear { neurons }
    }
}
impl Layer for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.data.shape().len(), 2);
        assert_eq!(x.data.shape()[1], self.neurons[0].w.data.shape()[0]);

        let x_shape = x.data.shape();
        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::zeros(vec![x_shape[0], self.neurons.len()]))));
        self.neurons.par_iter().enumerate().for_each(|(i, neuron)| {
            let mut out_write = out.write().unwrap();
            out_write.data.slice_mut(s![.., i]).assign(&neuron.forward(x).data.into_shape(IxDyn(&[x_shape[0]])).unwrap());
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    fn parameters(&self) -> Vec<Tensor> {
        let out = Arc::new(RwLock::new(Vec::new()));
        self.neurons.par_iter().for_each(|neuron| {
            for param in neuron.parameters().iter() {
                out.write().unwrap().push(param.clone());
            }
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
}

pub struct Conv2d {
    pub w: Tensor,
    pub b: Tensor,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: usize,
    pub padding: usize,
}
impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut rng = rand::thread_rng(); // TODO: fixed random seed
        let w = Tensor::new(ArrayD::from_shape_vec(
            IxDyn(&[out_channels, in_channels, kernel_size, kernel_size]), 
            (0..out_channels * in_channels * kernel_size * kernel_size).map(|_| rng.gen_range(-0.01..0.01)).collect()).unwrap());
        let b = Tensor::new(ArrayD::from_shape_vec(
            IxDyn(&[1, out_channels, 1, 1]), 
            (0..out_channels).map(|_| rng.gen_range(-0.01..0.01)).collect()).unwrap());

        Conv2d {
            w,
            b,
            in_channels,
            out_channels,
            kernel_size: (kernel_size, kernel_size),
            stride,
            padding,
        }
    }
}
impl Layer for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.conv2d(&self.w, self.stride, self.padding) + self.b.clone()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![self.w.clone(), self.b.clone()] }
}

pub struct BatchNorm2d {
    pub gamma: Tensor,
    pub beta: Tensor,
}
impl BatchNorm2d {
    pub fn new(channels: usize) -> Self {
        let mut rng = rand::thread_rng(); // TODO: fixed random seed
        let _gamma = vec![rng.gen_range(-0.01..0.01); channels];
        let gamma: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[channels]), _gamma).unwrap());
        let _beta = vec![rng.gen_range(-0.01..0.01); channels];
        let beta: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[channels]), _beta).unwrap());
        BatchNorm2d { gamma, beta }
    }
}
impl Layer for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: implement batchnorm
        x.clone()
    }
    fn parameters(&self) -> Vec<Tensor> { vec![self.gamma.clone(), self.beta.clone()] }
}

pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
}
impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self { MaxPool2d { kernel_size: (kernel_size, kernel_size) } }
}
impl Layer for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor { x.max_pool2d(self.kernel_size) }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}

pub struct Flatten;
impl Layer for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        assert!(x.data.shape().len() == 4);
        let out = x.clone();
        let out_shape = out.data.shape();
        let out_len: usize = out_shape.iter().product();
        let new_shape = IxDyn(&[out_shape[0], out_len / out_shape[0]]);
        let reshaped_out_data = out.data.into_shape(new_shape).unwrap();
        Tensor { data: reshaped_out_data }
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}
impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self { Model { layers } }
    pub fn update_weights(&self, learning_rate: f64) {
        self.parameters().par_iter().for_each(|tensor| {
            tensor.data.par_iter().for_each(|value| {
                let mut value_write = value.0.write().unwrap();
                let grad = value_write.grad;
                value_write.data -= learning_rate * grad;
            });
        });
    }
}
impl Layer for Model {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out: Tensor = x.clone();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for layer in self.layers.iter() {
            for param in layer.parameters().iter() {
                out.push(param.clone());
            }
        }
        out
    }
}