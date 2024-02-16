use crate::tensor::Tensor;
use rayon::prelude::*;
use rand::Rng;
use std::sync::{Arc, RwLock};
use ndarray::{ArrayD, IxDyn};

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
        let w: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, nin]), _w).unwrap());
        let _b = vec![rng.gen_range(-0.01..0.01)];
        let b: Tensor = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[1, 1]), _b).unwrap());
        Neuron { w, b }
    }
}
impl Layer for Neuron {
    fn forward(&self, x: &Tensor) -> Tensor { self.w.dot(x) + self.b.clone() }
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
        let x_shape = x.data.shape();
        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::zeros(vec![self.neurons.len(), x_shape[1]]))));
        self.neurons.par_iter().enumerate().for_each(|(i, neuron)| {
            let temp = neuron.forward(x);
            for j in 0..x_shape[1] {
                out.write().unwrap().data[[i, j]] = temp.data[[0, j]].clone();
            }
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

// pub struct Conv2d {
//     pub w: Tensor,
//     pub b: Tensor,
//     pub kernel_size: usize,
//     pub stride: usize,
//     pub padding: usize,
//     pub bias: bool,
// }
// impl Conv2d {
//     pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
//         let mut rng = rand::thread_rng(); // TODO: fixed random seed       
//     }
// }

pub struct Model {
    pub layers: Vec<Box<dyn Layer>>,
}
impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self { Model { layers } }
    pub fn update_weights(&self, learning_rate: f32) {
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