use crate::tensor::Tensor;
use rand::Rng;

pub trait Layer {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
    fn zerograd(&self) {
        for tensor in self.parameters().iter() {
            for i in tensor.data.iter() {
                i.0.write().unwrap().grad = 0.0;
            }
        }
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
        let w: Tensor = Tensor::new(vec![rng.gen_range(-1.0..1.0); nin], vec![1, nin]);
        let b: Tensor = Tensor::new(vec![rng.gen_range(-1.0..1.0)], vec![1, 1]);
        Neuron { w, b }
    }
}
impl Layer for Neuron {
    fn forward(&self, x: &Tensor) -> Tensor { self.w.dot(&x) + self.b.clone() }
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
        let mut out: Tensor = Tensor::new(vec![0.0; x.shape[1] * self.neurons.len()], vec![self.neurons.len(), x.shape[1]]);
        for i in 0..self.neurons.len() {
            let temp = self.neurons[i].forward(x);
            for j in 0..x.shape[1] {
                out.set(i, j, temp.get(0, j));
            }
        }
        out
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for neuron in self.neurons.iter() {
            for param in neuron.parameters().iter() {
                out.push(param.clone());
            }
        }
        out
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
    pub fn update_weights(&self, learning_rate: f64) {
        for tensor in self.parameters().iter() {
            for i in tensor.data.iter() {
                let grad = i.0.read().unwrap().grad;
                i.0.write().unwrap().data -= learning_rate * grad;
            }
        }
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