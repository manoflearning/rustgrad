use crate::tensor::Tensor;
use rand::Rng;

pub struct Neuron {
    pub w: Tensor,
    pub b: Tensor,
    pub act: fn(&Tensor) -> Tensor,
}

impl Neuron {
    pub fn new(nin: usize, act: fn(&Tensor) -> Tensor) -> Self {
        let mut rng = rand::thread_rng(); // TODO: fixed random seed

        let w: Tensor = Tensor::new(vec![(0..nin).map(|_| rng.gen_range(-1.0..1.0)).collect()], vec![1, nin]);
        let b: Tensor = Tensor::new(vec![vec![rng.gen_range(-1.0..1.0)]], vec![1, 1]);
        Neuron { w, b, act }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out: Tensor = self.w.dot(&x) + self.b.clone();
        (self.act)(&out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let out: Vec<Tensor> = vec![self.w.clone(), self.b.clone()];
        out
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, act: fn(&Tensor) -> Tensor) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin, act)).collect();
        Layer { neurons }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out: Tensor = Tensor::new(vec![vec![0.0; x.shape[1]]; self.neurons.len()], vec![self.neurons.len(), x.shape[1]]);
        for i in 0..self.neurons.len() {
            for j in 0..x.shape[1] {
                out.data[i][j] = self.neurons[i].forward(x).data[0][j].clone();
            }
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for neuron in self.neurons.iter() {
            for param in neuron.parameters().iter() {
                out.push(param.clone());
            }
        }
        out
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..nouts.len() {
            if i == 0 {
                layers.push(Layer::new(nin, nouts[i], Tensor::relu));
            } else {
                layers.push(Layer::new(nouts[i - 1], nouts[i], Tensor::relu));
            }
        }
        NeuralNetwork { layers }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out: Tensor = x.clone();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out.softmax()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for layer in self.layers.iter() {
            for param in layer.parameters().iter() {
                out.push(param.clone());
            }
        }
        out
    }

    pub fn zerograd(&self) {
        for tensor in self.parameters().iter() {
            for i in tensor.data.iter() {
                for j in i.iter() {
                    j.0.borrow_mut().grad = 0.0;
                }
            }
        }
    }

    pub fn update_weights(&self, learning_rate: f64) {
        for tensor in self.parameters().iter() {
            for i in tensor.data.iter() {
                for j in i.iter() {
                    let grad = j.0.borrow().grad;
                    j.0.borrow_mut().data -= learning_rate * grad;
                }
            }
        }
    }
}