use crate::engine::Value;
use rand::Rng;

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}
    
impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<Value> = (0..nin).map(|_| Value::new(rng.gen_range(-1.0..1.0))).collect();
        let b: Value = Value::new(rng.gen_range(-1.0..1.0));
        Neuron { w, b }
    }
    
    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let out: Value = self.w.iter().zip(x.iter()).map(|(wi, xi)| wi.clone() * xi.clone()).sum::<Value>() + self.b.clone();
        out.relu()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut out: Vec<Value> = self.w.clone();
        out.push(self.b.clone());
        out
    }
}
    
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Layer { neurons }
    }
    
    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let out: Vec<Value> = self.neurons.iter().map(|neuron| neuron.forward(x)).collect();
        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        for neuron in self.neurons.iter() {
            out.extend(neuron.parameters());
        }
        out
    }
}
    
pub struct MLP {
    pub layers: Vec<Layer>,
}
    
impl MLP {
    pub fn new(nin: usize, nouts: &Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..nouts.len() {
            if i == 0 {
                layers.push(Layer::new(nin, nouts[i]));
            } else {
                layers.push(Layer::new(nouts[i - 1], nouts[i]));
            }
        }
        MLP { layers }
    }
    
    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let mut out: Vec<Value> = x.clone();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        for layer in self.layers.iter() {
            out.extend(layer.parameters());
        }
        out
    }

    pub fn zerograd(&self) {
        for param in self.parameters().iter() {
            param.0.borrow_mut().grad = 0.0;
        }
    }
}