use crate::value::Value;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<Value>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>) -> Self {
        Tensor { 
            data: data.iter().map(|x| Value::new(*x)).collect(), 
            shape: vec![data.len()],
        }
    }

    pub fn data(&self) -> Vec<f64> { self.data.iter().map(|x| x.clone().data()).collect() }
    pub fn grad(&self) -> Vec<f64> { self.data.iter().map(|x| x.clone().grad()).collect() }

    pub fn backward(&self) {
        for x in self.data.iter() {
            x.clone().backward();
        }
    }

    pub fn pow(&self, other: f64) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.clone().pow(other)).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn exp(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.clone().exp()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn relu(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.clone().relu()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.clone().sigmoid()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn tanh(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.clone().tanh()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}

use std::ops::{Add, Sub, Mul, Div, Neg};
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: Tensor) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(x, y)| x.clone() + y.clone()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f64) -> Self::Output { self + Tensor::new(vec![other]) }
}
impl Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self]) + other }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(x, y)| x.clone() - y.clone()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { self + -Tensor::new(vec![other]) }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self]) + -other }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(x, y)| x.clone() * y.clone()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![other]) }
}
impl Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self]) * other }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().zip(other.data.iter()).map(|(x, y)| x.clone() / y.clone()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![other]).pow(-1.0) }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self]) * other.pow(-1.0) }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().map(|x| -x.clone()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}