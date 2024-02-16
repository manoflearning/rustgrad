use crate::value::Value;
use rayon::prelude::*;
use std::cmp::max;
use std::sync::{Arc, RwLock};
use std::ops::{Add, Div, Mul, Neg, Sub};
use ndarray::{ArrayD, IxDyn, Zip};

// TODO: support 1D, 3D, 4D ... tensors

#[derive(Clone, Debug)]
pub struct Tensor {pub data: ArrayD<Value>}

// base ops: add, mul, neg
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: Tensor) -> Self::Output {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        let row_len = max(self_shape[0], other_shape[0]);
        let col_len = max(self_shape[1], other_shape[1]);

        assert!(row_len % self_shape[0] == 0 && row_len % other_shape[0] == 0, "row_len: {} self_shape[0]: {} other_shape[0]: {}", row_len, self_shape[0], other_shape[0]);
        assert!(col_len % self_shape[1] == 0 && col_len % other_shape[1] == 0, "col_len: {} self_shape[1]: {} other_shape[1]: {}", col_len, self_shape[1], other_shape[1]);

        let mut out = Tensor::new(ArrayD::zeros(vec![row_len, col_len]));
        Zip::from(&mut out.data)
        .and(self.data.broadcast(IxDyn(&[row_len, col_len])).unwrap())
        .and(other.data.broadcast(IxDyn(&[row_len, col_len])).unwrap())
        .par_for_each(|r, a, b| {
            *r = a.clone() + b.clone();
        });
        out
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], other)) + self }
}
impl Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], self)) + other }
}
impl Mul<Tensor> for Tensor { 
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        let row_len = max(self_shape[0], other_shape[0]);
        let col_len = max(self_shape[1], other_shape[1]);

        assert!(row_len % self_shape[0] == 0 && row_len % other_shape[0] == 0, "row_len: {} self_shape[0]: {} other_shape[0]: {}", row_len, self_shape[0], other_shape[0]);
        assert!(col_len % self_shape[1] == 0 && col_len % other_shape[1] == 0, "col_len: {} self_shape[1]: {} other_shape[1]: {}", col_len, self_shape[1], other_shape[1]);

        let mut out = Tensor::new(ArrayD::zeros(vec![row_len, col_len]));
        Zip::from(&mut out.data)
        .and(self.data.broadcast(IxDyn(&[row_len, col_len])).unwrap())
        .and(other.data.broadcast(IxDyn(&[row_len, col_len])).unwrap())
        .par_for_each(|r, a, b| {
            *r = a.clone() * b.clone();
        });
        out
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], other)) * self }
}
impl Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], self)) * other }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor { data: self.data.mapv(|x| -x) }
    }
}

// more ops: sub, div
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output { self + -other }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { -Tensor::new(ArrayD::from_elem(vec![1, 1], other)) + self }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], self)) + -other }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], other)) * self }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1, 1], self)) * other.pow(-1.0) }
}

impl Tensor {
    pub fn new(data: ArrayD<f64>) -> Tensor { Tensor { data: data.mapv(|x| Value::new(x)), } }

    // base ops: sum, pow, exp, relu, log
    pub fn sum(&self) -> Tensor {
        Tensor { data: ArrayD::from_elem(IxDyn(&[1, 1]), self.data.par_iter().map(|x| x.clone()).sum()) }
    }
    pub fn pow(&self, other: f64) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.pow(other); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.pow(other)) }
    }
    pub fn exp(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.exp(); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.exp()) }
    }
    pub fn relu(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.relu(); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.relu()) }
    }
    pub fn log(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.log(); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.log()) }
    }

    // more ops: dot, softmax, sigmoid, tanh
    pub fn dot(&self, other: &Tensor) -> Tensor {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::zeros(vec![self_shape[0], other_shape[1]]))));
        (0..self_shape[0]).into_par_iter().for_each(|i| {
            (0..other_shape[1]).for_each(|j| {
                let sum = (0..self_shape[1]).into_par_iter().map(|k|
                    self.data[[i, k]].clone() * other.data[[k, j]].clone()
                ).sum::<Value>();
                out.write().unwrap().data[[i, j]] = sum;
            });
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn softmax(&self) -> Tensor {
        let self_shape = self.data.shape();

        assert_eq!(self_shape.len(), 2);
        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::zeros(self.data.raw_dim()))));
        (0..self_shape[1]).into_par_iter().for_each(|c| {
            let mut tmp: Vec<Value> = Vec::new();
            for r in 0..self_shape[0] {
                tmp.push(self.data[[r, c]].exp());
            }
            let sum: Value = tmp.iter().map(|x| x.clone()).sum();
            let mut out_write = out.write().unwrap();
            for (r, _) in tmp.iter().enumerate().take(self_shape[0]) {
                out_write.data[[r, c]] = tmp[r].clone() / sum.clone();
            }
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn sigmoid(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.sigmoid(); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.sigmoid()) }
    }
    pub fn tanh(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x: &mut Value| { *x = x.tanh(); });
        Tensor { data: out }
        // Tensor { data: self.data.mapv(|x| x.tanh()) }
    }

    // backward prop
    pub fn backward(&self) {
        self.data.iter().for_each(|i| { i.backward(); });
    }

    // misc
    pub fn transpose(&self) -> Tensor {
        let self_shape = self.data.shape();
        let mut out = Tensor::new(ArrayD::zeros(vec![self_shape[1], self_shape[0]]));
        (0..self_shape[0]).for_each(|i| {
            (0..self_shape[1]).for_each(|j| {
                out.data[[j, i]] = self.data[[i, j]].clone();
            });
        });
        out
    }
    pub fn slice(&self, s: std::ops::Range<usize>) -> Tensor {
        let self_shape = self.data.shape();
        assert_eq!(self_shape.len(), 2);
        assert!(s.start < s.end);
        assert!(s.end <= self_shape[1]);
        let mut out = Tensor::new(ArrayD::zeros( vec![self_shape[0], s.end - s.start]));
        (0..self_shape[0]).for_each(|i| {
            (s.start..s.end).for_each(|j| {
                out.data[[i, j - s.start]] = self.data[[i, j]].clone();
            })
        });
        out
    }
    pub fn requires_grad(&self, requires_grad: bool) {
        self.data.par_iter().for_each(|i| { i.0.write().unwrap().requires_grad = requires_grad; });
    }
    pub fn data(&self) -> ArrayD<f64> { self.data.mapv(|x| x.0.read().unwrap().data) }
    pub fn grad(&self) -> ArrayD<f64> { self.data.mapv(|x| x.0.read().unwrap().grad) }
}