use crate::value::Value;
use rayon::prelude::*;
use std::cmp::max;
use std::sync::{Arc, RwLock};
use std::ops::{Add, Div, Mul, Neg, Sub};
use ndarray::{ArrayD, IxDyn};

// TODO: support 1D, 3D, 4D ... tensors
// TODO: support broadcasting

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
        (0..row_len).for_each(|i| {
            (0..col_len).for_each(|j| {
                out.data[[i, j]] = self.data[[i % self_shape[0], j % self_shape[1]]].clone() + other.data[[i % other_shape[0], j % other_shape[1]]].clone();
            });
        });
        out
    }
}
impl Add<f32> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f32) -> Self::Output { Tensor::new(ArrayD::from_elem(self.data.raw_dim(), other)) + self }
}
impl Add<Tensor> for f32 {
    type Output = Tensor;
    fn add(self: f32, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(other.data.raw_dim(), self)) + other }
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
        (0..row_len).for_each(|i| {
            (0..col_len).for_each(|j| {
                out.data[[i, j]] = self.data[[i % self_shape[0], j % self_shape[1]]].clone() * other.data[[i % other_shape[0], j % other_shape[1]]].clone();
            });
        });
        out
    }
}
impl Mul<f32> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f32) -> Self::Output { Tensor::new(ArrayD::from_elem(self.data.raw_dim(), other)) * self }
}
impl Mul<Tensor> for f32 {
    type Output = Tensor;
    fn mul(self: f32, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(other.data.raw_dim(), self)) * other }
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
impl Sub<f32> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f32) -> Self::Output { -Tensor::new(ArrayD::from_elem(self.data.raw_dim(), other)) + self }
}
impl Sub<Tensor> for f32 {
    type Output = Tensor;
    fn sub(self: f32, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(other.data.raw_dim(), self)) + -other }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f32> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f32) -> Self::Output { Tensor::new(ArrayD::from_elem(self.data.raw_dim(), other)) * self }
}
impl Div<Tensor> for f32 {
    type Output = Tensor; 
    fn div(self: f32, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(other.data.raw_dim(), self)) * other.pow(-1.0) }
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Tensor { Tensor { data: data.mapv(|x| Value::new(x)), } }

    // base ops: pow, exp, relu, log
    pub fn pow(&self, other: f32) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.pow(other)) }
    }
    pub fn exp(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.exp()) }
    }
    pub fn relu(&self) -> Tensor {
        // let mut out_data = ArrayD::from_elem(self.data.shape(), Value::new(0.0));
        // (0..self.data.shape()[0]).into_par_iter().for_each(|i| {
        //     let slice_self_data = self.data.index_axis(Axis(0), i);
        //     let mut slice_out_data = out_data.index_axis_mut(Axis(0), i);
    
        //     slice_out_data
        //         .iter_mut()
        //         .zip(slice_self_data.iter())
        //         .for_each(|(x, y)| {
        //             *x = y.relu();
        //         });
        // });
        // Tensor { data: out_data }
        Tensor { data: self.data.mapv(|x| x.relu()) }
    }
    pub fn log(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.log()) }
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
            for (r, _) in tmp.iter().enumerate().take(self_shape[0]) {
                out.write().unwrap().data[[r, c]] = tmp[r].clone() / sum.clone();
            }
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn sigmoid(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.sigmoid()) }
    }
    pub fn tanh(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.tanh()) }
    }
    pub fn sum(&self) -> Tensor {
        Tensor { data: ArrayD::from_elem(IxDyn(&[1, 1]), self.data.iter().map(|x| x.clone()).sum()) }
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
        self.data.iter().for_each(|i| { i.0.write().unwrap().requires_grad = requires_grad; });
    }
    pub fn data(&self) -> ArrayD<f32> { self.data.mapv(|x| x.0.read().unwrap().data) }
    pub fn grad(&self) -> ArrayD<f32> { self.data.mapv(|x| x.0.read().unwrap().grad) }
}