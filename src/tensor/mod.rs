use rayon::prelude::*;
use crate::value::Value;
use std::sync::{Arc, RwLock};
use std::cmp::max;
use std::ops::{Add, Div, Mul, Neg, Sub};

// TODO: redifine tensor operations using ndarray, simd
// TODO: support 1D, 3D, 4D ... tensors

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<Value>,
    pub shape: Vec<usize>,
}

// base ops: add, mul, neg
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len])));
        (0..row_len).into_par_iter().for_each(|i| (0..col_len).for_each(|j| {
            out.write().unwrap().set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            + other.get(i % other.shape[0], j % other.shape[1]));
        }));
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f64) -> Self::Output { self + Tensor::new(vec![other], vec![1, 1]) }
}
impl Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) + other }
}
impl Mul<Tensor> for Tensor { 
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len])));
        (0..row_len).into_par_iter().for_each(|i| (0..col_len).for_each(|j| {
            out.write().unwrap().set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            * other.get(i % other.shape[0], j % other.shape[1]));
        }));
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![other], vec![1, 1]) }
}
impl Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) * other }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.par_iter().map(|x| -x.clone()).collect(),
            shape: self.shape.clone(),
        }
    }
}

// more ops: sub, div
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output { self + -other }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { self + -Tensor::new(vec![other], vec![1, 1]) }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) + -other }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![other], vec![1, 1]).pow(-1.0) }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) * other.pow(-1.0) }
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {        
        let out = Arc::new(RwLock::new(Tensor {
            data: Vec::new(),
            shape: shape.clone()
        }));
        out.write().unwrap().data.resize(data.len(), Value::new(0.0));
        if shape.len() == 2 {
            (0..shape[0]).into_par_iter().for_each(|i| {
                (0..shape[1]).for_each(|j| {
                    let hash_num = out.read().unwrap().hash(&[i, j]);
                    out.write().unwrap().data[hash_num] = Value::new(data[hash_num]);
                })
            });
        } else { panic!("shape length not supported"); }

        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }

    // base ops: pow, exp, relu, log
    pub fn pow(&self, other: f64) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.pow(other)).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn exp(&self) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.exp()).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn relu(&self) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.relu()).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn log(&self) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.log()).collect(),
            shape: self.shape.clone(),
        }
    }

    // more ops: dot, softmax, sigmoid, tanh
    pub fn dot(&self, other: &Tensor) -> Tensor {
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; self.shape[0] * other.shape[1]], vec![self.shape[0], other.shape[1]])));
        (0..self.shape[0]).into_par_iter().for_each(|i| {
            (0..other.shape[1]).for_each(|j| {
                let sum = (0..self.shape[1]).into_par_iter().map(|k|
                    self.get(i, k) * other.get(k, j)
                ).sum::<Value>();
                let hash_num = out.read().unwrap().hash(&[i, j]);
                out.write().unwrap().data[hash_num] = sum;
            });
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; self.data.len()], self.shape.clone())));
        (0..self.shape[1]).into_par_iter().for_each(|c| {
            let mut tmp: Vec<Value> = Vec::new();
            for r in 0..self.shape[0] {
                tmp.push(self.get(r, c).exp());
            }
            let sum: Value = tmp.par_iter().map(|x| x.clone()).sum();
            for (r, _) in tmp.iter().enumerate().take(self.shape[0]) {
                let hash_num = out.read().unwrap().hash(&[r, c]);
                out.write().unwrap().data[hash_num] = tmp[r].clone() / sum.clone();
            }
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.sigmoid()).collect(),
            shape: self.shape.clone(),
        }
    }
    pub fn tanh(&self) -> Tensor {
        Tensor {
            data: self.data.par_iter().map(|x| x.tanh()).collect(),
            shape: self.shape.clone(),
        }
    }

    // backward prop
    pub fn backward(&self) {
        self.data.iter().for_each(|i| { i.backward(); });
    }

    // misc
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; self.data.len()], vec![self.shape[1], self.shape[0]])));
        (0..self.shape[0]).into_par_iter().for_each(|i| {
            (0..self.shape[1]).for_each(|j| {
                let hash_num = out.read().unwrap().hash(&[j, i]);
                out.write().unwrap().data[hash_num] = self.get(i, j);
            });
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn slice(&self, s: std::ops::Range<usize>) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert!(s.start < s.end);
        assert!(s.end <= self.shape[1]);
        let out = Arc::new(RwLock::new(Tensor::new(vec![0.0; (s.end - s.start) * self.shape[0]], vec![self.shape[0], s.end - s.start])));
        (0..self.shape[0]).into_par_iter().for_each(|i| {
            (s.start..s.end).for_each(|j| {
                out.write().unwrap().set(i, j - s.start, self.get(i, j));
            })
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn hash(&self, coor: &[usize]) -> usize {
        let mut out: usize = 0;
        for (i, _) in coor.iter().enumerate() {
            out = out * self.shape[i] + coor[i];
        }
        out
    }
    pub fn get(&self, x: usize, y: usize) -> Value { self.data[self.hash(&[x, y])].clone() }
    pub fn set(&mut self, x: usize, y: usize, value: Value) {
        let hash_num = self.hash(&[x, y]);
        self.data[hash_num] = value;
    }
    pub fn data(&self) -> Vec<f64> { self.data.par_iter().map(|x| x.0.read().unwrap().data).collect() }
    pub fn grad(&self) -> Vec<f64> { self.data.par_iter().map(|x| x.0.read().unwrap().grad).collect() }
}