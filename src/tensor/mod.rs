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
        
        let mut out_shape = Vec::new();
        for i in 0..max(self_shape.len(), other_shape.len()) {
            if i < self_shape.len() && i < other_shape.len() {
                out_shape.push(max(self_shape[i], other_shape[i]));
            } else if i < self_shape.len() {
                out_shape.push(self_shape[i]);
            } else {
                out_shape.push(other_shape[i]);
            }
        }

        let mut out = Tensor::new(ArrayD::zeros(out_shape.clone()));
        Zip::from(&mut out.data)
        .and(self.data.broadcast(IxDyn(&out_shape)).unwrap())
        .and(other.data.broadcast(IxDyn(&out_shape)).unwrap())
        .par_for_each(|r, a, b| {
            *r = a.clone() + b.clone();
        });
        out
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], other)) + self }
}
impl Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], self)) + other }
}
impl Mul<Tensor> for Tensor { 
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        let mut out_shape = Vec::new();
        for i in 0..max(self_shape.len(), other_shape.len()) {
            if i < self_shape.len() && i < other_shape.len() {
                out_shape.push(max(self_shape[i], other_shape[i]));
            } else if i < self_shape.len() {
                out_shape.push(self_shape[i]);
            } else {
                out_shape.push(other_shape[i]);
            }
        }

        let mut out = Tensor::new(ArrayD::zeros(out_shape.clone()));
        Zip::from(&mut out.data)
        .and(self.data.broadcast(IxDyn(&out_shape)).unwrap())
        .and(other.data.broadcast(IxDyn(&out_shape)).unwrap())
        .par_for_each(|r, a, b| {
            *r = a.clone() * b.clone();
        });
        out
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], other)) * self }
}
impl Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], self)) * other }
}
impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = -x.clone(); });
        Tensor { data: out }
    }
}

// more ops: sub, div
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output { self + -other }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { -Tensor::new(ArrayD::from_elem(vec![1], other)) + self }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], self)) + -other }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], other)) * self }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(ArrayD::from_elem(vec![1], self)) * other.pow(-1.0) }
}

impl Tensor {
    pub fn new(data: ArrayD<f64>) -> Tensor { Tensor { data: data.mapv(|x| Value::new(x)), } }

    // base ops: sum, pow, exp, relu, log
    pub fn sum(&self) -> Tensor {
        Tensor { data: ArrayD::from_elem(IxDyn(&[1]), self.data.par_iter().map(|x| x.clone()).sum()) }
    }
    pub fn pow(&self, other: f64) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.pow(other); });
        Tensor { data: out }
    }
    pub fn exp(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.exp(); });
        Tensor { data: out }
    }
    pub fn relu(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.relu(); });
        Tensor { data: out }
    }
    pub fn log(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.log(); });
        Tensor { data: out }
    }

    // more ops: conv2d, max_pool2d, dot, softmax, sigmoid, tanh
    pub fn conv2d(&self, w: &Tensor, s: usize, p: usize) -> Tensor {
        // support 4d tensors only
        let x_shape = self.data.shape();
        let w_shape = w.data.shape();
        assert_eq!(x_shape.len(), 4);
        assert_eq!(w_shape.len(), 4);

        let out_shape = vec![x_shape[0], w_shape[0], (x_shape[2] + 2 * p - w_shape[2]) / s + 1, (x_shape[3] + 2 * p - w_shape[3]) / s + 1];
        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::from_elem(out_shape.clone(), 0.0))));
        (0..out_shape[0]).into_par_iter().for_each(|i| {
            (0..out_shape[1]).into_par_iter().for_each(|j| {
                (0..out_shape[2]).into_par_iter().for_each(|k| {
                    (0..out_shape[3]).into_par_iter().for_each(|l| {
                        let values = Arc::new(RwLock::new(Vec::new()));

                        (0..w_shape[1]).into_par_iter().for_each(|m| {
                            (0..w_shape[2]).into_par_iter().for_each(|n| {
                                (0..w_shape[3]).into_par_iter().for_each(|o| {
                                    let x = (k * s + n) as isize - p as isize;
                                    let y = (l * s + o) as isize - p as isize;
                                    if 0 <= x && x < x_shape[2] as isize && 0 <= y && y < x_shape[3] as isize {
                                        values.write().unwrap().push(self.data[[i, m, x as usize, y as usize]].clone() * w.data[[j, m, n, o]].clone());
                                    }
                                });
                            });
                        });

                        out.write().unwrap().data[[i, j, k, l]] = values.read().unwrap().par_iter().map(|x| x.clone()).sum();
                    });
                });
            });
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn max_pool2d(&self, f: (usize, usize)) -> Tensor {
        assert!(self.data.shape()[2] % f.0 == 0);
        assert!(self.data.shape()[3] % f.1 == 0);

        let out = Arc::new(RwLock::new(Tensor::new(ArrayD::from_elem(vec![
            self.data.shape()[0], 
            self.data.shape()[1], 
            self.data.shape()[2] / f.0, 
            self.data.shape()[3] / f.1], 0.0))));
        
        (0..self.data.shape()[0]).into_par_iter().for_each(|i| {
            (0..self.data.shape()[1]).into_par_iter().for_each(|j| {
                (0..self.data.shape()[2] / f.0).into_par_iter().for_each(|k| {
                    (0..self.data.shape()[3] / f.1).into_par_iter().for_each(|l| {
                        let mut max_val = Value::new(f64::NEG_INFINITY);
                        (0..f.0).for_each(|m| {
                            (0..f.1).for_each(|n| {
                                if max_val.data() < self.data[[i, j, k * f.0 + m, l * f.1 + n]].data() {
                                    max_val = self.data[[i, j, k * f.0 + m, l * f.1 + n]].clone();
                                }
                            });
                        });
                        out.write().unwrap().data[[i, j, k, l]] = max_val;
                    });
                });
            });
        });
        Arc::try_unwrap(out).unwrap().into_inner().unwrap()
    }
    pub fn dot(&self, other: &Tensor) -> Tensor {
        // support 1d, 2d tensors only
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();

        if self_shape.len() == 1 && other_shape.len() == 1 {
            assert_eq!(self_shape[0], other_shape[0]);
            let mut out = self.data.clone();
            Zip::from(&mut out)
                .and(&other.data)
                .par_for_each(|x, y| { *x = x.clone() * y.clone(); });
            Tensor { data: ArrayD::from_elem(IxDyn(&[1]), out.into_iter().map(|x| x.clone()).sum()) }
        }
        else if self_shape.len() == 1 && other_shape.len() == 2 {
            assert_eq!(self_shape[0], other_shape[0]);
            let mut out = ArrayD::from_elem(vec![1, other_shape[1]], Value::new(0.0));
            (0..other_shape[1]).for_each(|j| {
                let mut values = Vec::new();
                (0..self_shape[0]).for_each(|k| {
                    values.push(self.data[[k]].clone() * other.data[[k, j]].clone());
                });
                out[[0, j]] = values.par_iter().map(|x| x.clone()).sum();
            });
            Tensor { data: out }
        }
        else if self_shape.len() == 2 && other_shape.len() == 1 {
            assert_eq!(self_shape[1], other_shape[0]);
            let mut out = ArrayD::from_elem(vec![self_shape[0], 1], Value::new(0.0));
            (0..self_shape[0]).for_each(|i| {
                let mut values = Vec::new();
                (0..self_shape[1]).for_each(|k| {
                    values.push(self.data[[i, k]].clone() * other.data[[k]].clone());
                });
                out[[i, 0]] = values.par_iter().map(|x| x.clone()).sum();
            });
            Tensor { data: out }
        }
        else if self_shape.len() == 2 && other_shape.len() == 2 {
            assert_eq!(self_shape[1], other_shape[0]);
            let mut out = ArrayD::from_elem(vec![self_shape[0], other_shape[1]], Value::new(0.0));
            (0..self_shape[0]).for_each(|i| {
                (0..other_shape[1]).for_each(|j| {
                    let mut values = Vec::new();
                    (0..self_shape[1]).for_each(|k| {
                        values.push(self.data[[i, k]].clone() * other.data[[k, j]].clone());
                    });
                    out[[i, j]] = values.par_iter().map(|x| x.clone()).sum();
                });
            });
            Tensor { data: out }
        }
        else { panic!("Not supported shape"); }
    }
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.data.shape().len(), 2);
        let mut out = ArrayD::from_elem(self.data.shape(), Value::new(0.0));
        let out_h = out.shape()[0];
        let out_w = out.shape()[1];
        (0..out_h).for_each(|i| {
            let self_slice = self.slice(i..i + 1).exp();
            let sum_val = self_slice.sum().data[0].clone();
            (0..out_w).for_each(|j| {
                out[[i, j]] = self_slice.data[[0, j]].clone() / sum_val.clone();
            });
        });
        Tensor { data:out }
    }
    pub fn sigmoid(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x| { *x = x.sigmoid(); });
        Tensor { data: out }
    }
    pub fn tanh(&self) -> Tensor {
        let mut out = self.data.clone();
        Zip::from(&mut out)
            .par_for_each(|x: &mut Value| { *x = x.tanh(); });
        Tensor { data: out }
    }

    // backward prop
    pub fn backward(&self) {
        self.data.iter().for_each(|i| { i.backward(); });
    }

    // misc
    pub fn slice(&self, s: std::ops::Range<usize>) -> Tensor {
        let self_shape = self.data.shape();
        let mut out_shape = vec![];
        (0..self_shape.len()).for_each(|i| {
            if i == 0 { out_shape.push(s.end - s.start); }
            else { out_shape.push(self_shape[i]); }
        });

        let mut out = ArrayD::from_elem(out_shape.clone(), Value::new(0.0));
        if self_shape.len() == 2 {
            (0..out_shape[0]).for_each(|i| {
                (0..out_shape[1]).for_each(|j| {
                    out[[i, j]] = self.data[[i + s.start, j]].clone();
                });
            });
        }
        else if self_shape.len() == 3 {
            (0..out_shape[0]).for_each(|i| {
                (0..out_shape[1]).for_each(|j| {
                    (0..out_shape[2]).for_each(|k| {
                        out[[i, j, k]] = self.data[[i + s.start, j, k]].clone();
                    });
                });
            });
        }
        else if self_shape.len() == 4 {
            (0..out_shape[0]).for_each(|i| {
                (0..out_shape[1]).for_each(|j| {
                    (0..out_shape[2]).for_each(|k| {
                        (0..out_shape[3]).for_each(|l| {
                            out[[i, j, k, l]] = self.data[[i + s.start, j, k, l]].clone();
                        });
                    });
                });
            });
        }
        else { panic!("Not supported shape {:?}", self_shape); }

        Tensor { data: out }
    }
    pub fn requires_grad(&self, requires_grad: bool) {
        self.data.par_iter().for_each(|i| { i.0.write().unwrap().requires_grad = requires_grad; });
    }
    pub fn mean(&self) -> Tensor {
        assert_eq!(self.data.shape().len(), 4);
        let self_size = self.data.shape()[0] * self.data.shape()[2] * self.data.shape()[3];
        let mut out = ArrayD::from_elem(vec![1, self.data.shape()[1], 1, 1], Value::new(0.0));
        (0..self.data.shape()[1]).for_each(|i| {
            let mut values = Vec::new();
            (0..self.data.shape()[0]).for_each(|j| {
                (0..self.data.shape()[2]).for_each(|k| {
                    (0..self.data.shape()[3]).for_each(|l| {
                        values.push(self.data[[j, i, k, l]].clone());
                    });
                });
            });
            out[[0, i, 0, 0]] = values.par_iter().map(|x| x.clone()).sum::<Value>() / self_size as f64;
        });
        Tensor { data: out }
    }
    pub fn var(&self) -> Tensor {
        assert_eq!(self.data.shape().len(), 4);
        let self_size = self.data.shape()[0] * self.data.shape()[2] * self.data.shape()[3];
        let mut out = ArrayD::from_elem(vec![1, self.data.shape()[1], 1, 1], Value::new(0.0));
        (0..self.data.shape()[1]).for_each(|i| {
            let mut values = Vec::new();
            (0..self.data.shape()[0]).for_each(|j| {
                (0..self.data.shape()[2]).for_each(|k| {
                    (0..self.data.shape()[3]).for_each(|l| {
                        values.push(self.data[[j, i, k, l]].clone());
                    });
                });
            });
            let mean = values.par_iter().map(|x| x.clone()).sum::<Value>() / self_size as f64;
            out[[0, i, 0, 0]] = values.par_iter().map(|x| (x.clone() - mean.clone()).pow(2.0)).sum::<Value>() / self_size as f64;
        });
        Tensor { data: out }
    }
    pub fn data(&self) -> ArrayD<f64> { self.data.mapv(|x| x.0.read().unwrap().data) }
    pub fn grad(&self) -> ArrayD<f64> { self.data.mapv(|x| x.0.read().unwrap().grad) }
}