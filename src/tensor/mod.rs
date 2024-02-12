use crate::value::Value;

// support 2D tensors only
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<Vec<Value>>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<Vec<f64>>, shape: Vec<usize>) -> Self {
        assert_eq!(shape.len(), 2);
        assert_eq!(data.len(), shape[0]);
        assert_eq!(data[0].len(), shape[1]);
        let mut out = Tensor {
            data: Vec::new(),
            shape: shape.clone(),
        };
        for i in 0..shape[0] {
            let mut tmp = Vec::new();
            for j in 0..shape[1] {
                tmp.push(Value::new(data[i][j]));
            }
            out.data.push(tmp);
        }
        out
    }

    pub fn data(&self) -> Vec<Vec<f64>> { self.data.iter().map(|x| x.iter().map(|y| y.data()).collect()).collect() }
    pub fn grad(&self) -> Vec<Vec<f64>> { self.data.iter().map(|x| x.iter().map(|y| y.grad()).collect()).collect() }

    pub fn backward(&self) {
        for i in self.data.iter() {
            for j in i.iter() { j.backward(); }
        }
    }

    pub fn log(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.log()).collect()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn pow(&self, other: f64) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.pow(other)).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn exp(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.exp()).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn relu(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.relu()).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn sigmoid(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.sigmoid()).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn tanh(&self) -> Tensor {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| y.tanh()).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        // println!("({}, {})", other.shape[0], other.shape[1]);
        // println!("({}, {})", self.shape[0], self.shape[1]);

        assert_eq!(self.shape[1], other.shape[0]);
        let mut out = Tensor::new(vec![vec![0.0; other.shape[1]]; self.shape[0]], vec![self.shape[0], other.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                out.data[i][j] = (0..self.shape[1]).map(
                    |k| self.data[i][k].clone() * other.data[k][j].clone()
                ).sum();
            }
        }
        return out
    }

    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let mut out = Tensor::new(vec![vec![0.0; self.shape[0]]; self.shape[1]], vec![self.shape[1], self.shape[0]]);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                out.data[j][i] = self.data[i][j].clone();
            }
        }
        out
    }

    pub fn softmax(&self) -> Tensor {
        let mut out = Tensor::new(vec![vec![0.0; self.shape[1]]; self.shape[0]], self.shape.clone());
        for c in 0..self.shape[1] {
            let mut tmp = Vec::new();
            for r in 0..self.shape[0] {
                tmp.push(self.data[r][c].exp());
            }
            let sum: Value = tmp.iter().map(|x| x.clone()).sum();
            for r in 0..self.shape[0] {
                out.data[r][c] = tmp[r].clone() / sum.clone();
            }
        }
        out
    }
}

use std::cmp::max;
use std::ops::{Add, Div, Mul, Neg, Sub};
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![vec![0.0; col_len]; row_len], vec![row_len, col_len]);
        for i in 0..row_len {
            for j in 0..col_len {
                out.data[i][j] = self.data[i % self.shape[0]][j % self.shape[1]].clone() + other.data[i % other.shape[0]][j % other.shape[1]].clone();
            }
        }
        out
    }
}
impl Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: f64) -> Self::Output { self + Tensor::new(vec![vec![other]], vec![1, 1]) }
}
impl Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![vec![self]], vec![1, 1]) + other }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![vec![0.0; col_len]; row_len], vec![row_len, col_len]);
        for i in 0..row_len {
            for j in 0..col_len {
                out.data[i][j] = self.data[i % self.shape[0]][j % self.shape[1]].clone() - other.data[i % other.shape[0]][j % other.shape[1]].clone();
            }
        }
        out
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { self + -Tensor::new(vec![vec![other]], vec![1, 1]) }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![vec![self]], vec![1, 1]) + -other }
}

// element-wise multiplication
impl Mul<Tensor> for Tensor { 
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![vec![0.0; col_len]; row_len], vec![row_len, col_len]);
        for i in 0..row_len {
            for j in 0..col_len {
                out.data[i][j] = self.data[i % self.shape[0]][j % self.shape[1]].clone() * other.data[i % other.shape[0]][j % other.shape[1]].clone();
            }
        }
        out
    }
}
impl Mul<f64> for Tensor {
    type Output = Tensor;
    fn mul(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![vec![other]], vec![1, 1]) }
}
impl Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![vec![self]], vec![1, 1]) * other }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![vec![0.0; col_len]; row_len], vec![row_len, col_len]);
        for i in 0..row_len {
            for j in 0..col_len {
                out.data[i][j] = self.data[i % self.shape[0]][j % self.shape[1]].clone() / other.data[i % other.shape[0]][j % other.shape[1]].clone();
            }
        }
        out
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![vec![other]], vec![1, 1]).pow(-1.0) }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![vec![self]], vec![1, 1]) * other.pow(-1.0) }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let out = Tensor {
            data: self.data.iter().map(|x| x.iter().map(|y| -y.clone()).collect()).collect(),
            shape: self.shape.clone(),
        };
        out
    }
}