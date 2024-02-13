use crate::value::Value;

// TODO: support 1D, 3D, 4D ... tensors

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<Value>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn hash(&self, coor: &Vec<usize>) -> usize {
        assert!(coor.len() == self.shape.len());
        let mut out: usize = 0;
        for i in 0..coor.len() {
            if coor[i] >= self.shape[i] {
                println!("coor: {:?}", coor);
                println!("shape: {:?}", self.shape);
                panic!("coordinate out of range");
            }
            out = out * self.shape[i] + coor[i];
        }
        if out >= self.data.len() {
            println!("coor: {:?}", coor);
            println!("shape: {:?}", self.shape);
            println!("out: {}", out);
            println!("data.len(): {}", self.data.len());
            panic!("coordinate out of range");
        }
        out
    }
    pub fn get(&self, x: usize, y: usize) -> Value { self.data[self.hash(&vec![x, y])].clone() }
    pub fn set(&mut self, x: usize, y: usize, value: Value) {
        let hash_num = self.hash(&vec![x, y]);
        self.data[hash_num] = value;
    }

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {        
        let mut out = Tensor {
            data: Vec::new(),
            shape: shape.clone(),
        };
        
        out.data.resize(data.len(), Value::new(0.0));

        if shape.len() == 2 {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let hash_num = out.hash(&vec![i, j]);
                    out.set(i, j, Value::new(data[hash_num]));
                }
            }
        } else { panic!("shape length not supported"); }

        out
    }

    pub fn data(&self) -> Vec<f64> { self.data.iter().map(|x| x.0.read().unwrap().data).collect() }
    pub fn grad(&self) -> Vec<f64> { self.data.iter().map(|x| x.0.read().unwrap().grad).collect() }

    pub fn backward(&self) {
        for i in self.data.iter() { i.backward(); }
    }

    pub fn log(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.log()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn pow(&self, other: f64) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.pow(other)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn exp(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.exp()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn relu(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.relu()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.sigmoid()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x.tanh()).collect(),
            shape: self.shape.clone(),
        }
    }

    // TODO: support 1D, 3D, 4D ... tensors
    pub fn dot(&self, other: &Tensor) -> Tensor {
        // println!("({}, {})", other.shape[0], other.shape[1]);
        // println!("({}, {})", self.shape[0], self.shape[1]);

        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);
        let mut out = Tensor::new(vec![0.0; self.shape[0] * other.shape[1]], vec![self.shape[0], other.shape[1]]);
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                out.set(i, j, (0..self.shape[1]).map(|k| self.get(i, k) * other.get(k, j)).sum());
            }
        }
        return out
    }

    // TODO: support 1D, 3D, 4D ... tensors
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let mut out = Tensor::new(vec![0.0; self.data.len()], vec![self.shape[1], self.shape[0]]);
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    // TODO: support 1D, 3D, 4D ... tensors
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        let mut out: Tensor = Tensor::new(vec![0.0; self.data.len()], self.shape.clone());
        for c in 0..self.shape[1] {
            let mut tmp: Vec<Value> = Vec::new();
            for r in 0..self.shape[0] {
                tmp.push(self.get(r, c).exp());
            }
            
            let sum: Value = tmp.iter().map(|x| x.clone()).sum();
            
            for r in 0..self.shape[0] {
                out.set(r, c, tmp[r].clone() / sum.clone());
            }

            let test_sum: f64 = tmp.iter().map(|x| x.clone()).sum::<Value>().data() / sum.data();
            if !(0.999 < test_sum && test_sum < 1.001) {
                println!("out: {:?}", tmp.iter().map(|x| x.data()).collect::<Vec<f64>>());
                println!("sum: {}", test_sum);
                panic!("sum is not 1.0");
            }
        }
        out
    }

    // TODO: support 1D, 3D, 4D ... tensors
    pub fn slice(&self, s: std::ops::Range<usize>) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert!(s.start < s.end);
        assert!(s.end <= self.shape[1]);
        let mut out = Tensor::new(vec![0.0; (s.end - s.start) * self.shape[0]], vec![self.shape[0], s.end - s.start]);
        for i in 0..self.shape[0] {
            for j in s.start..s.end {
                out.set(i, j - s.start, self.get(i, j));
            }
        }
        out
    }
}

// TODO: fix broadcasting bug
use std::cmp::max;
use std::ops::{Add, Div, Mul, Neg, Sub};
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len]);
        (0..row_len).for_each(|i| (0..col_len).for_each(|j| {
            out.set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            + other.get(i % other.shape[0], j % other.shape[1]));
        }));
        out
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

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len]);
        (0..row_len).for_each(|i| (0..col_len).for_each(|j| {
            out.set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            - other.get(i % other.shape[0], j % other.shape[1]));
        }));
        out
    }
}
impl Sub<f64> for Tensor {
    type Output = Tensor;
    fn sub(self: Tensor, other: f64) -> Self::Output { self + -Tensor::new(vec![other], vec![1, 1]) }
}
impl Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) + -other }
}

impl Mul<Tensor> for Tensor { 
    type Output = Tensor;
    fn mul(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len]);
        (0..row_len).for_each(|i| (0..col_len).for_each(|j| {
            out.set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            * other.get(i % other.shape[0], j % other.shape[1]));
        }));
        out
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

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: Tensor) -> Self::Output {
        let row_len = max(self.shape[0], other.shape[0]);
        let col_len = max(self.shape[1], other.shape[1]);
        let mut out = Tensor::new(vec![0.0; row_len * col_len], vec![row_len, col_len]);
        (0..row_len).for_each(|i| (0..col_len).for_each(|j| {
            out.set(i, j, self.get(i % self.shape[0], j % self.shape[1]) 
            / other.get(i % other.shape[0], j % other.shape[1]));
        }));
        out
    }
}
impl Div<f64> for Tensor {
    type Output = Tensor;
    fn div(self: Tensor, other: f64) -> Self::Output { self * Tensor::new(vec![other], vec![1, 1]).pow(-1.0) }
}
impl Div<Tensor> for f64 {
    type Output = Tensor; 
    fn div(self: f64, other: Tensor) -> Self::Output { Tensor::new(vec![self], vec![1, 1]) * other.pow(-1.0) }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        Tensor {
            data: self.data.iter().map(|x| -x.clone()).collect(),
            shape: self.shape.clone(),
        }
    }
}