// base ops (defined as itself):
// add, mul, neg, sum, pow, exp, relu, log
// more ops (defined as a combination of base ops):
// sub, div, sigmoid, tanh

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::{Arc, RwLock};
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct RawValue {
    pub id: usize,
    pub data: f64,
    pub grad: f64,
    pub children: Vec<(Value, f64)>,
    pub requires_grad: bool,
}

#[derive(Clone, Debug)]
pub struct Value(pub Arc<RwLock<RawValue>>);

// base ops: add, mul, neg, sum
impl Add<Value> for Value {
    type Output = Value;
    fn add(self, other: Value) -> Self::Output {
        let out = Value::new(self.data() + other.data());
        {
            let mut out_write = out.0.write().unwrap();
            out_write.children.push((self, 1.0));
            out_write.children.push((other, 1.0));
        }
        out
    }
}
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, other: f64) -> Self::Output { self + Value::new(other) }
}
impl Add<Value> for f64 {
    type Output = Value;
    fn add(self: f64, other: Value) -> Self::Output { Value::new(self) + other }
}
impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self: Value, other: Value) -> Self::Output {
        let self_data = self.data();
        let other_data = other.data();
        let out = Value::new(self_data * other_data);
        {
            let mut out_write = out.0.write().unwrap();
            // we only need to backprop if the gradient is non-zero
            if other_data != 0.0 { out_write.children.push((self, other_data)); }
            if self_data != 0.0 { out_write.children.push((other, self_data)); }
        }
        out
    }
}
impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self: Value, other: f64) -> Self::Output { self * Value::new(other) }
}
impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self: f64, other: Value) -> Self::Output { Value::new(self) * other }
}
impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        let out = Value::new(-self.data());
        out.0.write().unwrap().children.push((self, -1.0));
        out
    }
}
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        let out = Value::new(0.0);
        {
            let mut out_write = out.0.write().unwrap();
            for v in iter {
                out_write.data += v.data();
                out_write.children.push((v, 1.0));
            }
        }
        out
    }
}

// more ops: sub, div
impl Sub<Value> for Value {
    type Output = Value;
    fn sub(self: Value, other: Value) -> Self::Output { self + -other }
}
impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self: Value, other: f64) -> Self::Output { self + -Value::new(other) }
}
impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self: f64, other: Value) -> Self::Output { Value::new(self) + -other }
}
impl Div<Value> for Value {
    type Output = Value;
    fn div(self: Value, other: Value) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f64> for Value {
    type Output = Value;
    fn div(self: Value, other: f64) -> Self::Output { self * Value::new(other).pow(-1.0) }
}
impl Div<Value> for f64 {
    type Output = Value; 
    fn div(self: f64, other: Value) -> Self::Output { Value::new(self) * other.pow(-1.0) }
}

pub static COUNTER: AtomicUsize = AtomicUsize::new(1);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Arc::new(RwLock::new(RawValue {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            data,
            grad: 0.0,
            children: Vec::new(),
            requires_grad: true,
        })))
    }

    // base ops: pow, exp, relu, log
    pub fn pow(&self, other: f64) -> Value {
        let out = Value::new(self.data().powf(other));
        // we only need to backprop if the input is non-zero
        if other != 0.0 { out.0.write().unwrap().children.push((self.clone(), other * self.data().powf(other - 1.0))); }
        out
    }
    pub fn exp(&self) -> Value {
        let out = Value::new(self.data().exp());
        out.0.write().unwrap().children.push((self.clone(), self.data().exp()));
        out
    }
    pub fn relu(&self) -> Value {
        let out = Value::new(self.data().max(0.0));
        // we only need to backprop if the input is greater than 0
        if self.data() > 0.0 { out.0.write().unwrap().children.push((self.clone(), 1.0)); }
        out
    }
    pub fn log(&self) -> Value {
        let out = Value::new(self.data().ln());
        out.0.write().unwrap().children.push((self.clone(), 1.0 / self.data()));
        out
    }

    // more ops: sigmoid, tanh
    pub fn sigmoid(&self) -> Value {
        1.0 / (1.0 + (-self.clone()).exp())
    }
    pub fn tanh(&self) -> Value {
        ((2.0 * self.clone()).exp() - 1.0) / ((2.0 * self.clone()).exp() + 1.0)
    }

    // backward prop
    pub fn backward(&self) {
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        fn build_topo(v: Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            visited.insert(v.id());
            for (child, _) in v.0.read().unwrap().children.iter() {
                let child_read = child.0.read().unwrap();
                if !visited.contains(&child.id()) 
                && child_read.requires_grad
                && !child_read.children.is_empty() {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v);
        }
        build_topo(self.clone(), &mut topo, &mut visited);

        self.0.write().unwrap().grad = 1.0;
        for v in topo.iter().rev() { v._backward(); }
    }
    pub fn _backward(&self) {
        let self_read = self.0.read().unwrap();
        let grad = self_read.grad;
        for (child, weight) in self_read.children.iter() {
            child.0.write().unwrap().grad += grad * weight;
        }
    }

    // misc
    pub fn id(&self) -> usize { self.0.read().unwrap().id }
    pub fn data(&self) -> f64 { self.0.read().unwrap().data }
    pub fn grad(&self) -> f64 { self.0.read().unwrap().grad }
}