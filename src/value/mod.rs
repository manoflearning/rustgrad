// base ops (defined as itself):
// add, mul, neg, pow, exp, relu, log
// more ops (defined as a combination of base ops):
// sub, div, sum, sigmoid, tanh

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::{Arc, RwLock};
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct RawValue {
    pub id: usize,
    pub data: f32,
    pub grad: f32,
    pub children: Vec<(Value, f32)>,
    pub requires_grad: bool,
}

#[derive(Clone, Debug)]
pub struct Value(pub Arc<RwLock<RawValue>>);

// base ops: add, mul, neg
impl Add<Value> for Value {
    type Output = Value;
    fn add(self, other: Value) -> Self::Output {
        let out = Value::new(self.data() + other.data());
        out.0.write().unwrap().children.push((self.clone(), 1.0));
        out.0.write().unwrap().children.push((other.clone(), 1.0));
        out
    }
}
impl Add<f32> for Value {
    type Output = Value;
    fn add(self, other: f32) -> Self::Output { self + Value::new(other) }
}
impl Add<Value> for f32 {
    type Output = Value;
    fn add(self: f32, other: Value) -> Self::Output { Value::new(self) + other }
}
impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self: Value, other: Value) -> Self::Output {
        let out = Value::new(self.data() * other.data());
        out.0.write().unwrap().children.push((self.clone(), other.data()));
        out.0.write().unwrap().children.push((other.clone(), self.data()));
        out
    }
}
impl Mul<f32> for Value {
    type Output = Value;
    fn mul(self: Value, other: f32) -> Self::Output { self * Value::new(other) }
}
impl Mul<Value> for f32 {
    type Output = Value;
    fn mul(self: f32, other: Value) -> Self::Output { Value::new(self) * other }
}
impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        let out = Value::new(-self.data());
        out.0.write().unwrap().children.push((self.clone(), -1.0));
        out
    }
}

// more ops: sub, div, sum
impl Sub<Value> for Value {
    type Output = Value;
    fn sub(self: Value, other: Value) -> Self::Output { self + -other }
}
impl Sub<f32> for Value {
    type Output = Value;
    fn sub(self: Value, other: f32) -> Self::Output { self + -Value::new(other) }
}
impl Sub<Value> for f32 {
    type Output = Value;
    fn sub(self: f32, other: Value) -> Self::Output { Value::new(self) + -other }
}
impl Div<Value> for Value {
    type Output = Value;
    fn div(self: Value, other: Value) -> Self::Output { self * other.pow(-1.0) }
}
impl Div<f32> for Value {
    type Output = Value;
    fn div(self: Value, other: f32) -> Self::Output { self * Value::new(other).pow(-1.0) }
}
impl Div<Value> for f32 {
    type Output = Value; 
    fn div(self: f32, other: Value) -> Self::Output { Value::new(self) * other.pow(-1.0) }
}
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, value| acc + value)
    }
}

pub static COUNTER: AtomicUsize = AtomicUsize::new(1);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Arc::new(RwLock::new(RawValue {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            data,
            grad: 0.0,
            children: Vec::new(),
            requires_grad: true,
        })))
    }

    // base ops: pow, exp, relu, log
    pub fn pow(&self, other: f32) -> Value {
        let out = Value::new(self.data().powf(other));
        out.0.write().unwrap().children.push((self.clone(), other * self.data().powf(other - 1.0)));
        out
    }
    pub fn exp(&self) -> Value {
        let out = Value::new(self.data().exp());
        out.0.write().unwrap().children.push((self.clone(), self.data().exp()));
        out
    }
    pub fn relu(&self) -> Value {
        let out = Value::new(self.data().max(0.0));
        out.0.write().unwrap().children.push((self.clone(), if self.data() > 0.0 { 1.0 } else { 0.0 }));
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
        self.0.write().unwrap().grad = 1.0;
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        fn build_topo(v: Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            visited.insert(v.id());
            for (child, _) in v.0.read().unwrap().children.iter() {
                if !visited.contains(&child.id()) 
                && child.0.read().unwrap().requires_grad
                && !child.0.read().unwrap().children.is_empty() {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v);
        }
        build_topo(self.clone(), &mut topo, &mut visited);
        for v in topo.iter().rev() { v._backward(); }
    }
    pub fn _backward(&self) {
        let grad = self.0.read().unwrap().grad;
        self.0.read().unwrap().children.iter().for_each(|(child, x)| {
            child.0.write().unwrap().grad += x * grad;
        });
    }

    // misc
    pub fn id(&self) -> usize { self.0.read().unwrap().id }
    pub fn data(&self) -> f32 { self.0.read().unwrap().data }
    pub fn grad(&self) -> f32 { self.0.read().unwrap().grad }
}