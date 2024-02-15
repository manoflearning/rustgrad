// base ops (defined as itself):
// add, mul, neg, pow, exp, relu, log
// more ops (defined as a combination of base ops):
// sub, div, sum, sigmoid, tanh

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::{Arc, RwLock};
use std::iter::Sum;

#[derive(Clone, Debug)]
pub struct RawValue {
    pub id: usize,
    pub data: f64,
    pub grad: f64,
    pub children: Vec<(Value, f64)>,
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
        let out = Value::new(self.data() * other.data());
        out.0.write().unwrap().children.push((self.clone(), other.data()));
        out.0.write().unwrap().children.push((other.clone(), self.data()));
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
        out.0.write().unwrap().children.push((self.clone(), -1.0));
        out
    }
}

// more ops: sub, div, sum
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
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, value| acc + value)
    }
}

pub static COUNTER: AtomicUsize = AtomicUsize::new(1);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Arc::new(RwLock::new(RawValue {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            data,
            grad: 0.0,
            children: Vec::new(),
        })))
    }

    // base ops: pow, exp, relu, log
    pub fn pow(&self, other: f64) -> Value {
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
    pub fn _backward(&self) {
        // each value has at most 2 children, so do not use par_iter
        self.0.read().unwrap().children.iter().for_each(|(child, x)| {
            child.0.write().unwrap().grad += x * self.0.read().unwrap().grad;
        });
    }
    pub fn backward(&self) { // TODO: optimize
        self.0.write().unwrap().grad = 1.0;
        
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        fn build_topo(v: Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            visited.insert(v.id());
            for (child, _) in v.0.read().unwrap().children.iter() {
                if !visited.contains(&child.id()) {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v.clone());
        }
        build_topo(self.clone(), &mut topo, &mut visited);

        for v in topo.iter().rev() { v._backward(); }
    }

    // misc
    pub fn id(&self) -> usize { self.0.read().unwrap().id }
    pub fn data(&self) -> f64 { self.0.read().unwrap().data }
    pub fn grad(&self) -> f64 { self.0.read().unwrap().grad }
}