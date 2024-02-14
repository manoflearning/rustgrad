// TODO: delete value module and implement operations in tensor module

// base operator (defined as itself):
// add, mul, neg, pow, exp, relu, log
// more operator (defined as a combination of base operators):
// sub, div, sigmoid, tanh

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
pub struct RawValue {
    pub id: usize,
    pub data: f64,
    pub grad: f64,
    pub children: Vec<(Value, f64)>,
}

#[derive(Clone, Debug)]
pub struct Value(pub Arc<RwLock<RawValue>>);

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

    pub fn id(&self) -> usize { self.0.read().unwrap().id }
    pub fn data(&self) -> f64 { self.0.read().unwrap().data }
    pub fn grad(&self) -> f64 { self.0.read().unwrap().grad }

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
                    build_topo(Value(Arc::clone(&child.0)), topo, visited);
                }
            }
            topo.push(Value(Arc::clone(&v.0)));
        }
        build_topo(Value(Arc::clone(&self.0)), &mut topo, &mut visited);

        topo.reverse();
        for v in topo.iter() { v._backward(); }
    }

    pub fn pow(&self, other: f64) -> Value {
        let out = Value::new(self.data().powf(other));
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), other * self.data().powf(other - 1.0)));
        out
    }

    pub fn exp(&self) -> Value {
        let out = Value::new(self.data().exp());
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), self.data().exp()));
        out
    }
    
    pub fn relu(&self) -> Value {
        let out = Value::new(self.data().max(0.0));
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), if self.data() > 0.0 { 1.0 } else { 0.0 }));
        out
    }

    pub fn sigmoid(&self) -> Value {
        1.0 / (1.0 + -Value(Arc::clone(&self.0)).exp())
    }

    pub fn tanh(&self) -> Value {
        ((2.0 * Value(Arc::clone(&self.0))).exp() - 1.0) / ((2.0 * Value(Arc::clone(&self.0))).exp() + 1.0)
    }

    pub fn log(&self) -> Value {
        let out = Value::new(self.data().ln());
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), 1.0 / self.data()));
        out
    }
}

use std::iter::Sum;
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        iter.fold(Value::new(0.0), |acc, value| acc + value)
    }
}

impl Add<Value> for Value {
    type Output = Value;
    fn add(self, other: Value) -> Self::Output {
        let out = Value::new(self.data() + other.data());
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), 1.0));
        out.0.write().unwrap().children.push((Value(Arc::clone(&other.0)), 1.0));
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

impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self: Value, other: Value) -> Self::Output {
        let out = Value::new(self.data() * other.data());
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), other.data()));
        out.0.write().unwrap().children.push((Value(Arc::clone(&other.0)), self.data()));
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

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        let out = Value::new(-self.data());
        out.0.write().unwrap().children.push((Value(Arc::clone(&self.0)), -1.0));
        out
    }
}