use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;

pub struct RawValue {
    pub id: usize,
    pub data: f64,
    pub grad: f64,
    pub children: Vec<(Value, f64)>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<RawValue>>);

pub static COUNTER: AtomicUsize = AtomicUsize::new(1);

impl Value {
    pub fn new(data: f64) ->Self {
        Value(Rc::new(RefCell::new(RawValue {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            data,
            grad: 0.0,
            children: Vec::new(),
        })))
    }

    pub fn id(&self) -> usize { self.0.borrow().id }
    pub fn data(&self) -> f64 { self.0.borrow().data }
    pub fn grad(&self) -> f64 { self.0.borrow().grad }

    // pub fn clone(&self) -> Value { Value(Rc::clone(&self.0)) }

    pub fn _backward(&self) {
        for (child, x) in self.0.borrow().children.iter() {
            child.0.borrow_mut().grad += x * self.0.borrow().grad;
        }
    }

    pub fn backward(&self) {
        self.0.borrow_mut().grad = 1.0;
            
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(v: Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            visited.insert(v.id());
            for (child, _) in v.0.borrow().children.iter() {
                if !visited.contains(&child.id()) {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v.clone());
        }
        build_topo(self.clone(), &mut topo, &mut visited);
            
        topo.reverse();

        for v in topo.iter() {
            v._backward();
        }
    }

    pub fn pow(&self, other: f64) -> Value {
        let out = Value::new(self.data().powf(other));
        out.0.borrow_mut().children.push((self.clone(), other * self.data().powf(other - 1.0)));
        out
    }

    pub fn relu(&self) -> Value {
        let out = Value::new(self.data().max(0.0));
        if self.data() > 0.0 {
            out.0.borrow_mut().children.push((self.clone(), 1.0))
        } else {
            out.0.borrow_mut().children.push((self.clone(), 0.0))
        }
        out
    }
}

use std::iter::Sum;
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(iter: I) -> Self {
        let out: Value = iter.fold(Value::new(0.0), |acc, value| acc + value);
        out
    }
}

// operator overloading
use std::ops::{Add, Sub, Mul, Div, Neg};

impl Add<Value> for Value {
    type Output = Value;
    fn add(self, other: Self) -> Self::Output {
        let out = Value::new(self.data() + other.data());
        out.0.borrow_mut().children.push((self.clone(), 1.0));
        out.0.borrow_mut().children.push((other.clone(), 1.0));
        out
    }
}

impl Sub<Value> for Value {
    type Output = Value;
    fn sub(self, other: Self) -> Self::Output {
        self + -other
    }
}

impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self, other: Self) -> Self::Output {
        let out = Value::new(self.data() * other.data());
        out.0.borrow_mut().children.push((self.clone(), other.data()));
        out.0.borrow_mut().children.push((other.clone(), self.data()));
        out
    }
}

impl Div<Value> for Value {
    type Output = Value;
    fn div(self, other: Self) -> Self::Output {
        self * other.pow(-1.0)
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Self::Output {
        let out = Value::new(-self.data());
        out.0.borrow_mut().children.push((self.clone(), -1.0));
        out
    }
}