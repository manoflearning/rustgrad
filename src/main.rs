mod engine;
use engine::Value;

fn main() {
    // test forward and backward

    let x: Value = Value::new(-4.0);
    let z: Value = Value::new(2.0) * x.clone() + Value::new(2.0) + x.clone();
    let q: Value = z.relu() + z.clone() * x.clone();
    let h: Value = (z.clone() * z.clone()).relu();
    let y: Value = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    println!("Value(data={}, grad={}) ", y.data(), y.grad());
    println!("Value(data={}, grad={}) ", x.data(), x.grad());
}