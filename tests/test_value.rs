extern crate rustgrad;
use rustgrad::Value;

#[test]
fn test_value_sanity_check() {
    let x: Value = Value::new(-4.0);
    let z: Value = 2.0 * x.clone() + 2.0 + x.clone();
    let q: Value = z.relu() + z.clone() * x.clone();
    let h: Value = (z.clone() * z.clone()).relu();
    let y: Value = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(x.data(), -4.0);
    assert_eq!(x.grad(), 46.0);
    assert_eq!(y.data(), -20.0);
    assert_eq!(y.grad(), 1.0);
}