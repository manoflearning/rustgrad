extern crate rustgrad;
use rustgrad::Tensor;

#[test]
fn test_tensor_sanity_check() {
    let x: Tensor = Tensor::new(vec![-4.0]);
    let z: Tensor = 2.0 * x.clone() + 2.0 + x.clone();
    let q: Tensor = z.relu() + z.clone() * x.clone();
    let h: Tensor = (z.clone() * z.clone()).relu();
    let y: Tensor = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(x.data(), vec![-4.0]);
    assert_eq!(x.grad(), vec![46.0]);
    assert_eq!(y.data(), vec![-20.0]);
    assert_eq!(y.grad(), vec![1.0]);
}