extern crate rustgrad;
use rustgrad::Tensor;

#[test]
fn test_tensor() {
    let x = Tensor::new(vec![-4.0]);
    let z = 2.0 * x.clone() + 2.0 + x.clone();
    let q = z.relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(x.data(), vec![-4.0]);
    assert_eq!(x.grad(), vec![46.0]);
    assert_eq!(y.data(), vec![-20.0]);
    assert_eq!(y.grad(), vec![1.0]);
}

#[test]
fn test_tensor_matmul() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0]);
    let y = Tensor::new(vec![5.0, 6.0, 7.0, 8.0]);
    let z = x.dot(y.clone());
    z.backward();

    assert_eq!(z.data(), vec![70.0]);
    assert_eq!(z.grad(), vec![1.0]);
    assert_eq!(x.grad(), vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y.grad(), vec![1.0, 2.0, 3.0, 4.0]);
}