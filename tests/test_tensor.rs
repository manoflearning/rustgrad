extern crate rustgrad;
use rustgrad::Tensor;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_tensor() {
    let x = Tensor::new(ArrayD::from_elem(IxDyn(&[1, 1]), -4.0));
    let z = 2.0 * x.clone() + 2.0 + x.clone();
    let q = z.relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(x.data(), ArrayD::from_elem(IxDyn(&[1, 1]), -4.0));
    assert_eq!(x.grad(), ArrayD::from_elem(IxDyn(&[1, 1]), 46.0));
    assert_eq!(y.data(), ArrayD::from_elem(IxDyn(&[1, 1]), -20.0));
    assert_eq!(y.grad(), ArrayD::from_elem(IxDyn(&[1, 1]), 1.0));
}

#[test]
fn test_tensor_mul() {
    let a = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());
    let b = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![5.0])
        .unwrap());
    let c = a.clone() * b.clone();
    c.backward();

    assert_eq!(c.data(), ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![5.0, 10.0, 15.0, 20.0])
        .unwrap());
    assert_eq!(c.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![1.0, 1.0, 1.0, 1.0])
        .unwrap());
    assert_eq!(a.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![5.0, 5.0, 5.0, 5.0])
        .unwrap());
    assert_eq!(b.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![10.0])
        .unwrap());
}

#[test]
fn test_tensor_dot() {
    let x = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());
    let y = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap());
    let z = x.dot(&y);
    z.backward();

    assert_eq!(z.data(), ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![70.0])
        .unwrap());
    assert_eq!(z.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![1.0])
        .unwrap());
    assert_eq!(x.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 4]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap());
    assert_eq!(y.grad(), ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());

    let p = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());
    let q = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap());
    let r = p.transpose().dot(&q);
    r.backward();

    assert_eq!(r.data(), ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![70.0])
        .unwrap());
    assert_eq!(r.grad(), ArrayD::from_shape_vec(
        IxDyn(&[1, 1]), vec![1.0])
        .unwrap());
    assert_eq!(p.grad(), ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![5.0, 6.0, 7.0, 8.0])
        .unwrap());
    assert_eq!(q.grad(), ArrayD::from_shape_vec(
        IxDyn(&[4, 1]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());
}

#[test]
fn test_tensor_softmax() {
    let x = Tensor::new(ArrayD::from_shape_vec(
        IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0])
        .unwrap());
    let y = x.softmax();
    y.backward();

    assert_eq!(y.data(), ArrayD::from_shape_vec(
        IxDyn(&[2, 2]), vec![0.11920293, 0.119202934, 0.88079715, 0.8807971])
        .unwrap());
    assert_eq!(y.grad(), ArrayD::from_shape_vec(
        IxDyn(&[2, 2]), vec![1.0, 1.0, 1.0, 1.0])
        .unwrap())
}