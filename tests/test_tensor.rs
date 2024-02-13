extern crate rustgrad;
use rustgrad::Tensor;

#[test]
fn test_tensor() {
    let x = Tensor::new(vec![vec![-4.0]], vec![1, 1]);
    let z = 2.0 * x.clone() + 2.0 + x.clone();
    let q = z.relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(x.data(), vec![vec![-4.0]]);
    assert_eq!(x.grad(), vec![vec![46.0]]);
    assert_eq!(y.data(), vec![vec![-20.0]]);
    assert_eq!(y.grad(), vec![vec![1.0]]);
}

#[test]
fn test_tensor_mul() {
    let a = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![2, 2]);
    let b = Tensor::new(vec![vec![5.0]], vec![1, 1]);
    let c = a.clone() * b.clone();
    c.backward();

    assert_eq!(c.data(), vec![vec![5.0, 10.0], vec![15.0, 20.0]]);
    assert_eq!(c.grad(), vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    assert_eq!(a.grad(), vec![vec![5.0, 5.0], vec![5.0, 5.0]]);
    assert_eq!(b.grad(), vec![vec![10.0]]);
}

#[test]
fn test_tensor_dot() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let x = Tensor::new(vec![vec![1.0, 2.0, 3.0, 4.0]], vec![1, 4]);
    let y = Tensor::new(vec![vec![5.0], vec![6.0], vec![7.0], vec![8.0]], vec![4, 1]);
    let z = x.dot(&y);
    z.backward();

    assert_eq!(z.data(), vec![vec![70.0]]);
    assert_eq!(z.grad(), vec![vec![1.0]]);
    assert_eq!(x.grad(), vec![vec![5.0, 6.0, 7.0, 8.0]]);
    assert_eq!(y.grad(), vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]);

    let p = Tensor::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]], vec![4, 1]);
    let q = Tensor::new(vec![vec![5.0], vec![6.0], vec![7.0], vec![8.0]], vec![4, 1]);
    let r = p.transpose().dot(&q);
    r.backward();

    assert_eq!(r.data(), vec![vec![70.0]]);
    assert_eq!(r.grad(), vec![vec![1.0]]);
    assert_eq!(p.grad(), vec![vec![5.0], vec![6.0], vec![7.0], vec![8.0]]);
    assert_eq!(q.grad(), vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]);
}