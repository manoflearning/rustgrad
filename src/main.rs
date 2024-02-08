mod engine;
use engine::Value;

mod nn;
use nn::MLP;

fn main() {
    // test forward and backward

    let x: Value = Value::new(-4.0);
    let z: Value = 2.0 * x.clone() + 2.0 + x.clone();
    let q: Value = z.relu() + z.clone() * x.clone();
    let h: Value = (z.clone() * z.clone()).relu();
    let y: Value = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    println!("Value(data={}, grad={}) ", y.data(), y.grad());
    println!("Value(data={}, grad={}) ", x.data(), x.grad());

    // test MLP

    // let xs: Vec<Vec<Value>> = vec![
    //     vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
    //     vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
    //     vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
    //     vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    // ];
    // let ys: Vec<Value> = vec![Value::new(1.0), Value::new(0.0), Value::new(0.0), Value::new(1.0)];

    // let mlp: MLP = MLP::new(3, &vec![4, 4, 1]);
    // for i in 0..1000 {
    //     let mut ypred: Vec<Value> = Vec::new();
    //     for j in 0..xs.len() {
    //         ypred.push(mlp.forward(&xs[j])[0].clone());
    //     }

    //     let loss: Value = ypred.iter().zip(ys.iter()).map(|(yhat, y)| (yhat.clone() - y.clone()).pow(2.0)).sum();

    //     mlp.zerograd();
    //     loss.backward();

    //     for param in mlp.parameters().iter() {
    //         param.0.borrow_mut().data -= 0.01 * param.grad();
    //     }
        
    //     if i % 100 == 0 {
    //         println!("Epoch: {}, Loss: {}", i, loss.data());
    //         println!("Predictions: {:?}", ypred.iter().map(|v| v.data()).collect::<Vec<f64>>());
    //     }
    // }
}