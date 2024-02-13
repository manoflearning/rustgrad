extern crate rustgrad;
use ndarray::prelude::*;
use rustgrad::{Value, Tensor, Layer, ReLU, Softmax, Linear, Model};
use std::time::Instant;

use mnist::*;

fn fetch_mnist() -> (Tensor, Tensor, Tensor, Tensor) {
    // load MNIST dataset (how to use: https://docs.rs/mnist/latest/mnist/)
    // TODO: implement dataloader without any dependencies

    let Mnist {
        trn_img,
        trn_lbl,
        val_img: _,
        val_lbl: _,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_digit()
        .finalize();

    let _x_train = Array3::from_shape_vec((60_000, 28, 28), trn_img)
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.0);
    
    let _x_test = Array3::from_shape_vec((10_000, 28, 28), tst_img)
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.0);
    
    let _y_train = Array2::from_shape_vec((60_000, 1), trn_lbl)
    .expect("Error converting labels to Array2 struct");
    
    let _y_test: ArrayBase<ndarray::OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::from_shape_vec((10_000, 1), tst_lbl)
    .expect("Error converting labels to Array2 struct");

    let mut x_train = Tensor::new(vec![0.0; 784 * 60_000], vec![784, 60_000]);
    let mut x_test = Tensor::new(vec![0.0; 28 * 28 * 10_000], vec![28 * 28, 10_000]);
    let mut y_train = Tensor::new(vec![0.0; 10 * 60_000], vec![10, 60_000]);
    let mut y_test = Tensor::new(vec![0.0; 10 * 10_000], vec![10, 10_000]);

    for i in 0..60_000 {
        for j in 0..784 {
            x_train.set(j, i, Value::new(_x_train[[i, j / 28, j % 28]]));
        }
        y_train.set(_y_train[[i, 0]] as usize, i, Value::new(1.0));
    }
    for i in 0..10_000 {
        for j in 0..784 {
            x_test.set(j, i, Value::new(_x_test[[i, j / 28, j % 28]]));
        }
        y_test.set(_y_test[[i, 0]] as usize, i, Value::new(1.0));
    }

    (x_train, y_train, x_test, y_test)
}

fn cross_entropy_loss(y_pred: &Tensor, y: &Tensor) -> Tensor {
    let mut out = Tensor::new(vec![0.0; y_pred.shape[1]], vec![1, y_pred.shape[1]]);
    for i in 0..y_pred.shape[1] {
        out.set(0, i, 
            -((0..y_pred.shape[0]).map(|k| 
                y.get(k, i) * y_pred.get(k, i).log()
            ).sum::<Value>())
        );
    }
    out
}

fn get_test_acc(y_pred: &Tensor, y_test: &Tensor) -> f64 {
    let mut correct = 0;
    let mut total = 0;

    for i in 0..y_pred.shape[1] {
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for j in 0..y_pred.shape[0] {
            if y_pred.get(j, i).data() > max_val {
                max_val = y_pred.get(j, i).data();
                max_idx = j;
            }
        }
        if y_test.get(max_idx, i).data() == 1.0 {
            correct += 1;
        }
        total += 1;
    }
    correct as f64 / total as f64
}

pub fn main() {
    let total_time_start: Instant = Instant::now();

    let (x_train, y_train, x_test, y_test) = fetch_mnist();
    println!("Data Loading Done");

    let train_size: usize = 60_000;

    // parameters
    let learning_rate: f64 = 0.001;
    let n_epochs: usize = 15;
    let batch_size: usize = 50;

    // initialize model
    let model = Model::new(vec![
        Box::new(Linear::new(784, 128)), Box::new(ReLU),
        Box::new(Linear::new(128, 64)), Box::new(ReLU),
        Box::new(Linear::new(64, 10)), Box::new(Softmax),
    ]);

    // training loop
    for epoch in 0..n_epochs {
        let mut cost_sum: f64 = 0.0;
        let mut time_sum: Vec<f64> = vec![0.0; 4];
        // train using mini-batches
        for i in 0..train_size / batch_size {
            // get the next batch
            let batch_time_start: Instant = Instant::now();

            let start = i * batch_size;
            let end = (i + 1) * batch_size;
            let batch_data: Tensor = x_train.slice(start..end);
            let batch_labels: Tensor = y_train.slice(start..end);
            
            time_sum[0] += (Instant::now() - batch_time_start).as_secs_f64();

            // forward pass
            let forward_time_start: Instant = Instant::now();

            let y_pred: Tensor = model.forward(&batch_data);
            
            for j in 0..y_pred.shape[1] {
                let mut sum: f64 = 0.0;
                for i in 0..y_pred.data.len() {
                    assert!(-0.001 <= y_pred.get(i, j).data() && y_pred.get(i, j).data() < 1.001);
                    sum += y_pred.get(i, j).data();
                }

                if !(0.999 < sum && sum < 1.001) {
                    println!("sum: {}", sum);
                    println!("shape: {:?}", y_pred.shape);
                    
                    for i in 0..y_pred.data.len() {
                        print!("{:.4} ", y_pred.get(i, j).data());
                    }

                    panic!("sum is not 1.0");
                }
            }

            time_sum[1] += (Instant::now() - forward_time_start).as_secs_f64();

            // calculate loss
            let loss_time_start: Instant = Instant::now();

            let loss: Tensor = cross_entropy_loss(&y_pred, &batch_labels);
            (0..loss.shape[0]).for_each(|i| (0..loss.shape[1]).for_each(|j| {
                assert!(loss.get(i, j).data() >= 0.0);
            }));

            let cost: Value = loss.data.iter().map(|x| x.clone()).sum();
            cost_sum += cost.data();

            time_sum[2] += (Instant::now() - loss_time_start).as_secs_f64();

            // backpropagation and weight update
            let backward_time_start: Instant = Instant::now();

            model.zerograd();
            cost.backward();
            model.update_weights(learning_rate);

            time_sum[3] += (Instant::now() - backward_time_start).as_secs_f64();
        }
        cost_sum /= train_size as f64;

        println!("Epoch {:5}: Loss {:5.2} Time: {:5.2}s (batch: {:5.2}s, forward: {:5.2}s, loss: {:5.2}s, backward: {:.4}s)", 
        epoch, cost_sum, time_sum.iter().sum::<f64>(), time_sum[0], time_sum[1], time_sum[2], time_sum[3]);
    }

    // test the trained model
    let accuracy = get_test_acc(&model.forward(&x_test), &y_test);
    println!("Test Accuracy: {:5.2}%", accuracy * 100.0);

    println!("Total Time: {:5.2}s", (Instant::now() - total_time_start).as_secs_f64());
}