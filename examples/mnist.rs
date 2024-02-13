extern crate rustgrad;
use ndarray::prelude::*;
use rustgrad::{Value, Tensor, NeuralNetwork};
use std::time::Instant;

use mnist::*;

pub fn cross_entropy_loss(predictions: &Tensor, labels: &Tensor) -> Tensor {
    let mut out = Tensor::new(vec![vec![0.0; predictions.shape[1]]; 1], vec![1, predictions.shape[1]]);
    for i in 0..predictions.shape[1] {
        out.data[0][i] = -((0..predictions.shape[0]).map(|k| 
            labels.data[k][i].clone() * predictions.data[k][i].log()
        ).sum::<Value>());
    }
    out
}

fn count_correct(predictions: &Tensor, labels: &Tensor) -> usize {
    let mut correct = 0;
    for i in 0..predictions.data[0].len() {
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for j in 0..predictions.data.len() {
            if predictions.data[j][i].data() > max_val {
                max_val = predictions.data[j][i].data();
                max_idx = j;
            }
        }
        if labels.data[max_idx][i].data() == 1.0 {
            correct += 1;
        }
    }
    correct
}

pub fn main() {
    // std::env::set_var("RUST_BACKTRACE", "1");

    let total_time_start: Instant = Instant::now();

    // load MNIST dataset (how to use: https://docs.rs/mnist/latest/mnist/)
    // TODO: implement dataloader without any dependencies
    let train_size = 600;
    let test_size = 100;

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

    let _train_data = Array3::from_shape_vec((60_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);
    let _train_labels: Array2<f64> = Array2::from_shape_vec((60_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);
    let _test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);
    let _test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    let mut train_data = Tensor::new(vec![vec![0.0; train_size]; 28 * 28], vec![28 * 28, train_size]);
    let mut train_labels = Tensor::new(vec![vec![0.0; train_size]; 10], vec![10, train_size]);

    for i in 0..train_size {
        for j in 0..28 * 28 {
            train_data.data[j][i] = Value::new(_train_data[[i, j / 28, j % 28]]);
        }
        train_labels.data[_train_labels[[i, 0]] as usize][i] = Value::new(1.0);
    }

    let mut test_data = Tensor::new(vec![vec![0.0; test_size]; 28 * 28], vec![28 * 28, test_size]);
    let mut test_labels = Tensor::new(vec![vec![0.0; test_size]; 10], vec![10, test_size]);

    for i in 0..test_size {
        for j in 0..28 * 28 {
            test_data.data[j][i] = Value::new(_test_data[[i, j / 28, j % 28]]);
        }
        test_labels.data[_test_labels[[i, 0]] as usize][i] = Value::new(1.0);
    }

    println!("Data Loading Done");

    // parameters
    let learning_rate: f64 = 0.001;
    let n_epochs: i32 = 15;
    let batch_size: usize = 5;

    let img_size: usize = 28 * 28;
    let n_classes: usize = 10;

    // initialize the neural network
    let network = NeuralNetwork::new(img_size, vec![128, 64, n_classes]);

    // training loop
    for epoch in 0..n_epochs {
        let mut cost_sum: f64 = 0.0;
        let mut time_sum: Vec<f64> = vec![0.0; 4];
        // train using mini-batches
        for i in 0..train_size / batch_size {
            // get the next batch
            let batch_time_start: Instant = Instant::now();

            let start = i * batch_size;
            let end = (i + 1) * batch_size; // [start, end)
            let batch_data: Tensor = train_data.slice(start..end);
            let batch_labels: Tensor = train_labels.slice(start..end);
            
            time_sum[0] += (Instant::now() - batch_time_start).as_secs_f64();

            // forward pass
            let forward_time_start: Instant = Instant::now();

            let predictions: Tensor = network.forward(&batch_data);
            
            for j in 0..predictions.data[0].len() {
                let mut sum: f64 = 0.0;
                for i in 0..predictions.data.len() {
                    assert!(-0.001 <= predictions.data[i][j].data() && predictions.data[i][j].data() < 1.001);
                    sum += predictions.data[i][j].data();
                }

                if !(0.999 < sum && sum < 1.001) {
                    println!("sum: {}", sum);
                    println!("shape: {:?}", predictions.shape);
                    
                    for i in 0..predictions.data.len() {
                        print!("{:.4} ", predictions.data[i][j].data());
                    }

                    assert!(0.999 < sum && sum < 1.001);
                }
            }

            time_sum[1] += (Instant::now() - forward_time_start).as_secs_f64();

            // calculate loss
            let loss_time_start: Instant = Instant::now();

            let loss: Tensor = cross_entropy_loss(&predictions, &batch_labels);
            (0..loss.data.len()).for_each(|i| (0..loss.data[i].len()).for_each(|j| {
                assert!(loss.data[i][j].data() >= 0.0);
            }));

            let cost: Value = loss.data.iter().map(|x| x.iter().map(|y| y.clone()).sum()).sum::<Value>();
            cost_sum += cost.data();

            time_sum[2] += (Instant::now() - loss_time_start).as_secs_f64();

            // backpropagation and weight update
            let backward_time_start: Instant = Instant::now();

            network.zerograd();
            cost.backward(); // or loss.backward() ??
            network.update_weights(learning_rate);

            time_sum[3] += (Instant::now() - backward_time_start).as_secs_f64();
        }
        cost_sum /= train_size as f64;

        println!("Epoch {}: Loss {:.4} Time: {:.4}s (batch: {:.4}s, forward: {:.4}s, loss: {:.4}s, backward: {:.4}s)", epoch, cost_sum, time_sum.iter().sum::<f64>(), time_sum[0], time_sum[1], time_sum[2], time_sum[3]);
    }

    // test the trained model
    let mut correct = 0;
    let mut total = 0;

    for i in 0..test_size / batch_size {
        let start = i * batch_size;
        let end = (i + 1) * batch_size;
        let batch_data: Tensor = test_data.slice(start..end);
        let batch_labels: Tensor = test_labels.slice(start..end);
        let test_predictions = network.forward(&batch_data);
        correct += count_correct(&test_predictions, &batch_labels);
        total += batch_size;
    }
    let accuracy = correct as f64 / total as f64;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    println!("Total Time: {:.4}s", (Instant::now() - total_time_start).as_secs_f64());
}