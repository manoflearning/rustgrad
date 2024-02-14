extern crate rustgrad;
use ndarray::prelude::*;
use rustgrad::*;
use std::{cmp::min, time::Instant};
use mnist::*;

fn fetch_mnist(train_size: usize, test_size: usize) -> (Tensor, Tensor, Tensor, Tensor) { // TODO: optimize
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
            .training_set_length(train_size as u32)
            .validation_set_length(0)
            .test_set_length(test_size as u32)
            .finalize();

    let _x_train = Array3::from_shape_vec((train_size, 28, 28), trn_img)
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.0);
    
    let _x_test = Array3::from_shape_vec((test_size, 28, 28), tst_img)
    .expect("Error converting images to Array3 struct")
    .map(|x| *x as f64 / 256.0);
    
    let _y_train = Array2::from_shape_vec((train_size, 1), trn_lbl)
    .expect("Error converting labels to Array2 struct");
    
    let _y_test: ArrayBase<ndarray::OwnedRepr<u8>, Dim<[usize; 2]>> = Array2::from_shape_vec((test_size, 1), tst_lbl)
    .expect("Error converting labels to Array2 struct");

    let mut x_train = Tensor::new(vec![0.0; 784 * train_size], vec![784, train_size]);
    let mut x_test = Tensor::new(vec![0.0; 784 * test_size], vec![784, test_size]);
    let mut y_train = Tensor::new(vec![0.0; 10 * train_size], vec![10, train_size]);
    let mut y_test = Tensor::new(vec![0.0; 10 * test_size], vec![10, test_size]);

    (0..train_size).for_each(|i| {
        (0..784).for_each(|j| {
            let hash_num = x_train.hash(&[j, i]);
            x_train.data[hash_num] = Value::new(_x_train[[i, j / 28, j % 28]]);
        });
        let hash_num = y_train.hash(&[_y_train[[i, 0]] as usize, i]);
        y_train.data[hash_num] = Value::new(1.0);
    });
    (0..test_size).for_each(|i| {
        (0..784).for_each(|j| {
            let hash_num = x_test.hash(&[j, i]);
            x_test.data[hash_num] = Value::new(_x_test[[i, j / 28, j % 28]]);
        });
        let hash_num = y_test.hash(&[_y_test[[i, 0]] as usize, i]);
        y_test.data[hash_num] = Value::new(1.0);
    });

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
        if y_test.get(max_idx, i).data() == 1.0 as f64 {
            correct += 1;
        }
        total += 1;
    }
    correct as f64 / total as f64
}

pub fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let total_time_start: Instant = Instant::now();

    // load MNIST dataset
    let data_loading_time_start: Instant = Instant::now();

    let train_size: usize = 60_000;
    let test_size: usize = 10_000;
    let (x_train, y_train, x_test, y_test) = fetch_mnist(train_size, test_size);
    
    println!("Data Loading Done \t\t| Time: {:7.2}s", (Instant::now() - data_loading_time_start).as_secs_f64());

    // parameters
    let learning_rate: f64 = 0.001;
    let n_epochs: usize = 15;
    let batch_size: usize = 128;

    // initialize model
    let model_time_start: Instant = Instant::now();
    
    let model = Model::new(vec![
        Box::new(Linear::new(784, 128)), Box::new(ReLU),
        Box::new(Linear::new(128, 64)), Box::new(ReLU),
        Box::new(Linear::new(64, 10)), Box::new(Softmax),
    ]);
    
    println!("Model Initialization Done \t| Time: {:7.2}s", (Instant::now() - model_time_start).as_secs_f64());

    // training loop
    let trainig_time_start: Instant = Instant::now();

    for epoch in 0..n_epochs {
        let mut time_sum: Vec<f64> = vec![0.0; 4];

        let mut cost_sum: f64 = 0.0;
        // train using mini-batches
        for i in 0..(train_size + batch_size - 1) / batch_size {
            // get the next batch
            let batch_time_start: Instant = Instant::now();

            let (start, end) = (i * batch_size, min((i + 1) * batch_size, train_size));
            let batch_data: Tensor = x_train.slice(start..end);
            let batch_labels: Tensor = y_train.slice(start..end);
            
            time_sum[0] += (Instant::now() - batch_time_start).as_secs_f64();

            // forward pass
            let forward_time_start: Instant = Instant::now();
            
            let y_pred: Tensor = model.forward(&batch_data);

            time_sum[1] += (Instant::now() - forward_time_start).as_secs_f64();

            // calculate loss
            let loss_time_start: Instant = Instant::now();

            let loss: Tensor = cross_entropy_loss(&y_pred, &batch_labels);
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

        println!("Epoch {:5}, Loss {:5.2} \t| Time: {:7.2}s (batch: {:5.2}s, forward: {:5.2}s, loss: {:5.2}s, backward: {:.4}s)", epoch, cost_sum, time_sum.iter().sum::<f64>(), time_sum[0], time_sum[1], time_sum[2], time_sum[3]);
    }
    
    println!("Training Done \t\t\t| Time: {:7.2}s", (Instant::now() - trainig_time_start).as_secs_f64());

    // test the trained model
    let accuracy_time_start: Instant = Instant::now();

    let y_pred = model.forward(&x_test);
    let accuracy = get_test_acc(&y_pred, &y_test);
    
    println!("Test Accuracy: {:5.2}% \t\t| Time: {:7.2}s", accuracy * 100.0, (Instant::now() - accuracy_time_start).as_secs_f64());

    println!("All Process Done \t\t| Time: {:7.2}s", (Instant::now() - total_time_start).as_secs_f64());
}