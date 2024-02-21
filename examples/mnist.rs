extern crate rustgrad;
use ndarray::prelude::*;
use rustgrad::*;
use std::{cmp::min, time::Instant};
use mnist::*;

fn fetch_mnist(train_size: usize, test_size: usize) -> (Tensor, Tensor, Tensor, Tensor) {
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

    let x_train = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[train_size, 1, 28, 28]), trn_img)
    .expect("Error converting images to ArrayD struct")
    .map(|x| *x as f64 / 256.0));
    
    let x_test = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[test_size, 1, 28, 28]), tst_img)
    .expect("Error converting images to ArrayD struct")
    .map(|x| *x as f64 / 256.0));
    
    let y_train = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[train_size, 1]), trn_lbl)
    .expect("Error converting labels to ArrayD struct")
    .map(|x| *x as f64));
    
    let y_test = Tensor::new(ArrayD::from_shape_vec(IxDyn(&[test_size, 1]), tst_lbl)
    .expect("Error converting labels to ArrayD struct")
    .map(|x| *x as f64));

    // TODO: standardization

    (x_train, y_train, x_test, y_test)
}

fn cross_entropy_loss(y_pred: &Tensor, y: &Tensor) -> Tensor {
    let y_pred_shape = y_pred.data.shape();
    let mut out = Tensor::new(ArrayD::zeros(vec![y_pred_shape[0], 1]));
    for i in 0..y_pred_shape[0] {
        out.data[[i, 0]] = -y_pred.data[[i, y.data[[i, 0]].data() as usize]].log();
    }
    out
}

fn get_test_acc(y_pred: &Tensor, y_test: &Tensor) -> usize {
    let mut correct = 0;

    let y_pred_shape = y_pred.data.shape();
    for i in 0..y_pred_shape[0] {
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for j in 0..y_pred_shape[1] {
            if y_pred.data[[i, j]].data() > max_val {
                max_val = y_pred.data[[i, j]].data();
                max_idx = j;
            }
        }
        if y_test.data[[i, 0]].data() == max_idx as f64 { correct += 1; }
    }
    correct
}

pub fn main() {
    std::env::set_var("RUST_BACKTRACE", "full");

    println!("--------------------------------------------------------------------------------");
    let total_time_start: Instant = Instant::now();

    // load MNIST dataset
    let data_loading_time_start: Instant = Instant::now();

    let train_size: usize = 6;
    let test_size: usize = 1;
    let (x_train, y_train, x_test,  y_test) = fetch_mnist(train_size, test_size);
    x_train.requires_grad(false);
    y_train.requires_grad(false);
    x_test.requires_grad(false);
    y_test.requires_grad(false);

    println!("Data Loading Done \t\t| Time: {:10.2}s", (Instant::now() - data_loading_time_start).as_secs_f64());

    // parameters
    let learning_rate: f64 = 0.005;
    let n_epochs: usize = 15;
    let batch_size: usize = 1;

    // initialize model
    let model_time_start: Instant = Instant::now();

    // CNN
    let model = Model::new(vec![
        Box::new(Conv2d::new(1, 32, 5, 1, 0)), Box::new(ReLU),
        Box::new(Conv2d::new(32, 32, 5, 1, 0)), Box::new(ReLU),
        Box::new(BatchNorm2d::new(32)), Box::new(MaxPool2d::new(2)),
        Box::new(Conv2d::new(32, 64, 3, 1, 0)), Box::new(ReLU),
        Box::new(Conv2d::new(64, 64, 3, 1, 0)), Box::new(ReLU),
        Box::new(BatchNorm2d::new(64)), Box::new(MaxPool2d::new(2)),
        Box::new(Flatten),
        Box::new(Linear::new(576, 10)), Box::new(Softmax),
    ]);

    // 3 Layer MLP
    // let model = Model::new(vec![
    //     Box::new(Flatten),
    //     Box::new(Linear::new(784, 128)),
    //     Box::new(ReLU),
    //     Box::new(Linear::new(128, 64)),
    //     Box::new(ReLU),
    //     Box::new(Linear::new(64, 10)),
    //     Box::new(Softmax),
    // ]);

    model.layers.iter().for_each(|layer| {
        layer.requires_grad(false);
    });

    println!("Model Initialization Done \t| Time: {:10.2}s", (Instant::now() - model_time_start).as_secs_f64());

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
            let data: Tensor = x_train.slice(start..end);
            let targets: Tensor = y_train.slice(start..end);

            time_sum[0] += (Instant::now() - batch_time_start).as_secs_f64();

            // forward pass
            let forward_time_start: Instant = Instant::now();

            let y_pred: Tensor = model.forward(&data);

            time_sum[1] += (Instant::now() - forward_time_start).as_secs_f64();

            // calculate loss
            let loss_time_start: Instant = Instant::now();

            let loss: Tensor = cross_entropy_loss(&y_pred, &targets);
            let cost: Tensor = loss.sum();
            cost_sum += cost.data()[[0]];

            time_sum[2] += (Instant::now() - loss_time_start).as_secs_f64();

            // backpropagation and weight update
            let backward_time_start: Instant = Instant::now();

            model.zerograd();
            cost.backward();
            model.update_weights(learning_rate);

            time_sum[3] += (Instant::now() - backward_time_start).as_secs_f64();
        }
        cost_sum /= train_size as f64;

        println!("Epoch {:5}, Loss {:5.2} \t| Time: {:10.2}s (batch: {:5.2}s, forward: {:5.2}s, loss: {:5.2}s, backward: {:5.2}s)", epoch, cost_sum, time_sum.iter().sum::<f64>(), time_sum[0], time_sum[1], time_sum[2], time_sum[3]);
    }

    println!("Training Done \t\t\t| Time: {:10.2}s", (Instant::now() - trainig_time_start).as_secs_f64());

    // test the trained model
    let accuracy_time_start: Instant = Instant::now();

    let mut correct = 0;
    for i in 0..(test_size + batch_size - 1) / batch_size {
        let (start, end) = (i * batch_size, min((i + 1) * batch_size, test_size));
        let x_test = x_test.slice(start..end);
        let y_test = y_test.slice(start..end);
        let y_pred = model.forward(&x_test);
        correct += get_test_acc(&y_pred, &y_test);
    }

    println!("Test Accuracy: {:5.2}% \t\t| Time: {:10.2}s", (correct as f64 / test_size as f64) * 100.0, (Instant::now() - accuracy_time_start).as_secs_f64());

    println!("All Process Done \t\t| Time: {:10.2}s", (Instant::now() - total_time_start).as_secs_f64());
    println!("--------------------------------------------------------------------------------");
}