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

    let _x_train = ArrayD::from_shape_vec(IxDyn(&[train_size, 784]), trn_img)
    .expect("Error converting images to ArrayD struct")
    .map(|x| *x as f32 / 256.0);
    
    let _x_test = ArrayD::from_shape_vec(IxDyn(&[test_size, 784]), tst_img)
    .expect("Error converting images to ArrayD struct")
    .map(|x| *x as f32 / 256.0);
    
    let _y_train = ArrayD::from_shape_vec(IxDyn(&[train_size, 1]), trn_lbl)
    .expect("Error converting labels to ArrayD struct")
    .map(|x| *x as f32);
    
    let _y_test = ArrayD::from_shape_vec(IxDyn(&[test_size, 1]), tst_lbl)
    .expect("Error converting labels to ArrayD struct")
    .map(|x| *x as f32);

    let mut y_train = Tensor::new(ArrayD::zeros(IxDyn(&[10, train_size])));
    let mut y_test = Tensor::new(ArrayD::zeros(IxDyn(&[10, test_size])));
    for i in 0..train_size {
        y_train.data[[_y_train[[i, 0]] as usize, i]] = Value::new(1.0);
    }
    for i in 0..test_size {
        y_test.data[[_y_test[[i, 0]] as usize, i]] = Value::new(1.0);
    }

    (Tensor::new(_x_train).transpose(), y_train,
    Tensor::new(_x_test).transpose(), y_test)
}

fn cross_entropy_loss(y_pred: &Tensor, y: &Tensor) -> Tensor {
    let y_pred_shape = y_pred.data.shape();
    assert_eq!(y_pred_shape, y.data.shape());
    let mut out = Tensor::new(ArrayD::zeros(vec![1, y_pred_shape[1]]));
    for i in 0..y_pred_shape[1] {
        out.data[[0, i]] = -((0..y_pred_shape[0]).map(|k| 
            y.data[[k, i]].clone() * y_pred.data[[k, i]].log()
        ).sum::<Value>());
    }
    out
}

fn get_test_acc(y_pred: &Tensor, y_test: &Tensor) -> usize {
    let mut correct = 0;

    let y_pred_shape = y_pred.data.shape();
    for i in 0..y_pred_shape[1] {
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for j in 0..y_pred_shape[0] {
            if y_pred.data[[j, i]].data() > max_val {
                max_val = y_pred.data[[j, i]].data();
                max_idx = j;
            }
        }
        if y_test.data[[max_idx, i]].data() == 1.0 as f32 { correct += 1; }
    }
    correct
}

pub fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    println!("--------------------------------------------------------------------------------");
    let total_time_start: Instant = Instant::now();

    // load MNIST dataset
    let data_loading_time_start: Instant = Instant::now();

    let train_size: usize = 60;
    let test_size: usize = 10;
    let (x_train, y_train, x_test, y_test) = fetch_mnist(train_size, test_size);
    x_train.requires_grad(false);
    y_train.requires_grad(false);
    x_test.requires_grad(false);
    y_test.requires_grad(false);

    println!("Data Loading Done \t\t| Time: {:10.2}s", (Instant::now() - data_loading_time_start).as_secs_f32());

    // parameters
    let learning_rate: f32 = 0.005;
    let n_epochs: usize = 15;
    let batch_size: usize = 1;

    // initialize model
    let model_time_start: Instant = Instant::now();
    
    let model = Model::new(vec![
        Box::new(Linear::new(784, 128)), Box::new(ReLU),
        Box::new(Linear::new(128, 64)), Box::new(ReLU),
        Box::new(Linear::new(64, 10)), Box::new(Softmax),
    ]);
    model.layers[0].requires_grad(false); // is it really correct?
    model.layers[2].requires_grad(false); // is it really correct?
    model.layers[4].requires_grad(false); // is it really correct?

    println!("Model Initialization Done \t| Time: {:10.2}s", (Instant::now() - model_time_start).as_secs_f32());

    // training loop
    let trainig_time_start: Instant = Instant::now();

    for epoch in 0..n_epochs {
        let mut time_sum: Vec<f32> = vec![0.0; 4];

        let mut cost_sum: f32 = 0.0;
        // train using mini-batches
        for i in 0..(train_size + batch_size - 1) / batch_size {
            // get the next batch
            let batch_time_start: Instant = Instant::now();

            let (start, end) = (i * batch_size, min((i + 1) * batch_size, train_size));
            let batch_data: Tensor = x_train.slice(start..end);
            let batch_labels: Tensor = y_train.slice(start..end);
            
            time_sum[0] += (Instant::now() - batch_time_start).as_secs_f32();

            // forward pass
            let forward_time_start: Instant = Instant::now();
            
            let y_pred: Tensor = model.forward(&batch_data);

            time_sum[1] += (Instant::now() - forward_time_start).as_secs_f32();

            // calculate loss
            let loss_time_start: Instant = Instant::now();

            let loss: Tensor = cross_entropy_loss(&y_pred, &batch_labels);
            let cost: Tensor = loss.sum();
            cost_sum += cost.data()[[0, 0]];

            time_sum[2] += (Instant::now() - loss_time_start).as_secs_f32();

            // backpropagation and weight update
            let backward_time_start: Instant = Instant::now();

            model.zerograd();
            cost.backward();
            model.update_weights(learning_rate);

            time_sum[3] += (Instant::now() - backward_time_start).as_secs_f32();
        }
        cost_sum /= train_size as f32;

        println!("Epoch {:5}, Loss {:5.2} \t| Time: {:10.2}s (batch: {:5.2}s, forward: {:5.2}s, loss: {:5.2}s, backward: {:.4}s)", epoch, cost_sum, time_sum.iter().sum::<f32>(), time_sum[0], time_sum[1], time_sum[2], time_sum[3]);
    }

    println!("Training Done \t\t\t| Time: {:10.2}s", (Instant::now() - trainig_time_start).as_secs_f32());

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

    println!("Test Accuracy: {:5.2}% \t\t| Time: {:10.2}s", (correct as f32 / test_size as f32) * 100.0, (Instant::now() - accuracy_time_start).as_secs_f32());

    println!("All Process Done \t\t| Time: {:10.2}s", (Instant::now() - total_time_start).as_secs_f32());
    println!("--------------------------------------------------------------------------------");
}