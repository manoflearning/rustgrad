use ndarray::ArrayD;
use rand::{rngs::StdRng, Rng, SeedableRng};
use crate::{Tensor, Value};

pub fn zeros(shape: Vec<usize>) -> Tensor { Tensor::new(ArrayD::zeros(shape)) }
pub fn ones(shape: Vec<usize>) -> Tensor { Tensor::new(ArrayD::ones(shape)) }

pub fn rand(shape: Vec<usize>, seed: u64) -> Tensor {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = (0..shape.iter().product::<usize>()).map(|_| rng.gen_range(0.0..1.0)).collect();
    Tensor::new(ArrayD::from_shape_vec(shape, data).unwrap())
}

pub fn multinomial(input: &Tensor, num_samples: usize, replacement: bool) -> Tensor {
    assert_eq!(input.data.shape().len(), 2, "Error: input must be 2D");

    let mut rng = rand::thread_rng();
    let mut out = Tensor::new(ArrayD::zeros(vec![input.data.shape()[0], num_samples]));

    if !replacement {
        unimplemented!();
    }
    else {
        for i in 0..input.data.shape()[0] {
            let mut sum = 0.0;
            for j in 0..input.data.shape()[1] {
                sum += input.data[[i, j]].data();
            }
            for j in 0..num_samples {
                let mut r = rng.gen_range(0.0..sum);
                for k in 0..input.data.shape()[1] {
                    r -= input.data[[i, k]].data();
                    if r <= 0.0 {
                        out.data[[i, j]] = Value::new(k as f64);
                        break;
                    }

                    assert!(k != input.data.shape()[1] - 1, "Error: multinomial failed");
                }
            }
        }
    }
    out
}