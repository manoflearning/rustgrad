# RustGrad

RustGrad is a small deep learning framework developed from scratch in Rust.
It is a toy project for understanding neural networks, but the long-term goal is to develop it into a practical and useful framework.

## Features

### MNIST

With RustGrad, you can train on the MNIST dataset. Although it's currently slow and a bit rough around the edges, it's fully functional! It supports operations like ```Linear```, ```ReLU```, ```Conv2d```, ```BatchNorm2d```, ```MaxPool2d```, etc. For more details, please check out [examples/mnist.rs](https://github.com/manoflearning/rustgrad/blob/master/examples/mnist.rs).

## Inspired by

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad) by tiny corp