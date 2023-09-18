# JAX Hyena Hierarchy

A (data-parallel) JAX/Flax implementation of the [Hyena Hierarchy](https://arxiv.org/abs/2302.10866) paper. Trained on the Tiny Shakespeare dataset.

## Description

Hyena is an attention replacement with subquadratic time complexity. This let's it process significantly longer sequences, yielding state-of-the-art results on long-sequence learning tasks. It does so by learning implicitly parametrized convolutional filters, with which the input sequence is convolved (alternating with pointwise multiplications of input projections). Subquadratic scaling is achieved by convolving using the FFT (convolution theorem).

In this repository, data parallelism is implemented using JAX's jit API and explicit sharding, as opposed to pmap. This makes it possible to explore model parallelism more easily by simply redefining the hardware mesh & sharding rules (at the expense of being more verbose).

Code was tested on a TPU v3-8, but should work on GPUs as well.

## Getting Started

hyena.py: Definition of layers and the model itself.

demo.ipynb: Training of Hyena model on the Tiny Shakespeare dataset.

helpers.py: Various helpers for training & sharding.