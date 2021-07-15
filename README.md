# Scale-Translation-Equivariant Networks with Decomposed Convolutions

Author: Daniel Treiman (dan.treiman@gmail.com)

Stack:  Python3, Tensorflow, Keras

Purpose:
An implementation of decomposed convolutions introduced in the paper:

Zhu, Wei, et al. “Scaling-Translation-Equivariant Networks with Decomposed Convolutional Filters.” ArXiv:1909.11193 [Cs, Stat], May 2021. arXiv.org, http://arxiv.org/abs/1909.11193.


This library is organized into two parts: decomposed convolution, and st-equivariant convolution.

## decomposed_conv.py

See in Colab: https://drive.google.com/file/d/1T8XuDvqw_5SXJxhIwzN7qlGE3aFcbTOw/view?usp=sharing

This file provides convolution functions and layers with kernels decomposed in the Fourier basis for 1D convolution,
and the Bessel basis in the case of 2D convolution.

Decomposed kernels decouple the effective kernel size from the number of trainable parameters.



## scale_equivariant.py

Provides functions for performing S-T equivariant convolution by broadcasting image tensors across a scale dimension,
performing decomposed convolution with weight-sharing between scales, and scale-equivariant average pooling.

