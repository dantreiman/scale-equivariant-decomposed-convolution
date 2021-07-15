"""Implements decomposed convolutions using fourier-bessel basis."""
import math
import scipy.special
import tensorflow as tf
import tensorflow.keras.backend as K



# -------------------- Decomposed Convolution in 1D using Fourier Basis --------------------


def kernel_expansion_1d(L, coeffs, alpha=0, masked=False):
    """Expands decomposed kernel on spatial grid of size L given fourier coefficients c.
    
        L (int) The size of the discretized kernel.
        coeffs (Tensor) The fourier coefficients.  May be any shape, as long as the summation axis is dimension 0.
        alpha (float) The scale.
        masked (bool) Mask kernel to its natural range (its range at alpha=0).
        """
    dx = 1.0 / tf.constant(L, dtype=tf.float32)
    x = tf.linspace(dx/2, 1-dx/2, L)
    x = (x - 0.5) * (2**alpha) + 0.5
    n_coeffs = tf.shape(coeffs)[0]
    PI = tf.constant(math.pi)
    xb = tf.expand_dims(x, 0)                        # 1, L
    n = tf.cast(tf.range(n_coeffs) + 1, tf.float32)  # n_coeffs    
    # broadcast_to
    n = tf.reshape(n, tf.concat([[-1], tf.ones(tf.rank(coeffs), dtype=tf.int32)], axis=0))
    xb = tf.reshape(xb, tf.concat([[1, L], tf.ones(tf.rank(coeffs) - 1, dtype=tf.int32)], axis=0))
    
    coeffs = tf.expand_dims(coeffs, 1)         # n_coeffs, 1   16,1,1,1,1
    terms = coeffs * tf.math.sin(PI * n * xb)  # n_coeffs, m
    if masked:
        mask = tf.cast((x >= dx/2) & (x <= 1-dx/2), tf.float32)
        mask = tf.reshape(mask, tf.concat([[1, -1], tf.ones(tf.rank(terms)-2, dtype=tf.int32)], axis=0))
        terms = terms * mask
    return tf.reduce_mean(terms, axis=0)


class DecomposedConv1D(tf.keras.layers.Layer):
    """Decomposed convolution over 1d input.

       Args:
         filters (int) The number of unstructured channels out.
         ksize (int) The discretized kernel size.
         n_coeffs (int) The number of coefficients (n_coeffs <= ksize).
         strides (int) The stride.
         padding (str) 'same' or 'valid'.
         use_bias (bool) Add bias term.
    """
    def __init__(self, filters, ksize, n_coeffs, strides=1, padding='valid', use_bias=False, **kwargs):
        self.filters = filters
        self.ksize = ksize
        self.n_coeffs = n_coeffs
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        super(DecomposedConv1D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'ksize': self.ksize,
            'n_coeffs': self.n_coeffs,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias
        })
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.coeffs = self.add_weight(name='coeffs', shape=(self.n_coeffs, self.in_channels, self.filters), dtype='float32',
                                      initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.use_bias:
            self.biases = self.add_weight(name='bias', shape=(1, 1, self.filters), dtype='float32',
                                          initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        super(DecomposedConv1D, self).build(input_shape)

    def call(self, x_in):
        kernel = kernel_expansion_1d(self.ksize, self.coeffs)
        result = K.conv1d(x_in, kernel, padding=self.padding, strides=self.strides)
        if self.use_bias:
            result = result + self.biases
        return result


# -------------------- Decomposed Convolution in 2D using Fourier-Bessel Basis --------------------


# Precomputes the first few zeros.  Decomposed kernel size K should be no larger than K_max.
K_max = 32

bessel_j0_zeros = scipy.special.jn_zeros(0, K_max)
bessel_j1_zeros = scipy.special.jn_zeros(1, K_max)

def add_dims_to_match(x, y):
    """Adds dimensions to x to match the dimensionality of y for tf.rank(x) <= tf.rank(y)."""
    x_shape = tf.shape(x)
    new_dims = tf.rank(y)-tf.rank(x)
    if new_dims > 0:
        new_shape = tf.concat((x_shape, tf.ones((new_dims,), dtype=tf.int32)), axis=0)
        return tf.reshape(x, new_shape)
    else:
        return x


def kernel_expansion_zero_order(L, coeffs, alpha=0, masked=False):
    """Computes the zero-order kernel of size k given coeffs.
    
    Args:
      L (int) The size of the discretized kernel (LxL).
      coeffs (Tensor) The fourier-bessel coefficients.  May be any shape, but the summation axis must be dimension 0.   
      alpha (float) The scale.
      masked (bool) Mask kernel to its natural range (its range at alpha=0).
    """
    
    # Note: dy = dx, since kernel is always LxL.
    k = tf.constant(2.0 / math.sqrt(2), dtype=tf.float32)
    dx = k / tf.constant(L, dtype=tf.float32)
    dy = dx
    
    x = tf.linspace(-k/2+dx/2, k/2-dx/2, L) * (2**alpha)
    y = tf.linspace(-k/2+dy/2, k/2-dy/2, L) * (2**alpha)
    
    X, Y = tf.meshgrid(x, y)
    
    r = tf.square(X) + tf.square(Y)
    theta = tf.math.atan2(Y, X)
    
    n = tf.shape(coeffs)[0]
    
    PI = tf.constant(math.pi)
    
    coeffs = tf.expand_dims(tf.expand_dims(coeffs, 1), 1)
    r = tf.expand_dims(r, 0)
    zeros = tf.constant(bessel_j0_zeros, dtype=tf.float32)[:n]
    zeros = tf.reshape(zeros, [n,1,1])
    # zero-order mode
    terms_0 = tf.math.special.bessel_j0(zeros * r)    
    terms_0 = add_dims_to_match(terms_0, coeffs)
    terms = coeffs * terms_0
    if masked:
        mask = tf.cast((r <= k/2), tf.float32)
        mask = tf.reshape(mask, tf.concat([[1, L, L], tf.ones(tf.rank(terms)-3, dtype=tf.int32)], axis=0))
        terms = terms * mask
    return tf.reduce_mean(terms, axis=0)


def kernel_expansion_first_order(L, coeffs_0, coeffs_1x, coeffs_1y, alpha=0, masked=False):
    """Computes the zero-order kernel of size k given coeffs.
    
    Args:
      L (int) The size of the discretized kernel (LxL).
      coeffs_0 (Tensor) The order 0 fourier-bessel coefficients.
      coeffs_1x (Tensor) The order 1 fourier-bessel coefficients (vertically symmetric).
      coeffs_1y (Tensor) The order 1 fourier-bessel coefficients (horizontally symmetric)
      alpha (float) The scale.
      masked (bool) Mask kernel to its natural range (its range at alpha=0).
    """
    
    # Note: dy = dx, since kernel is always LxL.
    k = tf.constant(2.0 / math.sqrt(2), dtype=tf.float32)
    dx = k / tf.constant(L, dtype=tf.float32)
    dy = dx
    
    x = tf.linspace(-k/2+dx/2, k/2-dx/2, L) * (2**alpha)
    y = tf.linspace(-k/2+dy/2, k/2-dy/2, L) * (2**alpha)
    
    X, Y = tf.meshgrid(x, y)
    
    r = tf.square(X) + tf.square(Y)
    theta = tf.math.atan2(Y, X)
    
    n = tf.shape(coeffs_0)[0]
    m = [tf.shape(coeffs_1x)[0], tf.shape(coeffs_1y)[0]]
    
    PI = tf.constant(math.pi)

    # Broadcast coefficients across spatial dimensions.
    coeffs_0 = tf.expand_dims(tf.expand_dims(coeffs_0, 1), 1)
    coeffs_1x = tf.expand_dims(tf.expand_dims(coeffs_1x, 1), 1)
    coeffs_1y = tf.expand_dims(tf.expand_dims(coeffs_1y, 1), 1)

    r = tf.expand_dims(r, 0)
    order_0_zeros = tf.constant(bessel_j0_zeros, dtype=tf.float32)[:n]
    order_1_zeros = tf.constant(bessel_j1_zeros, dtype=tf.float32)
    order_1_zeros_x = order_1_zeros[:m[0]]
    order_1_zeros_y = order_1_zeros[:m[1]]
    # Broadcast zeros across spatial dimensions.
    order_0_zeros = tf.reshape(order_0_zeros, [n,1,1])
    order_1_zeros_x = tf.reshape(order_1_zeros_x, [m[0],1,1])
    order_1_zeros_y = tf.reshape(order_1_zeros_y, [m[1],1,1])
    # zero-order mode
    terms_0 = tf.math.special.bessel_j0(order_0_zeros * r)
    # first-order mode
    terms_1x = tf.math.special.bessel_j1(order_1_zeros_x * r) * tf.math.cos(1 * theta)
    terms_1y = tf.math.special.bessel_j1(order_1_zeros_y * r) * tf.math.sin(1 * theta)
    # broadcast_to
    terms_0 = add_dims_to_match(terms_0, coeffs_0)
    terms_1x = add_dims_to_match(terms_1x, coeffs_1x)
    terms_1y = add_dims_to_match(terms_1y, coeffs_1y)
    terms = (
        tf.reduce_mean(coeffs_0 * terms_0, axis=0) +
        tf.reduce_mean(coeffs_1x * terms_1x, axis=0) +
        tf.reduce_mean(coeffs_1y * terms_1y, axis=0)
    )
    if masked:
        mask = tf.cast((r <= k/2), tf.float32)
        mask = tf.reshape(mask, tf.concat([[L, L], tf.ones(tf.rank(terms)-2, dtype=tf.int32)], axis=0))
        terms = terms * mask
    return terms


class DecomposedConv2D(tf.keras.layers.Layer): 
    """Decomposed convolution over 2d input.

       Args:
         filters (int) The number of unstructured channels out.
         ksize (int,int) The discretized (square) kernel size.
         n_coeffs_0 (int) The number of zero-order coefficients (n_coeffs <= ksize).
         n_coeffs_1 ((int,int)) The number of first-order coefficients (x and y).  If left to None, only zero order modes are included.
                                Zero-order modes are equivariant to rotation.
         strides ((int,int)) The stride.
         padding (str) 'same' or 'valid'.
         use_bias (bool) Add bias term.
    """
    def __init__(self, filters, ksize, n_coeffs_0, n_coeffs_1=None, strides=(1,1), padding='valid', use_bias=False, **kwargs):
        self.filters = filters
        if ksize[0] != ksize[1]:
            k = max(ksize[0], ksize[1])
            ksize = (k, k)
            print('Warning: only square kernels are supported, using %s.' % str(ksize))
        self.ksize = ksize
        self.n_coeffs_0 = n_coeffs_0
        self.n_coeffs_1 = n_coeffs_1
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        super(DecomposedConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'ksize': self.ksize,
            'n_coeffs_0': self.n_coeffs_0,
            'n_coeffs_1': self.n_coeffs_1,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias
        })
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.coeffs_0 = self.add_weight(name='coeffs_0', shape=(self.n_coeffs_0, self.in_channels, self.filters), dtype='float32',
                                        initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.n_coeffs_1 is not None:
            self.coeffs_1x = self.add_weight(name='coeffs_1x', shape=(self.n_coeffs_1[0], self.in_channels, self.filters), dtype='float32',
                                             initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
            self.coeffs_1y = self.add_weight(name='coeffs_1y', shape=(self.n_coeffs_1[1], self.in_channels, self.filters), dtype='float32',
                                             initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.use_bias:
            self.biases = self.add_weight(name='bias', shape=(1, 1, 1, self.filters), dtype='float32',
                                          initializer='zeros', trainable=True)
        super(DecomposedConv2D, self).build(input_shape)

    def call(self, x_in):
        if self.n_coeffs_1 is None:
            kernel = kernel_expansion_zero_order(self.ksize[0], self.coeffs_0)
        else:
            kernel = kernel_expansion_first_order(self.ksize[0], self.coeffs_0, self.coeffs_1x, self.coeffs_1y)
        result = K.conv2d(x_in, kernel, padding=self.padding, strides=self.strides)
        if self.use_bias:
            result = result + self.biases
        return result

