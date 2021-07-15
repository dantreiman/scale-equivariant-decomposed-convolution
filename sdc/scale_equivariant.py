"""Defines functions and classes for constructing scale equivariant convolutional networks."""
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

import decomposed_conv


def expand_scale_channel(x, n_scales):
    """Adds scale scannel dimension (second dimension) and tiles input over n_scales.  Assumes input dimension 0 is batch dimension."""
    repeats = tf.concat(
        [tf.constant([1, n_scales], dtype=tf.int32), tf.ones(tf.rank(x) - 1, dtype=tf.int32)],
        axis=0
    )
    return tf.tile(tf.expand_dims(x, axis=1), repeats)


class ExpandScaleChannel(tf.keras.layers.Layer):
    """Adds scale channel dimension (second dimension) and tiles input over n_scales.  Assumes input dimension 0 is batch dimension."""
    def __init__(self, n_scales, **kwargs):
        self.n_scales = n_scales
        super(ExpandScaleChannel, self).__init__(**kwargs)
        
    def call(self, x):
        return expand_scale_channel(x, self.n_scales)


class ReduceScaleChannel(tf.keras.layers.Layer):
    """Averages over scale dimension.  Assumes input dimension 0 is batch dimension."""
    def call(self, x):
        return tf.reduce_mean(x, 1)


# -------------------- Scale Equivariant Average Pooling --------------------


def gaussian_kernel(size, mean, std):
    d = tfp.distributions.Normal(tf.cast(mean, tf.float32), tf.cast(std, tf.float32))
    kernel = d.prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
    return kernel / tf.reduce_sum(kernel)


def gaussian_filter_1d(x, sigma):
    if sigma < 0.000001:
        return x
    n_c = tf.shape(x)[-1]
    size = int(4*sigma + 0.5)
    kernel = gaussian_kernel(size=size, mean=0.0, std=sigma)
    kernel = tf.tile(
        tf.reshape(kernel, (kernel.shape[0], 1, 1)), (1, n_c, n_c)
    )
    return tf.nn.conv1d(x, kernel, stride=1, padding='SAME')


class ScaleEquivariantAveragePooling(tf.keras.layers.Layer):
    """Scale-equivariant average pooling.

       Args:
         pool_size (int) The size of the area to pool over, in the largest scale.
         alphas (list) The list of scales (powers of 2).
         strides (int or list) The strides.
         padding (str) 'SAME' or 'VALID.
         scale_axis (int) The axis index corresponding to the scale channel.
                          By default, the scale channel is the second dimension.
    """
    def __init__(self, ksize, alphas, strides, padding='SAME', scale_axis=1, **kwargs):
        self.ksize = ksize
        self.alphas = alphas
        self.strides = strides
        self.padding = padding
        self.scale_axis = scale_axis
        super(ScaleEquivariantAveragePooling, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ksize': self.ksize,
            'alphas': self.alphas,
            'strides': self.strides,
            'padding': self.padding,
            'scale_axis': self.scale_axis,
        })
        return config

    def call(self, x_in): 
        """Implements scale equivariant average pooling
        
           Args:
             x_in (Tensor) A 1d input tensor with a scale channel [batch_size, t, a, c]
        """
        # Gaussian filter x_in according to scale.
        xs = tf.unstack(x_in, axis=self.scale_axis)
        
        scales_filtered = [
            tf.nn.avg_pool(gaussian_filter_1d(x, 2**alpha), ksize=self.ksize, strides=self.strides, padding=self.padding)
            for x, alpha in zip(xs, self.alphas)
        ]
                
        return tf.stack(scales_filtered, axis=self.scale_axis)


# -------------------- Scale Equivariant Convolution in 1D --------------------


def edge_repeat_pad(x, axis, padding):
    """Repeat-pad x along specified axis
    Args:
      x (Tensor) The tensor to pad.
      axis (int) The axis to repeat-pad.
      padding ((int,int)) The amount of padding to add to the beginning and end, respectively.
    """
    # Permute the padded (scale) axis as 0 so we can use [] slicing.
    if axis < 0:
        axis = K.ndim(x) - axis
    # swap_permutation = K.arange(0, x.ndim)
    # swap_permutation[axis] = 0
    # swap_permutation[0] = axis
    swap_permutation = K.concatenate([
        tf.convert_to_tensor([axis]),
        K.arange(1, axis),
        tf.convert_to_tensor([0]),
        K.arange(axis+1, K.ndim(x))
    ], axis=0)
    xt = tf.transpose(x, swap_permutation)
    # Repeat-pad the 0 axis
    slices = [
        tf.repeat(tf.expand_dims(xt[0], 0), repeats=padding[0], axis=0),
        xt,
        tf.repeat(tf.expand_dims(xt[-1], 0), repeats=padding[1], axis=0)
    ]
    padded_xt = tf.concat(slices, axis=0)
    return tf.transpose(padded_xt, swap_permutation)


class MultiscaleDecomposedConv1D(tf.keras.layers.Layer):
    """Decomposed convolution over 1d input.

       Args:
         filters (int) The number of unstructured channels out.
         ksize (int,int) The discretized kernel size (time, scale).
         scales ([float]) The list of scales (alphas).
         n_coeffs (int) Number of spatial fourier coefficients.
         ncoeffs_scale (int) Number of coefficients in scale dimension.
         strides ((int,int)) The strides (time,scale).
         padding (str) 'same' or 'valid' padding for time dimension.
         scale_padding (str) 'same' or 'valid' padding for scale dimensions.  edge-repeat padding is used for the scale dimension for 'same' convolution.
         use_bias (bool) Add bias term.
         return_kernels (bool) If true, return scaled kernels instead of convolution result.  Useful for debugging.
    """
    def __init__(self, filters, ksize, scales, n_coeffs, n_coeffs_scale=3, strides=(1,1), padding='same', scale_padding='same', use_bias=False, return_kernels=False, **kwargs):
        super(MultiscaleDecomposedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.ksize = ksize
        self.scales = scales
        self.n_coeffs = n_coeffs
        self.strides = strides
        self.padding = padding
        self.scale_padding = scale_padding
        self.use_bias = use_bias
        self.return_kernels = return_kernels
        # 'pointwise' convolution over scale dimension
        self.conv_scale = decomposed_conv.DecomposedConv1D(1, ksize[1], n_coeffs_scale, strides[1], padding='valid')  # Always use repeat padding in scale dimension.

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'ksize': self.ksize,
            'scales': self.scales,
            'n_coeffs': n_coeffs,
            'strides': self.strides,
            'padding': self.padding,
            'scale_padding': self.scale_padding,
            'use_bias': self.use_bias,
            'return_kernels': self.return_kernels
        })
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.coeffs = self.add_weight(name='coeffs', shape=(self.n_coeffs, self.in_channels, self.filters), dtype='float32',
                                      initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.use_bias:
            self.biases = self.add_weight(name='bias', shape=(1, 1, 1, self.filters), dtype='float32',
                                          initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        super(MultiscaleDecomposedConv1D, self).build(input_shape)


    def call(self, x_in):
        # Share weights over scale dimension, but render kernel scaled by 2^alpha
        x_in_shape = tf.shape(x_in)  # N, scale, time, channel
        n_rows = x_in_shape[0]    # How many rows in the batch
        n_scales_in = x_in_shape[1]  # How many scale channels in
        n_points_in = x_in_shape[2]  # 'Length'
        n_channels_in = x_in_shape[3]  # How many unstructured channels in
#         print('n_rows: %d, n_scales_in: %d, n_points_in: %d, n_channels_in: %d' % (n_rows, n_scales_in, n_points_in, n_channels_in))

        # Pointwise convolution over scale dimension.
        # Do convolution in scale
        scale_stride = self.strides[1]
        if self.scale_padding == 'same':
            padding = (x_in_shape[1] * (scale_stride-1)) - scale_stride + self.ksize[1]
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            n_scales_in += (pad_top + pad_bottom)
            x_in = edge_repeat_pad(x_in, axis=1, padding=(pad_top, pad_bottom))
        # Swaps time and scale dimensions
        xs = tf.transpose(x_in, [0,2,3,1])  # N, points, channel, scale
        xs_shape = tf.shape(xs)
        # Collapses time and channel dimensions
        xs = tf.reshape(xs, [-1, n_scales_in, 1])  # N x points x channel, scale, 1 ...
        xs = self.conv_scale(xs)
        n_scales_out = tf.shape(xs)[1]
        xs = tf.reshape(xs, [n_rows, n_points_in, n_channels_in, n_scales_out])  # N, points, channels, scale
        xs = tf.transpose(xs, [0,3,1,2])  # N, scale, points, channel
        # Multiscale spatial convolution.
        x_scales = []
        kernels = []
        for i, alpha in enumerate(self.scales):
            x_alpha = xs[:, i]  # N, points, channel
            kernel = decomposed_conv.kernel_expansion_1d(self.ksize[0], self.coeffs, alpha=alpha, masked=True) * tf.math.pow(2.0, alpha)
            kernels.append(kernel)
            x_scale = K.conv1d(x_alpha, kernel, padding=self.padding, strides=self.strides[0])  # N, points, out_channels
            x_scales.append(x_scale)
        xs = tf.stack(x_scales, axis=1)  # N, scale, points, out_channels
        if self.return_kernels:
            ks = tf.stack(kernels, axis=0)
            return ks  # Returns kernels instead of convolved input.
        if self.use_bias:
            return xs + self.biases
        return xs


# -------------------- Scale Equivariant Convolution in 2D --------------------


class MultiscaleDecomposedConv2D(tf.keras.layers.Layer):
    """Decomposed convolution over 2d input.

       Args:
         filters (int) The number of unstructured channels out.
         ksize (int,int) The discretized kernel size (space, scale).
         scales ([float]) The list of scales (alphas).
         n_coeffs_0 (int) Number of spatial mode-0 coefficients.
         n_coeffs_1 ((int,int)) Number of spatial mode-1 coefficients, or None.
         n_coeffs_scale (int) Number of coefficients in scale dimension.
         strides ((int,int,int)) The strides (h,w,scale).
         padding (str) 'same' or 'valid'.  Note: repeat-padding is used for the scale dimension for 'same' convolution.
         scale_padding (str) 'same' or 'valid' padding for scale dimensions.  edge-repeat padding is used for the scale dimension for 'same' convolution.
         use_bias (bool) Add bias term.
         return_kernels (bool) If true, return scaled kernels instead of convolution result.  Useful for debugging or visualization.
    """

    def __init__(self, filters, ksize, scales, n_coeffs_0, n_coeffs_1=None, n_coeffs_scale=3, strides=(1, 1, 1),
                 padding='same', scale_padding='same', use_bias=False, return_kernels=False, **kwargs):
        self.filters = filters
        if ksize[0] != ksize[1]:
            k = max(ksize[0], ksize[1])
            ksize = (k, k)
            print('Warning: only square kernels are supported, using %s.' % str(ksize))
        self.ksize = ksize
        self.scales = scales
        self.n_coeffs_0 = n_coeffs_0
        self.n_coeffs_1 = n_coeffs_1
        self.strides = strides
        self.padding = padding
        self.scale_padding = scale_padding
        self.use_bias = use_bias
        self.return_kernels = return_kernels
        # 'pointwise' convolution over scale dimension
        self.conv_scale = decomposed_conv.DecomposedConv1D(1, ksize[1], n_coeffs_scale, strides[1],
                                                           padding='valid')  # Always use repeat padding in scale dimension.
        super(MultiscaleDecomposedConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'ksize': self.ksize,
            'scales': self.scales,
            'n_coeffs_0': self.n_coeffs_0,
            'n_coeffs_1': self.n_coeffs_1,
            'strides': self.strides,
            'padding': self.padding,
            'scale_padding': self.scale_padding,
            'use_bias': self.use_bias,
            'return_kernels': self.return_kernels
        })
        return config

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.coeffs_0 = self.add_weight(name='coeffs_0', shape=(self.n_coeffs_0, self.in_channels, self.filters),
                                        dtype='float32',
                                        initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.n_coeffs_1 is not None:
            self.coeffs_1x = self.add_weight(name='coeffs_1x',
                                             shape=(self.n_coeffs_1[0], self.in_channels, self.filters),
                                             dtype='float32',
                                             initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
            self.coeffs_1y = self.add_weight(name='coeffs_1y',
                                             shape=(self.n_coeffs_1[1], self.in_channels, self.filters),
                                             dtype='float32',
                                             initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        if self.use_bias:
            self.biases = self.add_weight(name='bias', shape=(1, 1, 1, self.filters), dtype='float32',
                                          initializer='zeros', trainable=True)
        super(MultiscaleDecomposedConv2D, self).build(input_shape)

    def call(self, x_in):
        # Share weights over scale dimension, but render kernel scaled by 2^alpha
        x_in_shape = tf.shape(x_in)  # N, scale, height, width, channel
        n_rows = x_in_shape[0]  # How many rows in the batch
        n_scales_in = x_in_shape[1]  # How many scale channels in
        height = x_in_shape[2]
        width = x_in_shape[3]
        n_channels_in = x_in_shape[4]  # How many unstructured channels in
        #         print('n_rows: %d, n_scales_in: %d, height: %d, width: %d, n_channels_in: %d' % (n_rows, n_scales_in, height, width, n_channels_in))

        # Pointwise convolution over scale dimension.
        # Do convolution in scale
        scale_stride = self.strides[2]
        if self.scale_padding == 'same':
            padding = (x_in_shape[1] * (scale_stride - 1)) - scale_stride + self.ksize[1]
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            n_scales_in += (pad_top + pad_bottom)
            x_in = edge_repeat_pad(x_in, axis=1, padding=(pad_top, pad_bottom))
        # Swaps position and scale dimensions
        xs = tf.transpose(x_in, [0, 2, 3, 4, 1])  # N, height, width, channel, scale
        xs_shape = tf.shape(xs)
        # Collapses time and channel dimensions
        xs = tf.reshape(xs, [-1, n_scales_in, 1])  # N x height x width x channel, scale, 1 ...
        xs = self.conv_scale(xs)
        n_scales_out = tf.shape(xs)[1]
        xs = tf.reshape(xs, [n_rows, height, width, n_channels_in, n_scales_out])  # N, height, width, channels, scale
        xs = tf.transpose(xs, [0, 4, 1, 2, 3])  # N, scale, height, width, channel
        # Multiscale spatial convolution.
        x_scales = []
        kernels = []
        for i, alpha in enumerate(self.scales):
            x_alpha = xs[:, i]  # N, height, width, channel
            if self.n_coeffs_1 is None:
                kernel = decomposed_conv.kernel_expansion_zero_order(self.ksize[0], self.coeffs_0, alpha=alpha,
                                                                     masked=True)
            else:
                kernel = decomposed_conv.kernel_expansion_first_order(self.ksize[0], self.coeffs_0, self.coeffs_1x,
                                                                      self.coeffs_1y, alpha=alpha, masked=True)
            kernel = kernel * tf.math.pow(2.0, 2 * alpha)
            kernels.append(kernel)
            x_scale = K.conv2d(x_alpha, kernel, strides=self.strides[:2],
                               padding=self.padding)  # N, height, width, out_channels
            x_scales.append(x_scale)
        xs = tf.stack(x_scales, axis=1)  # N, scale, height, width, out_channels
        if self.return_kernels:
            ks = tf.stack(kernels, axis=0)
            return ks
        if self.use_bias:
            return xs + self.biases
        return xs
