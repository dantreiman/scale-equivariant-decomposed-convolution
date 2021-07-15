import boto3
import botocore
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.signal
import tensorflow as tf
import tensorflow.keras.backend as K
import sys

alivecor_module_path = '/ds/alivecor/src/python3'
if alivecor_module_path not in sys.path:
    sys.path.append(alivecor_module_path)
    
from atc.atc_reader import ATCReader


s3 = boto3.resource('s3')

def read_ecg_data(recording_id):
    # Download the ATC from S3
    s3_path = 'files/atc_raw/%s.atc' % recording_id
    atc_object = s3.Object('alivecor-feature-store', s3_path)
    try:
        atc_file = atc_object.get()['Body']
        reader = ATCReader(atc_file)
    except botocore.exceptions.ClientError as e:
        print('Failed to load ATC file: %s' % str(e))
        print('This should never happen, ATC files should have been verified already, maybe something went wrong talking to S3.')
        return None
    
    # Parsing and decoding the file
    leadI_data = reader.get_ecg_samples(1)
    return np.array(leadI_data, dtype=np.float32)


# Loads a test ECG
ecg = read_ecg_data('2wgq3pgqszrrkadvdlnbkbhwz')


test_x = np.expand_dims(np.expand_dims(ecg, -1), 0).astype(np.float32)
test_x.shape


class ScaleInvariantDepthwiseConvolution:
    """Scales signal to multiple scales, and performance a depthwise-separable convolution over space (first),
       then scale.
    """
    def __init__(self, base=2.0, exponent_range=(-1, 1), scale_channels=9, kernel_size=(7,3), **kwargs):
        """Constructor
        
           Args:
             base (float) The base of the scaling group.
             exponent_step (float, float) The step between adjacent scale channels.
             scale_channels (int) The number of scale channels.
             kernel_size ((int,int)) The size of the kernel
        """
        exponents = np.linspace(exponent_range[0], exponent_range[1], scale_channels)
        self.scale_factors = np.power(base, exponents)
        self.conv1 = tf.keras.layers.Conv2D(kernel_size=(kernel_size[0], 1), padding='valid', **kwargs)  # Conv over spatial dimension
        self.conv2 = tf.keras.layers.Conv2D(kernel_size=(1, kernel_size[1]), padding='valid', **kwargs)  # Conv over scale dimension
        scale_channels_after_conv = scale_channels - (kernel_size[1] - 1)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(1, scale_channels_after_conv), padding='valid')  # Max pool over scale dimension

    def __call__(self, x):
        x_images = tf.expand_dims(x, 2) # N, L, 1, C
        L = tf.shape(x_images)[1]
        new_lengths = [int(np.ceil(tf.cast(L, tf.float32)*s)) for s in self.scale_factors]
        print(new_lengths)
        scaled = [tf.image.resize(x_images, [l, 1]) for l in new_lengths]
        conv_scaled = [self.conv1(r) for r in scaled]
        rescaled = [tf.image.resize(y, [L, 1]) for y in conv_scaled]
        print(rescaled[0].shape)
        block = tf.concat(rescaled, axis=2)
        print(block.shape)
        y = self.conv2(block)
        print(y.shape)
        y = self.max_pool(y)
        return tf.squeeze(y, axis=2)

    
dc = ScaleInvariantDepthwiseConvolution(filters=8)

s = dc(test_x)

print(s.shape)

print('Done')
