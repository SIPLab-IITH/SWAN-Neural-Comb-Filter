import tensorflow as tf
import numpy as np
from scipy import signal

IDCT = tf.signal.dct(tf.eye(128), type=3, norm='ortho')

def log(x):
    epsilon=1e-8
    x = tf.math.maximum(x, epsilon)
    return tf.math.log(x)

def stft_mag(wav, frame_length=512, frame_step=256, fft_length=512):
    X = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    return tf.math.abs(X)

class LossLayer(tf.keras.layers.Layer):
    def call(self, clean_spec, noisy_spec, pitch,  est_mask):
        #ref_mask = tf.clip_by_value(clean_spec/(noisy_spec+2.), 0., 1.)
        #mask_loss = tf.reduce_mean(tf.math.abs(ref_mask-est_mask), axis=-1)

        #est_spec = est_mask*noisy_spec
        #cep_loss = ceploss(clean_spec, est_spec, pitch)

        return 0,0 # For testing, no loss will be used

def dummy_loss(dummy, y_pred):
    return y_pred

def ceploss(clean_spec, est_spec, pitch):
    epsilon=1.
    cep_peak = tf.cast(8000/pitch, tf.int32)
    print(clean_spec.shape, est_spec.shape)
    clean_spec_log = log(epsilon+clean_spec[:,:128])
    est_spec_log = log(epsilon+est_spec[:,:128])

    cep_error = tf.matmul(clean_spec_log - est_spec_log, IDCT)
    error = cep_error**2
    data_shape=tf.shape(error)
    E = [tf.gather_nd(error, tf.stack([tf.range(data_shape[0]), cep_peak+window], axis=-1))
            for window in range(-7,8,1)]
    cep_loss = tf.reduce_mean(tf.stack(E, axis=-1))
    return cep_loss
