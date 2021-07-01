import tensorflow as tf
import numpy as np
from scipy import signal


def ceploss(y_true, y_pred):
    
    cep_peak = tf.cast(y_pred[:,:,-1], tf.int32)
    y_pred = y_pred[:,:, :257]

    y_true = tf.math.log(y_true + 1e-5)
    y_pred = tf.math.log(y_pred + 1e-5)
    
    y_true_cep=tf.signal.dct(y_true, norm='ortho', type=3) 
    y_pred_cep=tf.signal.dct(y_pred, norm='ortho', type=3)

    error = (y_true_cep - y_pred_cep)**2
    
    data_shape=tf.shape(error)
    ii, jj = tf.meshgrid(*(tf.range(data_shape[i]) 
        for i in range(2)), indexing='ij')
    E = [tf.gather_nd(error, tf.stack([ii, jj, cep_peak+window], axis=-1)) 
            for window in range(-7,8,1)]
    return tf.reduce_mean(tf.stack(E, axis=-1), axis=-1)
