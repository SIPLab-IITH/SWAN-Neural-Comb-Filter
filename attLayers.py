import tensorflow as tf


def symmetric_mask(shape, context=10):
    """Creates a symmetric boolean mask over the last 2 dimensions."""
    row_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.math.logical_and(
            tf.abs(row_index - col_index) <= context,
            tf.abs(row_index - col_index) > 0)


def causal_mask(shape, context=10):
    """Creates a symmetric boolean mask over the last 2 dimensions."""
    row_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.math.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return (row_index - col_index) <= context

class MultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, model_size, n_heads, layer_size, context=None):
        super(MultiHeadAttn, self).__init__()
        self.model_size = model_size
        self.layer_size = layer_size
        self.n_heads = n_heads
        self.context = context
        regularizer = tf.keras.regularizers.L2(1e-5)
        initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1e-4)

        self.qlin = [tf.keras.layers.Dense(self.layer_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False) for _ in range(n_heads)]
        self.vlin = [tf.keras.layers.Dense(self.layer_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False) for _ in range(n_heads)]
        self.attup = tf.keras.layers.Dense(self.model_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False)
  
    def call(self, inputs):
        
        heads=[]
        for i in range(self.n_heads):   
            Q = self.qlin[i](inputs)
            V = self.vlin[i](inputs)
            scores = tf.matmul(Q, V, transpose_b=True)
            scores = scores / 16.
            scores_shape = tf.shape(scores)
            mask_shape = tf.concat(
            [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)

            context_mask = symmetric_mask(mask_shape)
            scores = scores - 1.e9 * tf.cast(tf.math.logical_not(context_mask), dtype=tf.float32)
            weights = tf.keras.activations.softmax(scores, axis=-1)
            C = tf.matmul(weights, V)
            heads.append(C)
            
        A = self.attup(tf.concat(heads, axis=2)) 
        return A
    
    def get_config(self):
        return {"model_size": self.model_size, "n_heads": self.n_heads, "layer_size": self.layer_size, "context": self.context}
        
class MultiHeadAttn_Causal(tf.keras.layers.Layer):
    def __init__(self, model_size, n_heads, layer_size, context=None):
        super(MultiHeadAttn_Causal, self).__init__()
        self.model_size = model_size
        self.layer_size = layer_size
        self.n_heads = n_heads
        self.context = context
        regularizer = tf.keras.regularizers.L2(1e-5)
        initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=1e-4)

        self.qlin = [tf.keras.layers.Dense(self.layer_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False) for _ in range(n_heads)]
        self.vlin = [tf.keras.layers.Dense(self.layer_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False) for _ in range(n_heads)]
        self.attup = tf.keras.layers.Dense(self.model_size,kernel_regularizer=regularizer, kernel_initializer = initializer, use_bias = False)
  
    def call(self, inputs):
        
        heads=[]
        for i in range(self.n_heads):   
            Q = self.qlin[i](inputs)
            V = self.vlin[i](inputs)
            scores = tf.matmul(Q, V, transpose_b=True)
            scores = scores / 16.
            scores_shape = tf.shape(scores)
            mask_shape = tf.concat(
            [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)

            context_mask = causal_mask(mask_shape)
            scores = scores - 1.e9 * tf.cast(tf.math.logical_not(context_mask), dtype=tf.float32)
            weights = tf.keras.activations.softmax(scores, axis=-1)
            C = tf.matmul(weights, V)
            heads.append(C)
            
        A = self.attup(tf.concat(heads, axis=2)) 
        return A
    
    def get_config(self):
        return {"model_size": self.model_size, "n_heads": self.n_heads, "layer_size": self.layer_size, "context": self.context}
