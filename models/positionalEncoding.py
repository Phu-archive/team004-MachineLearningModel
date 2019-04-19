import tensorflow as tf
import numpy as np
import math

class PositionalEncoding(tf.keras.Model):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()

        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = 1 / np.power(10000, (2 * (np.arange(0, model_dim, 2) // 2))/model_dim)

        sin_term = np.sin(position * div_term)
        cos_term = np.cos(position * div_term)

        self.pe = np.concatenate([sin_term, cos_term], axis=-1)
        self.pe = np.expand_dims(self.pe, 0)

    def call(self, x):
        x = x + tf.Variable(self.pe[:, :tf.shape(x)[1]], dtype=tf.float32)
        return x
