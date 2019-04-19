import tensorflow as tf
import numpy as np

class InBetweenSubLayer(tf.keras.Model):
    def __init__(self, model_dim, drop_prob, eps=1e-6):
        super().__init__()
        # self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(drop_prob)

    def call(self, from_layer, input_layer, is_training=True):
        from_layer = self.dropout(from_layer, training=is_training)
        normalized_layer = tf.contrib.layers.layer_norm(input_layer + from_layer)

        return normalized_layer
