import tensorflow as tf
from models import positionalEncoding
import math

class PreprocessInput(tf.keras.Model):
    def __init__(self, vocab_size, model_dim, drop_prob):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim)
        self.pos_encode = positionalEncoding.PositionalEncoding(model_dim)
        self.model_dim = model_dim
        self.dropout = tf.keras.layers.Dropout(drop_prob)

    def call(self, inp, is_training):
        embedding = self.embedding(inp)
        embedding *= math.sqrt(self.model_dim)
        embedding = self.dropout(embedding, training=is_training)
        return self.pos_encode(embedding)
