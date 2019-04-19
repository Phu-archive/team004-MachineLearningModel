import tensorflow as tf

class PointWiseFeedForward(tf.keras.Model):
    def __init__(self, model_dim, between_dim):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(between_dim, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(model_dim)

    def call(self, inp):
        return self.layer2(self.layer1(inp))
