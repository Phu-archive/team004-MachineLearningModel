import tensorflow as tf
from models import attention, betweenSubLayer, pointwiseFeedForward
from models import preprocessInput

class EncoderSubLayer(tf.keras.Model):
    def __init__(self, model_dim, num_head, drop_prob, pointWise_dim):
        super().__init__()

        self.multi_headed = attention.MultiHeadAttention(model_dim, num_head)
        self.in_between = betweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

        self.point_wise = pointwiseFeedForward.PointWiseFeedForward(model_dim, pointWise_dim)
        self.in_between2 = betweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

    def call(self, inp, is_training, mask):
        attn, weight_attention = self.multi_headed(inp, inp, inp, mask)
        after_attention = self.in_between(
            attn,
            inp,
            is_training
        )
        output_sub = self.in_between2(
            self.point_wise(after_attention),
            after_attention,
            is_training
        )

        return output_sub, weight_attention

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer):
        super().__init__()
        self.preprocess = preprocessInput.PreprocessInput(vocab_size, model_dim, drop_prob)
        self.model_dim = model_dim
        self.num_sublayer = num_sublayer

        self.all_sublayers = [
            EncoderSubLayer(model_dim, num_head, drop_prob, pointWise_dim)
            for _ in range(num_sublayer)
        ]

    def call(self, inp, mask, is_training):
        inp = tf.cast(self.preprocess(inp, is_training), dtype=tf.float32)
        layer_weight = []
        for sub in self.all_sublayers:
            inp, weight = sub(inp, is_training, mask)
            layer_weight.append(weight)

        return inp, layer_weight
