import tensorflow as tf
from models import attention, betweenSubLayer, pointwiseFeedForward
from models import preprocessInput

class DecoderSubLayer(tf.keras.Model):
    def __init__(self, model_dim, num_head, drop_prob, pointWise_dim):
        super().__init__()
        self.model_dim = model_dim

        self.masked_multi_headed = attention.MultiHeadAttention(model_dim, num_head)
        self.in_between = betweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

        self.multi_headed_encode = attention.MultiHeadAttention(model_dim, num_head)
        self.in_between2 = betweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

        self.point_wise = pointwiseFeedForward.PointWiseFeedForward(model_dim, pointWise_dim)
        self.in_between3 = betweenSubLayer.InBetweenSubLayer(model_dim, drop_prob)

    def call(self, encoded, inp, look_ahead_mask, padding_mask, is_training):
        attn1, weight1 = self.masked_multi_headed(inp, inp, inp, mask=look_ahead_mask)
        first_layer = self.in_between(
            attn1,
            inp,
            is_training=is_training
        )

        attn2, weight2 = self.multi_headed_encode(first_layer, encoded, encoded, mask=padding_mask)
        second_layer = self.in_between2(
            attn2,
            first_layer,
            is_training=is_training
        )

        last_layer = self.in_between3(
            self.point_wise(second_layer),
            second_layer,
            is_training=is_training
        )

        return last_layer, (weight1, weight2)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer):
        super().__init__()
        self.preprocess = preprocessInput.PreprocessInput(vocab_size, model_dim, drop_prob)
        self.num_sublayer = num_sublayer
        self.model_dim = model_dim

        self.all_sublayers = [
            DecoderSubLayer(model_dim, num_head, drop_prob, pointWise_dim)
            for _ in range(num_sublayer)
        ]

        self.final_linear = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded, target, look_ahead_mask, padding_mask, is_training):
        inp_target = self.preprocess(target, is_training)
        all_weight = []
        for sub in self.all_sublayers:
            inp_target, weights = sub(encoded, inp_target, look_ahead_mask, padding_mask, is_training)
            all_weight.append(weights)

        return self.final_linear(inp_target), all_weight
