import tensorflow as tf
import math

class Attention(tf.keras.Model):
    def __init__(self, dim_k, model_dim):
        super().__init__()
        self.dim_k = dim_k

        self.weight_query = tf.keras.layers.Dense(dim_k, use_bias=False)
        self.weight_key = tf.keras.layers.Dense(dim_k, use_bias=False)
        self.weight_value = tf.keras.layers.Dense(dim_k, use_bias=False)

    def call(self, query, key, value, mask=None):
        query, key, value = self.weight_query(query), self.weight_key(key), self.weight_value(value)
        pre_score = tf.matmul(query, key, transpose_b=True)/math.sqrt(self.dim_k)

        if mask is not None:
            pre_score += (mask * -1e9)

        score = tf.nn.softmax(pre_score)
        return tf.matmul(score, value), score

class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_dim, num_head):
        super().__init__()
        assert model_dim%num_head == 0, "The model dimension should be in multiple of the number of heads"

        self.model_dim = model_dim
        self.num_head = num_head
        self.dim_all = model_dim//num_head

        self.weight_output = tf.keras.layers.Dense(model_dim, use_bias=False)
        self.all_attention_head = [Attention(self.dim_all, model_dim) for _ in range(num_head)]

    def call(self, query, key, value, mask=None):
        all_heads = [
            head(query, key, value, mask=mask)
            for head in self.all_attention_head
        ]
        all_heads, all_weight = zip(*all_heads)
        all_heads, all_weight = list(all_heads), list(all_weight)
        concat_heads = tf.concat(all_heads, axis=-1)
        return self.weight_output(concat_heads), all_weight
