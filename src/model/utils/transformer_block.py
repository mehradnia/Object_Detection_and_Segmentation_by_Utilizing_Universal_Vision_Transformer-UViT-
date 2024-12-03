import tensorflow as tf
from tensorflow.keras import layers, models


class TransformerBlock(layers.Layer):
    def __init__(self, depth: int, num_heads: int, mlp_factor: int, dropout_rate: float):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=depth, dropout=dropout_rate)

        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = models.Sequential([
            layers.Dense(int(depth * mlp_factor), activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(depth),
            layers.Dropout(dropout_rate)
        ])

    def call(self, x, training=False):
        attn_output = self.attn(self.norm1(
            x), self.norm1(x), training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        mlp_output = self.mlp(self.norm2(x), training=training)
        x = x + mlp_output

        return x
