import tensorflow as tf
from tensorflow.keras import layers

from transformer.transformer_block import TransformerBlock


class UViT(tf.keras.Model):
    def __init__(self, image_size: int, patch_size: int, hidden_size: int, num_heads: int, mlp_factor: int, dropout_rate: float, num_blocks: int, num_classes: int):
        super(UViT, self).__init__()

        self.patch_embedding = layers.Conv2D(
            filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid'
        )

        num_tokens = (image_size // patch_size) ** 2
        self.positional_embedding = self.add_weight(
            name='pos_embed', shape=(1, num_tokens, hidden_size), initializer='random_normal'
        )

        self.transformer_blocks = [
            TransformerBlock(hidden_size, num_heads, mlp_factor, dropout_rate) for _ in range(num_blocks)
        ]

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.head = layers.Dense(num_classes)

    def call(self, x, training=False):

        x = self.patch_embed(x)

        batch_size, height, width, channels = tf.shape(x)
        x = tf.reshape(x, [batch_size, height * width, channels])

        x = x + self.positional_embedding

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm(x)

        # x = tf.reduce_mean(x, axis=1) // diasble multi-scale feature
        x = self.head(x)

        return x
