import tensorflow as tf

from configs.config import DROPOUT

from src.layers.attention import (
    MultiHeadSelfAttention
)


class FeedForwardNetwork(
    tf.keras.layers.Layer
):
    def __init__(
        self,
        embed_dim,
        ff_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim=embed_dim
        self.ff_dim=ff_dim

        self.dense_1=tf.keras.layers.Dense(
            ff_dim,
            activation="gelu"
        )

        self.dense_2=tf.keras.layers.Dense(
            embed_dim
        )

    def call(
        self,
        inputs
    ):
        x=self.dense_1(
            inputs
        )

        x=self.dense_2(
            x
        )

        return x

    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "embed_dim":self.embed_dim,
                "ff_dim":self.ff_dim
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config
    ):
        return cls(**config)


class TransformerBlock(
    tf.keras.layers.Layer
):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout=DROPOUT,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.dropout_rate=dropout

        self.attention=MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        self.feed_forward=FeedForwardNetwork(
            embed_dim=embed_dim,
            ff_dim=ff_dim
        )

        self.norm_1=tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.norm_2=tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout_1=tf.keras.layers.Dropout(
            dropout
        )

        self.dropout_2=tf.keras.layers.Dropout(
            dropout
        )

    def call(
        self,
        inputs,
        training=False
    ):
        attention_output,attention_weights=(
            self.attention(
                inputs
            )
        )

        attention_output=self.dropout_1(
            attention_output,
            training=training
        )

        x=self.norm_1(
            inputs+attention_output
        )

        feed_forward_output=self.feed_forward(
            x
        )

        feed_forward_output=self.dropout_2(
            feed_forward_output,
            training=training
        )

        output=self.norm_2(
            x+feed_forward_output
        )

        return output,attention_weights

    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "embed_dim":self.embed_dim,
                "num_heads":self.num_heads,
                "ff_dim":self.ff_dim,
                "dropout":self.dropout_rate
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config
    ):
        return cls(**config)
