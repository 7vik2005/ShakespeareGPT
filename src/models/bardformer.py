import tensorflow as tf

from configs.config import (
    SEQUENCE_LENGTH,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FF_DIM,
    DROPOUT
)

from src.layers.positional_encoding import (
    PositionalEncoding
)

from src.layers.transformer_block import (
    TransformerBlock
)


class BardFormer(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_position=SEQUENCE_LENGTH,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size=vocab_size
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.ff_dim=ff_dim
        self.dropout_rate=dropout
        self.max_position=max_position

        self.embedding=tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.positional_encoding=PositionalEncoding(
            max_position=max_position,
            embed_dim=embed_dim
        )

        self.dropout=tf.keras.layers.Dropout(
            dropout
        )

        self.transformer_blocks=[
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ]

        self.final_norm=tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.output_projection=tf.keras.layers.Dense(
            vocab_size
        )

    def call(
        self,
        inputs,
        training=False,
        return_attention=False
    ):
        x=self.embedding(
            inputs
        )

        x*=tf.math.sqrt(
            tf.cast(
                self.embed_dim,
                tf.float32
            )
        )

        x=self.positional_encoding(
            x
        )

        x=self.dropout(
            x,
            training=training
        )

        attention_maps=[]

        for block in self.transformer_blocks:
            x,attention_weights=block(
                x,
                training=training
            )

            if return_attention:
                attention_maps.append(
                    attention_weights
                )

        x=self.final_norm(
            x
        )

        logits=self.output_projection(
            x
        )

        if return_attention:
            return logits,attention_maps

        return logits

    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "vocab_size":self.vocab_size,
                "embed_dim":self.embed_dim,
                "num_heads":self.num_heads,
                "num_layers":self.num_layers,
                "ff_dim":self.ff_dim,
                "dropout":self.dropout_rate,
                "max_position":self.max_position
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config
    ):
        return cls(**config)
