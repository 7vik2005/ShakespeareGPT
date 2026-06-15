import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def create_causal_mask(
        self,
        sequence_length
    ):
        mask=1-tf.linalg.band_part(
            tf.ones(
                (
                    sequence_length,
                    sequence_length
                ),
                dtype=tf.float32
            ),
            -1,
            0
        )

        mask=mask[
            tf.newaxis,
            tf.newaxis,
            :,
            :
        ]

        return mask

    def call(
        self,
        query,
        key,
        value
    ):
        scores=tf.matmul(
            query,
            key,
            transpose_b=True
        )

        depth=tf.cast(
            tf.shape(key)[-1],
            tf.float32
        )

        scores=scores/tf.math.sqrt(
            depth
        )

        sequence_length=tf.shape(
            scores
        )[-1]

        mask=self.create_causal_mask(
            sequence_length
        )

        scores=scores-mask*1e9

        attention_weights=tf.nn.softmax(
            scores,
            axis=-1
        )

        output=tf.matmul(
            attention_weights,
            value
        )

        return output,attention_weights

    def get_config(self):
        return super().get_config()


class MultiHeadSelfAttention(
    tf.keras.layers.Layer
):
    def __init__(
        self,
        embed_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)

        if embed_dim%num_heads!=0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads

        self.query_dense=tf.keras.layers.Dense(
            embed_dim
        )

        self.key_dense=tf.keras.layers.Dense(
            embed_dim
        )

        self.value_dense=tf.keras.layers.Dense(
            embed_dim
        )

        self.output_dense=tf.keras.layers.Dense(
            embed_dim
        )

        self.attention=ScaledDotProductAttention()

    def split_heads(
        self,
        tensor,
        batch_size
    ):
        tensor=tf.reshape(
            tensor,
            (
                batch_size,
                -1,
                self.num_heads,
                self.head_dim
            )
        )

        tensor=tf.transpose(
            tensor,
            perm=[0,2,1,3]
        )

        return tensor

    def combine_heads(
        self,
        tensor,
        batch_size
    ):
        tensor=tf.transpose(
            tensor,
            perm=[0,2,1,3]
        )

        tensor=tf.reshape(
            tensor,
            (
                batch_size,
                -1,
                self.embed_dim
            )
        )

        return tensor

    def call(
        self,
        inputs
    ):
        batch_size=tf.shape(
            inputs
        )[0]

        query=self.query_dense(
            inputs
        )

        key=self.key_dense(
            inputs
        )

        value=self.value_dense(
            inputs
        )

        query=self.split_heads(
            query,
            batch_size
        )

        key=self.split_heads(
            key,
            batch_size
        )

        value=self.split_heads(
            value,
            batch_size
        )

        attention_output,attention_weights=(
            self.attention(
                query,
                key,
                value
            )
        )

        attention_output=self.combine_heads(
            attention_output,
            batch_size
        )

        output=self.output_dense(
            attention_output
        )

        return output,attention_weights

    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "embed_dim":self.embed_dim,
                "num_heads":self.num_heads
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config
    ):
        return cls(**config)
