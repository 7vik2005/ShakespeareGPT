import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        max_position,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.max_position=max_position
        self.embed_dim=embed_dim

        self.positional_encoding=self._create_encoding()

    def _get_angles(
        self,
        positions,
        dimensions
    ):
        angle_rates=1/np.power(
            10000,
            (2*(dimensions//2))/self.embed_dim
        )

        return positions*angle_rates

    def _create_encoding(self):
        positions=np.arange(
            self.max_position
        )[:,np.newaxis]

        dimensions=np.arange(
            self.embed_dim
        )[np.newaxis,:]

        angle_radians=self._get_angles(
            positions,
            dimensions
        )

        angle_radians[:,0::2]=np.sin(
            angle_radians[:,0::2]
        )

        angle_radians[:,1::2]=np.cos(
            angle_radians[:,1::2]
        )

        positional_encoding=angle_radians[
            np.newaxis,
            ...
        ]

        return tf.cast(
            positional_encoding,
            tf.float32
        )

    def call(
        self,
        inputs
    ):
        sequence_length=tf.shape(
            inputs
        )[1]

        return (
            inputs
            +
            self.positional_encoding[
                :,
                :sequence_length,
                :
            ]
        )

    def get_config(self):
        config=super().get_config()

        config.update(
            {
                "max_position":self.max_position,
                "embed_dim":self.embed_dim
            }
        )

        return config

    @classmethod
    def from_config(
        cls,
        config
    ):
        return cls(**config)
