import tensorflow as tf

from configs.config import (
    TEMPERATURE,
    TOP_K,
    TOP_P
)


class Sampler:
    def __init__(
        self,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P
    ):
        self.temperature=temperature
        self.top_k=top_k
        self.top_p=top_p

    def apply_temperature(
        self,
        logits
    ):
        return logits/self.temperature

    def apply_top_k(
        self,
        logits
    ):
        if self.top_k<=0:
            return logits

        values,_=tf.math.top_k(
            logits,
            k=self.top_k
        )

        minimum_value=values[:,-1]

        minimum_value=tf.expand_dims(
            minimum_value,
            axis=-1
        )

        logits=tf.where(
            logits<minimum_value,
            tf.ones_like(logits)*-1e9,
            logits
        )

        return logits

    def apply_top_p(
        self,
        logits
    ):
        if self.top_p>=1.0:
            return logits

        sorted_logits=tf.sort(
            logits,
            direction="DESCENDING"
        )

        sorted_indices=tf.argsort(
            logits,
            direction="DESCENDING"
        )

        cumulative_probs=tf.cumsum(
            tf.nn.softmax(
                sorted_logits
            ),
            axis=-1
        )

        sorted_mask=(
            cumulative_probs>
            self.top_p
        )

        sorted_mask=tf.concat(
            [
                tf.zeros_like(
                    sorted_mask[:,:1]
                ),
                sorted_mask[:,1:]
            ],
            axis=-1
        )

        scatter_mask=tf.scatter_nd(
            indices=tf.expand_dims(
                sorted_indices,
                axis=-1
            ),
            updates=tf.cast(
                sorted_mask,
                tf.float32
            ),
            shape=tf.shape(
                logits,
                out_type=tf.int64
            )
        )

        logits=tf.where(
            scatter_mask>0,
            tf.ones_like(logits)*-1e9,
            logits
        )

        return logits

    def sample(
        self,
        logits
    ):
        logits=self.apply_temperature(
            logits
        )

        logits=self.apply_top_k(
            logits
        )

        logits=self.apply_top_p(
            logits
        )

        next_token=tf.random.categorical(
            logits,
            num_samples=1
        )

        return tf.squeeze(
            next_token,
            axis=-1
        )
