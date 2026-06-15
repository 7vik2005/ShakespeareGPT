from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from configs.config import (
    ATTENTION_HEATMAP_FILE,
    BEST_MODEL_PATH,
    SEQUENCE_LENGTH
)

from src.data.tokenizer import (
    CharacterTokenizer
)

from src.models.bardformer import (
    BardFormer
)


class AttentionVisualizer:
    def __init__(self):
        self.tokenizer=CharacterTokenizer()

        self.tokenizer.load()

        self.vocab_size=(
            self.tokenizer.vocab_size
        )

        self.model=self.load_model()

    def load_model(self):
        model=BardFormer(
            vocab_size=self.vocab_size
        )

        dummy_input=tf.zeros(
            (
                1,
                SEQUENCE_LENGTH
            ),
            dtype=tf.int32
        )

        model(dummy_input)

        if not Path(
            BEST_MODEL_PATH
        ).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {BEST_MODEL_PATH}"
            )

        model.load_weights(
            BEST_MODEL_PATH
        )

        return model

    def encode_prompt(
        self,
        prompt
    ):
        return self.tokenizer.transform(
            prompt
        )

    def get_attention_maps(
        self,
        prompt
    ):
        token_ids=self.encode_prompt(
            prompt
        )

        context=token_ids[
            -SEQUENCE_LENGTH:
        ]

        if len(context)<SEQUENCE_LENGTH:
            padding=[
                0
            ]*(
                SEQUENCE_LENGTH
                -
                len(context)
            )

            context=(
                padding
                +
                context
            )

        inputs=tf.convert_to_tensor(
            [context],
            dtype=tf.int32
        )

        _,attention_maps=self.model(
            inputs,
            training=False,
            return_attention=True
        )

        return attention_maps,context

    def plot_attention_head(
        self,
        attention_matrix,
        title,
        output_path
    ):
        plt.figure(
            figsize=(10,8)
        )

        plt.imshow(
            attention_matrix,
            aspect="auto"
        )

        plt.colorbar()

        plt.title(
            title
        )

        plt.xlabel(
            "Key Positions"
        )

        plt.ylabel(
            "Query Positions"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

    def visualize(
        self,
        prompt,
        layer_index=0,
        head_index=0
    ):
        attention_maps,context=(
            self.get_attention_maps(
                prompt
            )
        )

        attention_matrix=(
            attention_maps[
                layer_index
            ][0,head_index]
            .numpy()
        )

        self.plot_attention_head(
            attention_matrix=
            attention_matrix,

            title=
            f"Layer {layer_index+1} Head {head_index+1}",

            output_path=
            ATTENTION_HEATMAP_FILE
        )

        return ATTENTION_HEATMAP_FILE
