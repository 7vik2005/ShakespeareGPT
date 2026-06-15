from pathlib import Path

import tensorflow as tf

from configs.config import (
    BEST_MODEL_PATH,
    GENERATED_TEXT_FILE,
    MAX_GENERATION_LENGTH,
    SEQUENCE_LENGTH,
    TEMPERATURE
)

from src.data.tokenizer import (
    CharacterTokenizer
)

from src.models.bardformer import (
    BardFormer
)


class TextGenerator:
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

    def decode_tokens(
        self,
        tokens
    ):
        return self.tokenizer.inverse_transform(
            tokens
        )

    def sample_next_token(
        self,
        logits
    ):
        logits=logits/TEMPERATURE

        next_token=tf.random.categorical(
            logits,
            num_samples=1
        )

        return int(
            next_token.numpy()[0][0]
        )

    def generate(
        self,
        prompt,
        max_length=MAX_GENERATION_LENGTH
    ):
        token_ids=self.encode_prompt(
            prompt
        )

        generated=list(
            token_ids
        )

        for _ in range(max_length):
            context=generated[
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

            logits=self.model(
                inputs,
                training=False
            )

            next_token_logits=(
                logits[
                    :,
                    -1,
                    :
                ]
            )

            next_token=(
                self.sample_next_token(
                    next_token_logits
                )
            )

            generated.append(
                next_token
            )

        return self.decode_tokens(
            generated
        )

    def save_output(
        self,
        text
    ):
        with open(
            GENERATED_TEXT_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(text)
