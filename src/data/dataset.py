import tensorflow as tf

from configs.config import (
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    BATCH_SIZE,
    SHUFFLE_BUFFER,
    DATASET_STRIDE
)

from src.data.tokenizer import CharacterTokenizer


class ShakespeareDataset:
    def __init__(self):
        self.tokenizer=CharacterTokenizer()

        try:
            self.tokenizer.load()

        except FileNotFoundError:
            raise ValueError(
                "Tokenizer artifacts not found. Run tokenizer.fit() before creating datasets."
            )

    def load_encoded_text(self):
        text=self.tokenizer.load_text()

        encoded_text=self.tokenizer.transform(
            text
        )

        return encoded_text

    def split_text(self,encoded_text):
        split_index=int(
            len(encoded_text)*TRAIN_SPLIT
        )

        train_text=encoded_text[:split_index]

        validation_text=encoded_text[split_index:]

        return train_text,validation_text

    def create_character_dataset(
        self,
        encoded_text
    ):
        return tf.data.Dataset.from_tensor_slices(
            encoded_text
        )

    def create_windows(
        self,
        dataset
    ):
        dataset=dataset.window(
            SEQUENCE_LENGTH+1,
            shift=DATASET_STRIDE,
            drop_remainder=True
        )

        dataset=dataset.flat_map(
            lambda window:window.batch(
                SEQUENCE_LENGTH+1
            )
        )

        return dataset

    def create_input_target_pairs(
        self,
        dataset
    ):
        dataset=dataset.map(
            lambda window:(
                window[:-1],
                window[1:]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        return dataset

    def prepare_dataset(
        self,
        encoded_text,
        training=True
    ):
        dataset=self.create_character_dataset(
            encoded_text
        )

        dataset=self.create_windows(
            dataset
        )

        dataset=self.create_input_target_pairs(
            dataset
        )

        if training:
            dataset=dataset.shuffle(
                SHUFFLE_BUFFER
            )

        dataset=dataset.batch(
            BATCH_SIZE,
            drop_remainder=True
        )

        dataset=dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset

    def get_datasets(self):
        encoded_text=self.load_encoded_text()

        train_text,validation_text=self.split_text(
            encoded_text
        )

        train_dataset=self.prepare_dataset(
            train_text,
            training=True
        )

        validation_dataset=self.prepare_dataset(
            validation_text,
            training=False
        )

        return (
            train_dataset,
            validation_dataset
        )

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def decode_sample(
        self,
        encoded_sequence
    ):
        return self.tokenizer.inverse_transform(
            encoded_sequence
        )
