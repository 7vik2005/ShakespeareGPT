import keras_tuner as kt
import tensorflow as tf

from configs.config import (
    TUNER_DIR,
    TUNER_MAX_TRIALS,
    TUNER_EXECUTIONS_PER_TRIAL,
    SEQUENCE_LENGTH
)

from src.data.dataset import (
    ShakespeareDataset
)

from src.models.bardformer import (
    BardFormer
)


class BardFormerHyperModel(
    kt.HyperModel
):
    def __init__(
        self,
        vocab_size
    ):
        self.vocab_size=vocab_size

    def build(
        self,
        hp
    ):
        embed_dim=hp.Choice(
            "embed_dim",
            values=[
                128,
                256,
                384,
                512
            ]
        )

        num_heads=hp.Choice(
            "num_heads",
            values=[
                4,
                8
            ]
        )

        num_layers=hp.Choice(
            "num_layers",
            values=[
                4,
                6,
                8
            ]
        )

        ff_dim=hp.Choice(
            "ff_dim",
            values=[
                512,
                1024,
                2048
            ]
        )

        dropout=hp.Choice(
            "dropout",
            values=[
                0.1,
                0.2,
                0.3
            ]
        )

        learning_rate=hp.Choice(
            "learning_rate",
            values=[
                1e-4,
                5e-4,
                1e-3
            ]
        )

        model=BardFormer(
            vocab_size=self.vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_position=SEQUENCE_LENGTH
        )

        dummy_input=tf.zeros(
            (
                1,
                SEQUENCE_LENGTH
            ),
            dtype=tf.int32
        )

        model(dummy_input)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ]
        )

        return model


class BardFormerTuner:
    def __init__(self):
        self.dataset_builder=(
            ShakespeareDataset()
        )

        self.vocab_size=(
            self.dataset_builder
            .get_vocab_size()
        )

    def get_datasets(self):
        return (
            self.dataset_builder
            .get_datasets()
        )

    def create_tuner(self):
        hypermodel=(
            BardFormerHyperModel(
                self.vocab_size
            )
        )

        tuner=kt.RandomSearch(
            hypermodel=hypermodel,
            objective="val_loss",
            max_trials=
            TUNER_MAX_TRIALS,
            executions_per_trial=
            TUNER_EXECUTIONS_PER_TRIAL,
            overwrite=True,
            directory=TUNER_DIR,
            project_name="bardformer_search"
        )

        return tuner

    def search(
        self,
        epochs=3
    ):
        train_dataset,val_dataset=(
            self.get_datasets()
        )

        tuner=self.create_tuner()

        tuner.search(
            train_dataset,
            validation_data=
            val_dataset,
            epochs=epochs
        )

        return tuner

    def get_best_hyperparameters(
        self,
        tuner
    ):
        return tuner.get_best_hyperparameters(
            num_trials=1
        )[0]

    def print_best_results(
        self,
        tuner
    ):
        best_hp=(
            self.get_best_hyperparameters(
                tuner
            )
        )

        print("\nBest Hyperparameters")
        print("-"*60)

        for key,value in (
            best_hp.values.items()
        ):
            print(
                f"{key}: {value}"
            )
