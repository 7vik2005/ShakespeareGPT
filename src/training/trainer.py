import math
import time

import tensorflow as tf

from configs.config import (
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINTS_DIR,
    CHECKPOINT_NAME,
    TENSORBOARD_LOG_DIR,
    SEQUENCE_LENGTH
)

from src.data.dataset import (
    ShakespeareDataset
)

from src.models.bardformer import (
    BardFormer
)


class TrainingProgressCallback(
    tf.keras.callbacks.Callback
):
    def on_train_begin(
        self,
        logs=None
    ):
        self.training_start_time=time.time()

        print("\n"+"="*70)
        print("Training Started")
        print("="*70)

    def on_epoch_begin(
        self,
        epoch,
        logs=None
    ):
        self.epoch_start_time=time.time()

        print(
            f"\nEpoch {epoch+1}"
        )

        print("-"*70)

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):
        logs=logs or {}

        epoch_time=(
            time.time()
            -
            self.epoch_start_time
        )

        train_loss=logs.get(
            "loss",
            0.0
        )

        train_accuracy=logs.get(
            "accuracy",
            0.0
        )

        val_loss=logs.get(
            "val_loss",
            0.0
        )

        val_accuracy=logs.get(
            "val_accuracy",
            0.0
        )

        perplexity=math.exp(
            val_loss
        )

        learning_rate=(
            self.model.optimizer.learning_rate
        )

        if isinstance(
            learning_rate,
            tf.Variable
        ):
            learning_rate=float(
                learning_rate.numpy()
            )
        else:
            learning_rate=float(
                learning_rate
            )

        print(
            f"Train Loss       : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : {train_accuracy:.4%}"
        )

        print(
            f"Validation Loss  : {val_loss:.4f}"
        )

        print(
            f"Validation Acc   : {val_accuracy:.4%}"
        )

        print(
            f"Perplexity       : {perplexity:.4f}"
        )

        print(
            f"Learning Rate    : {learning_rate:.8f}"
        )

        print(
            f"Epoch Time       : {epoch_time:.2f}s"
        )

    def on_train_end(
        self,
        logs=None
    ):
        total_time=(
            time.time()
            -
            self.training_start_time
        )

        print("\n"+"="*70)
        print("Training Finished")
        print(
            f"Total Time: {total_time:.2f}s"
        )
        print("="*70)


class CheckpointNotificationCallback(
    tf.keras.callbacks.Callback
):
    def __init__(self):
        super().__init__()

        self.best_loss=float("inf")

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):
        logs=logs or {}

        current_loss=logs.get(
            "val_loss"
        )

        if current_loss is None:
            return

        if current_loss<self.best_loss:
            self.best_loss=current_loss

            print(
                "\nBest model updated"
            )


class Trainer:
    def __init__(self):
        self.dataset_builder=(
            ShakespeareDataset()
        )

        vocab_size=(
            self.dataset_builder
            .get_vocab_size()
        )

        if vocab_size==0:
            raise ValueError(
                "Tokenizer artifacts not found."
            )

        self.vocab_size=vocab_size

        self.model=BardFormer(
            vocab_size=vocab_size
        )

    def get_datasets(self):
        return (
            self.dataset_builder
            .get_datasets()
        )

    def build_model(self):
        dummy_input=tf.zeros(
            (
                1,
                SEQUENCE_LENGTH
            ),
            dtype=tf.int32
        )

        self.model(
            dummy_input
        )

        return self.model

    def get_optimizer(self):
        return tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )

    def get_loss(self):
        return (
            tf.keras.losses
            .SparseCategoricalCrossentropy(
                from_logits=True
            )
        )

    def get_metrics(self):
        return [
            tf.keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]

    def get_callbacks(self):
        checkpoint_callback=(
            tf.keras.callbacks
            .ModelCheckpoint(
                filepath=
                CHECKPOINTS_DIR
                /
                CHECKPOINT_NAME,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
        )

        tensorboard_callback=(
            tf.keras.callbacks
            .TensorBoard(
                log_dir=
                TENSORBOARD_LOG_DIR,
                histogram_freq=1
            )
        )

        early_stopping_callback=(
            tf.keras.callbacks
            .EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        )

        return [
            checkpoint_callback,
            tensorboard_callback,
            early_stopping_callback,
            CheckpointNotificationCallback(),
            TrainingProgressCallback()
        ]

    def compile_model(self):
        self.model.compile(
            optimizer=
            self.get_optimizer(),

            loss=
            self.get_loss(),

            metrics=
            self.get_metrics()
        )

    def train(self):
        train_dataset,validation_dataset=(
            self.get_datasets()
        )

        self.build_model()

        self.compile_model()

        history=self.model.fit(
            train_dataset,
            validation_data=
            validation_dataset,
            epochs=EPOCHS,
            callbacks=
            self.get_callbacks(),
            verbose=0
        )

        return history
