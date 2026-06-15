from datetime import datetime

from configs.config import (
    DATA_PATH,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    create_directories
)

from src.data.tokenizer import (
    CharacterTokenizer
)

from src.data.dataset import (
    ShakespeareDataset
)

from src.training.trainer import (
    Trainer
)


def print_header():
    print("\n"+"="*80)
    print("BardFormer")
    print("GPT-Style Character-Level Transformer")
    print("="*80)


def prepare_tokenizer():
    tokenizer=CharacterTokenizer()

    try:
        tokenizer.load()

        print(
            "\nTokenizer artifacts found"
        )

    except FileNotFoundError:
        print(
            "\nTokenizer artifacts not found"
        )

        print(
            "Building tokenizer..."
        )

        tokenizer.fit()

        print(
            "Tokenizer artifacts created"
        )

    return tokenizer


def print_dataset_statistics(
    tokenizer
):
    text=tokenizer.load_text()

    total_characters=len(text)

    vocabulary_size=(
        tokenizer.vocab_size
    )

    estimated_sequences=(
        total_characters
        -
        SEQUENCE_LENGTH
    )

    estimated_batches=(
        estimated_sequences
        //
        BATCH_SIZE
    )

    print("\nDataset Information")
    print("-"*80)

    print(
        f"Dataset File        : {DATA_PATH}"
    )

    print(
        f"Total Characters    : {total_characters:,}"
    )

    print(
        f"Vocabulary Size     : {vocabulary_size}"
    )

    print(
        f"Sequence Length     : {SEQUENCE_LENGTH}"
    )

    print(
        f"Batch Size          : {BATCH_SIZE}"
    )

    print(
        f"Approx Sequences    : {estimated_sequences:,}"
    )

    print(
        f"Approx Batches      : {estimated_batches:,}"
    )


def print_sample_vocabulary(
    tokenizer
):
    print("\nVocabulary Preview")
    print("-"*80)

    preview=tokenizer.vocab[:50]

    print(preview)


def print_model_statistics():
    trainer=Trainer()

    model=trainer.build_model()

    print("\nModel Information")
    print("-"*80)

    print(
        f"Vocabulary Size     : {trainer.vocab_size}"
    )

    print(
        f"Total Parameters    : {model.count_params():,}"
    )


def verify_dataset():
    dataset_builder=(
        ShakespeareDataset()
    )

    train_dataset,val_dataset=(
        dataset_builder.get_datasets()
    )

    train_batches=(
        train_dataset
        .cardinality()
        .numpy()
    )

    validation_batches=(
        val_dataset
        .cardinality()
        .numpy()
    )

    print("\nTraining Dataset")
    print("-"*80)

    print(
        f"Training Batches    : {train_batches:,}"
    )

    print(
        f"Validation Batches  : {validation_batches:,}"
    )

    for x,y in train_dataset.take(1):
        print(
            f"Input Shape         : {x.shape}"
        )

        print(
            f"Target Shape        : {y.shape}"
        )

        break


def main():
    start_time=datetime.now()

    print_header()

    create_directories()

    tokenizer=prepare_tokenizer()

    print_dataset_statistics(
        tokenizer
    )

    print_sample_vocabulary(
        tokenizer
    )

    verify_dataset()

    print_model_statistics()

    print("\nStarting Training")
    print("-"*80)

    trainer=Trainer()

    trainer.train()

    end_time=datetime.now()

    duration=end_time-start_time

    print("\nTraining Run Completed")
    print("-"*80)

    print(
        f"Started  : {start_time}"
    )

    print(
        f"Finished : {end_time}"
    )

    print(
        f"Duration : {duration}"
    )


if __name__=="__main__":
    main()
