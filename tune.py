from datetime import datetime

from configs.config import (
    create_directories
)

from src.data.tokenizer import (
    CharacterTokenizer
)

from src.tuning.tuner import (
    BardFormerTuner
)


def verify_tokenizer():
    tokenizer=CharacterTokenizer()

    try:
        tokenizer.load()

        print(
            f"Vocabulary Size: {tokenizer.vocab_size}"
        )

    except FileNotFoundError:
        print(
            "Tokenizer artifacts not found."
        )

        print(
            "Building tokenizer..."
        )

        tokenizer.fit()

        print(
            f"Vocabulary Size: {tokenizer.vocab_size}"
        )

    return tokenizer


def print_header():
    print("\n"+"="*80)
    print("BardFormer Hyperparameter Search")
    print("="*80)


def main():
    start_time=datetime.now()

    print_header()

    create_directories()

    verify_tokenizer()

    tuner=BardFormerTuner()

    print("\nStarting Hyperparameter Search")
    print("-"*80)

    search=tuner.search(
        epochs=3
    )

    print("\nSearch Complete")
    print("-"*80)

    tuner.print_best_results(
        search
    )

    end_time=datetime.now()

    print("\nFinished")
    print("-"*80)

    print(
        f"Started  : {start_time}"
    )

    print(
        f"Finished : {end_time}"
    )

    print(
        f"Duration : {end_time-start_time}"
    )


if __name__=="__main__":
    main()
