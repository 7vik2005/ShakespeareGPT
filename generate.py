from datetime import datetime

from src.inference.generate import (
    TextGenerator
)


def print_header():
    print("\n"+"="*80)
    print("BardFormer Text Generation")
    print("="*80)


def get_prompt():
    prompt=input(
        "\nEnter Prompt: "
    ).strip()

    if not prompt:
        prompt="KING:"

    return prompt


def main():
    start_time=datetime.now()

    print_header()

    prompt=get_prompt()

    print("\nLoading Model...")
    print("-"*80)

    generator=TextGenerator()

    print("Model Loaded")

    print("\nGenerating Text...")
    print("-"*80)

    generated_text=generator.generate(
        prompt=prompt
    )

    print("\nGenerated Text")
    print("="*80)

    print(generated_text)

    print("="*80)

    generator.save_output(
        generated_text
    )

    end_time=datetime.now()

    print(
        f"\nOutput saved successfully"
    )

    print(
        f"Generation Time: {end_time-start_time}"
    )


if __name__=="__main__":
    main()
