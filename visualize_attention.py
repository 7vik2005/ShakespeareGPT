from src.visualization.attention_visualizer import (
    AttentionVisualizer
)


def main():
    prompt=input(
        "Prompt: "
    ).strip()

    if not prompt:
        prompt="KING:"

    visualizer=AttentionVisualizer()

    file_path=visualizer.visualize(
        prompt=prompt
    )

    print(
        f"\nSaved: {file_path}"
    )


if __name__=="__main__":
    main()
