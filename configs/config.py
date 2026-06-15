from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent

DATA_DIR=PROJECT_ROOT/"data"
RAW_DATA_DIR=DATA_DIR/"raw"

ARTIFACTS_DIR=PROJECT_ROOT/"artifacts"
CHECKPOINTS_DIR=PROJECT_ROOT/"checkpoints"
LOGS_DIR=PROJECT_ROOT/"logs"
OUTPUTS_DIR=PROJECT_ROOT/"outputs"
TUNER_DIR=PROJECT_ROOT/"tuner_results"

DATA_PATH=RAW_DATA_DIR/"shakespeare.txt"

VOCAB_PATH=ARTIFACTS_DIR/"vocabulary.pkl"
CHAR2IDX_PATH=ARTIFACTS_DIR/"char2idx.pkl"
IDX2CHAR_PATH=ARTIFACTS_DIR/"idx2char.pkl"

MODEL_NAME="shakespearegpt"

CHECKPOINT_NAME="best_model.weights.h5"

BEST_MODEL_PATH=(
    CHECKPOINTS_DIR/
    CHECKPOINT_NAME
)

SEQUENCE_LENGTH=256

TRAIN_SPLIT=0.9

BATCH_SIZE=32

SHUFFLE_BUFFER=100000

DATASET_STRIDE=8

EMBED_DIM=256

NUM_HEADS=8

NUM_LAYERS=6

FF_DIM=1024

DROPOUT=0.1

LEARNING_RATE=1e-4

EPOCHS=20

TEMPERATURE=0.8

TOP_K=20

TOP_P=0.9

MAX_GENERATION_LENGTH=1000

RANDOM_SEED=42

TUNER_MAX_TRIALS=15

TUNER_EXECUTIONS_PER_TRIAL=1

TENSORBOARD_LOG_DIR=(
    LOGS_DIR/
    "tensorboard"
)

GENERATED_TEXT_FILE=(
    OUTPUTS_DIR/
    "generated_text.txt"
)

ATTENTION_HEATMAP_FILE=(
    OUTPUTS_DIR/
    "attention_heatmap.png"
)

TRAINING_CURVE_FILE=(
    OUTPUTS_DIR/
    "training_curves.png"
)

DEFAULT_PROMPT="KING:"


def create_directories():
    directories=[
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        OUTPUTS_DIR,
        TUNER_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )
