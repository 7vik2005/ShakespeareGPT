<div align="center">

# ShakespeareGPT

### A GPT-Style Decoder-Only Transformer Language Model Built Completely From Scratch Using TensorFlow

<p>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org)[![NLP](https://img.shields.io/badge/NLP-Transformer-blue?style=for-the-badge)](#)[![GPT](https://img.shields.io/badge/GPT-Decoder%20Only-green?style=for-the-badge)](#)[![Attention](https://img.shields.io/badge/Attention-Multi%20Head-red?style=for-the-badge)](#)[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

</p>

<p align="center">

A complete implementation of a GPT-style Transformer architecture built from scratch using TensorFlow, featuring custom Multi-Head Self Attention, Positional Encoding, Causal Masking, Hyperparameter Tuning, Text Generation, and Attention Visualization.

</p>

</div>

---

# Table of Contents

- Overview
- Motivation
- Key Features
- Architecture Overview
- Transformer Components
- Project Structure
- Dataset
- Training Pipeline
- Hyperparameter Tuning
- Text Generation
- Attention Visualization
- Installation
- Project Setup
- Running The Project
- Results
- Technologies Used
- Future Improvements
- Learning Outcomes
- References
- License
- Author

---

# Overview

ShakespeareGPT is a decoder-only Transformer Language Model inspired by modern GPT architectures and the groundbreaking paper:

**Attention Is All You Need (Vaswani et al., 2017)**

The project implements the complete Transformer pipeline manually without relying on TensorFlow's built-in MultiHeadAttention layer.

Every major Transformer component has been implemented from scratch, including:

- Scaled Dot Product Attention
- Multi Head Self Attention
- Sinusoidal Positional Encoding
- Causal Attention Masking
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Autoregressive Text Generation
- Hyperparameter Optimization
- Attention Visualization

The model is trained on Shakespeare's literary works and learns to generate text one character at a time in an autoregressive fashion.

Unlike high-level implementations that hide important details behind library abstractions, ShakespeareGPT focuses on understanding how modern Large Language Models work internally.

The objective of this project is not merely to generate text but to provide a practical implementation of the concepts that power systems such as:

- GPT
- GPT-2
- GPT-3
- GPT-4
- Claude
- Gemini
- LLaMA

while maintaining complete transparency over every architectural component.

---

# Motivation

Large Language Models have fundamentally transformed Natural Language Processing and Generative AI.

Modern systems are built upon Transformer architectures that utilize attention mechanisms to learn long-range dependencies and contextual relationships within text.

While many developers use pre-trained models through APIs, relatively few understand how these models operate internally.

The goal of ShakespeareGPT is to bridge that gap by building a complete Transformer architecture from first principles.

This project was developed to answer questions such as:

- How does self-attention actually work?
- Why are Transformers more effective than RNNs and LSTMs?
- How do GPT models generate coherent text?
- What role does positional encoding play?
- Why is causal masking necessary?
- How can attention maps be visualized and interpreted?
- How can hyperparameter tuning improve model performance?

By implementing every major component manually, ShakespeareGPT provides a deeper understanding of modern language modeling systems.

---

# Key Features

## Transformer Architecture

- GPT-Style Decoder-Only Architecture
- Custom Multi-Head Self Attention
- Scaled Dot Product Attention
- Sinusoidal Positional Encoding
- Causal Masking
- Layer Normalization
- Residual Connections
- Feed Forward Networks
- GELU Activation Function

## Training Infrastructure

- TensorFlow Dataset Pipeline
- Dynamic Sequence Generation
- Efficient Data Streaming
- Automatic Checkpoint Saving
- Early Stopping
- TensorBoard Integration
- Validation Monitoring
- Perplexity Tracking

## Hyperparameter Optimization

- Keras Tuner Integration
- Automatic Architecture Search
- Embedding Dimension Optimization
- Transformer Depth Optimization
- Attention Head Optimization
- Feed Forward Dimension Optimization
- Learning Rate Optimization
- Dropout Optimization

## Text Generation

- Prompt-Based Generation
- Temperature Sampling
- Top-K Sampling
- Top-P (Nucleus) Sampling
- Configurable Generation Length
- Autoregressive Decoding

## Visualization

- Attention Heatmap Generation
- Attention Head Inspection
- Layer-wise Attention Analysis
- Transformer Interpretability Tools

---

# Architecture Overview

ShakespeareGPT follows a decoder-only Transformer architecture similar to GPT-style language models.

The model receives a sequence of characters and learns to predict the next character given all previous characters.

```text
  Input Characters
        │
        ▼
Character Embedding
        │
        ▼
Positional Encoding
        │
        ▼
Transformer Block × N
        │
        ▼
Layer Normalization
        │
        ▼
Vocabulary Projection
        │
        ▼
     Softmax
        │
        ▼
Next Character Prediction
```

The architecture is specifically designed for autoregressive language modeling where each prediction depends only on previously observed characters.

---

# Transformer Components

## Character Embeddings

Each input character is mapped into a dense vector representation.

Example:

```text
A → [0.23, -0.81, 0.52, ...]
B → [-0.14, 0.91, -0.44, ...]
C → [0.77, -0.31, 0.12, ...]
```

These learned embeddings allow the model to represent semantic relationships between characters.

---

## Positional Encoding

Transformers process all tokens simultaneously.

Unlike recurrent architectures, they do not inherently understand sequence order.

To solve this issue, ShakespeareGPT uses sinusoidal positional encodings introduced in the original Transformer paper.

Mathematically:

PE(pos,2i)=sin(pos/10000^(2i/d_model))

PE(pos,2i+1)=cos(pos/10000^(2i/d_model))

This allows the model to incorporate positional information while maintaining the ability to generalize to unseen sequence lengths.

---

## Scaled Dot Product Attention

Attention is computed using:

Attention(Q,K,V)=softmax(QKᵀ/√d)V

Where:

- Q = Query Matrix
- K = Key Matrix
- V = Value Matrix
- d = Attention Dimension

This mechanism enables each character to selectively focus on relevant context from previous positions.

---

## Multi Head Self Attention

Rather than learning a single attention distribution, multiple attention heads operate in parallel.

Benefits include:

- Learning multiple contextual relationships simultaneously
- Capturing long-range dependencies
- Improving representation quality
- Better modeling of linguistic structure

Each head learns different aspects of the sequence.

Some heads may focus on punctuation while others focus on speaker names, sentence boundaries, or semantic context.

---

## Causal Masking

Future information must not be visible during training.

Consider:

```text
Input:
TO BE OR N

Target:
O
```

The model should only observe characters before the target.

Causal masks ensure that future positions are hidden by assigning large negative values before the softmax operation.

This guarantees true autoregressive behavior.

---

## Feed Forward Network

Every Transformer block contains a position-wise feed forward network:

```text
   Dense(ff_dim)
        │
        ▼
       GELU
        │
        ▼
   Dense(embed_dim)
```

This network enables non-linear transformations of learned representations and significantly increases the model's expressive power.

---

## Residual Connections

Residual connections help stabilize training and improve gradient flow.

Instead of learning complete transformations, layers learn residual updates:

Output = Layer(x) + x

This allows deeper Transformer architectures to train effectively.

---

## Layer Normalization

Layer normalization stabilizes activations and improves convergence.

It is applied after each residual connection within the Transformer block.

Benefits include:

- Faster convergence
- Improved stability
- Reduced internal covariate shift
- Better gradient propagation

---

# Project Structure

The project follows a modular architecture to ensure maintainability, scalability, and separation of concerns.

```text
ShakespeareGPT/
│
├── configs/
│   └── config.py
│
├── data/
│   └── raw/
│       └── shakespeare.txt
│
├── src/
│   │
│   ├── data/
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   │
│   ├── layers/
│   │   ├── positional_encoding.py
│   │   ├── attention.py
│   │   └── transformer_block.py
│   │
│   ├── models/
│   │   └── bardformer.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── tuning/
│   │   └── tuner.py
│   │
│   ├── inference/
│   │   ├── sampling.py
│   │   └── generate.py
│   │
│   └── visualization/
│       └── attention_visualizer.py
│
├── train.py
├── tune.py
├── generate.py
├── visualize_attention.py
│
├── artifacts/
├── checkpoints/
├── logs/
├── outputs/
├── tuner_results/
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

---

# Module Breakdown

## configs/

Contains all configurable hyperparameters and project settings.

Examples:

- Learning Rate
- Batch Size
- Sequence Length
- Embedding Dimension
- Number of Attention Heads
- Number of Transformer Layers
- Generation Parameters

Centralizing configuration makes experimentation significantly easier.

---

## src/data/

Responsible for all dataset processing operations.

### tokenizer.py

Implements a custom character-level tokenizer.

Responsibilities:

- Vocabulary Creation
- Character-to-Index Mapping
- Index-to-Character Mapping
- Encoding
- Decoding
- Artifact Persistence

Generated Artifacts:

```text
vocabulary.pkl
char2idx.pkl
idx2char.pkl
```

---

### dataset.py

Builds TensorFlow datasets used during training.

Responsibilities:

- Sequence Creation
- Sliding Window Generation
- Dataset Splitting
- Batching
- Prefetching
- Pipeline Optimization

Output:

```python
(train_dataset, validation_dataset)
```

---

## src/layers/

Contains custom Transformer building blocks.

### positional_encoding.py

Implements sinusoidal positional encodings from the original Transformer paper.

Purpose:

Allow the model to understand sequence order.

---

### attention.py

Implements:

- Scaled Dot Product Attention
- Multi Head Self Attention
- Causal Masking

This file forms the core of the Transformer architecture.

---

### transformer_block.py

Defines a complete Transformer block consisting of:

- Multi Head Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections
- Dropout

---

## src/models/

### bardformer.py

Defines the complete GPT-style architecture.

Components:

- Embedding Layer
- Positional Encoding
- Transformer Stack
- Final Projection Layer

This file assembles all lower-level components into a complete language model.

---

## src/training/

### trainer.py

Responsible for model training.

Features:

- Model Compilation
- Training Loop
- Checkpointing
- TensorBoard Logging
- Early Stopping
- Perplexity Calculation
- Progress Reporting

---

## src/tuning/

### tuner.py

Performs hyperparameter optimization.

Search Space Includes:

- Embedding Dimension
- Attention Heads
- Transformer Depth
- Feed Forward Dimension
- Learning Rate
- Dropout

---

## src/inference/

### sampling.py

Implements advanced generation strategies.

Supported Methods:

- Temperature Sampling
- Top-K Sampling
- Top-P Sampling

These techniques improve generation diversity and realism.

---

### generate.py

Responsible for text generation.

Pipeline:

Prompt
↓
Tokenization
↓
Inference
↓
Sampling
↓
Decoding
↓
Generated Text

---

## src/visualization/

### attention_visualizer.py

Provides interpretability tools.

Features:

- Attention Extraction
- Heatmap Generation
- Layer Inspection
- Head Inspection

---

# Dataset

The model is trained on Shakespeare's literary works.

The corpus contains:

- Tragedies
- Comedies
- Historical Plays
- Sonnets
- Character Dialogues
- Monologues

The complete corpus provides a rich source of language patterns suitable for training an autoregressive language model.

---

# Why Character-Level Modeling?

Most modern language models use:

- Word Tokenization
- Subword Tokenization
- Byte Pair Encoding (BPE)

This project intentionally uses character-level tokenization because it provides a deeper understanding of sequence modeling fundamentals.

Advantages:

- Simpler tokenizer implementation
- No Out-of-Vocabulary tokens
- Better understanding of Transformer mechanics
- Full control over preprocessing

Challenges:

- Longer training times
- Longer dependencies
- More difficult learning problem

Despite these challenges, character-level models are excellent educational tools for understanding language modeling.

---

# Dataset Processing Pipeline

The raw corpus undergoes the following transformations:

```text
 Raw Shakespeare Corpus
          │
          ▼
  Character Tokenization
          │
          ▼
  Vocabulary Creation
          │
          ▼
   Character Encoding
          │
          ▼
 Sliding Window Generation
          │
          ▼
  Input / Target Pairs
          │
          ▼
 Train / Validation Split
          │
          ▼
TensorFlow Dataset Pipeline
```

---

# Training Pipeline

The training workflow is fully automated.

Running:

```bash
python train.py
```

executes the following sequence:

```text
Create Project Directories
          │
          ▼
Build Tokenizer Artifacts
          │
          ▼
Load Shakespeare Dataset
          │
          ▼
Create TensorFlow Datasets
          │
          ▼
Build Transformer Model
          │
          ▼
    Compile Model
          │
          ▼
     Train Model
          │
          ▼
Evaluate Validation Metrics
          │
          ▼
  Save Best Checkpoint
          │
          ▼
Generate TensorBoard Logs
```

---

# Training Metrics

The following metrics are monitored during training.

### Training Loss

Measures prediction error on the training dataset.

Lower values indicate better model performance.

---

### Validation Loss

Measures generalization performance on unseen data.

Used for:

- Model Selection
- Early Stopping
- Hyperparameter Evaluation

---

### Accuracy

Character-level prediction accuracy.

Represents the percentage of correctly predicted next characters.

---

### Perplexity

A standard metric used for evaluating language models.

Perplexity is computed as:

```text
Perplexity = exp(loss)
```

Lower perplexity indicates better predictive capability.

---

# Checkpointing Strategy

The best model is automatically saved using validation loss.

Checkpoint Path:

```text
checkpoints/best_model.weights.h5
```

Only the highest-performing checkpoint is preserved.

This prevents storage waste while ensuring the best model remains available for inference.

---

# TensorBoard Integration

Training logs are automatically generated.

Launch TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Available Visualizations:

- Training Loss Curves
- Validation Loss Curves
- Accuracy Curves
- Learning Dynamics
- Weight Histograms

## TensorBoard provides valuable insights into model behavior and training progress.

# Hyperparameter Tuning

ShakespeareGPT includes an automated hyperparameter optimization pipeline powered by Keras Tuner.

The objective is to discover the best-performing Transformer configuration by evaluating multiple architectural combinations.

The tuning process explores different values for:

| Hyperparameter         | Search Space       |
| ---------------------- | ------------------ |
| Embedding Dimension    | 128, 256, 384, 512 |
| Attention Heads        | 4, 8               |
| Transformer Layers     | 4, 6, 8            |
| Feed Forward Dimension | 512, 1024, 2048    |
| Dropout Rate           | 0.1, 0.2, 0.3      |
| Learning Rate          | 1e-4, 5e-4, 1e-3   |

---

## Running Hyperparameter Search

Execute:

```bash
python tune.py
```

The tuning pipeline performs:

```text
 Build Model
      │
      ▼
Train Candidate Model
      │
      ▼
Evaluate Validation Loss
      │
      ▼
Record Results
      │
      ▼
Generate Next Candidate
      │
      ▼
Select Best Configuration
```

The best hyperparameters are automatically reported after the search completes.

---

# Text Generation Pipeline

After training, ShakespeareGPT can generate entirely new Shakespeare-style text.

The generation process is autoregressive.

At every step:

```text
  Previous Context
        │
        ▼
Transformer Prediction
        │
        ▼
  Sampling Strategy
        │
        ▼
   Next Character
        │
        ▼
 Append To Sequence
        │
        ▼
      Repeat
```

The newly generated character becomes part of the context for the next prediction.

---

## Generation Techniques

### Temperature Sampling

Controls randomness during generation.

Low Temperature:

```text
0.3 - 0.7
```

Characteristics:

- Conservative
- Predictable
- Less Creative

---

High Temperature:

```text
1.0 - 1.5
```

Characteristics:

- Creative
- Diverse
- Less Stable

---

Default:

```python
TEMPERATURE = 0.8
```

---

### Top-K Sampling

Restricts sampling to the K most probable candidates.

Example:

```text
Top K = 20
```

Only the 20 most likely characters remain eligible for selection.

Benefits:

- Reduces noise
- Improves coherence
- Prevents unlikely outputs

---

### Top-P (Nucleus) Sampling

Rather than keeping a fixed number of candidates, Top-P keeps the smallest set whose cumulative probability exceeds P.

Example:

```python
TOP_P = 0.9
```

This allows dynamic adaptation depending on prediction confidence.

Benefits:

- More natural text
- Better diversity
- Improved generation quality

---

# Running Text Generation

Execute:

```bash
python generate.py
```

Example Prompt:

```text
KING:
```

Example Output:

```text
KING:
My lord, the heavens shine upon thee this night.
The stars themselves bear witness to thy glory,
And every wind doth whisper of thy noble deeds.
```

Generated text is automatically saved to:

```text
outputs/generated_text.txt
```

---

# Attention Visualization

One of the most powerful aspects of Transformer models is the ability to inspect attention patterns.

ShakespeareGPT includes a complete visualization pipeline for analyzing learned attention distributions.

---

## Why Attention Visualization Matters

Attention heatmaps allow us to understand:

- Which characters influence predictions
- Long-range dependencies
- Context utilization
- Model reasoning behavior

This makes Transformers significantly more interpretable than many other neural architectures.

---

## Running Attention Visualization

Execute:

```bash
python visualize_attention.py
```

Example Prompt:

```text
KING:
```

Output:

```text
outputs/attention_heatmap.png
```

---

## Attention Matrix

For a sequence length of:

```python
256
```

Each attention head produces:

```python
256 × 256
```

attention scores.

For:

```python
8 heads
```

and

```python
6 layers
```

the model learns:

```python
48 independent attention patterns
```

which collectively model contextual relationships.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/7vik2005/ShakespeareGPT.git

cd ShakespeareGPT
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Verify Installation

```bash
python --version
```

Expected:

```text
Python 3.9+
```

---

# Dataset Setup

Download the Shakespeare corpus and place it inside:

```text
data/raw/shakespeare.txt
```

Expected structure:

```text
data/
└── raw/
    └── shakespeare.txt
```

The training pipeline automatically handles:

- Vocabulary Creation
- Character Encoding
- Artifact Generation

No manual preprocessing is required.

---

# Running The Project

## Train Model

```bash
python train.py
```

---

## Hyperparameter Search

```bash
python tune.py
```

---

## Generate Text

```bash
python generate.py
```

---

## Visualize Attention

```bash
python visualize_attention.py
```

---

# Results

Representative results obtained using the default configuration:

| Metric                 | Value       |
| ---------------------- | ----------- |
| Vocabulary Size        | 97          |
| Sequence Length        | 256         |
| Embedding Dimension    | 256         |
| Attention Heads        | 8           |
| Transformer Layers     | 6           |
| Feed Forward Dimension | 1024        |
| Batch Size             | 32          |
| Parameters             | ~12 Million |
| Validation Accuracy    | 53.4%       |
| Validation Loss        | 1.82        |
| Perplexity             | 6.17        |
| Training Epochs        | 20          |

---

## Sample Generated Text

Prompt:

```text
KING:
```

Generated:

```text
KING:
What sayest thou, my noble friend?
The heavens whisper through the silent air,
And all the stars bear witness unto fate.
```

---

# Technologies Used

### Languages

- Python

### Deep Learning

- TensorFlow
- Keras

### Scientific Computing

- NumPy

### Visualization

- Matplotlib
- TensorBoard

### Hyperparameter Optimization

- Keras Tuner

### NLP

- Transformer Architecture
- Attention Mechanisms
- Autoregressive Language Modeling

---

# Future Improvements

Potential future extensions include:

- Byte Pair Encoding (BPE)
- Word-Level Tokenization
- Rotary Positional Embeddings
- Flash Attention
- Mixed Precision Training
- Weight Tying
- Distributed Multi-GPU Training
- Quantization
- Fine-Tuning Support
- LoRA Integration
- Transformer Scaling Experiments
- Instruction Tuning

---

# Learning Outcomes

This project demonstrates practical understanding of:

### Deep Learning

- Neural Networks
- Optimization
- Representation Learning

### Natural Language Processing

- Language Modeling
- Sequence Prediction
- Text Generation

### Transformer Architectures

- Self Attention
- Multi Head Attention
- Positional Encoding
- Decoder-Only Design

### Software Engineering

- Modular Design
- Training Pipelines
- Experiment Management
- Configuration Management

### MLOps Foundations

- Checkpointing
- Logging
- Hyperparameter Optimization
- Model Evaluation

---

# References

## Attention Is All You Need

Ashish Vaswani
Noam Shazeer
Niki Parmar
Jakob Uszkoreit
Llion Jones
Aidan N. Gomez
Łukasz Kaiser
Illia Polosukhin

NeurIPS 2017

https://arxiv.org/abs/1706.03762

---

## TensorFlow Documentation

https://www.tensorflow.org

---

## Keras Tuner Documentation

https://keras.io/keras_tuner

---

# License

This project is licensed under the MIT License.

See the LICENSE file for complete details.

---

# Author

## Satvik Jambagi

AI Engineer • Full Stack Developer • Machine Learning Enthusiast

---

If you found this project useful, consider starring the repository.
