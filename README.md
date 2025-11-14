# GPT-2 Scratch Implementation

A from-scratch implementation of a GPT-2 (Generative Pre-trained Transformer) model with training, checkpointing, and text generation capabilities.

## Overview

This project implements a GPT-2 transformer model from scratch using PyTorch. The implementation includes:
- Full transformer architecture with causal self-attention
- Training loop with automatic checkpointing
- Periodic snapshot saving during training
- Automatic training termination when target loss is reached
- Text generation using the trained model

## Features

### Model Architecture
- **Causal Self-Attention**: Multi-head attention with causal masking
- **Transformer Blocks**: Layer normalization, attention, and MLP layers
- **Position Embeddings**: Learnable positional encodings
- **Weight Sharing**: Token embeddings shared with output projection

### Training Features
- **Automatic Snapshot Saving**: Periodic checkpoints saved every N steps
- **Target Loss Training**: Training automatically stops when loss reaches 0.099999
- **Final Model Saving**: Complete model checkpoint saved when target is reached
- **Progress Tracking**: Real-time loss monitoring and step counting

### Generation Features
- **Top-k Sampling**: Configurable top-k token sampling (default: 50)
- **Text Generation**: Generate multiple sequences from sample input
- **Configurable Length**: Adjustable maximum generation length

## Requirements

- Python 3.7+
- PyTorch (CPU, CUDA, or MPS supported)
- tiktoken (for GPT-2 tokenization)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpt2_scratch
```

2. Install dependencies:
```bash
pip install torch tiktoken transformers
```

3. Prepare your training data:
   - Create a file named `input.txt` in the project root
   - Add your training text data to this file

## Usage

### Basic Training

Simply run the training script:
```bash
python train.py
```

The script will:
1. Load and tokenize data from `input.txt`
2. Initialize a GPT model with default configuration
3. Train until loss reaches 0.099999
4. Save periodic snapshots in `snapshots/` directory
5. Save final model as `final_model.pt`
6. Generate sample predictions using the trained model

### Training Configuration

You can modify the following parameters in `train.py`:

```python
# Model configuration (GPTConfig)
block_size = 1024      # Maximum sequence length
vocab_size = 50257     # Vocabulary size (GPT-2 standard)
n_layer = 12          # Number of transformer layers
n_head = 12           # Number of attention heads
n_embd = 768          # Embedding dimension

# Training configuration
B = 4                 # Batch size
T = 32                # Sequence length
lr = 3e-4             # Learning rate
target_loss = 0.099999  # Target loss to stop training
snapshot_interval = 10   # Save snapshot every N steps

# Generation configuration
num_return_sequences = 5  # Number of sequences to generate
max_length = 30           # Maximum generation length
```

### Model Architecture

The model follows the GPT-2 architecture:

- **Embedding Layer**: Token and position embeddings
- **Transformer Blocks**: 12 layers (configurable) with:
  - Layer normalization
  - Causal self-attention (12 heads)
  - MLP with GELU activation
- **Output Layer**: Linear projection to vocabulary size

### Training Process

1. **Data Loading**: 
   - Text is loaded from `input.txt`
   - Tokenized using GPT-2 tokenizer (tiktoken)
   - Batched into sequences of length T

2. **Training Loop**:
   - Forward pass through the model
   - Loss calculation (cross-entropy)
   - Backward pass and optimization
   - Periodic snapshot saving
   - Loss monitoring

3. **Checkpointing**:
   - Snapshots saved to `snapshots/snapshot_step_N.pt`
   - Each snapshot contains:
     - Model state dict
     - Optimizer state dict
     - Step number
     - Current loss

4. **Final Model**:
   - Saved as `final_model.pt` when target loss is reached
   - Contains model, optimizer, config, and metadata

### Text Generation

After training completes, the script automatically:
1. Loads the final model
2. Encodes a sample input text
3. Generates text using top-k sampling
4. Decodes and prints the generated sequences

## File Structure

```
gpt2_scratch/
├── train.py              # Main training script
├── input.txt             # Training data (required)
├── README.md             # This file
├── snapshots/            # Periodic checkpoints (created during training)
│   ├── snapshot_step_10.pt
│   ├── snapshot_step_20.pt
│   └── ...
└── final_model.pt        # Final trained model (created when target loss reached)
```

## Model Details

### Default Configuration
- **Parameters**: ~124M (GPT-2 small equivalent)
- **Layers**: 12
- **Heads**: 12
- **Embedding Dimension**: 768
- **Vocabulary Size**: 50,257
- **Block Size**: 1024

### Weight Initialization
- Standard normal initialization (std=0.02)
- Special scaling for residual connections
- Proper initialization for attention and MLP projections

## Customization

### Using Your Own Data
1. Replace `input.txt` with your training data
2. Ensure the file is readable and contains text

### Changing Sample Input
Modify the sample text in the prediction section:
```python
sample_text = "Your custom input text here"
```

### Loading Pretrained Models
The code includes support for loading GPT-2 pretrained weights:
```python
model = GPT.from_pretrained('gpt2')  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
```

### Resuming Training
To resume from a checkpoint:
```python
checkpoint = torch.load('snapshots/snapshot_step_N.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Device Support

The script automatically detects and uses the best available device:
- **CUDA**: If NVIDIA GPU is available
- **MPS**: If Apple Silicon GPU is available
- **CPU**: Fallback option

## Notes

- Training will continue indefinitely until the target loss (0.099999) is reached
- Snapshots are saved periodically to prevent data loss
- The model uses GPT-2 tokenization (tiktoken encoding)
- Random seeds are set for reproducibility (1337 for training, 42 for generation)

## Troubleshooting

### Out of Memory
- Reduce batch size (`B`) or sequence length (`T`)
- Use a smaller model (reduce `n_layer`, `n_embd`)

### Training Too Slow
- Use GPU if available (CUDA or MPS)
- Increase batch size if memory allows
- Reduce sequence length

### Loss Not Decreasing
- Check learning rate (try different values)
- Verify training data quality
- Ensure data is properly formatted

## License

[Add your license here]

## Acknowledgments

This implementation is inspired by:
- OpenAI's GPT-2 architecture
- Andrej Karpathy's nanoGPT
- The transformers library

