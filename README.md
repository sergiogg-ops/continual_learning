# Continual Learning Backend

A PyTorch-based framework for continual learning in neural machine translation (NMT) tasks. This repository implements several continual learning techniques including Elastic Weight Consolidation (EWC) and Knowledge Distillation to mitigate catastrophic forgetting when training models on sequential tasks.

## Features

- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting by regularizing important parameters based on Fisher Information Matrix
- **Knowledge Distillation**: Transfers knowledge from a teacher model to maintain performance on previous tasks
- **Fisher Information Matrix Computation**: Efficient diagonal FIM calculation for EWC
- **Neural Machine Translation**: Built on HuggingFace Transformers for sequence-to-sequence tasks
- **PyTorch Lightning Integration**: Streamlined training with automatic logging and callbacks
- **Flexible Configuration**: YAML-based configuration system for reproducible experiments

## Repository Structure

```
continual_learning/
├── clearn.py                 # Main training script with continual learning losses
├── fim.py                    # Fisher Information Matrix computation
├── train_vgg_mnist.py        # VGG model training on MNIST (alternative task)
├── environment.yaml          # Conda environment specification
├── config/                   # Configuration files for different experiments
│   ├── preliminary.yaml      # Initial training configuration
│   ├── distillation.yaml     # Knowledge distillation settings
│   └── ewc.yaml             # EWC training configuration
├── data/                     # Training and validation data
│   ├── train.{en,es}        # Training data (English/Spanish)
│   └── valid.{en,es}        # Validation data
├── models/                   # Saved models and Fisher matrices
│   └── fisher.pth           # Precomputed Fisher Information Matrix
└── logs/                     # TensorBoard logs and checkpoints
    └── ewc/                 # Experiment-specific logs
```

## Installation

### Using Conda (Recommended)

```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate environment
conda activate clearn
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n clearn python=3.14

# Activate environment
conda activate clearn

# Install dependencies
pip install torch torchvision pytorch-lightning tensorboard transformers
pip install einops sacrebleu ConfigArgParse lxml safetensors

# Dependencies from Helsinki-NLP/opus-mt-es-en
pip install sentencepiece
```

## Quick Start

### 1. Preliminary Training

Train an initial model on the first task:

```bash
python clearn.py --config config/preliminary.yaml \
    --model Helsinki-NLP/opus-mt-es-en 
```

### 2. Compute Fisher Information Matrix

Calculate the Fisher Information Matrix for EWC:

```bash
python fim.py data/train.es data/train.en models/fisher.pth \
    --model logs/preliminary/final_model.pth \
    --batch_size 32 \
    --size 10000
```

### 3. Continual Learning with EWC

Train on a new task while preserving knowledge from the previous task:

```bash
python clearn.py --config config/ewc.yaml 
```

## Configuration Options

All training parameters can be specified via command-line arguments or YAML config files:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path or name of pretrained model | None |
| `--teacher` | Path to teacher model for distillation | None |
| `--fim` | Path to Fisher Information Matrix for EWC | None |
| `--distillation` | Knowledge distillation loss weight | 0.0 |
| `--ewc` | EWC regularization strength (λ) | 0.0 |
| `--batch_size` | Training batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--max_epochs` | Maximum training epochs | None |
| `--max_steps` | Maximum training steps | -1 |
| `--size` | Number of training samples to use | -1 (all) |
| `--early_stop` | Enable early stopping | False |
| `--patience` | Early stopping patience | 2 |
| `--seed` | Random seed | 42 |

### Example Configuration Files

**config/ewc.yaml**:
```yaml
model: Helsinki-NLP/opus-mt-es-en
teacher: Helsinki-NLP/opus-mt-es-en
fim: models/fisher.pth
ewc: 2000              # EWC strength
distillation: 0.1      # Distillation weight
batch_size: 16
lr: 1e-5
max_steps: 20000
early_stop: true
```

## Implementation Details

### Continual Learning Loss

The total loss combines three components:

```python
L_total = L_CE + (λ_EWC / 2) * L_EWC + λ_distill * L_KD
```

Where:
- **L_CE**: Cross-entropy loss on current task
- **L_EWC**: EWC regularization term (∑ F_i * (θ_i - θ*_i)²)
- **L_KD**: KL divergence between student and teacher predictions

### Fisher Information Matrix

The diagonal of the empirical Fisher Information Matrix is computed as:

```python
F_i = E[(∂log P(y|x; θ) / ∂θ_i)²]
```

This approximates parameter importance for the previous task.

## Features by Script

### `clearn.py`
- Main training loop with PyTorch Lightning
- Custom loss function combining CE, EWC, and distillation
- Translation data loading and preprocessing
- BLEU score evaluation
- TensorBoard logging

### `fim.py`
- Efficient diagonal FIM computation
- Batch-wise gradient accumulation
- Progress tracking with tqdm
- Supports variable dataset sizes

### `train_vgg_mnist.py`
- Alternative computer vision task
- VGG model on MNIST dataset
- HuggingFace Trainer integration

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Metrics tracked:
- `train_loss`: Training loss (total)
- `val_loss`: Validation cross-entropy loss
- `val_BLEU`: BLEU score on validation set

## Data Format

The code supports two data formats:

1. **Moses Format**: Plain text files with one sentence per line
   ```
   Hello world
   This is a test
   ```

2. **XML Format**: TMX-style XML with DOC and SEG tags
   ```xml
   <DOC>
     <SEG>Hello world</SEG>
     <SEG>This is a test</SEG>
   </DOC>
   ```

## Advanced Usage

### Custom Model Architecture

Replace the model loading function in `clearn.py`:

```python
def load_model(path):
    model = YourCustomModel.from_pretrained(path)
    tok = YourCustomTokenizer.from_pretrained(path)
    return model, tok
```

### Multiple Sequential Tasks

For training on multiple tasks sequentially:

```bash
# Task 1
python clearn.py --config config/task1.yaml --experiment task1

# Compute FIM for task 1
python fim.py data/task1_train.src data/task1_train.tgt models/fisher_task1.pth \
    --model logs/task1/final_model.pth

# Task 2 with EWC
python clearn.py --config config/task2.yaml \
    --model logs/task1/final_model.pth \
    --fim models/fisher_task1.pth \
    --ewc 1000 \
    --experiment task2
```

[comment]: # (## Citation)


## References

- Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks." PNAS.
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network." arXiv.

## License

This project is available for academic and research purposes.

## Troubleshooting

### Common Issues

**Issue**: KeyError with tokenizer inputs
- **Solution**: Ensure you're passing properly formatted tokenizer outputs to the model. Use `**tok_inputs.to(device)` instead of passing the BatchEncoding object directly.

**Issue**: Out of memory during training
- **Solution**: Reduce `batch_size` or increase `accum_steps` for gradient accumulation.

**Issue**: Fisher matrix computation is slow
- **Solution**: Reduce `--size` parameter to use fewer samples for FIM approximation.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
