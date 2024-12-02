# MNIST Classification with CI/CD Pipeline

[![ML Pipeline](https://github.com/althafkk/S5MNISTCICDCLASS/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/althafkk/S5MNISTCICDCLASS/actions/workflows/ml-pipeline.yml)

A deep learning project that implements a CNN model for MNIST digit classification with automated testing and CI/CD pipeline integration.

## Project Structure

project/
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions workflow configuration
├── model.py # Neural network model architecture
├── train.py # Training script
├── test_model.py # Model testing and validation
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules

## Model Architecture
- 3-layer CNN with batch normalization and dropout
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: < 25,000
- Target accuracy: > 95%

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest
- numpy

## Setup and Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training
Run the training script:

```bash
python train.py
```

Training Configuration:
- Batch size: 16
- Learning rate: 0.008
- Optimizer: AdamW with weight decay 1e-5
- Data augmentation: minimal rotation (2°) and translation (0.02)
- Label smoothing: 0.01

### Testing
Run the test suite:

```bash
python test_model.py
```

The test script validates:
1. Model parameter count (< 25,000)
2. Model accuracy (> 95%)

## Model Files
Models are saved with the format:
```
model_mnist_[TIMESTAMP]_epoch[N]_acc[XX.X].pth
```
- TIMESTAMP: YYYYMMDD_HHMMSS
- N: Epoch number
- XX.X: Accuracy percentage

## CI/CD Pipeline
The GitHub Actions workflow:
1. Sets up Python environment
2. Installs dependencies
3. Trains model
4. Runs validation tests
5. Archives model artifact

## Notes
- CPU-based training
- Automated dataset download
- Real-time training/testing output
- Model files excluded from git

## Troubleshooting
1. For test output visibility:

```bash
pytest test_model.py -v -s
```

2. For accuracy issues:
- Verify training parameters
- Check data augmentation settings
- Run multiple training iterations

3. For parameter count issues:
- Review model architecture
- Check layer dimensions

This README provides:
1. Project overview
2. Setup instructions
3. Usage guidelines
4. File structure
5. Important configurations
6. Troubleshooting tips

