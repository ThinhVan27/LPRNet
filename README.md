# LPRNet: License Plate Recognition with Deep Learning

## 👀 Overview
This project implements **LPRNet** (License Plate Recognition Network), inspired from [LPRNet: Liscense Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447) and modified for a downstream task - embedding into an computing device like FPGA. Therefore, it should be a light-weight model, real-time inference and significant accuracy.


## 📦 Installation
```bash
# Clone the repository
git clone <repository_url>
cd LPRNet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start

### Prepare Your Data

The project expects data in the following structure (this dataset is refactored myself from the public dataset in [ICPR 2026 Competition on Low-Resolution License Plate Recognition](https://icpr26lrlpr.github.io/#schedule), with train/test split ratio 0.8/0.2). You can download my custom dataset via script:
```
bash scripts/download_ds.sh
```
Dataset structure:
```
data/
├── train/
│   ├── Scenario-A/
│   │   ├── Brazilian/
│   │   │   ├── track_00002/
│   │   │   │   ├── hr-001.png
│   │   │   │   ├── hr-002.png
│   │   │   │   └── annotations.json
│   │   │   └── ... (more tracks)
│   │   └── Mercosur/
│   │       └── ... (more tracks)
│   └── Scenario-B/
│       ├── Brazilian/
│       └── Mercosur/
└── test/
    ├── Scenario-A/
    │   ├── Brazilian/
    │   └── Mercosur/
    └── Scenario-B/
        ├── Brazilian/
        └── Mercosur/
```

Each track directory should contain:
- **Image files** (`hr-001.png`, `hr-002.png`, etc.) - 5 images per track
- **annotations.json** - Contains the license plate text in JSON format:
  ```json
  {
    "plate_text": "ABC1234"
  }
  ```

### Configure Training Parameters

Edit `config/train_config.yaml` to customize training settings:

```yaml
max_epoch: 500                    # Maximum training epochs
lpr_max_len: 7                    # Maximum license plate length
dropout_rate: 0.5                 # Dropout rate for regularization
train_batch_size: 128             # Training batch size
test_batch_size: 128              # Testing batch size
learning_rate: 0.0001             # Initial learning rate
momentum: 0.95                    # SGD momentum
weight_decay: 0.00005             # L2 regularization
gamma: 0.8                        # Learning rate decay factor
topk: 5                           # Beam search top-k
mode: "beam"                      # Decoding mode ("beam" or "greedy")
cuda: true                        # Enable GPU acceleration
```

### Train the Model

```bash
# Train LPRNet from scratch
python train.py

# The script will:
# - Load training data from the specified directories
# - Initialize network weights using Kaiming initialization
# - Train using SGD optimizer with cosine annealing scheduler
# - Log metrics to Weights & Biases (configure credentials if needed)
# - Save checkpoints at specified intervals
```

### Test the Model

```bash
# Test pre-trained or trained model
python test.py

# Configure test parameters in config/test_config.yaml:
# - pretrained_model: path to model weights
# - test_img_dirs: path to test data
# - cuda: enable GPU acceleration
```

### Post-Training Quantization (Optional)

```bash
# Quantize SmallLPRNet for edge deployment
python post_quan.py

# Note: Full FPGA embedding is out of scope for this project
```


## 📂 Project Structure

```
LPRNet/
├── README.md                          # Project overview
├── PROJECT_DESCRIPTION.md             # This file
├── LICENSE                            # Apache License 2.0
├── requirements.txt                   # Python dependencies
│
├── train.py                           # Training script
├── test.py                            # Testing script
├── post_quan.py                       # Post-training quantization script
├── utils.py                           # Utility functions and parser
│
├── config/                            # Configuration files
│   ├── train_config.yaml              # Training hyperparameters
│   ├── test_config.yaml               # Testing parameters
│   └── quan_config.yaml               # Quantization parameters
│
├── model/                             # Neural network architectures
│   ├── __init__.py
│   ├── LPRNet.py                      # Full LPRNet architecture
│   └── small_LPRNet.py                # Lightweight SmallLPRNet variant
│
├── data/                              # Data handling and preprocessing
│   ├── __init__.py
│   ├── load_data.py                   # Dataset loader and character mapping
│   ├── data_augment.py                # Data augmentation strategies
│   ├── download_data.py               # Data download utilities
│   ├── train/                         # Training dataset
│   │   ├── Scenario-A/
│   │   │   ├── Brazilian/
│   │   │   └── Mercosur/
│   │   └── Scenario-B/
│   │       ├── Brazilian/
│   │       └── Mercosur/
│   └── test/                          # Test dataset
│       ├── Scenario-A/
│       │   ├── Brazilian/
│       │   └── Mercosur/
│       └── Scenario-B/
│           ├── Brazilian/
│           └── Mercosur/
│
├── weights/                           # Pre-trained model weights
│   ├── LPRNet/
│   │   ├── best.pth                   # Best LPRNet weights
│   │   └── last.pth                   # Last LPRNet checkpoint
│   └── small_LPRNet/
│       ├── best.pth                   # Best SmallLPRNet weights
│       └── last.pth                   # Last SmallLPRNet checkpoint
│
└── scripts/                           # Utility scripts
    └── download_ds.sh                 # Dataset download script
```

## 🏗️ Model Architectures

- LPRNet (Full Version): built based-on the paper with some tiny modification.

- SmallLPRNet (Lightweight Version): optimized for futher deployment in FPGA.

## 📈 Training Details

### Loss Function
- Connectionist Temporal Classification loss.
- Applied to predicted logits vs. ground truth character indices

### Optimizer
- **SGD with momentum** (momentum=0.95)
- **Learning rate**: 0.0001 (initial)
- **Weight decay**: 0.00005 (L2 regularization)

### Learning Rate Scheduling
- Cosine annealing with linear warmup
- Decay factor: 0.8

### Data Augmentation
- Random rotations
- Brightness/contrast adjustments
- Denoising and sharpening

## 🔍 Decoding Strategies

### Greedy Decoding
- Selects the character with highest probability at each position
- Fast inference with single forward pass
- Baseline accuracy

### Beam Search Decoding
- Maintains k best hypotheses during decoding
- Top-k parameter: 5
- Improved accuracy at computational cost
- Better handling of ambiguous predictions

## 📊 Evaluation Metrics

- **Sequence Recognition Rate (SRR)**: Exact plate match accuracy

## 📜 License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for full details.


## 📚 Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{lprnet2024,
  title={LPRNet: License Plate Recognition with Deep Learning},
  author={ThinhVan27},
  year={2024},
  publisher={Ho Chi Minh City University of Technology},
  howpublished={\url{https://github.com/your-repo/LPRNet}}
}
```
