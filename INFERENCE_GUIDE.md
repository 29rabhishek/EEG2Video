# EEG2Video Inference Guide

This guide provides step-by-step instructions for preparing data and running inference with the EEG2Video framework.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation Pipeline](#data-preparation-pipeline)
3. [Model Training Pipeline](#model-training-pipeline)
4. [Inference Pipeline](#inference-pipeline)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/XuanhaoLiu/EEG2Video.git
cd EEG2Video

# Create conda environment
conda create -n eegvideo python=3.8
conda activate eegvideo

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Acquisition

1. Fill out the SEED-DV [License file](https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc)
2. [Apply](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/) for the dataset
3. Download the SEED-DV dataset after approval
4. Download pre-trained Stable Diffusion v1.4 model to `./checkpoints/stable-diffusion-v1-4/`

### 3. Expected Dataset Structure

```
SEED-DV/
├── raw_eeg/           # Raw EEG signals (62 channels @ 200Hz)
├── videos/            # Stimulus videos (40 classes)
└── metadata/          # Session and trial information
```

## Data Preparation Pipeline

### Step 1: Segment Raw EEG Signals

**Script**: `EEG_preprocessing/segment_raw_signals_200Hz.py`

**Purpose**: Segment continuous EEG recordings into trials aligned with video stimuli.

**Input**: 
- Raw EEG data files (`.mat` or `.npy` format)
- 62 channels sampled at 200Hz

**Output**:
- Segmented EEG trials for each video clip

**Usage**:
```bash
cd EEG_preprocessing
python segment_raw_signals_200Hz.py \
    --input_path /path/to/raw_eeg \
    --output_path /path/to/segmented_eeg \
    --sampling_rate 200
```

**Expected Output Format**:
- Shape: `(num_subjects, num_trials, num_channels, num_timepoints)`
- Example: `(7, 40, 62, 1200)` for 7 subjects, 40 videos, 62 channels, 6 seconds @ 200Hz

### Step 2: Extract DE/PSD Features

**Script**: `EEG_preprocessing/extract_DE_PSD_features_1per1s.py`

**Purpose**: Extract frequency-domain features (Differential Entropy and Power Spectral Density) from segmented EEG.

**Feature Bands**:
- Delta (1-4 Hz)
- Theta (4-8 Hz)
- Alpha (8-14 Hz)
- Beta (14-31 Hz)
- Gamma (31-50 Hz)

**Input**: Segmented EEG signals
**Output**: DE/PSD features

**Usage**:
```bash
python extract_DE_PSD_features_1per1s.py \
    --input_path /path/to/segmented_eeg \
    --output_path /path/to/de_psd_features \
    --window_size 1
```

**Expected Output Format**:
- Shape: `(num_subjects, num_trials, num_channels, num_windows, num_bands)`
- Example: `(7, 40, 62, 5, 5)` for 5 frequency bands across 5 time windows (1s each)

**Note**: For different temporal resolutions, use `extract_DE_PSD_features_1per2s.py`

### Step 3: Prepare Text Embeddings

**Purpose**: Generate CLIP text embeddings for video captions to train the semantic predictor.

**Input**: Video captions from `dataset/BLIP/`

**Process**:
```python
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# Load CLIP model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Process captions
captions = []  # Load from dataset/BLIP/*.txt
inputs = tokenizer(captions, padding=True, return_tensors="pt")
text_embeddings = text_model(**inputs).last_hidden_state

# Save embeddings
np.save("text_embeddings.npy", text_embeddings.cpu().numpy())
```

**Expected Output**:
- Shape: `(num_videos, 77, 768)` - CLIP text embedding format

## Model Training Pipeline

### Step 1: Train Semantic Predictor (EEG → Text Embeddings)

**Script**: `EEG2Video/EEG2Video_New/Semantic/eeg_text.py`

**Purpose**: Train a model to map EEG features to CLIP text embedding space.

**Configuration**:
```python
# Edit paths in eeg_text.py
eeg_data_path = "/path/to/de_psd_features/sub1.npy"
text_embedding_path = "/path/to/text_embeddings.npy"
```

**Usage**:
```bash
cd EEG2Video/EEG2Video_New/Semantic
python eeg_text.py
```

**Training Details**:
- Architecture: MLP (310 → 10000 → 10000 → 10000 → 10000 → 59136)
- Input: 310-dim (62 channels × 5 frequency bands, averaged over time)
- Output: 59136-dim (77 × 768, CLIP text embedding)
- Loss: MSE between predicted and ground truth text embeddings
- Epochs: 200
- Optimizer: Adam (lr=5e-4)
- Scheduler: CosineAnnealing

**Expected Output**:
- `semantic_predictor.pt` - Trained model checkpoint

### Step 2: Train Seq2Seq Latent Predictor

**Script**: `EEG2Video/EEG2Video_New/Seq2Seq/my_autoregressive_transformer.py`

**Purpose**: Train a transformer to predict video latent sequences from EEG.

**Input**:
- EEG features (DE/PSD)
- Video latent codes (extracted from videos using VAE encoder)

**Architecture**:
- Transformer-based autoregressive model
- Predicts latent codes frame by frame

**Usage**:
```bash
cd EEG2Video/EEG2Video_New/Seq2Seq
python my_autoregressive_transformer.py \
    --eeg_path /path/to/de_psd_features \
    --video_latents_path /path/to/video_latents
```

**Expected Output**:
- `seq2seq_model.pt` - Trained Seq2Seq model checkpoint
- `latents.npy` - Predicted latent codes without noise

### Step 3: Generate DANA Latents

**Script**: `EEG2Video/EEG2Video_New/DANA/add_noise.py`

**Purpose**: Add adaptive noise to latent codes using diffusion process.

**Input**: Latents from Seq2Seq model

**Usage**:
```bash
cd EEG2Video/EEG2Video_New/DANA
python add_noise.py \
    --input_latents /path/to/latents.npy \
    --output_latents /path/to/latents_add_noise.npy
```

**Expected Output**:
- `latents_add_noise.npy` - Latent codes with adaptive noise

### Step 4: Fine-tune Video Diffusion Model

**Script**: `EEG2Video/EEG2Video_New/Generation/train_finetune_videodiffusion.py`

**Purpose**: Fine-tune Stable Diffusion for EEG-conditioned video generation.

**Configuration**: Edit `configs/all_40_video.yaml`

```yaml
pretrained_model_path: "./checkpoints/stable-diffusion-v1-4"
output_dir: "./outputs/40_classes_video_200_epoch"

train_data:
  video_path: "path/to/training/videos"
  n_sample_frames: 6
  width: 512
  height: 288
  sample_frame_rate: 2

learning_rate: 3e-5
train_batch_size: 10
max_train_steps: 6000  # Adjust based on dataset size
trainable_modules:
  - "attn1.to_q"
  - "attn2.to_q"
  - "attn_temp"
```

**Usage**:
```bash
cd EEG2Video/EEG2Video_New/Generation
python train_finetune_videodiffusion.py \
    --config configs/all_40_video.yaml
```

**Training Details**:
- Base model: Stable Diffusion v1.4
- Architecture: UNet3D with temporal attention
- Fine-tuned modules: Cross-attention query projections and temporal attention
- Mixed precision: FP16
- Memory optimization: Gradient checkpointing, xformers attention

**Expected Output**:
- Fine-tuned UNet model in `./outputs/40_classes_video_200_epoch/unet/`

## Inference Pipeline

### Quick Start Inference

**Script**: `EEG2Video/EEG2Video_New/Generation/inference_eeg2video.py`

**Purpose**: Generate videos from test EEG data.

### Step-by-Step Inference Process

#### 1. Prepare Test EEG Data

```python
# Load and preprocess test EEG
eeg_data_path = "/path/to/test_eeg_features.npy"
eegdata = np.load(eeg_data_path)

# Shape: (num_subjects, num_trials, num_channels, num_windows, num_bands)
# Example: (7, 40, 62, 5, 5)
```

#### 2. Configure Inference Paths

Edit `inference_eeg2video.py`:

```python
# Model paths
pretrained_eeg_encoder_path = '/path/to/semantic_predictor.pt'
pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/40_classes_video_200_epoch"

# Latent paths
latents_add_noise = np.load('./models/latents_add_noise.npy')  # With DANA
latents = np.load('./models/latents.npy')  # Without DANA

# Data path
eeg_data_path = "/path/to/test_eeg_features.npy"
```

#### 3. Run Inference

```bash
cd EEG2Video/EEG2Video_New/Generation
python inference_eeg2video.py
```

#### 4. Inference Modes

The script supports three inference modes:

**Full Model (Recommended)**:
```python
woSeq2Seq = False
woDANA = False
# Uses both Seq2Seq predicted latents and DANA noise addition
```

**Without DANA**:
```python
woSeq2Seq = False
woDANA = True
# Uses Seq2Seq latents but no adaptive noise
```

**Without Seq2Seq (No Initial Latents)**:
```python
woSeq2Seq = True
# Uses only semantic predictor, no latent guidance
```

#### 5. Output

Generated videos are saved as GIF files:
- Full model: `./40_Classes_Fullmodel/{i}.gif`
- Without DANA: `./40_Classes_woDANA/{i}.gif`
- Without Seq2Seq: `./40_Classes_woSeq2Seq/{i}.gif`

### Inference Parameters

Key parameters in the inference pipeline:

```python
video = pipe(
    model,                      # Semantic predictor model
    eeg_test[i:i+1,...],       # Test EEG features
    latents=latents_add_noise,  # Initial latent codes (optional)
    video_length=6,             # Number of frames (default: 6)
    height=288,                 # Video height
    width=512,                  # Video width
    num_inference_steps=100,    # Diffusion steps (higher = better quality)
    guidance_scale=12.5         # Classifier-free guidance scale
).videos
```

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                          │
└─────────────────────────────────────────────────────────────┘
1. Raw EEG (62 channels @ 200Hz)
   ↓ [segment_raw_signals_200Hz.py]
2. Segmented EEG trials
   ↓ [extract_DE_PSD_features_1per1s.py]
3. DE/PSD Features (62 × 5 × 5)
   ↓ [Average over time]
4. Flattened Features (310-dim)

┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                            │
└─────────────────────────────────────────────────────────────┘
5. Train Semantic Predictor (EEG → Text Embeddings)
   ↓ [Semantic/eeg_text.py]
6. Train Seq2Seq Model (EEG → Latent Codes)
   ↓ [Seq2Seq/my_autoregressive_transformer.py]
7. Generate DANA Latents (Add Adaptive Noise)
   ↓ [DANA/add_noise.py]
8. Fine-tune Video Diffusion Model
   ↓ [Generation/train_finetune_videodiffusion.py]

┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE                                 │
└─────────────────────────────────────────────────────────────┘
9. Load Test EEG → Semantic Predictor → Text Embeddings
   + Load Pretrained Latents (from Seq2Seq + DANA)
   ↓ [Generation/inference_eeg2video.py]
10. Video Diffusion Model → Generated Videos (GIF)
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Solutions**:
- Enable xformers: `pipe.enable_xformers_memory_efficient_attention()`
- Enable VAE slicing: `pipe.enable_vae_slicing()`
- Reduce batch size in training
- Use gradient checkpointing
- Use mixed precision (FP16)

#### 2. Poor Reconstruction Quality

**Check**:
- Are all three components (Semantic, Seq2Seq, DANA) properly trained?
- Are the pretrained latents from the correct test subject?
- Try increasing `num_inference_steps` (e.g., 100 → 150)
- Adjust `guidance_scale` (try 7.5 - 15.0)

#### 3. Shape Mismatch Errors

**Verify**:
- EEG data shape: `(num_subjects, num_trials, 62, num_windows, 5)`
- After averaging: `(num_samples, 310)` where 310 = 62 × 5
- Text embeddings: `(num_samples, 77, 768)`
- Latent codes: `(num_samples, 4, 6, 36, 64)` for 6 frames at 288×512

#### 4. Missing Pretrained Models

**Download Required Models**:
- Stable Diffusion v1.4: Download from HuggingFace
- CLIP models: Auto-downloaded by transformers library
- Pretrained semantic predictor and latents: Train following this guide or request from authors

### Performance Optimization

**For Faster Inference**:
- Reduce `num_inference_steps` (but may decrease quality)
- Use smaller video resolution (e.g., 256×256)
- Process fewer frames (e.g., 4 instead of 6)
- Enable model compilation (PyTorch 2.0+)

**For Better Quality**:
- Increase `num_inference_steps` (up to 150)
- Use full precision (FP32) if memory allows
- Train models for more epochs
- Use larger batch sizes during training
- Fine-tune on more diverse data

## Additional Resources

- **Paper**: [EEG2Video: Towards Decoding Dynamic Visual Perception from EEG Signals](https://nips.cc/virtual/2024/poster/95156)
- **Project Website**: [https://bcmi.sjtu.edu.cn/home/eeg2video/](https://bcmi.sjtu.edu.cn/home/eeg2video/)
- **Dataset Application**: [https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/)

## Contact

For questions or issues:
- **Xuanhao Liu**: haogram_sjtu@sjtu.edu.cn
- **Tianyi Zhou**: 213212387@seu.edu.cn
