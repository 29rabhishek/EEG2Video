# Quick Start Guide for Inference

This is a condensed reference for running inference with EEG2Video. For detailed explanations, see [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).

## 🚀 Quick Setup

```bash
# 1. Clone and install
git clone https://github.com/XuanhaoLiu/EEG2Video.git
cd EEG2Video
conda create -n eegvideo python=3.8
conda activate eegvideo
pip install -r requirements.txt

# 2. Download Stable Diffusion v1.4 to ./checkpoints/stable-diffusion-v1-4/
# 3. Apply for and download SEED-DV dataset
```

## 📊 Data Preparation (Recommended: Use New Version)

### Option A: Using EEG2Video_New (Recommended)

```bash
# Step 1: Preprocess EEG
cd EEG_preprocessing
python extract_DE_PSD_features_1per1s.py \
    --input_path /path/to/raw_eeg \
    --output_path /path/to/features

# Step 2: Train Semantic Predictor (EEG → Text Embeddings)
cd ../EEG2Video/EEG2Video_New/Semantic
# Edit eeg_text.py with your data paths
python eeg_text.py

# Step 3: Train Seq2Seq Model (EEG → Latents)
cd ../Seq2Seq
python my_autoregressive_transformer.py

# Step 4: Generate DANA Latents (Add Noise)
cd ../DANA
python add_noise.py

# Step 5: Fine-tune Video Diffusion
cd ../Generation
python train_finetune_videodiffusion.py --config configs/all_40_video.yaml
```

### Option B: Using Original EEG2Video

```bash
# Step 1: Preprocess EEG (same as above)

# Step 2: Train Semantic Predictor
cd EEG2Video/models
python train_semantic_predictor.py

# Step 3: Fine-tune Video Diffusion
cd ..
python train_finetune_videodiffusion.py
```

## 🎬 Run Inference

### Using EEG2Video_New (Recommended)

```bash
cd EEG2Video/EEG2Video_New/Generation

# Edit inference_eeg2video.py:
# 1. Set pretrained_eeg_encoder_path to your semantic_predictor.pt
# 2. Set my_model_path to your fine-tuned model
# 3. Set eeg_data_path to your test EEG features
# 4. Set latents paths to your Seq2Seq + DANA outputs

python inference_eeg2video.py
```

### Using Original EEG2Video

```bash
cd EEG2Video

# Edit inference_eeg2video.py with your model and data paths
python inference_eeg2video.py
```

## 📁 Key File Paths

**Input Data**:
- Raw EEG: `*.npy` with shape `(n_subjects, n_trials, 62, n_timepoints)`
- Features: `*.npy` with shape `(n_subjects, n_trials, 62, n_windows, 5)`

**Model Checkpoints**:
- Semantic Predictor: `semantic_predictor.pt` or `eeg2text_40_eeg.pt`
- Seq2Seq Latents: `latents.npy`
- DANA Latents: `latents_add_noise.npy`
- Fine-tuned Model: `./outputs/40_classes_video_200_epoch/unet/`

**Output**:
- Generated videos: `./40_Classes_Fullmodel/*.gif`

## 🎯 Inference Modes

Edit these flags in `inference_eeg2video.py`:

```python
# Full model (best quality)
woSeq2Seq = False
woDANA = False

# Without DANA (no adaptive noise)
woSeq2Seq = False
woDANA = True

# Without Seq2Seq (no latent guidance)
woSeq2Seq = True
```

## ⚙️ Key Parameters

**Training**:
- Semantic Predictor: 200 epochs, lr=5e-4
- Video Diffusion: ~6000 steps (depends on data size), lr=3e-5

**Inference**:
- `video_length`: 6 frames (default)
- `height`: 288, `width`: 512
- `num_inference_steps`: 100 (higher = better quality)
- `guidance_scale`: 12.5 (range: 7.5-15.0)

## 🔧 Troubleshooting

**Out of Memory**:
```python
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
# Reduce batch size or video resolution
```

**Poor Quality**:
- Increase `num_inference_steps` to 150
- Adjust `guidance_scale` (try 10.0-15.0)
- Ensure all models are properly trained

**Shape Errors**:
- Verify EEG features: `(n_samples, 310)` where 310 = 62 × 5
- Check latents: `(n_samples, 4, 6, 36, 64)` for 6 frames

## 📖 Full Documentation

- **[Repository Structure](REPOSITORY_STRUCTURE.md)** - Detailed component descriptions
- **[Inference Guide](INFERENCE_GUIDE.md)** - Complete step-by-step instructions
- **[Main README](README.md)** - Project overview and demos

## 💬 Support

- Issues: [GitHub Issues](https://github.com/XuanhaoLiu/EEG2Video/issues)
- Email: haogram_sjtu@sjtu.edu.cn (Xuanhao Liu)
