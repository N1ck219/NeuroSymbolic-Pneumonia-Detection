# A Neuro-Symbolic Hybrid Architecture for Robust Pneumonia Detection 

![Architecture Diagram](assets/schema_rete.png)

## Overview
This repository contains the official implementation of the project **"A Neuro-Symbolic Hybrid Architecture for Robust Pneumonia Detection and Localization in Chest X-rays"**.
The system is designed to tackle visual localization and image recognition challenges in medical environments, particularly focusing on pneumonia detection from Chest X-rays (CXRs) utilizing the RSNA Pneumonia dataset.

It proposes a novel hybrid architecture that combines the intrinsic noise-suppression capabilities of **Cellular Neural Networks (CeNNs)**, the deep feature extraction of a **DenseNet-121** backbone, and the explicit medical reasoning of an **Answer Set Programming (ASP)** logic module.

## Key Features
- **Cellular Neural Network (CeNN) Frontend**: A bio-inspired Continuous-Time dynamic layer ensuring structural resilience against extreme Gaussian noise (up to 90%) where standard CNNs collapse.
- **Multi-Task Learning (MTL)**: Simultaneous pathology classification and bounding box regression (localization) through a shared feature extractor.
- **CBAM Attention**: Integrated Convolutional Block Attention Module to focus on salient lung opacity regions and suppress background artifacts.
- **Neuro-Symbolic Semantic Validation (ASP)**: Post-inference processing using `clingo` to enforce anatomical and geometric rules. It acts as a safety-first "logic filter", identifying "hallucinations" (reducing False Positives) and rescuing sub-threshold clinical cases (reducing False Negatives).

## Project Structure
- `code/models.py`: Defines the Neural Network architectures (CeNN frontend, DenseNet backbone, CBAM, Classification & Regression heads).
- `code/train.py`: Handles the multi-task training loop, incorporating weighted sampling, layer freezing, and gradient accumulation.
- `code/run_neurosymbolic.py`: Core script to execute the DL inference and pass the statistical output to the ASP logic solver.
- `code/asp_solver.py` & `code/logic_rules/`: Contains the clingo logic integration and the declarative constraints (`.lp` files) for semantic validation (e.g., "Smart Heart" and "Organic shape" rules).
- `code/dataset.py`: Handles dataset processing, bounded box scaling, and Albumentations transformations (including noise injection).
- `code/profile_hardware.py`: Benchmarks inference speed, hardware complexity (M parameters, FLOPs), and VRAM usage.
- `code/test_all_noise.py`: Generates the degradation curve of the proposed model versus baselines strictly against Gaussian Noise.
- `code/settings.py`: Centralized configuration for device mapping, explicit paths, and hyper-parameters.
- `latex_code/`: LaTeX source files and figures for the detailed academic report.

## Setup and Installation

### Requirements
The project requires a Python environment and several deep learning packages, including PyTorch, `torchxrayvision` (for medical pre-trained weights extraction), and the `clingo` module for the ASP logic solver.

Install dependencies via:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
1. Download the **RSNA Pneumonia Detection Challenge** dataset from Kaggle.
2. Update the `CSV_PATH` and `TRAIN_IMG_PATH` variables inside `code/settings.py` to point to your local `stage_2_train_labels.csv` file and `stage_2_train_images` folder respectively.
3. Pre-trained Weights: Due to GitHub's file size limits, the trained .pth files are hosted externally. Download the pre-trained models from this [OneDrive Repository](https://aauklagenfurt-my.sharepoint.com/:f:/g/personal/n1flego_edu_aau_at/IgDaPeY5viWtRZkUomqMzEMnAbtf6HHt5YgZ103JDwj8bAA?e=We7fMm) and place them inside the trained_models/ directory before running inference scripts.

### Usage

### 1. Training the Model
To start the multi-task training procedure:
```bash
cd code
python train.py
```
This handles class balancing, initial partial model freezing for transfer learning, and gradient accumulation suited for running on consumer GPUs.

### 2. Neuro-Symbolic Inference
Run the hybrid pipeline (Deep Learning + ASP logic gating) to evaluate images and generate confusion matrices. You can test different rule configurations (e.g., `rules_v20.lp` or `rules_v29.lp`) by editing the called file path inside `run_neurosymbolic.py`.
```bash
python run_neurosymbolic.py
```

### 3. Noise Robustness Testing
Assess model degradation under different scales of additive Gaussian noise (from 0% up to 90%). This script directly compares the proposed Hybrid CeNN against standard baselines (ResNet50, EfficientNet-B0, SE-ResNet50) and an Ablation Model.
```bash
python test_all_noise.py
```

### 4. Hardware Profiling
Generate metrics related to throughput, parameter count, and FLOPs.
```bash
python profile_hardware.py
```
The resulting comparison matrix will be saved in `results/hardware_profiling`.

## License
Provided under the MIT License (see `LICENSE` file for details).

## Acknowledgments
Author: **Nicola Flego**

*Based on the academic paper co-authored with Filippo Pilutti, Mohamed El Bahnasawi, and Kyandoghere Kyamakya (University of Klagenfurt).*
