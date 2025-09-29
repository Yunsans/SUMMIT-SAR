# SUMMIT: A SAR Foundation Model with Multiple Auxiliary Tasks Enhanced Intrinsic Characteristics
[SUMMIT: A SAR Foundation Model with Multiple Auxiliary Tasks Enhanced Intrinsic Characteristics](https://doi.org/10.1016/j.jag.2025.104624)
## Overview 
This repository hosts the official implementation of SUMMIT, a state-of-the-art (SOTA) foundation model tailored for Synthetic Aperture Radar (SAR) image understanding. Proposed in the paper "SUMMIT: A SAR foundation model with multiple auxiliary tasks enhanced intrinsic characteristics" (published in International Journal of Applied Earth Observation and Geoinformation, 2025), SUMMIT addresses the limitations of existing deep learning methods in SAR processing—such as neglecting SAR’s intrinsic physical properties and poor cross-task generalization—by integrating self-supervised auxiliary tasks and SAR-specific prior knowledge.

## Key Contributions
1. Large-Scale SAR Dataset (MuSID)Constructed the Multi-sensor SAR Image Dataset (MuSID) with over 560,000 SAR images, covering diverse scenarios (aircraft, ships, bridges, harbors), resolutions (0.1–25 m), and sensors (Gaofen-3, Sentinel-1, TerraSARX, etc.). It supports large-scale self-supervised pre-training for SAR foundation models.

2. Multi-Auxiliary-Task Pre-Training FrameworkProposed three self-supervised auxiliary tasks (SSATs) to enhance SAR feature learning: Masked Image Modeling (MIM): Learns robust structural representations of SAR images. Self-Supervised Denoising: Mitigates speckle noise (a unique artifact of SAR imaging) and improves noise resistance. Spatial Scattering Feature (SSF) Enhancement: Preserves geometric consistency by extracting edge features (via Canny algorithm) and scattering point features (via Harris corner detection).

3. Auxiliary Task Coordination Module (ATCM)Designed ATCM to dynamically balance and fuse the three auxiliary tasks. Unlike simple task aggregation, ATCM aligns each task with the optimal stage of the learning process (e.g., denoising at input level, edge reconstruction at output level), ensuring effective integration of SAR physical properties into feature learning.

## Model Architecture

![SUMMIT Architecture](overall.pdf "SUMMIT Framework")

SUMMIT is built on a Vision Transformer (ViT). Pre-Training StageInput: MuSID dataset (448×448 resized images). Process: ATCM coordinates MIM, denoising, and SSF enhancement tasks. The shared ViT encoder learns SAR-specific features, with a decoder optimizing multi-task reconstruction loss.

## Environment Setup
```bash
conda create -n summit python=3.8
conda activate summit
pip install -r requirements.txt
