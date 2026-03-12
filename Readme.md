# FetMRIE: Unsupervised Fetal Brain MRI Image Enhancement with Anatomical Fidelity via Adaptive State-Matching Denoising Diffusion
**Official Implementation of MICCAI 2026 Paper Submission**
## Overview
This repository contains the implementation of FetMRIE, a fully unsupervised framework for enhancing fetal brain MRI images. Our method employs a two-stage pipeline combining latent-space denoising with adaptive diffusion-based structural restoration to achieve superior tissue contrast enhancement while preserving anatomical fidelity.
## Key Features
- **Fully Unsupervised**: No paired training data required
- **Two-Stage Framework**: VAE initial denoising + Diffusion-based structural restoration  
- **Adaptive State Matching**: Case-specific noise level adaptation during inference
- **Anatomical Preservation**: Maintains cortical folds and tissue boundaries
  
## Code Availability
The complete source code has been provided. Some of them are originally from MONAI, we deeply appreciate it!

## Pre-trained Weights
Pre-trained model weights can be provided upon paper acceptance. 
