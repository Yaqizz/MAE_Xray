# Self-Supervised Learning with Masked Autoencoders (MAE)

**PyTorch Lightning + Vision Transformer (ViT)**

This project implements a self-supervised image reconstruction pipeline using Masked Autoencoders (MAE) with a Vision Transformer backbone. Experiments are conducted on the PneumoniaMNIST dataset (chest X-rays).

## Highlights

- Built a Vision Transformer-based MAE in PyTorch Lightning
- Implemented custom patchification, random masking, and reconstruction modules
- Trained on PneumoniaMNIST for self-supervised learning
- Evaluated learned embeddings via t-SNE visualization

## Requirements
- torch
- torchvision
- pytorch-lightning
- torchmetrics
- matplotlib
- numpy
- pillow
- medmnist
