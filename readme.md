# End-to-End Vision-to-Motion Model with Auxiliary Segmentation Module for Indoor Navigation

Source code for the paper [End-to-End Vision-to-Motion Model with Auxiliary Segmentation Module for Indoor Navigation](https://dl.acm.org/doi/abs/10.1145/3342999.3343007).
<img width="1450" height="460" alt="image" src="https://github.com/user-attachments/assets/4dc09164-4bce-45cb-af58-40c312a30604" />

## File Structure

- **config**: Configuration files for the model and training parameters.
- **generator**: Data generation scripts for training.
- **inference**: Utilities for running inference using the pre-trained model.
- **model**: Scripts for building the model architecture.
- **train**: Training utilities and scripts.
- **visualization**: Tools for visualizing results.
- **evaluation**: Metrics and evaluation scripts for the pre-trained model.

## Usage

```bash
# 0. Configure the setup in config.py

# 1. Start training
python train.py

# 2. Run inference and testing
python inference.py
```
