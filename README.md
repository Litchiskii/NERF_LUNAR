# NERF_LUNAR
This project aims to apply Neural Radiance Fields (NeRF) to the Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO-NAC) dataset to generate high-quality 3D reconstructions of the lunar surface. The LRO-NAC dataset provides high-resolution images of the Moon, which are used in conjunction with NeRF to create detailed and realistic 3D models.

## Neural Radiance Fields (NeRF) Model

This repository contains a Python script (`nerf.py`) implementing a Neural Radiance Field (NeRF) model for generating novel views of a scene from 2D images. Key components include:

- **NerfModel Class:** Defines the neural network architecture for NeRF.
- **render_rays Function:** Renders colors and densities along rays to generate images.
- **train Function:** Handles training of the NeRF model using ray datasets.
- **test Function:** Generates and saves rendered images for evaluation.

### Usage

The script loads training and testing datasets, trains the NeRF model, and renders images for evaluation.



### NeRF Trial Implementation (trial.py)
This repository contains a trial implementation of Neural Radiance Fields (NeRF) in the trial.py file. This implementation is based on the original paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Ben Mildenhall et al.

Overview
The trial.py file is a simplified version of NeRF and is not fully functional. It serves as a starting point for understanding the core concepts of NeRF, such as positional encoding, MLP architecture, and volume rendering.

Disclaimer
This implementation is just a trial and is not fully functional. I have taken help from various source codes and tried to combine them into a cohesive script. However, it may not produce the expected results and is intended for educational purposes only.

Code Explanation
Positional Encoding Function
Converts 3D input coordinates into a higher-dimensional space using sine and cosine functions.
NeRF Model
An MLP with several linear layers and ReLU activations.
Processes both positional and view directions separately and combines them for the final RGB and density outputs.
Volume Rendering Function
Integrates color and density outputs from the NeRF model along each ray.
Accumulates colors using weights derived from the alpha values.
Training Function
Optimizes the NeRF model using an Adam optimizer and MSE loss.
Iterates through the dataset, computing predictions and updating model weights.
Dummy Dataset
A simple dataset class to generate random rays and target images for training.
Main Function
Sets up model and dataset, then trains the model.
Renders a test image from random rays.
