# NERF_LUNAR
This project aims to apply Neural Radiance Fields (NeRF) to the Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO-NAC) dataset to generate high-quality 3D reconstructions of the lunar surface. The LRO-NAC dataset provides high-resolution images of the Moon, which are used in conjunction with NeRF to create detailed and realistic 3D models.

## Neural Radiance Fields (NeRF) Model

This repository contains a Python script (`nerf.py`) implementing a Neural Radiance Field (NeRF) model for generating novel views of a scene from 2D images. Key components include:

- **NerfModel Class:** Defines the neural network architecture for NeRF.
- **render_rays Function:** Renders colors and densities along rays to generate images.
- **train Function:** Handles training of the NeRF model using ray datasets.
- **test Function:** Generates and saves rendered images for evaluation.

The script loads training and testing datasets, trains the NeRF model, and renders images for evaluation.



## NeRF Trial Implementation (trial.py)

This repository contains a trial implementation of Neural Radiance Fields (NeRF) in the ('trial.py') file. This implementation is based on the original paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" .
- **Overview**
The trial.py file is a simplified version of NeRF and is not fully functional. It serves as a starting point for understanding the core concepts of NeRF, such as positional encoding, MLP architecture, and volume rendering.

- **Disclaimer**
This implementation is just a trial and is not fully functional. I have taken help from various source codes and tried to combine them into a cohesive script. However, it may not produce the expected results and is intended for educational purposes only.

The code in trial.py includes a positional encoding function that converts 3D input coordinates into a higher-dimensional space using sine and cosine functions. The NeRF model itself is an MLP with several linear layers and ReLU activations, which processes both positional and view directions separately and combines them for the final RGB and density outputs. The volume rendering function integrates color and density outputs from the NeRF model along each ray and accumulates colors using weights derived from the alpha values. The training function optimizes the NeRF model using an Adam optimizer and MSE loss, iterating through the dataset, computing predictions, and updating model weights. A dummy dataset is used to generate random rays and target images for training. Finally, the main function sets up the model and dataset, then trains the model and renders a test image from random rays.


## Dependencies

This project relies on the following libraries:

- `torch` for building and training neural networks
- `numpy` for numerical computations
- `tqdm` for progress bar
- `matplotlib` for plotting and visualization
