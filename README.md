# NERF_LUNAR
This project aims to apply Neural Radiance Fields (NeRF) to the Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO-NAC) dataset to generate high-quality 3D reconstructions of the lunar surface. The LRO-NAC dataset provides high-resolution images of the Moon, which are used in conjunction with NeRF to create detailed and realistic 3D models.

#nerf.py 
This script implements a Neural Radiance Field (NeRF) model for generating novel views of a scene from 2D images. Key components include:

NerfModel Class: Defines the neural network architecture.
render_rays Function: Renders colors and densities along rays.
train Function: Trains the NeRF model on ray datasets.
test Function: Generates and saves rendered images.
Usage
Loads training and testing datasets, trains the NeRF model, and renders images for evaluation.
