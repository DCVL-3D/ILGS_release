/*
 * Copyright (C) 2025, DCVL-3D
 * ILGS_release research group, https://github.com/DCVL-3D/ILGS_release
 * All rights reserved.
 * ------------------------------------------------------------------------
 * Modified from codes in Gaussian-Splatting 
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_OBJECTS 16 // Default 16, identity encoding
#define NUM_SEMANTIC_CHANNELS 3 // Subject to change
#define BLOCK_X 16
#define BLOCK_Y 16

#endif