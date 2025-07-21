# Installation

Our codebase is developed based on Ubuntu 18.04 and Pytorch framework.

### Installation with conda

```bash
# We suggest to create a new conda environment with python version 3.8
conda create --name ILGS python=3.8

# Activate conda environment
conda activate ILGS

# Install Pytorch that is compatible with your CUDA version
# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ILGS
git clone https://github.com/DCVL-3D/ILGS_release.git --recursive
cd ILGS_release

pip install -r requirements.txt

pip install submodules/ILGS-rasterization
pip install submodules/simple-knn
```
### Install Segment-Anything-Langsplt and download the checkpoint SAM


`Segment-Anything-Langsplt` : [segment-anything-langsplat](https://github.com/minghanqin/segment-anything-langsplat) </br>
`SAM checkpoints from` [SAM](https://github.com/facebookresearch/segment-anything)
