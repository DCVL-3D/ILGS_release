<h1 align="center">Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation</h1>
<!-- <p align="center">
  <a href="https://iccv.thecvf.com//"><img src="https://img.shields.io/badge/ICCV-2025-blue.svg?style=flat-square"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%E2%9D%A4-red.svg?style=flat-square"></a>
</p> -->

Official Pytorch implementation [**"Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation"**](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.pdf) <br>
[SungMin Jang](https://github.com/jsmbank), and [Wonjun Kim](https://sites.google.com/view/dcvl) (Corresponding Author) <br>
🌸***IEEE/CVF International Conference on Computer Vision, (ICCV)***, Oct. 2025.🌸

[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=IdJYIZfuwfE&t=5s)

![ Training pipeline ](documents/Pipeline.jpeg)
<p align="center">[ Training pipeline ]</p>

## :eyes: Overview

We propose **Identity-aware Language Gaussian Splatting (ILGS)**, a simple yet powerful method to resolve view-inconsistent language embeddings for open-vocabulary 3D semantic segmentation.

ILGS introduces two key components: an **identity-aware semantic consistency loss** to ensure consistent features across views, and a **progressive mask expanding** scheme for precise boundary segmentation.

We provide:

- ✅ **Full Implementation**: The complete source code for ILGS.
- 🪄 **3D Editing**: Examples of object removal, color modification, and resizing.

## 💿 Installation

For detailed instructions on how to set up the environment and install dependencies, please refer to our [Installation.md](documents/Installation.md) 

## 📂 Download

We provide guidelines to download datasets. </br>
Please check [Download.md](documents/Download.md) for more information.

## ✅ Full Implementation

To get started with the full implementation of our open-vocabulary segmentation model, please follow the instructions in the [Implementation.md](documents/Implementation.md).


## 🪄 3D Editing 
For instructions on our 3D editing features, please refer to the [Editing.md](documents/Edit.md). This guide explains how to launch the interactive demos and use functionalities like object removal, color modification, and resizing.

## Results

**Below are the semantic segmentation results of our proposed method on the LERF-Mask benchmark.**
![result](documents/Result.jpg)

## License

This project is licensed under the **Apache License 2.0**, with the exception of certain components derived from the [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) project.

- **Apache License 2.0**: All original code written for ILGS is released under the Apache 2.0 license. See [LICENSE](./LICENSE).
- **Non-commercial License (Inria & MPII)**: Some parts of the code are based on Gaussian Splatting, which is licensed for **non-commercial research use only**. See [LICENSE_GAUSSIAN_SPLATTING.md](./LICENSE_GAUSSIAN_SPLATTING.md) for full terms.

Please ensure that you comply with both licenses when using this repository.


## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) funded by the Korea Government (MSIT) under Grant RS-2023-NR076462.

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping)  </br>
[minghanqin/LangSplat](https://github.com/minghanqin/LangSplat)  </br>

## Citation
If you find our work useful for your project, please consider citing the following paper.
```
@inproceedings{jsmbankILGS,
  title={Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation},
  author={SungMin Jang and Wonjun Kim},
  booktitle={Proceedings of the International Conference on Computer Vision},
  year={2025}
}
```

