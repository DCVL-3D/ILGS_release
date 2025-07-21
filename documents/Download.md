# Download

 Download datasets and pseudo labels for training.

```bash
mkir -p datasets  # datasets
cd datasets
```

## LERF-Mask dataset

***We use the same data from the previous project [Gaussian-Grouping](https://github.com/lkeab/gaussian-grouping)***

You can download LERF-Mask dataset from [this hugging-face link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask). 
Test set of LERF-Mask dataset includes 2-4 novel view images. The mask annotations are saved in `test_mask` folder. The name of each mask image corresponds to the input text-prompt.
```
lerf_mask
|____figurines
| |____distorted
| |____images
| |____images_train
| |____object_mask
| |____sparse
| |____stereo
| |____test_mask
|   |____<novel view 0>
|   | |____<text prompt 0>.png
|   | |____...
|   |____<novel view 1>
|   | |____<text prompt 0>.png
|   | |____...
|____ramen
| |____...
|____teatime
| |____...
```

