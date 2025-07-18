## Precessing dataset

1. gengerate language features of the scenes. 
```
python preprocess.py --dataset_path $dataset_path 
```
2. train the Autoencoder and get the 3-dim feature
```
# train the autoencoder
cd autoencoder
python train.py --dataset_name $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --output ae_ckpt

# get the 3-dims language feature of the scene
python test.py --dataset_name $dataset_path --output
```

## Train the ILGS
```
# Example teatime dataset
bash script/train_lerf.sh lerf/teatime 1
```

## Open-vocabulary segmentation 
```
# Example teatime dataset
python render_all.py -m output/lerf/teatime --skip_train

# decode 3-dim language features to 512-dim
python feature_projector.py lerf/figurines

# make segmenttaion mask 
python make_test_mask.py
```

## Evaluaiton 
```
# Example teatime dataset
python script/eval_lerf_mask.py teatime
```