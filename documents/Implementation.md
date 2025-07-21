# ILGS Implementation

A quick-start guide for preprocessing scene language features, training ILGS, performing open-vocabulary segmentation, and evaluating results.

---

## 1. Dataset Preprocessing

Generate language features of the scenes:

```bash
python preprocess.py --dataset_path $dataset_path
```

Train the autoencoder and extract 3-dimensional scene features:

```bash
# Train the autoencoder
cd autoencoder
python train.py \
  --dataset_name $dataset_path \
  --encoder_dims 256 128 64 32 3 \
  --decoder_dims 16 32 64 128 256 256 512 \
  --lr 0.0007 \
  --output ae_ckpt

# Extract 3-D features
python test.py --dataset_name $dataset_path --output
```

---

## 2. Train the ILGS

Train ILGS on your dataset (example: `teatime`):

```bash
bash script/train_lerf.sh lerf/teatime 1
```

---

## 3. Open-Vocabulary Segmentation

1. Render and skip training:

   ```bash
   python render_all.py -m output/lerf/teatime --skip_train
   ```
2. Decode 3-D features to 512-D:

   ```bash
   python feature_projector.py lerf/teatime
   ```
3. Generate segmentation masks:

   ```bash
   python segmentation_mask.py -m output/lerf/teatime
   ```

---

## 4. Evaluation

Evaluate segmentation masks against ground truth:

```bash
python script/eval_lerf_mask.py -m output/lerf/teatime
```

---
