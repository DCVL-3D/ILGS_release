#!/bin/bash

# Validate the number of arguments using a case statement
case "$#" in
    2)
        # Correct number of arguments, proceed to assign them
        dataset_name="$1"
        scale="$2"
        ;;
    *)
        # Any other number of arguments is incorrect
        echo "Usage: $0 <dataset_name> <scale>" >&2
        exit 1
        ;;
esac

dataset_folder="data/$dataset_name"

# Check for the existence of the dataset folder
if [ ! -d "$dataset_folder" ]; then
    echo "Error: Cannot find dataset directory at '$dataset_folder'." >&2
    exit 2
fi

echo "Dataset: $dataset_name, Scale: $scale, Path: $dataset_folder ... ready to process."


# ILGS training
python train.py    -s $dataset_folder -r ${scale}  -m output/${dataset_name} --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python render.py -m output/${dataset_name} --num_classes 256
