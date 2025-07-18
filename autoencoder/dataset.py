# This code is based on or brought from LangSplat.
# Original repository: https://github.com/minghanqin/LangSplat
#
# Copyright (c) 2024 Minghan Qin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        self.data_dic = {}
        for i in range(len(data_names)):
            features = np.load(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            self.data_dic[name] = features.shape[0] 
            if i == 0:
                data = features
            else:
                data = np.concatenate([data, features], axis=0)
        self.data = data

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 