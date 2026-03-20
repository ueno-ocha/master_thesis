import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, male_data, female_data):
        self.male_data = male_data
        self.female_data = female_data

        # すべての (occupation, index) の組み合わせを列挙
        self.index_map = []
        for occ in male_data.keys():
            count = len(male_data[occ])
            for i in range(count):
                self.index_map.append((occ, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        occ, i = self.index_map[idx]

        text_male = self.male_data[occ][i]
        text_female = self.female_data[occ][i]

        # ペアで返す例
        return {
            "occupation": occ,
            "male": text_male,
            "female": text_female
        }
