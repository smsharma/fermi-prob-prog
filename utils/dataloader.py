import numpy as np
import torch
from bisect import bisect


class BigDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, target_paths):
        self.data_memmaps = [np.load(path, mmap_mode='r') for path in data_paths]
        self.target_memmaps = [np.load(path, mmap_mode='r') for path in target_paths]
        self.start_indices = [0] * len(data_paths)
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        data = self.data_memmaps[memmap_index][index_in_memmap]
        target = self.target_memmaps[memmap_index][index_in_memmap]
        return torch.from_numpy(data), torch.from_numpy(target)
