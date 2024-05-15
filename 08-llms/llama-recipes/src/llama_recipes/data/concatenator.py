# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    '''ConcatDataset的作用是对输入的数据集进行预处理，
       将数据集中的样本根据指定的chunk_size进行切分，以便于后续的批量处理。'''
    def __init__(self, dataset, chunk_size=4096):
        # 输入的数据集
        self.dataset = dataset
        # 指定了每个数据块的大小，默认值为4096
        self.chunk_size = chunk_size
        # 用于存储预处理后的数据样本
        self.samples = []
        # 用于暂存处理过程中的数据
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            # 对于dataset中的每个样本，将其字段加入到buffer中。
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            # 当buffer中任一字段的长度超过chunk_size时，执行循环。
            while len(next(iter(buffer.values()))) > self.chunk_size:
                # 在循环内部，将buffer中每个字段的前chunk_size个元素作为一个新的样本加入到self.samples。
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                # 然后更新buffer，保留每个字段中剩余的元素。
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        # 根据索引返回对应的样本
        return self.samples[idx]

    def __len__(self):
        # 返回预处理后的样本总数
        return len(self.samples)
