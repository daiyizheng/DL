# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random
from itertools import islice

import numpy as np
import torch

'''
torch.utils.data.BatchSampler，用于在深度学习训练过程中根据数据长度来创建批次。
这种采样器特别适用于处理长度不一的数据，例如文本或序列数据。
'''
class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    '''用于对数据进行批处理'''
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        '''定义构造函数，接收数据源、批次大小、是否丢弃最后一批（如果不满足批次大小）以及是否随机打乱数据。'''
        # 检查数据源中的第一个元素是否是字典类型
        if isinstance(next(iter(data_source)), dict):
            # 找到第一个元素的第一个键
            first_key = next(iter(next(iter(data_source)).keys()))
            # 创建了一个长度列表，其中每个元素对应于数据源中每个字典的第一个键的长度。
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            # 如果数据源中的元素不是字典，这行代码假设数据源是直接的序列（如字符串列表），并计算每个序列的长度。
            self.lengths = [len(d) for d in data_source]
        # 在训练过程中每个批次中的样本数量
        self.batch_size = batch_size
        # 是否在最后一个批次样本数不足时丢弃它
        self.drop_last = drop_last
        # 是否随机打乱数据
        self.shuffle = shuffle

    def __iter__(self):
        '''定义迭代器，用于生成批次。'''
        # 根据数据长度对索引进行排序
        ids = np.argsort(self.lengths)
        # 如果设置了丢弃最后一批，则相应地调整索引。
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]
        # 根据批次大小分割排序后的索引
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]
        # 如果设置了打乱顺序，则随机打乱批次。
        if self.shuffle:
            random.shuffle(batches)
        # 逐个返回批次
        for b in batches:
            yield b

    def __len__(self):
        '''定义返回批次总数的方法'''
        # 根据是否丢弃最后一批和批次大小来计算总批次数
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        '''
        data_source: 数据源。
        batch_size: 每个批次的大小。
        num_replicas: 分布式训练中的副本总数。
        rank: 当前副本的编号。
        shuffle: 是否在每个纪元开始时打乱数据。
        seed: 随机种子，用于确保不同副本的数据打乱方式一致。
        '''
        random.seed(seed)
        # 创建一个 LengthBasedBatchSampler 实例来处理数据
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank
        
    def __iter__(self):
        '''定义迭代器，用于在分布式环境中生成批次。'''
        # 计算分布式环境中每个副本应处理的最大批次数量
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        # 使用islice从self.batch_sampler中为每个副本切割出一部分数据。
        # 每个副本处理的数据批次由其rank决定。
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
         
    def __len__(self):
        '''返回每个副本应处理的批次数量'''
        # 计算总批次数除以副本数
        return len(self.batch_sampler) // self.num_replicas
            