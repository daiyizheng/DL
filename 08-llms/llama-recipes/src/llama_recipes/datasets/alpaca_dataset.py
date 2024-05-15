# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        # 包含数据集的配置信息
        self.ann = json.load(open(dataset_config.data_path))
        # 指定是使用训练集还是其他部分，默认为"train"
        if partition == "train":
            self.ann = self.ann[200:]
        else:
            self.ann = self.ann[:200]
        # 用于文本编码的分词器
        self.tokenizer = tokenizer

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.ann)

    def __getitem__(self, index):
        '''根据索引返回单个样本'''
        # 在损失计算中忽略某些标签
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        # 根据数据项ann是否包含input字段，选择不同的提示文本（prompt）格式。
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        # 将提示文本和输出（ann["output"]）拼接起来形成完整的样本
        example = prompt + ann["output"]
        # 使用提供的分词器将文本提示（prompt）编码为整数序列
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        # 将eos_token_id（结束符标记）添加到编码后的样本末尾
        example.append(self.tokenizer.eos_token_id)
        # 将完整的样本（包括提示和输出）转换为PyTorch张量
        example = torch.tensor(
            example, dtype=torch.int64
        )
        # 创建标签（labels），并将提示部分的标签设为-1（表示忽略）。
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        # 创建掩码（example_mask和label_mask）以指示有效的标签位置。
        # 表示example中的每个元素是否大于等于0。
        example_mask = example.ge(0)
        # 表示labels中的每个元素是否大于等于0。
        label_mask = labels.ge(0)
        # 对于不在掩码内的位置，将输入标记为0，将标签设置为IGNORE_INDEX。
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
