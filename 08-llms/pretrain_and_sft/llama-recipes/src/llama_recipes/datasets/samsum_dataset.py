# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_preprocessed_samsum(dataset_config, tokenizer, split):
    # 从datasets库加载samsum数据集的指定部分
    dataset = datasets.load_dataset("samsum", split=split)
    # 定义一个用于生成摘要任务提示的模板字符串
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    )

    def apply_prompt_template(sample):
        '''该函数将每个样本的对话内容填充到提示模板中，并与摘要一起返回。'''
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    # 处理数据集中的每个样本，并移除原有的列。
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        '''对每个样本的提示文本和摘要进行分词编码'''
        # 在编码的提示文本前添加开始符号（bos_token），并在摘要后添加结束符号（eos_token）。
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        # 构建每个样本的input_ids（输入ID）、attention_mask（注意力掩码）和labels（标签）。
        # 标签中对应提示文本的部分设置为-100（在损失计算中忽略）。
        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample
    # 处理数据集中的每个样本
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
