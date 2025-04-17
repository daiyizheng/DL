# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# inspect模块提供了一系列用于获取对象信息的函数，比如检查对象类型、获取源码、检查类或函数的参数等。
import inspect
# asdict函数用于将dataclass实例转换为字典。
from dataclasses import asdict
# 分布式计算模块，用于在多个计算节点之间进行数据通信和同步。
import torch.distributed as dist
# 在分布式训练过程中对数据进行采样
from torch.utils.data import DistributedSampler
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
# 该函数用于默认的数据整理方式
from transformers import default_data_collator
# 专门用于序列到序列（Seq2Seq）模型的数据整理器
from transformers.data import DataCollatorForSeq2Seq

from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler
from llama_recipes.utils.dataset_utils import DATASET_PREPROC


def update_config(config, **kwargs):
    '''函数目的是更新配置对象的属性。函数可以处理多种类型的配置对象，包括单个配置、配置元组或列表。
       它还允许通过具有层级结构的关键字参数来更新配置。'''
    # 如果config是一个元组或列表，函数将对其中的每一个元素递归地调用update_config，以此来更新每个配置对象。
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        # 如果config是单个对象，函数会遍历kwargs中的每一项。
        for k, v in kwargs.items():
            # 如果config有一个属性与kwargs中的键（k）相同，就使用setattr更新该属性的值为kwargs中相应的值（v）。
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                # 如果关键字参数的键（k）包含点号（.），这意味着参数指向一个更复杂的配置结构。
                # 这时，代码会将这个键分割成配置名（config_name）和参数名（param_name）。
                config_name, param_name = k.split(".")
                # 如果config的类型名与config_name相匹配，并且config有一个名为param_name的属性，
                # 那么就更新这个属性的值。
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # 如果config中没有名为param_name的属性，则打印一个警告信息，说明这个配置不接受该参数。
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                # 如果遇到未知的参数并且config是train_config的实例，函数也会打印一个警告信息。
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config, kwargs):
    # 创建了一个名为configs的元组，包含三个配置对象：lora_config、llama_adapter_config和prefix_config。
    configs = (lora_config, llama_adapter_config, prefix_config)
    # 创建了一个名为peft_configs的元组，包含三个配置类：LoraConfig、AdaptionPromptConfig 和 PrefixTuningConfig。
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    # 创建了一个名为names的元组，通过遍历configs元组中的每个配置对象，获取它们的类名（去除类名末尾的"_config"）。
    names = tuple(c.__name__.rstrip("_config") for c in configs)
    # 确保train_config的peft_method属性值存在于names元组中。如果不存在，则抛出一个异常，提示“Peft配置未找到”。
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    # 根据train_config.peft_method的值，从configs元组中选择对应的配置对象，并创建该配置对象的实例。
    config = configs[names.index(train_config.peft_method)]()
    # 更新 config 对象的属性，**kwargs 表示接收任意数量的关键字参数。
    update_config(config, **kwargs)
    # 将config对象转换为字典形式
    params = asdict(config)
    # 根据train_config.peft_method的值，从peft_configs元组中选择对应的配置类，并使用params字典中的参数创建这个类的实例。
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(train_config, kwargs):
    # 创建了一个名为names的元组，它包含DATASET_PREPROC字典的所有键。
    # 这些键可能代表不同的数据集预处理配置的名称。
    names = tuple(DATASET_PREPROC.keys())
    # 用于检查train_config中的dataset属性是否在names元组中。如果不在，程序将抛出异常，异常信息为“未知的数据集”。
    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"
    # 首先使用inspect.getmembers函数获取datasets模块的所有成员（可能是类、函数、常量等），然后创建一个字典，其中的键和值分别是成员的名称和对应的对象。
    # 接着，它使用train_config.dataset作为键从这个字典中检索对应的对象，并调用这个对象（可能是一个类或者工厂函数）来创建一个数据集配置的实例。
    dataset_config = {k:v for k, v in inspect.getmembers(datasets)}[train_config.dataset]()
    # 调用update_config函数来更新dataset_config实例
    update_config(dataset_config, **kwargs)
    # 返回最终生成的数据集配置实例
    return  dataset_config


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
    '''根据训练配置、数据集、分词器和模式（训练或验证）生成数据加载器（dataloader）的关键字参数'''
    # 用于存储将要返回的关键字参数
    kwargs = {}
    # 根据模式确定批量大小（batch_size）, 如果模式是"train"，使用训练配置中的 batch_size_training；否则使用 val_batch_size。
    batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
    # 判断训练配置中的批处理策略是否为"padding"
    if train_config.batching_strategy == "padding":
        # 如果启用了FSDP（Fully Sharded Data Parallel），则配置分布式长度基础批处理采样器
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
        else:
            # 如果没有启用FSDP，使用普通的长度基础批处理采样器
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
        # 用于处理批处理中的数据
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
    elif train_config.batching_strategy == "packing":
        # 如果批处理策略是"packing"
        if train_config.enable_fsdp:
            # 如果启用了FSDP，配置分布式采样器
            kwargs["sampler"] = DistributedSampler(
            dataset,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=mode=="train",
        )
        kwargs["batch_size"] = batch_size # 设置批量大小
        kwargs["drop_last"] = True # 在数据集大小不是批量大小的整数倍时，最后一个不完整的批次将被丢弃。
        kwargs["collate_fn"] = default_data_collator
    else:
        # 如果批处理策略既不是"padding"也不是"packing, 抛出异常，表示批处理策略未知。
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")
    # 返回包含配置好的关键字参数的字典
    return kwargs
