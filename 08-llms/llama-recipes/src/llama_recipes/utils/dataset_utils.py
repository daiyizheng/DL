# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
# partial用于部分应用一个函数，即固定某些参数的值并返回一个新函数。
from functools import partial
# Path类用于创建和操作文件系统路径。
from pathlib import Path

import torch

from llama_recipes.datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
)


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    用于从指定的Python文件中加载模块。
    在实际应用中，这可以让用户动态地加载和运行Python代码，这在某些场景下非常有价值，例如在插件系统或动态代码执行环境中。
    """
    # 使用Path类处理输入的文件路径py_file，获取文件的名称
    module_name = Path(py_file).name
    # 创建一个加载器，用于从源文件加载模块。
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    # 通过加载器创建模块规范（specification）。规范描述了如何加载模块。
    spec = importlib.util.spec_from_loader(module_name, loader)
    # 根据给定的规范创建一个新的模块对象。
    module = importlib.util.module_from_spec(spec)
    # 使用之前创建的加载器来执行模块，实际上是加载模块的内容。
    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    '''这个函数的主要作用是动态加载指定Python文件中的函数，并用于获取数据集。
       它通过模块路径和函数名的灵活组合，能够适应各种不同的数据集获取需求。'''
    # 冒号用来分隔模块路径和函数名
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"
    # 检查模块路径是否以.py结尾，确保它是一个Python文件。
    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    # 将模块路径转换为Path对象
    module_path = Path(module_path)
    # 检查这个路径是否确实指向一个文件
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")
    # 加载模块
    module = load_module_from_py_file(module_path.as_posix())
    try:
        # 尝试从加载的模块中获取名为func_name的函数，并调用它，
        # 传入dataset_config，tokenizer和split作为参数。如果成功，返回函数的结果。
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    '''这个函数的作用是根据提供的配置和分词器，动态地选择并调用相应的函数来获取和预处理数据集。'''
    # 检查dataset_config.dataset（数据集名称）是否在DATASET_PREPROC字典的键中
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        # 根据split参数的值（"train"或其他）来返回相应的数据集分割配置。
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
