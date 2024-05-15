# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/jfleg
# For download and preparation see: recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb

# 用于加载各种格式的数据集
from datasets import load_dataset
# 导入Path用于处理文件路径
from pathlib import Path
# 导入PyTorch的Dataset类，作为自定义数据集的基类。
from torch.utils.data import Dataset


class grammar(Dataset):
    def __init__(
        self,
        tokenizer, # 用于文本处理的分词器
        csv_name=None, # CSV文件的名称或路径，包含需要加载的数据。
    ):

        try:
            # 使用load_dataset函数加载CSV文件中的数据集。如果加载失败，则打印错误信息并抛出异常。
            self.dataset = load_dataset(
                "csv",
                data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
                delimiter=",",
            )
        except Exception as e:
            print("Loading of grammar dataset failed! Please see recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb for details on how to download the dataset.")
            raise e

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        # 保存传入的分词器实例，用于后续文本处理。
        self.tokenizer = tokenizer
        # 设置一个标志，决定是否打印文本，这里默认设置为不打印。
        self.print_text = False  # print_text

    def __len__(self):
        # 返回训练集的样本数
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        '''将数据集中的单个样本转换为模型可以处理的特征'''
        # Create prompt and tokenize contexts and questions

        if self.print_text:
            # 打印经过清理的文本
            print("Input Text: ", self.clean_text(example_batch["text"]))
        # 从样本中提取输入文本和目标文本
        input_ = example_batch["input"]
        target_ = example_batch["target"]
        # 构建一个提示字符串，用于模型的输入。
        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: "
        # 使用分词器对提示进行编码，添加句子开始标记（bos_token）但不添加特殊标记。
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        # 对目标文本进行编码，并添加句子结束标记（eos_token）。
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids, # 提示和标签的组合
            "attention_mask": [1] * len(prompt_ids + label_ids), # 一个全1的数组，表示所有的tokens都应该被模型考虑。
            "labels": [-100] * len(prompt_ids) + label_ids # 包含了用于训练的标签，其中输入部分的标签被设置为-100（通常用于忽略计算损失的标记）。
        }

        return sample

    def __getitem__(self, index):
        '''用于按索引获取数据集中的单个样本'''
        # 通过索引获取训练集中的样本，并调用convert_to_features方法将其转换为特征。
        return self.convert_to_features(self.dataset["train"][int(index)])


def get_dataset(
    dataset_config, tokenizer, csv_name=None
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    # 果没有提供csv_name参数，函数会构建一个默认的文件路径。
    if csv_name is None:
        # 使用Path库拼接当前工作目录和默认的数据集路径。
        currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
        print(f"Loading dataset {currPath}")
        # 将路径转换为字符串格式，以便可以被后续的函数调用。
        csv_name = str(currPath)
    # 使用提供的分词器和CSV文件名创建grammar类的实例
    dataset = grammar(
        tokenizer=tokenizer,
        csv_name=csv_name,
    )

    return dataset
