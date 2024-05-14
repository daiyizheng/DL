# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
# 导入PeftModel，它是一个预先训练好的模型，可以与基础模型合并。
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def main(base_model: str, # 基础模型的名称或路径
         peft_model: str, # 预先训练好的模型的名称或路径
         output_dir: str): # 合并后模型的保存路径
        
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False, # 不以8位精度加载模型
        torch_dtype=torch.float16, # 使用16位浮点数进行计算，以节省内存和提高计算速度。
        device_map="auto",
        offload_folder="tmp", 
    )
    # 加载分词器
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model
    )
    # 加载并合并预先训练好的模型
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )
    # 合并模型并卸载不再需要的部分
    model = model.merge_and_unload()
    # 将合并后的模型和分词器保存到指定的目录
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)