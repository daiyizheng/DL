# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig

# Function to load the main model for text generation
def load_model(model_name, quantization):
    '''用于加载Llama模型'''
    model = LlamaForCausalLM.from_pretrained(
        # 模型的名称或路径
        model_name,
        # 确保模型返回 transformers.file_utils.ModelOutput 类型的对象，而不是标准的元组。
        return_dict=True,
        # 一个布尔值，用于指示是否以8位量化模式加载模型。量化是一种优化技术，可以减少模型大小并提高运行速度，但可能会牺牲一些精度。
        load_in_8bit=quantization,
        # 自动确定模型应该加载到哪个设备（如 CPU、GPU）。
        device_map="auto",
        # 优化内存使用，特别是在 CPU 上运行时。
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    '''加载一个已经预训练好的 PeftModel'''
    # model：已加载的模型。
    # peft_model：性能优化模型的名称或路径。
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    '''从配置文件中加载Llama模型'''
    # 使用LlamaConfig.from_pretrained加载配置
    model_config = LlamaConfig.from_pretrained(config_path)
    # 根据加载的配置创建一个新的LlamaForCausalLM实例
    model = LlamaForCausalLM(config=model_config)
    return model
    
    