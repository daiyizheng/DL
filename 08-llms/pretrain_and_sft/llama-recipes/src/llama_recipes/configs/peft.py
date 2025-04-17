# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import List

@dataclass
class lora_config:
     # 表示LoRA的秩，即低秩矩阵的维度。
     r: int=8
     # LoRA的扩展因子，用于控制LoRA层的大小。
     lora_alpha: int=32
     # 一个字符串列表，指定要应用LoRA的目标模块（如Transformer中的某些子层）。
     target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
     # 指定是否为LoRA层添加偏置。
     bias= "none"
     # 指定任务类型，这里是因果语言模型
     task_type: str= "CAUSAL_LM"
     # LoRA层的dropout率
     lora_dropout: float=0.05
     # 指定是否在推理模式下使用LoRA
     inference_mode: bool = False

@dataclass
class llama_adapter_config:
     # 适配器的长度
     adapter_len: int= 10
     # 适配器中包含的层数
     adapter_layers: int= 30
     # 指定任务类型，这里同样是因果语言模型
     task_type: str= "CAUSAL_LM"

@dataclass
class prefix_config:
     '''前缀调优是一种调整大型语言模型的技术，它通过向模型输入的每个样本添加一些“虚拟”tokens来实现。'''
     # 虚拟token的数量
     num_virtual_tokens: int=30
     # 指定任务类型，同样是因果语言模型
     task_type: str= "CAUSAL_LM"    