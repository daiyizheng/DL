# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    '''用于配置神经网络模型的训练参数'''
    # 模型名称或模型文件的路径
    model_name: str="/root/autodl-tmp/llama-7b-chat-hf"
    # 是否启用全参数分片
    enable_fsdp: bool=False
    # 是否启用CPU卸载以优化FSDP的内存使用
    low_cpu_fsdp: bool=False
    # 是否在训练过程中运行验证
    run_validation: bool=True
    # 训练时的批量大小
    batch_size_training: int=8
    # 批处理策略，"packing"或"padding"
    batching_strategy: str="packing" #alternative: padding
    # 输入的上下文长度
    context_length: int=1024
    # 梯度累积步数
    gradient_accumulation_steps: int=1
    # 是否应用梯度裁剪
    gradient_clipping: bool = False
    # 梯度裁剪阈值
    gradient_clipping_threshold: float = 1.0
    # 训练的总轮数（epoch数）
    num_epochs: int=1
    # 数据加载时的工作线程数
    num_workers_dataloader: int=8
    # 学习率
    lr: float=1e-4
    # 权重衰减率
    weight_decay: float=0.0
    # 学习率衰减系数
    gamma: float= 0.85
    # 随机种子
    seed: int=42
    # 是否使用FP16精度
    use_fp16: bool=True
    # 是否使用混合精度训练
    mixed_precision: bool=True
    # 验证时的批量大小
    val_batch_size: int=2
    # 使用的数据集
    dataset = "alpaca_dataset"
    # 参数高效微调（PEFT）方法，如"lora"、"llama_adapter"或"prefix"。
    peft_method: str = "lora" # None , llama_adapter, prefix
    # 是否使用参数高效微调（PEFT）
    use_peft: bool=True
    # 保存PEFT模型的路径
    output_dir: str = "/root/autodl-tmp/PEFT/model"
    # 是否冻结部分层
    freeze_layers: bool = False
    # 冻结的层数
    num_freeze_layers: int = 1
    # 是否应用量化
    quantization: bool = True
    # 是否仅在一个GPU上训练
    one_gpu: bool = True
    # 是否保存模型
    save_model: bool = True
    # FSDP模型保存的根文件夹路径
    dist_checkpoint_root_folder: str="/root/autodl-tmp/FSDP/model" # will be used if using FSDP
    # FSDP模型微调后保存的文件夹名称
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    # 是否保存优化器状态
    save_optimizer: bool=False # will be used if using FSDP
    # 是否使用快速内核，如Flash Attention和Xformer。
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
