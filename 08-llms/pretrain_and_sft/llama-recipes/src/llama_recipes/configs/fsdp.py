# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 从Python的dataclasses模块中导入了dataclass装饰器,可以自动为类生成特殊方法，比如__init__()、__repr__()等。
from dataclasses import dataclass
# ShardingStrategy是一个枚举类型，用于指定模型参数在分布式训练中的分片策略。
from torch.distributed.fsdp import ShardingStrategy
# StateDictType是一个枚举类型，用于指定模型状态字典的保存和加载方式。
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    # 混合精度训练开关，默认为True。
    mixed_precision: bool=True
    # 是否使用FP16精度，默认为False。
    use_fp16: bool=False
    # 分片策略，默认为全参数分片。
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    # 模型状态字典的保存和加载类型，默认为分片状态字典。
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    # 是否开启FSDP的激活检查点，用于节省内存，默认为True。
    fsdp_activation_checkpointing: bool=True
    # 是否开启CPU卸载，以减少GPU的内存使用，默认为False。
    fsdp_cpu_offload: bool=False
    # 是否纯粹使用bf16精度，默认为False。
    pure_bf16: bool = False
    # 优化器类型，默认为"AdamW"。
    optimizer: str= "AdamW"
    
