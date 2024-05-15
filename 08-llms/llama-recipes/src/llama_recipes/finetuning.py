# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
# 用于处理版本号和包管理
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
# CPUOffload 用于 FSDP 训练中的内存优化
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
# 学习率调度器，用于调整学习率。
from torch.optim.lr_scheduler import StepLR
# 导入与 LLaMA 模型相关的类
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
# 导入 LLaMA 模型的解码层
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# 导入 FSDP 和训练的配置
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
# 数据集串联工具
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
# 获取预处理过的数据集
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    # 初始化训练配置和FSDP（完全分布式数据并行）配置
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # 使用传入的关键字参数更新这些配置
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    # 设置 CUDA、PyTorch 和 Python 的随机种子
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    # 如果启用了FSDP，则执行初始化步骤，包括设置本地排名、总排名和word_size（分布式训练中的总进程数）
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    # 如果已初始化分布式训练，则设置 CUDA 设备并清除 GPU 缓存。
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    # 加载预训练的模型, 根据配置选择是否使用缓存和量化.
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        如果启用了FSDP和低CPU内存使用的FSDP配置，只在rank 0加载模型，以节省CPU内存。
        这对于大型模型（如LLaMA 70B）很重要，因为这些模型可能会占用大量的CPU内存。
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # 解析当前PyTorch版本
        v = packaging.version.parse(torch.__version__)
        # 确保使用的是 2023年7月1日或之后的开发版本, 这是因为低CPU内存的FSDP可能需要最新版本的特定功能。
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        #  如果不是最新版本，抛出异常，要求安装最新的nightly构建版本。
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        # 只在分布式训练的主进程（rank 0）上加载模型
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name, # 加载预训练的LLaMA模型
                load_in_8bit=True if train_config.quantization else None, # 如果启用量化，则在加载时使用8位格式，这有助于进一步减少内存占用。
                device_map="auto" if train_config.quantization else None, # 自动将模型的部分放置在合适的设备上
                use_cache=use_cache, # 是否使用缓存
            )
        else:
            # 如果没有启用低CPU内存的FSDP配置，则正常加载模型。
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            # 在一个名为"meta"的PyTorch设备上下文中创建模型
            with torch.device("meta"):
                # 创建了一个模型实例
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    # 如果启用了 FSDP 并且配置了使用快速内核（use_fast_kernels），则尝试使用特定的加速技术，
    # 如 Flash Attention 或 Xformer 内存高效内核，这些技术可以加速微调过程。
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            # 将模型转换为使用更高效算法的版本。这可能需要 optimum 模块，如果该模块不存在，则输出错误消息。
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # Load the tokenizer and add special tokens
    # 加载与 LLaMA 模型相对应的分词器
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    # 将分词器的填充（pad）标记设置为结束（eos）标记的ID。
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 打印模型的大小和其他相关信息。如果启用了 FSDP，则使用当前进程的 rank；否则，默认为 0。
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    # 检查是否启用了量化
    if train_config.quantization:
        # 如果启用了量化，则准备模型进行 int8 训练。
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # 检查是否同时启用了FSDP和pure_bf16
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        # 将模型的数据类型转换为 bfloat16
        model.to(torch.bfloat16)
    #  检查是否启用了 PEFT
    if train_config.use_peft:
        # 生成 PEFT 配置
        peft_config = generate_peft_config(train_config, kwargs)
        # 获取适用于 PEFT 的模型
        model = get_peft_model(model, peft_config)
        # 打印可训练参数的信息
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    # 检查是否启用了 FSDP
    if train_config.enable_fsdp:
        # 如果未使用 PEFT 且需要冻结层
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)
        # 获取混合精度和自动包装策略
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # 获取 FSDP 自动包装策略
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        # 将模型包装为 FSDP 实例，配置多个选项，如 CPU offload、混合精度策略、分片策略等。
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        # 如果启用了 FSDP 的激活检查点，则应用相关设置。
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        # 如果没有启用量化和FSDP, 将模型移至 CUDA 设备。
        model.to("cuda")
    # 根据训练配置和其他关键字参数生成数据集的配置
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    # 使用分词器、数据集配置加载并预处理训练数据集。
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    # 检查是否未启用 FSDP 或者当前进程是主进程（rank 0）
    if not train_config.enable_fsdp or rank == 0:
        # 打印训练集的长度
        print(f"--> Training Set Length = {len(dataset_train)}")
    # 加载并预处理验证数据集
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    # 检查是否未启用 FSDP 或者是主进程，并打印验证集的长度。
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")
    # 如果训练配置中指定了"packing"作为批处理策略，则对训练数据集进行分块处理（ConcatDataset）。
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)
    # 根据训练配置、训练数据集和分词器获取数据加载器的参数。
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    # 使用获取到的参数创建训练数据加载器。参数中包括了工作线程数（num_workers）、是否固定内存（pin_memory）等。
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    # 如果训练配置中指定了需要运行验证，则进行验证数据加载器的创建。
    if train_config.run_validation:
        # 如果指定了"packing"批处理策略，则对验证数据集进行分块处理。
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)
        # 根据训练配置、验证数据集和分词器获取验证数据加载器的参数。
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
        # 使用获取到的参数创建验证数据加载器
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    #  如果配置中启用了pure_bf16和指定了anyprecision优化器，则使用AnyPrecisionAdamW。
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        # 否则，默认使用 optim.AdamW
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    # 创建一个步进学习率调度器，它会在每个 epoch 后更新学习率。
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    # 调用 train 函数开始训练过程
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    # 如果没有启用 FSDP 或者当前进程是主进程（rank 0），则打印训练结果。
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    # 如果脚本作为主程序运行，则使用fire.Fire将main函数转换为命令行接口。
    fire.Fire(main)
