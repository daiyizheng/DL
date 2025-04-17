# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
# contextlib是用于创建和管理上下文管理器的库，nullcontext是一个不执行任何操作的上下文管理器。
from contextlib import nullcontext
from pathlib import Path
# 用于包管理和版本控制的工具
from pkg_resources import packaging


import torch
# nccl是用于多GPU通信的库，常用于深度学习中的分布式训练。
import torch.cuda.nccl as nccl
# PyTorch的分布式训练模块
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
# ShardedGradScaler，这是一个用于梯度缩放的工具，特别是在分布式训练中。
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    '''这个函数用于设置分词器的参数'''
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    '''将字节转换为兆字节的结果。这里使用2**20是因为1MB等于2的20次方字节。'''
    return int(x / 2**20)

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    # 初始化梯度缩放器：根据配置选择使用ShardedGradScaler（用于分布式训练）或torch.cuda.amp.GradScaler（用于混合精度训练）。
    # 这有助于防止在使用较小的数据类型时出现的梯度下溢。
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    # 是否启用了FSDP（Fully Sharded Data Parallel）并设置world_size
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # 根据是否使用FP16（16位浮点数）精度，选择使用自动混合精度训练的上下文管理器或普通的上下文。
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        # 记录轮次开始时间
        epoch_start_time = time.perf_counter()
        # 使用MemoryTrace上下文管理器来追踪内存使用情况
        with MemoryTrace() as memtrace:  # track the memory usage
            # 设置模型为训练模式
            model.train()
            # 初始化总损失并计算每个轮次的总步数
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            # 初始化进度条tqdm用于可视化训练进程
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            # 遍历训练数据加载器（train_dataloader）：这个循环处理每个批次（batch）的数据。
            for step, batch in enumerate(train_dataloader):
                # 将批次数据移动到正确的设备上：如果启用了FSDP（完全分片数据并行），则将数据移动到local_rank指定的设备上。
                # 否则，数据移动到默认的CUDA设备（'cuda:0'）。
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')
                # 这里使用autocast上下文管理器来自动选择使用FP16或FP32精度计算模型的前向传播。
                # 这有助于加速训练并减少内存使用。
                with autocast():
                    # 计算损失：调用模型的前向传播并获取损失值。
                    loss = model(**batch).loss
                # 将损失除以梯度累积步数：这是实现梯度累积的关键步骤，允许在多个批次上累积梯度，从而进行更大批次大小的模拟。
                loss = loss / gradient_accumulation_steps
                # 累积总损失：这用于后续计算平均损失。
                total_loss += loss.detach().float()
                # 混合精度训练的梯度更新：如果启用了FP16，使用梯度缩放器（scaler）来处理梯度的反向传播和更新。
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    # 缩放损失并进行反向传播
                    scaler.scale(loss).backward()
                    # 检查当前步骤是否是梯度累积的最后一步，或者是否是训练数据的最后一个批次。
                    # 这个条件确保了梯度只在累积足够多的步骤后或者数据结束时被更新。
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        # 梯度裁剪：如果配置中启用了梯度裁剪并且裁剪阈值大于0，则进行梯度裁剪。这有助于防止梯度爆炸问题。
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            # 在进行梯度裁剪之前对缩放的梯度进行反缩放。这是混合精度训练中的一个重要步骤，用于确保梯度的数值稳定。
                            scaler.unscale_(optimizer)
                            # 根据是否启用FSDP来选择梯度裁剪的方法。如果启用了FSDP，使用model.clip_grad_norm_来裁剪梯度，
                            # 否则使用torch.nn.utils.clip_grad_norm_。
                            # 梯度裁剪是通过限制梯度的最大范数来防止梯度爆炸的一种技术。
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        # 执行优化器步骤，这里利用了梯度缩放器来适应FP16训练的需求。
                        scaler.step(optimizer)
                        # 更新梯度缩放器的状态，准备下一轮的梯度计算。
                        scaler.update()
                        # 清除模型参数的梯度。这是为了下一个批次的训练准备的，防止旧的梯度与新的梯度混合。
                        optimizer.zero_grad()
                        # 更新进度条，表示完成了一个梯度累积步骤。
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    # 反向传播, backward()函数会计算损失（loss）对模型参数的梯度。
                    loss.backward()
                    # 这里再次使用了之前的条件，即检查是否达到了梯度累积的步数，或者是否是训练数据的最后一个批次。
                    # 这确保了在适当的时机进行梯度更新。
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        # 如果配置中启用了梯度裁剪并且裁剪阈值大于0，则对梯度进行裁剪。
                        # 梯度裁剪有助于防止在训练过程中出现的梯度爆炸问题。
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            # 根据是否启用了FSDP（完全分片数据并行），选择适当的梯度裁剪方法。
                            # 如果启用了FSDP，使用model.clip_grad_norm_进行裁剪；
                            # 如果没有启用FSDP，则使用torch.nn.utils.clip_grad_norm_。
                            # 这两种方法都是通过限制梯度的范数来减少梯度过大的问题。
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        # 更新模型的参数, 优化器利用计算出的梯度来调整参数，以减少模型的损失。
                        optimizer.step()
                        # 在开始下一个训练批次之前，清除模型参数的梯度。
                        # 这是必要的步骤，因为如果不清除梯度，新的梯度会和旧的梯度累积在一起，导致不正确的参数更新。
                        optimizer.zero_grad()
                        # 更新进度条：记录当前步骤的完成情况。
                        pbar.update(1)
                # 设置进度条描述：显示当前训练周期、步骤和损失信息。
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
            # 关闭进度条：在一个训练周期结束后关闭进度条。
            pbar.close()
        # 计算并记录当前训练周期（epoch）的总耗时
        # time.perf_counter()提供了一个高精度的时间计数器，用于测量短时间间隔。
        epoch_end_time = time.perf_counter()-epoch_start_time
        # 将当前周期的耗时添加到epoch_times列表中
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        # 多设备上的损失汇总：如果使用了多个CUDA设备并且启用了FSDP，那么使用dist.all_reduce来汇总所有设备上的总损失。
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        # 计算训练周期的平均损失。这是通过将总损失除以训练数据加载器中的批次数量来实现的。
        train_epoch_loss = total_loss / len(train_dataloader)
        # 如果启用了FSDP，那么还需要将损失除以world_size，
        # 这是因为在分布式训练中，损失已经在所有设备上累加了。
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        # 计算困惑度（perplexity），它是衡量语言模型性能的一个常用指标。
        # 困惑度是损失的指数，较低的困惑度通常表示模型性能较好。
        train_perplexity = torch.exp(train_epoch_loss)
        # 将计算出的困惑度和损失添加到相应的列表中，用于跟踪训练过程中的性能变化。
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        # 首先检查是否启用了FSDP。如果启用了FSDP，那么只有在当前节点的rank为0时（通常代表主节点），才会打印内存使用情况。
        # 这是因为在分布式训练中，通常只需要从主节点收集和报告此类信息。
        if train_config.enable_fsdp:
            if rank==0:
                # 分配给CUDA的最大内存量
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                # 为CUDA保留的最大内存量
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                # CUDA的峰值活动内存使用量
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                # CUDA内存分配尝试的次数
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                # 训练过程中CPU内存使用的峰值
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        # Update the learning rate as needed
        # 更新学习率
        lr_scheduler.step()
        # 检查是否需要运行模型的验证过程
        if train_config.run_validation:
            # 调用一个评估函数，计算模型在验证集上的表现。这通常包括损失和其他指标，如困惑度（perplexity）。
            eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
            # 使用计时器记录模型保存操作的开始时间
            checkpoint_start_time = time.perf_counter()
            # 如果配置允许保存模型且当前验证损失小于之前最好的损失，则进行保存。
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                # 是否启用了完全分布式数据并行（FSDP）
                if train_config.enable_fsdp:
                    # 在分布式训练中，确保所有进程同步到此点。
                    dist.barrier()
                # 检查是否使用PEFT
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    # 保存预训练模型
                    model.save_pretrained(train_config.output_dir)
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    # 根据配置的不同，选择不同的方式来保存模型。FULL_STATE_DICT 表示保存完整的模型状态字典。
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        # 保存模型和优化器的状态
                        save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")
                        # 在使用分片状态字典（SHARDED_STATE_DICT）时保存模型和优化器
                        save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")

                    if not train_config.use_peft and  train_config.save_optimizer:
                        # 单独保存优化器的状态
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                if train_config.enable_fsdp:
                    dist.barrier()
            # 记录模型保存操作所需的总时间
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            # 判断用于检查当前的验证损失是否比之前记录的最低验证损失还要低
            if eval_epoch_loss < best_val_loss:
                # 如果当前验证损失更低，更新最低验证损失的记录。
                best_val_loss = eval_epoch_loss
                # 检查是否启用了完全分布式数据并行（FSDP）
                if train_config.enable_fsdp:
                    # 在分布式训练中，通常只有主进程（rank 0）负责输出信息。
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            # 将最佳验证损失和验证困惑度添加到相应的列表中
            val_loss.append(best_val_loss)
            val_prep.append(eval_ppl)
        # 同样检查FSDP的启用状态，根据是否启用FSDP和进程的等级，打印当前epoch的训练困惑度、训练损失和epoch运行时间。
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
    # 计算epoch时间、检查点时间、训练困惑度和训练损失的平均值
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    #  如果配置了运行验证，则计算验证的平均困惑度和损失。
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
    # 将计算出的平均值存储在一个字典中，这个字典用于返回训练过程中的结果。
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    #saving the training params including fsdp setting for reference.
    # 检查是否启用了FSDP且没有使用PEFT，如果是，保存训练参数和FSDP配置。
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    # 如果启用了完全分布式数据并行（FSDP），则从环境变量中获取世界大小（WORLD_SIZE），即参与分布式计算的节点总数。
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # 将模型设置为评估模式
    model.eval()
    # 初始化用于存储评估预测结果的列表和评估损失。
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    # 一个上下文管理器，用于追踪评估过程中的内存使用情况。
    with MemoryTrace() as memtrace:
        # 这个循环处理每一个评估批次
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            # 对于批次中的每个数据项（如输入数据、标签等）
            for key in batch.keys():
                # 将数据移动到适当的设备上。在分布式设置中，数据移动到本地排名对应的设备上；
                # 在非分布式设置中，数据移动到第一个CUDA设备上。
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            # 这个上下文管理器确保在这个范围内不计算梯度，这有助于减少内存消耗。
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                # 从模型输出中提取损失
                loss = outputs.loss
                # 累计评估损失。使用 detach() 来避免梯度的计算，float() 确保损失以浮点数形式累加。
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            # 从模型输出的logits中计算出预测的类别
            preds = torch.argmax(outputs.logits, -1)
            # 将预测转换为人类可读的格式。
            # 先用 detach().cpu().numpy() 将预测转移到CPU并转换为NumPy数组，然后使用分词器解码。
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    # 检查是否有多个CUDA设备且是否启用了完全分布式数据并行（FSDP）,
    # 如果处于一个分布式的训练环境中，需要在所有设备上聚合（reduce）评估损失。
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        # 使用PyTorch的分布式通信库来在所有进程间累加（reduce）评估损失。
        # all_reduce操作确保每个进程得到的eval_loss是所有进程上eval_loss之和。
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    # 计算每个epoch的平均评估损失, 通过将总损失除以数据加载器中批次的数量得到。
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    # 如果启用了FSDP，则进一步将评估损失除以世界大小（world_size），即分布式环境中的总进程数。
    # 这是必要的，因为之前的all_reduce操作将损失累加了world_size次。
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    # 计算评估困惑度（perplexity）
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    # 在分布式环境中，通常只有主进程（local_rank == 0）负责打印输出。
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
    # 函数返回计算得到的评估困惑度和评估epoch损失
    return eval_ppl, eval_epoch_loss

def freeze_transformer_layers(model, num_layer):
    '''冻结Transformer模型中的前num_layer层, 冻结层意味着在训练过程中不会更新这些层的参数（即梯度不会在这些层上传播）。'''
    for i, layer in enumerate(model.model.layers):
        # 检查当前层的索引是否小于要冻结的层数量。如果是，则冻结该层。
        if i < num_layer:
            # 迭代当前层中的所有参数
            for param in layer.parameters():
                # 参数的requires_grad属性设置为False，这意味着在训练过程中不会计算这些参数的梯度，从而“冻结”了这些层。
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    '''检查一个模型中每层的参数是否被冻结'''
    for i, layer in enumerate(model.base_model.model.model.layers):
        # 迭代当前层中的所有参数及其名称。
        for name, param in layer.named_parameters():
            print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    # 调用了PyTorch的分布式通信库 (torch.distributed 模块) 来初始化进程组。
    # 这里使用的是nccl后端，这是一种专为NVIDIA GPUs优化的通信后端，非常适合于CUDA上的分布式训练。
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes 设置一些环境变量，以便于调试分布式训练过程。"""
    # 设置环境变量以显示C++的堆栈跟踪。当PyTorch遇到底层错误时，这可以帮助调试。
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # 开启NCCL（NVIDIA Collective Communications Library）的异步错误处理。
    # 这样做可以在发生错误时提供更及时的反馈，特别是在多GPU环境中。
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    #如果启用，它会设置PyTorch分布式调试的详细级别。
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)

    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    # 调用了PyTorch的分布式通信库，销毁之前创建的进程组。
    # 这是分布式训练结束后的一个重要步骤，它有助于释放分布式训练中所使用的资源。
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    # 调用了PyTorch的CUDA库，来清空当前CUDA设备的缓存, 这个操作有助于释放不再需要的GPU内存.
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters
       获取模型参数的数据类型"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        #  将参数的名称和数据类型存储在字典中
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    # 只有当进程的排名是0（即主进程）时才执行打印操作
    if rank == 0:
        # 打印模型的名称
        print(f"--> Model {config.model_name}")
        # 计算模型中可训练参数的总数
        # # model.parameters() 返回模型的所有参数，p.numel() 返回每个参数的元素数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping
    这个函数 get_policies 用于获取混合精度训练和FSDP封装的策略。
    混合精度训练是一种用于加速神经网络训练的技术，
    它通过在训练中使用不同的数据类型（如FP32、FP16、BF16）来减少内存使用和提高运算速度。
    """
    # 一个条件表达式，检查是否支持bfloat16。它考虑了CUDA版本、PyTorch对bf16的支持、NCCL版本等因素。
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    # 这两行初始化两个变量，分别用于存储混合精度训练策略和FSDP封装策略。
    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    # 检查配置中是否启用了混合精度训练
    if cfg.mixed_precision:
        # 检查是否支持bf16
        bf16_ready = verify_bfloat_support
        # 如果支持bf16且没有指定使用FP16，则使用bf16策略。
        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            # 如果指定使用FP16，则使用FP16策略。
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            #  如果不支持bf16，则默认使用FP32，且不使用混合精度训练。
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    # 获取用于FSDP的包装策略
    wrapping_policy = get_llama_wrapper()
    # 返回混合精度训练策略和FSDP包装策略
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    用于将训练配置 (train_config) 和完全分布式数据并行（FSDP）配置 (fsdp_config) 保存到一个 YAML 文件中。
    这样做有助于将这些配置用于推理脚本，同时也作为未来参考的日志记录。
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    # 将 train_config 和 fsdp_config 对象转换为字典，并将所有值转换为字符串
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    # 使用字典解包将两个配置字典合并成一个
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    # 构建保存配置文件的文件夹名称
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )
    # 创建保存目录
    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    # 使用yaml.dump将配置字典转换成易于阅读的YAML格式。
    config_yaml = yaml.dump(train_params_dict, indent=4)
    # 创建 YAML 文件的完整路径并检查是否存在同名的目录
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")
