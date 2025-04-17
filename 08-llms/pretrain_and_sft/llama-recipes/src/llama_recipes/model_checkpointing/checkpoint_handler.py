# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time

from torch.distributed.fsdp import (
    # FSDP类是PyTorch中用于实现完全分片数据并行的主要类。
    FullyShardedDataParallel as FSDP,
    # 一个枚举类，定义了不同类型的状态字典（state dict），用于保存和加载模型的参数。
    StateDictType,
    # 配置类，用于创建非分片、非扁平化的模型参数的状态字典。
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    # 配置类，用于创建扁平化的模型参数的状态字典，仅在 FSDP 环境下可用。
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # 用于创建未扁平化但分片的参数的状态字典，可用于其他并行方案。
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    # 用于读取和写入文件系统中的模型检查点
    FileSystemReader,
    FileSystemWriter,
    # 函数用于保存和加载模型的状态字典
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    # 这些类提供了默认的保存和加载模型检查点的策略
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
# 将检查点模块导入为 dist_cp，方便使用
import torch.distributed._shard.checkpoint as dist_cp
# 导入 PyTorch 的分布式训练模块
import torch.distributed as dist


def get_date_of_run():
    """create date and time for file save uniqueness
    这个函数生成一个独特的时间戳字符串，用于文件保存时确保唯一性。
    example: 2022-05-07-08:31:12_PM'
    """
    # 使用 datetime.now().strftime 根据当前时间创建一个格式化的字符串。
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
# 定义了一个全状态字典配置（FullStateDictConfig），用于保存模型的状态。
# offload_to_cpu=True：在保存前将模型的参数从GPU卸载到CPU。
# rank0_only=True：仅在分布式训练环境中的rank 0（通常是主节点）上执行保存操作。
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    '''用于在分布式环境中加载以FSDP（FullyShardedDataParallel）方式分片存储的模型。
       这个过程涉及到从文件系统中读取模型状态字典，并将其加载到模型中。
    model：要加载状态的 PyTorch 模型。
    rank：在分布式训练环境中，当前进程的等级（rank）。
    cfg：配置对象，包含有关检查点和模型的信息。
       '''
    # torch.manual_seed(103)
    # 使用配置参数来构造包含检查点的文件夹路径
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )
    # Path.cwd() 获取当前工作目录，folder_name 是根据配置生成的检查点文件夹的名称。
    load_dir = Path.cwd() / folder_name
    # 检查指定的文件夹是否存在。如果不存在，且当前进程是主进程（rank == 0），则打印信息并退出函数。
    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    # 如果当前进程是主进程，则打印加载信息。
    if rank == 0:
         print(f"loading model from model path: {load_dir} ")
    # FileSystemReader 用于读取文件系统中的数据
    reader = FileSystemReader(load_dir)

    # 这是一个上下文管理器，用于指定在加载状态字典时应使用的类型，这里是分片状态字典（SHARDED_STATE_DICT）。
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 创建一个包含当前模型状态的字典
        checkpoint = {"model": model.state_dict()}
        # 如果当前进程是主进程，打印检查点的键信息。
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        # 使用 dist_cp.load_state_dict 从文件系统中加载状态字典
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        # 如果当前进程是主进程，打印加载完成的信息。
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir
    旨在保存分片式分布式训练（FSDP）环境中的模型和优化器的状态, 它使用PyTorch的分布式训练模块和文件系统处理功能。
    model: 要保存的模型。
    rank: 在分布式训练环境中的进程等级。
    cfg: 配置对象，包含模型和检查点的相关信息。
    optim: （可选）要保存的优化器。
    """
    # 使用配置参数构建检查点的文件夹名称
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )
    # Path.cwd() 获取当前工作目录，然后与 folder_name 结合形成完整的保存路径。
    save_dir = Path.cwd() / folder_name
    # 如果当前进程是主进程（rank == 0），则打印模型即将被保存的位置。
    if rank == 0:
        print(f"Saving model to {save_dir}")
    # 使用FileSystemWriter类实例化一个分布式写入器，用于将状态字典写入到文件系统。
    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    # 记录保存操作开始的时间，用于计算保存操作所需时间。
    t0 = time.perf_counter()
    # 使用上下文管理器来指定保存的状态字典类型
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 创建一个包含模型和（可选的）优化器状态的字典
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)
        # 调用dist_cp.save_state_dict函数保存状态字典，使用之前定义的分布式写入器和默认的保存计划器。
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    # 确保所有进程都已完成保存操作
    dist.barrier()
    # 记录保存操作结束的时间
    t1 = time.perf_counter()
    # 如果当前进程是主进程，打印保存完成的信息和所需时间。
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )

def save_model_checkpoint(
    model, # 要保存的 PyTorch 模型
    optimizer, # 用于训练模型的优化器
    rank, # 在分布式训练环境中的进程等级
    cfg, # 配置对象，包含有关模型和检查点的信息。
    epoch=1, # 当前训练的轮次，默认为 1。
):
    """saving model via rank0 cpu streaming and full_state_dict
    在分布式训练环境中保存模型的状态字典, 专注于仅由主进程（rank 0）执行的保存操作，且保存的是完整的状态字典，而不是分片的状态字典。
    """

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        # fullstate_save_policy 指定保存策略，通常包括将数据卸载到 CPU 并由主进程执行保存。
        # cpu_state 获取模型的状态字典。
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        print(f"--> saving model ...")
        # create save path 构建保存路径
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model 使用 torch.save 保存模型的状态字典
        torch.save(cpu_state, save_full_path)

        
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")
      


def load_model_checkpoint(model, rank, cfg):
    """这个函数的目的是在分布式训练环境中，仅在主节点（rank 0）上加载模型检查点。
       如果检查点不存在或其他节点尝试执行此函数，它将不执行任何操作。"""

    if rank != 0:
        return

    # 构造了模型检查点文件的路径
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # 检查构造的路径是否指向一个文件。如果不是文件，说明检查点不存在。
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return

    # 加载检查点文件
    model_checkpoint = torch.load(full_state_dict_model_path)
    # 将加载的检查点（一个状态字典）应用到传入的model上, 这样可以恢复模型到之前保存的状态。
    model.load_state_dict(model_checkpoint)

    
    print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    # 打印当前节点（rank）调用优化器状态保存的消息
    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...
    # 使用 FSDP.full_optim_state_dict 函数获取优化器的完整状态字典
    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")
    # 只有在主节点上执行保存操作
    if rank == 0:
        # 构造保存文件的文件夹名称
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        # 使用当前工作目录和文件夹名称构造保存目录的路径
        save_dir = Path.cwd() / folder_name
        # 创建保存目录，如果已存在则忽略。
        save_dir.mkdir(parents=True, exist_ok=True)
        # 构造优化器状态文件的名称，包括模型名和训练周期。
        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        # 获取优化器状态文件的完整路径
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")
        # 使用PyTorch的 torch.save 方法将优化器状态保存到指定路径
        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    这个函数的目的是在分布式训练环境中加载优化器的检查点
    """

    # 检查提供的优化器检查点路径是否指向一个文件。如果不是，表示检查点不存在。
    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        # 如果当前节点是主节点（rank 0），则使用torch.load加载优化器检查点文件
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model,model_path):
    '''这个函数的目的是加载一个分片模型的状态到单个GPU上'''
    # 创建一个 FileSystemReader 对象，用于从指定的 model_path 路径读取数据。FileSystemReader 可能是一个用于读取文件系统中数据的工具。
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict() # 包含了模型的当前状态
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict, # 指定要加载的状态字典
                storage_reader= FileSystemReader(model_path), # 再次创建一个 FileSystemReader 用于读取模型路径。
                no_dist=True, # 加载过程不涉及分布式计算，因为目标是单个GPU。
            )
    # 使用状态字典中的“model”键对应的值更新模型的状态。这是将加载的状态应用于传入的模型对象。
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model