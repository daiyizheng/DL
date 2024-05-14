# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

def fsdp_auto_wrap_policy(model, transformer_layer_name):
    '''此函数的核心价值在于为 FSDP 提供了一种灵活的自动包装机制，能够根据模型的具体层次结构和特性，智能地决定哪些部分需要进行分布式并行处理。
       用于创建一个自动包装策略，用于完全分布式数据并行（FSDP）中决定哪些模型层应当被包装。
       这个策略综合了基于 Lambda 函数的自定义策略和针对特定transformer层的策略。
    '''
    import functools

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        '''这个 Lambda 函数用于判断一个给定的模块是否应该被 FSDP 自动包装。
           它检查模块是否没有子模块、是否有权重且权重需要梯度。'''
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False
    # 使用 functools.partial 创建一个偏函数，该函数将 lambda_policy_fn 作为参数传递给 lambda_auto_wrap_policy。
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    # 创建一个偏函数用于transformer层的自动包装策略。
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # 指定一组类别作为transformer层
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )
    # 使用 _or_policy 将两种策略组合起来。
    # 这个组合策略会对每个模块应用两种策略，如果任一策略返回 True，则该模块将被 FSDP 包装。
    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    # 返回组合后的自动包装策略
    return auto_wrap_policy