# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model


def main(
    # 模型名称
    model_name,
    # 如果提供，则加载预先训练好的模型。
    peft_model: str=None,
    # 是否对模型进行量化，以减小模型大小和提高推理速度。
    quantization: bool=False,
    # 生成的最大tokens数量
    max_new_tokens =100, #The maximum numbers of tokens to generate
    # 包含提示文本的文件路径
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    # 是否使用采样方法生成文本
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    # 生成文本的最小长度
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    # 是否使用缓存加速解码
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    # 这些参数用于控制文本生成的策略和样式
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    # 启用不同的内容安全检查
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    # 用于tokenizer的最大填充长度
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    # 启用PyTorch加速变换器的SDPA，以提高性能。
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    # 是否提供了prompt_file
    if prompt_file is not None:
        # 如果提供了prompt_file，则断言该文件确实存在，否则抛出异常。
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        # 读取文件
        with open(prompt_file, "r") as f:
            # 读取文件所有行，并使用换行符\n将它们合并成一个字符串，这个字符串即为用户的提示文本。
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        # 如果没有提供prompt_file，则检查标准输入是否为终端。
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    # 根据提供的参数（是否启用各种安全检查功能）获取安全检查器的实例
    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_salesforce_content_safety,
                                        enable_llamaguard_content_safety
                                        )

    # Safety check of the user prompt
    # 对用户的提示文本进行安全检查。这里用到了列表推导式，对每个安全检查器执行检查操作。
    safety_results = [check(user_prompt) for check in safety_checker]
    # 判断所有的安全检查结果是否都是安全的
    are_safe = all([r[1] for r in safety_results])
    # 如果提示文本被认为是安全的（are_safe为真），则打印出安全确认消息和用户的提示文本。
    if are_safe:
        print("User prompt deemed safe.")
        print(f"User prompt:\n{user_prompt}")
    else:
        # 否则，如果有任何安全检查未通过，则打印出不安全的原因，
        # 并退出程序（使用sys.exit(1)表示错误状态退出）。
        print("User prompt deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
        print("Skipping the inference as the prompt is not safe.")
        sys.exit(1)  # Exit the program with an error status

    # Set the seeds for reproducibility
    # 设置了PyTorch的随机数种子，以确保实验的可重复性。
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    # 加载指定的模型。如果quantization参数为真，则会加载量化后的模型，这可以减小模型的大小并提高运行效率。
    model = load_model(model_name, quantization)
    # 如果提供了peft_model参数，则进一步使用load_peft_model函数来加载预先训练好的模型。
    # 这通常用于特定任务的微调模型。
    if peft_model:
        model = load_peft_model(model, peft_model)
    # 设置为评估模式
    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            #  如果导入成功，则将模型转换为使用更高效的内核，例如Flash Attention或Xformer内存高效内核。
            #  这可以在处理批量输入时加速推理。
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    # 加载与模型相对应的分词器
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # 将分词器的填充token设置为结束符token
    tokenizer.pad_token = tokenizer.eos_token
    # 使用分词器处理用户的提示文本，将其转换为模型可接受的格式。这包括填充和截断操作，以及将数据转换为PyTorch张量。
    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
    # 将处理后的数据移动到GPU上，以便进行高效计算。
    batch = {k: v.to("cuda") for k, v in batch.items()}
    # 开始计时
    start = time.perf_counter()
    # 在推理过程中不需要梯度计算，禁用梯度可以减少内存使用并提高计算效率。
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, # 是否使用采样策略
            top_p=top_p, # 控制生成文本多样性
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    # 计算推理过程所用的时间（毫秒）
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    # 使用分词器将生成的tokens解码成文本，并跳过特殊tokens。
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Safety check of the model output
    # 对生成的文本进行安全检查，同样使用列表推导式遍历所有安全检查器。
    safety_results = [check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt) for check in safety_checker]
    # 判断生成的文本是否全部通过安全检查
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        # 如果生成的文本安全，则打印确认信息和生成的文本。
        print("User input and model output deemed safe.")
        print(f"Model output:\n{output_text}")
    else:
        # 如果检查结果表明文本不安全，则打印不安全的原因。
        print("Model output deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
                

if __name__ == "__main__":
    fire.Fire(main)
