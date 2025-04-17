# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# 自动将Python函数转换成命令行接口
import fire
import os
import sys
sys.path.insert(0, "./src")

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=False, # Enable safety check woth Saleforce safety flan t5
    enable_llamaguard_content_safety: bool = False,
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    # 如果提供了prompt_file参数，会检查该文件是否存在。
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        # 根据prompt_file或标准输入读取对话文本
        dialogs= read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        # isatty()是一个方法，用于判断标准输入（stdin）是否连接到终端。
        # 如果返回False，说明标准输入来自一个管道或文件而不是交互式终端。
        # 在这种情况下，代码读取标准输入中的所有行（sys.stdin.readlines()），
        # 并将它们连接成一个字符串，每行之间用换行符分隔。
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        # 如果sys.stdin.isatty()返回True，则表明没有提供输入数据, 使用sys.exit(1)退出。
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialogs:\n{dialogs}")
    print("\n==================================\n")


    # Set the seeds for reproducibility
    # 设置了PyTorch的随机种子，以确保实验的可重复性。
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    # 加载指定名称的模型
    model = load_model(model_name, quantization)
    # 如果提供了peft_model（可能指代部分预训练模型），则使用load_peft_model来加载或修改模型。
    if peft_model:
        model = load_peft_model(model, peft_model)
    if use_fast_kernels:
        """
        如果启用了use_fast_kernels，代码尝试使用optimum.bettertransformer中的BetterTransformer来转换模型。
        这是为了利用特定的加速技术，如Flash Attention或Xformer的内存高效内核，以加快批量输入的推理速度。
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    # 加载了预训练的LLaMA Tokenizer，并向其中添加了特殊的填充token(<PAD>)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    # 格式化用户对话, 将文本分割成tokens、添加必要的特殊tokens等，以便模型能够正确理解和处理这些文本。
    chats = format_tokens(dialogs, tokenizer)
    # 不计算梯度
    with torch.no_grad():
        # 遍历chats中的每个对话（chat），idx是当前对话的索引。
        for idx, chat in enumerate(chats):
            # 构建一个或多个安全性检查器, 然后对每个对话内容进行安全性检查，汇总结果以确定是否所有检查都认为内容是安全的。
            safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_saleforce_content_safety,
                                        enable_llamaguard_content_safety
                                        )
            # Safety check of the user prompt
            safety_results = [check(dialogs[idx][0]["content"]) for check in safety_checker]
            are_safe = all([r[1] for r in safety_results])
            # 如果用户输入被认为是安全的，则继续；如果不安全，打印不安全的原因并退出程序。
            if are_safe:
                print(f"User prompt deemed safe.")
                print("User prompt:\n", dialogs[idx][0]["content"])
                print("\n==================================\n")
            else:
                print("User prompt deemed unsafe.")
                for method, is_safe, report in safety_results:
                    if not is_safe:
                        print(method)
                        print(report)
                print("Skipping the inferece as the prompt is not safe.")
                sys.exit(1)  # Exit the program with an error status
            # 将对话转换为PyTorch张量，并添加一个维度（batch维度），然后将数据移动到GPU上（如果可用）进行计算。
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            tokens= tokens.to("cuda:0")
            # 使用模型的generate方法生成文本
            outputs = model.generate(
                input_ids=tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
            # 使用tokenizer将模型的输出（outputs[0]）解码成文本。
            # 参数skip_special_tokens=True意味着在解码过程中将忽略特殊tokens.
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Safety check of the model output
            # 与先前对用户输入进行的类似，这里对模型的输出也进行安全性检查。
            # 使用相同的安全检查器（safety_checker）来评估生成文本的安全性。
            safety_results = [check(output_text) for check in safety_checker]
            are_safe = all([r[1] for r in safety_results])
            # 如果模型的输出被认为是安全的，则打印一条消息和模型生成的文本。
            # 如果输出被认为不安全，则打印一条警告信息和相关的安全报告。
            if are_safe:
                print("User input and model output deemed safe.")
                print(f"Model output:\n{output_text}")
                print("\n==================================\n")

            else:
                print("Model output deemed unsafe.")
                for method, is_safe, report in safety_results:
                    if not is_safe:
                        print(method)
                        print(report)



if __name__ == "__main__":
    # 使用了fire库来自动将main函数转换为命令行接口，使得可以通过命令行参数来配置和运行此脚本。
    fire.Fire(main)
