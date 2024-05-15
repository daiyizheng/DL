# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

import torch
from vllm import LLM
from vllm import LLM, SamplingParams


torch.cuda.manual_seed(42)
torch.manual_seed(42)

def load_model(model_name, tp_size=1):
    '''加载指定名称的模型'''
    # tensor_parallel_size参数可能用于设置模型并行度
    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm

def main(
    model,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    while True:
        # 如果没有提供初始提示，程序会请求用户输入一个提示。
        if user_prompt is None:
            user_prompt = input("Enter your prompt: ")
            
        print(f"User prompt:\n{user_prompt}")
        # 根据给定的采样参数（控制生成文本的随机性和创造性）创建一个SamplingParams实例。
        print(f"sampling params: top_p {top_p} and temperature {temperature} for this inference request")
        sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)
        
        # 调用模型的generate方法，生成基于用户提示的文本。
        outputs = model.generate(user_prompt, sampling_params=sampling_param)
        # 打印模型的输出
        print(f"model output:\n {user_prompt} {outputs[0].outputs[0].text}")
        # 请求用户输入下一个提示，或者按Enter键退出循环。
        user_prompt = input("Enter next prompt (press Enter to exit): ")
        if not user_prompt:
            break

def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1, # 张量并行大小
    max_new_tokens=100, # 生成文本的最大新tokens数
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    # 加载指定的模型
    model = load_model(model_name, tp_size)
    # 调用 main 函数，启动与模型的交互过程。
    main(model, max_new_tokens, user_prompt, top_p, temperature)

if __name__ == "__main__":
    # fire.Fire 接受 run_script 函数并创建一个命令行界面，
    # 该界面自动解析命令行参数并将它们传递给 run_script 函数。
    fire.Fire(run_script)
