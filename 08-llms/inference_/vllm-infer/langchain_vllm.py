# -*- encoding: utf-8 -*-
'''
Filename         :langchain_vllm.py
Description      :
Time             :2024/05/19 17:37:39
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from langchain_community.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key = "EMPTY",
    openai_api_base = "http://localhost:9908/v1",
    model_name = "xxx",
    max_tokens = 1024
)

print(llm("白术是什么中药?"))