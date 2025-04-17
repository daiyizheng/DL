# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
# 导入类型注解模块, List用于指定列表类型，Literal和TypedDict用于创建具有特定结构的字典类型。
from typing import List, Literal, TypedDict

# Role 是一个类型别名，表示角色类型，它只能是 "user" 或 "assistant" 两种字面量中的一种。
Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role # 角色
    content: str # 内容

# Dialog是Message类型的列表，代表一组对话消息。
Dialog = List[Message]
# 定义了一些特殊的标记字符串，用于在处理文本时标记不同部分的开始和结束。
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def format_tokens(dialogs, tokenizer):
    '''用于将对话数据转换成模型可以理解的格式'''
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            # 将系统消息（如果存在）与用户消息合并，确保对话以用户消息开始，交替进行。
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        # 使用tokenizer对每条消息进行编码，然后将编码后的消息组合成一个长序列。
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        # 确保对话以用户消息结束
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        # 将对话的每个部分编码成一个长序列，用于模型的输入。
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens
        

def read_dialogs_from_file(file_path):
    '''从文件中读取对话数据'''
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs
