# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools

# 标记指令的开始和结束。
B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    '''dialog（对话数据）和tokenizer（用于编码文本的tokenizer）。'''
    # 对话中的提示（用户输入）和回答（模型回复）分别被编码。提示前添加了开始标记（bos_token）和[INST]，后添加[/INST]。
    # 回答后添加了结束标记（eos_token）。
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    # 使用itertools.chain.from_iterable和zip将提示和回答的tokens交替合并，形成一整个对话的tokens序列。
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    # 用于模型训练的标签, 提示部分的tokens（偶数索引）被转换为-100（在PyTorch中通常用于忽略计算损失的tokens），而回答部分的tokens保持不变。
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]
    # 这里将对话tokens（dialog_tokens）和相应的标签（labels_tokens）合并成一个字典。
    # input_ids 是模型的输入tokens，而 labels 是用于模型训练时的目标输出。
    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }
    # 函数返回的字典中还包含了attention_mask,
    # attention_mask是一个列表，其中的每个元素对应于input_ids中的一个token。
    # 在这里，所有的值都设置为1，表示模型应该关注所有的tokens。
    # 这在处理变长输入时特别有用，可以帮助模型区分实际数据和填充数据。
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    # 加载特定的数据集
    dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)
    # 使用map方法对数据集中的每个样本进行预处理
    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True, # 以批处理的方式进行映射
        remove_columns=list(dataset.features),) # 移除数据集中不需要的列

    # 用于存储每个消息的子消息ID
    nodes = {}
    # 存储每个消息ID对应的文本内容
    messages = {}
    # 存储没有父消息的消息ID，即对话的起始消息
    root_ids = []

    for data in dataset:
        # 如果一个消息有parent_id，则将其添加到对应父消息的子消息列表中；
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            # 如果没有parent_id，则视为对话的根消息。
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        '''用于递归地构建完整的对话线索(thread)'''
        # follow函数接收当前的对话线索（thread）和当前消息的ID（current_id），并将当前消息的文本添加到对话线索中。
        thread = copy.copy(thread) + [messages[current_id]]
        # 如果当前消息ID在nodes字典中（即它有子消息），则递归地调用follow函数为每个子消息构建新的对话线索。
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            # 如果当前消息没有子消息，返回包含当前线索的列表。
            return [thread]

    def get_threads_from_root(root_id):
        #  存储所有从这个根消息开始的对话线索
        all_threads = []
        # 对于每个子消息 ID（在 nodes[root_id] 中），函数递归地调用 follow 函数来构建线索，
        # 并将结果添加到 all_threads。
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads
    # 首先，使用 filter 函数保留那些作为对话根部的消息。
    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    # 然后，使用 map 函数将每个根消息转换为一个或多个完整的对话线索。
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    # 最后，将所有线索展平为一个单一的线索列表。
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

    def to_dialog(thread):
        '''将线索转换为对话格式，其中包括角色（用户或助手）和内容。'''
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}
    # 使用 map 函数应用 to_dialog，将每个线索转换为格式化的对话。
    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
    # 最后，使用 map 函数应用 tokenize_dialog 函数，将对话格式化为模型训练所需的格式。这一步包括将文本转换为tokens。
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset
