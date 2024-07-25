'''
@Description: 
@Author: shenlei
@Date: 2024-01-15 13:06:56
@LastEditTime: 2024-01-15 13:08:02
@LastEditors: shenlei
'''
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

def reranker_tokenize_preproc(
    query: str, # 用户的查询语句
    passages: List[str], # 需要进行重排序的文本段落
    tokenizer=None,
    max_length: int=512,
    overlap_tokens: int=80, # 在分割长文本时，两个分片之间重叠的token数量
    ):
    assert tokenizer is not None, "Please provide a valid tokenizer for tokenization!"
    #  获取分词器定义的分隔符（通常是[SEP]）的ID，用于后续在输入序列中插入分隔符。
    sep_id = tokenizer.sep_token_id

    def _merge_inputs(chunk1_raw, chunk2):
        # 合并两个文本块的分词结果
        chunk1 = deepcopy(chunk1_raw)
        # 对chunk1的input_ids和attention_mask进行扩展，添加chunk2的对应信息，
        # 并在input_ids的末尾加上分隔符ID。
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        # 如果 chunk1 包含 token_type_ids，则为 chunk2 中的每个token（包括一个额外的分隔符）设置token类型ID为1，
        # 并将这些ID添加到 chunk1 的 token_type_ids 中。
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+1)]
            chunk1['token_type_ids'].extend(token_type_ids)
        # 返回更新后的 chunk1 对象
        return chunk1
    # 对查询query进行编码
    query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
    # 计算单个文段能有的最大长度。它由总允许的最大长度max_length减去查询编码的长度和一个额外的空间（用于分隔符）来确定。
    max_passage_inputs_length = max_length - len(query_inputs['input_ids']) - 1
    # 检查剩余可用于文段的长度是否合理（大于100个tokens）。如果不是，将断言失败并提示查询太长。
    assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
    # ：计算重叠tokens的实际数量
    overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length//4)
    
    res_merge_inputs = []
    res_merge_inputs_pids = []
    # 循环遍历所有传入的passages
    for pid, passage in enumerate(passages):
        # 对每个文段进行编码
        passage_inputs = tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
        # 计算每个编码后的文段长度
        passage_inputs_length = len(passage_inputs['input_ids'])
        # 如果文段长度小于或等于max_passage_inputs_length，则直接将查询和文段的编码合并
        if passage_inputs_length <= max_passage_inputs_length:
            qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
            res_merge_inputs.append(qp_merge_inputs)
            res_merge_inputs_pids.append(pid)
        else:
            '''
            主要作用是在文段长度超过了给定的最大长度限制时，将文段分割成多个小段，
            每个小段长度不超过max_passage_inputs_length，并且相邻的小段之间会有重叠部分，以保持上下文的连贯性。
            '''
            # 当前处理的文段小片段的起始索引
            start_id = 0
            # 循环处理，确保整个文段都被处理
            while start_id < passage_inputs_length:
                # 计算当前小片段的结束索引，确保了每个小片段的长度不会超过模型可以处理的最大长度。
                end_id = start_id + max_passage_inputs_length
                # 建一个新的字典sub_passage_inputs，它包含从当前起始索引start_id到结束索引end_id的文段片段。
                sub_passage_inputs = {k:v[start_id:end_id] for k,v in passage_inputs.items()}

                # 更新start_id以准备处理下一个小片段。如果end_id小于文段的总长度，说明还有文段没有被处理，
                # 这时将start_id设置为end_id减去重叠tokens数overlap_tokens_implt，这样下一个小片段会与当前小片段重叠，
                # 保持了文段的连贯性。如果end_id已经达到或超过了文段的总长度，说明这是最后一个小片段，
                # 直接将start_id设置为end_id，循环将在下一轮结束。
                start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id
                # 将查询的编码和当前处理的文段小片段的编码合并
                qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                # 每次合并后，将合并的结果和文段的索引（pid）分别添加到res_merge_inputs和res_merge_inputs_pids列表中。
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
    # 函数返回两个列表：res_merge_inputs（包含所有合并后的编码）和res_merge_inputs_pids（记录了每个合并编码对应的原文段索引）
    return res_merge_inputs, res_merge_inputs_pids