'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-01-19 12:11:05
@LastEditors: shenlei
'''
import logging
import torch

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

from .utils import reranker_tokenize_preproc

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('BCEmbedding.models.RerankerModel')


class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/bce-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        # for advanced preproc of tokenization
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
    
    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],  # 句子对
            batch_size: int = 256,
            max_length: int = 512,
            enable_tqdm: bool=True,
            **kwargs
        ):
        '''用于计算句子对相似度或者相关性得分'''
        if self.num_gpus > 1:
            # 在多GPU环境下，可以并行处理更多的数据，从而提高整体处理速度。
            batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
                # 当前批次的句子对
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
                # 对得分应用sigmoid函数，将logits转换为介于0和1之间的得分。
                # 这是因为模型的输出通常是logits，需要转换成概率表示相关性或相似度的得分。
                scores = torch.sigmoid(scores)
                scores_collection.extend(scores.cpu().numpy().tolist())
        # 果scores_collection只包含一个元素，则直接返回这个元素，这适用于只计算了一个句子对得分的情况。
        if len(scores_collection) == 1:
            return scores_collection[0]
        # 如果有多个得分，则返回整个得分列表。
        return scores_collection

    def rerank(
            self,
            query: str, # 一个字符串，表示用户的查询。
            passages: List[str], # 一个字符串列表，包含需要对其进行重排序的文段。
            batch_size: int=256,
            **kwargs # 接收任意数量的关键字参数
        ):
        '''对一组文段（passages）相对于一个查询（query）进行重排序'''
        # 先过滤掉非字符串类型或空字符串的文段，然后将每个文段裁剪到最多128000个字符，这是为了确保文段的有效性和减少处理过长文本的负担。
        passages = [p[:128000] for p in passages if isinstance(p, str) and 0 < len(p)]
        # 检查查询是否为空或者文段列表是否为空，这是一种快速失败机制，避免在输入数据无效时执行不必要的计算。
        if query is None or len(query) == 0 or len(passages) == 0:
            return {'rerank_passages': [], 'rerank_scores': []}
        
        # 将查询和每个文段组合成模型可以处理的格式，同时考虑到最大长度和令牌重叠的要求。
        sentence_pairs, sentence_pairs_pids = reranker_tokenize_preproc(
            query, passages, 
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            overlap_tokens=self.overlap_tokens,
            )

        # batch inference
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        # 存储所有查询-文段对的得分
        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(sentence_pairs), batch_size):
                # 对当前批次的查询-文段对进行填充，以确保它们具有相同的长度，方便模型处理。
                batch = self.tokenizer.pad(
                        sentence_pairs[k:k+batch_size],
                        padding=True, # 启用填充功能，这意味着所有的序列将被填充到当前批次中最长序列的长度。
                        max_length=None, # None意味着不强制序列长度上限，而是自动根据批次中最长的序列确定填充长度。
                        pad_to_multiple_of=None, # 允许用户指定填充长度必须是某个数的倍数。设置为None表示不启用这个功能。
                        return_tensors="pt"
                    )
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                scores = self.model(**batch_on_device, return_dict=True).logits.view(-1,).float()
                # 使用sigmoid函数将logits转换为介于0和1之间的概率值，表示每个文段与查询的匹配程度。
                scores = torch.sigmoid(scores)
                tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        # 所有元素初始化为0，这个列表用于存储每个文段的最高得分。
        merge_scores = [0 for _ in range(len(passages))]
        # 循环遍历sentence_pairs_pids（每个查询-文段对的文段索引）和tot_scores（计算得到的得分）
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            # 对于每个文段，将其对应的得分更新为已有得分和当前得分中的较大值。
            # 这是因为一个文段可能与查询组成多个查询-文段对（尤其是在文段被分割处理的情况下），
            # 而我们希望保留最高的得分作为该文段的最终得分。
            merge_scores[pid] = max(merge_scores[pid], score)

        # 使用NumPy的argsort函数对merge_scores进行降序排序，
        # 并获取排序后的索引，这些索引用于按得分高低排序文段。
        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        # 存储排序后的文段和对应的得分
        sorted_passages = []
        sorted_scores = []
        # 根据索引mid，将对应的得分添加到sorted_scores列表中。
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_passages.append(passages[mid])
        
        return {
            'rerank_passages': sorted_passages, # 根据得分降序排序后的文段
            'rerank_scores': sorted_scores, # 排序后文段对应的得分
            'rerank_ids': merge_scores_argsort.tolist() # 排序后的索引，这可以用来追踪排序后的文段在原列表中的位置。
        }
