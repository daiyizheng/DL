'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 18:52:01
@LastEditTime: 2024-01-07 00:17:13
@LastEditors: shenlei
'''
from typing import cast, List, Dict, Union

import numpy as np
from torch import nn
from BCEmbedding import EmbeddingModel
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.c_mteb.yd_dres_model')


class YDDRESModel(nn.Module):
    def __init__(
            self,
            model_name_or_path: str = None,
            pooler: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            passage_instruction_for_retrieval: str = None,
            batch_size: int = 160,
            max_length: int = 512,
            **kwargs
        ):
        self.model = EmbeddingModel(model_name_or_path=model_name_or_path, pooler=pooler, **kwargs)

        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.passage_instruction_for_retrieval = passage_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        # 一个布尔值，用于指示模型名称是否包含特定的字符串（"e5-base"或"e5-large"）
        self.instruction_for_all = "e5-base" in model_name_or_path or "e5-large" in model_name_or_path

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''用于编码查询文本，生成用于检索任务的嵌入。
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        logger.info(f'##BCEmbedding##: using encode_queries with instruction: {self.query_instruction_for_retrieval}')
        return self.model.encode(
            queries, # 需要编码的查询文本列表
            batch_size=self.batch_size,  # 批处理的大小
            max_length=self.max_length, # 文本的最大长
            normalize_to_unit=self.normalize_embeddings, # 是否对嵌入进行归一化
            query_instruction=self.query_instruction_for_retrieval, # 查询指令
            enable_tqdm=True # 开启进度条显示
            )

    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        logger.info(f'##BCEmbedding##: using encode_corpus with instruction: {self.passage_instruction_for_retrieval}')
        # 检查corpus列表的第一个元素是否为字典类型。
        # 这是为了处理两种不同格式的输入：一种是包含标题和文本的字典，另一种是直接的文本字符串。
        if isinstance(corpus[0], dict):
            # 将每个文档的标题和文本组合成一个字符串，并去除首尾空格。如果文档没有标题，将使用空字符串代替。
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            # 如果输入直接是文本字符串，直接使用输入的corpus列表。
            input_texts = corpus
        # 生成语料库文本的嵌入
        return self.model.encode(
            input_texts,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            query_instruction=self.passage_instruction_for_retrieval,
            enable_tqdm=True
            )
    
    def encode(
            self,
            sentences: Union[str, List[str]], # 参数sentence，这个参数可以是单个字符串（代表一个句子）或字符串列表（代表多个句子）。
            **kwargs
        ):
        # 是否对所有编码操作使用相同的指令
        if self.instruction_for_all:
            assert len(self.query_instruction_for_retrieval) > 0
            instruction = self.query_instruction_for_retrieval
        else:
            instruction = None
            
        logger.info(f'##BCEmbedding##: instruction for all: {self.instruction_for_all}; using encode with instruction: {instruction}')
        # 生成文本的嵌入
        return self.model.encode(
            sentences,
            batch_size=self.batch_size, 
            max_length=self.max_length,
            normalize_to_unit=self.normalize_embeddings,
            query_instruction=instruction,
            enable_tqdm=True
        )
