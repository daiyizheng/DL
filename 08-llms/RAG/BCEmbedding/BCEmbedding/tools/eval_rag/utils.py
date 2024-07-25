'''
@Description: 
@Author: shenlei
@Date: 2023-12-29 17:09:31
@LastEditTime: 2024-01-27 13:12:00
@LastEditors: shenlei
'''
import json
import os.path as osp
import pandas as pd

from typing import List
# 用于读取某个目录下的文件或数据
from llama_index import SimpleDirectoryReader
# 用于解析节点数据，将原始数据转换为特定的格式或结构
from llama_index.node_parser import SimpleNodeParser

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.indices.query.schema import QueryBundle, QueryType
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever

from datasets import load_dataset

# Prompt to generate questions
qa_generate_prompt_tmpl_en = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
Generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.\
"""

qa_generate_prompt_tmpl_zh = """\
以下是上下文信息。

---------------------
{context_str}
---------------------

深入理解上述给定的上下文信息，而不是你的先验知识，根据下面的要求生成问题。

要求：你是一位教授，你的任务是为即将到来的考试设置{num_questions_per_chunk}个问题。你应该严格基于上下文信息，来设置多种多样的问题。\
你设置的问题不要包含选项，也不要以“问题1”或“问题2”为开头。\
将问题限制在所提供的上下文信息中。\
"""

def load_dataset_from_huggingface(dataset_name='maidalun1020/CrosslingualMultiDomainsDataset'):
    '''从Hugging Face的数据集库中加载指定的数据集，并对加载的数据进行处理，最后以字典形式返回处理后的数据集。'''
    # 从Hugging Face加载指定名称的数据集。split='dev'指定了加载数据集的哪个部分，这里是开发集（dev set）。
    datasets_raw = load_dataset(dataset_name, split='dev')
    datasets = {}
    # ? 这里可能报错？dataset_raw是一个对象，而不是一个列表
    for dataset_raw in datasets_raw:
        for k in dataset_raw:
            # 将每个键对应的值从JSON字符串解析成Python对象
            dataset_raw[k] = json.loads(dataset_raw[k])
        # 从dataset_raw中移除键为'pdf_file'的项，并将其值赋给变量pdf_file
        pdf_file = dataset_raw.pop('pdf_file')
        datasets[pdf_file] = dataset_raw
    return datasets

# function to clean the dataset
def filter_qa_dataset(qa_dataset):
    # Extract keys from queries and relevant_docs that need to be removed
    # 这个qa_dataset应该是一个包含查询（queries）、相关文档（relevant_docs）和文档语料（corpus）的问答数据集对象。
    # 定义了一个集合推导式，用于找出需要从查询（queries）中移除的键
    queries_relevant_docs_keys_to_remove = {
        k for k, v in qa_dataset.queries.items()
        if 'Here are 2' in v or 'Here are two' in v
    }

    # Filter queries and relevant_docs using dictionary comprehensions
    # 遍历qa_dataset.queries中的每个键值对，只保留那些键不在queries_relevant_docs_keys_to_remove集合中的项。
    filtered_queries = {
        k: v for k, v in qa_dataset.queries.items()
        if k not in queries_relevant_docs_keys_to_remove
    }
    # 遍历qa_dataset.relevant_docs中的每个键值对，只保留那些键不在queries_relevant_docs_keys_to_remove集合中的项。
    filtered_relevant_docs = {
        k: v for k, v in qa_dataset.relevant_docs.items()
        if k not in queries_relevant_docs_keys_to_remove
    }

    # Create a new instance of EmbeddingQAFinetuneDataset with the filtered data
    return EmbeddingQAFinetuneDataset(
        queries=filtered_queries,
        corpus=qa_dataset.corpus,
        relevant_docs=filtered_relevant_docs
    )

def display_results(embedding_name, reranker_name, eval_results):
    # 存储每个评估结果的指标字典
    metric_dicts = []
    for eval_result in eval_results:
        # 从当前的评估结果中提取指标值字典
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    # 创建一个pandas DataFrame，这个DataFrame包含所有评估结果的指标。
    full_df = pd.DataFrame(metric_dicts)
    # 计算hit_rate指标的平均值，并将其转换为百分比形式。
    hit_rate = 100*full_df["hit_rate"].mean()
    # 计算mrr（平均倒数排名）指标的平均值，并将其转换为百分比形式。
    mrr = 100*full_df["mrr"].mean()
    # 这个DataFrame包含嵌入式模型的名称、重排序器的名称、计算得到的平均hit_rate、平均mrr和评估结果的数量。
    metric_df = pd.DataFrame(
        {"Embedding": [embedding_name], "Reranker": [reranker_name], "hit_rate": [hit_rate], "mrr": [mrr], "nums": [len(eval_results)]}
    )
    return metric_df

def extract_data_from_pdf(pdf_paths,  # PDF文件路径或路径列表
                          llm=None,
                          chunk_size=512, # 文档分块大小
                          split=[0,36], # 用于指定要处理的文档范围
                          qa_generate_prompt_tmpl=qa_generate_prompt_tmpl_en # 用于生成QA对的模板
                          ):
    if isinstance(pdf_paths, str) and osp.exists(pdf_paths):
        pdf_paths = [pdf_paths]
    # 从给定的PDF文件路径列表中读取数据，并调用load_data方法加载文档数据。
    documents = SimpleDirectoryReader(input_files=pdf_paths).load_data()
    # 创建一个SimpleNodeParser实例
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    # 根据split参数指定的范围，从加载的文档数据中提取节点。
    nodes = node_parser.get_nodes_from_documents(documents[split[0]:split[1]])
    # 生成QA数据集
    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
    )
    # filter out pairs with phrases `Here are 2 questions based on provided context`
    # 过滤QA数据集，移除包含特定短语的QA对。
    qa_dataset = filter_qa_dataset(qa_dataset)
    # 返回提取的节点和过滤后的QA数据集
    return nodes, qa_dataset

# Define Retriever
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search
    执行向量搜索和知识图谱搜索"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        reranker: None,
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._reranker = reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query.
        该方法接收一个QueryBundle实例作为参数，返回一个NodeWithScore实例的列表，表示检索到的节点及其得分。
        """
        # 根据query_bundle执行向量搜索，并将结果保存在retrieved_nodes中。
        retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
        # 检查是否有重排序器
        if self._reranker is None:
            # 将检索到的节点列表裁剪为前5个节点
            retrieved_nodes = retrieved_nodes[:5]
        else:
            # 对检索到的节点进行重排序，并更新retrieved_nodes列表。
            retrieved_nodes = self._reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        return retrieved_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.
        直接调用了同步的_retrieve方法来执行实际的检索操作。这意味着虽然方法本身是异步的，但实际执行过程可能并不涉及异步IO操作。
        """
        return self._retrieve(query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        # 如果str_or_query_bundle是一个字符串，则将其封装成一个QueryBundle实例。
        if isinstance(str_or_query_bundle, str):
            str_or_query_bundle = QueryBundle(str_or_query_bundle)
        # 调用_aretrieve方法执行异步检索，并使用await关键字等待其完成。
        return await self._aretrieve(str_or_query_bundle)