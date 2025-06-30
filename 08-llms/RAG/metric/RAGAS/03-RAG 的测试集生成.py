#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :03-RAG 的测试集生成.py
@Description  :
@Time         :2025/06/30 10:47:13
@Author       :flow-laic
@Version      :1.0
'''
import os
from langchain_community.document_loaders import DirectoryLoader

path = "/Users/a1-6/Documents/projects/DL/08-llms/RAG/metric/datasets"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=os.environ["api_key"], base_url=os.environ["base_url"]))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.environ["api_key"], base_url=os.environ["base_url"]))

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)