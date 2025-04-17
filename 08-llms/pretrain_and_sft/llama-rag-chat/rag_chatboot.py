# -*- encoding: utf-8 -*-
'''
Filename         :rag_chatboot.py
Description      :
Time             :2024/05/24 09:46:58
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from dotenv import load_dotenv
load_dotenv(dotenv_path="10-langchain/.env")

from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 数据处理
DATA_PATH = "08-llms/data"
DB_FAISS_PATH = "08-llms/data/vectorstore/db_faiss"


# 使用PyPDFDirectoryLoader加载整个目录 (还可以使用 PyPDFLoader 加载单个文件)

loader = PyPDFDirectoryLoader(DATA_PATH)

documents = loader.load()

# 将加载的文档分割成更小的块
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 200)

splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs = {"device": "cpu"})


# 将embedding存入向量数据库
db = FAISS.from_documents(splits, embedding=embeddings)

# 保存到本地
db.save_local(DB_FAISS_PATH)

import langchain
from queue import Queue
from typing import Any
from langchain_community.llms import VLLMOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from anyio.from_thread import start_blocking_portal

langchain.debug = True

# 向量数据文件路径
DB_FAISS_PATH = "08-llms/data/vectorstore/db_faiss"

# LLaMA-2 7B host port
LLAMA2_7B_HOSTPATH = "http://localhost:9909/v1"

model_dict = {
    "7b-chat" : LLAMA2_7B_HOSTPATH,
}

# 系统提示
system_message = {"role" : "system",
                  "content" : "You are a helpful assistant."}
# 加载embedding

embeddings = HuggingFaceEmbeddings(model_name = "/slurm/home/yrd/shaolab/daiyizheng/resources/hf_weights/sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs = {"device" : "cpu"})

# 加载 db

db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


# 创建一个模型服务实例
# API : https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/openai.py#L144

llm = VLLMOpenAI(
    openai_api_key = "EMPTY",
    openai_api_base = LLAMA2_7B_HOSTPATH,
    model_name = "/slurm/home/yrd/shaolab/daiyizheng/resources/hf_weights/Qwen/Qwen1.5-7B",
    max_tokens = 300, 
    streaming=True
)

# 模板

template = """
[INST]利用以下内容回答问题。如果没有提供上下文，请像人工智能助手一样回答。
{context}
问题: {question} [/INST]
"""

# retriever 检索器

retriever = db.as_retriever(search_kwargs = {"k": 6})

# 定义 chain

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type_kwargs = {
        "prompt": PromptTemplate(
            template = template,
            input_variables = ["context", "question"],
        ),
    }
)

# 测试

result = qa_chain({"query": "老年人糖脂代谢病主要有哪些病症? "})

print(1)