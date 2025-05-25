# -*- encoding: utf-8 -*-
'''
Filename         :03_LCEL.py
Description      :
Time             :2024/04/18 14:46:21
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from langchain_openai import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv,find_dotenv

# 加载.env文件中的环境变量
load_dotenv(find_dotenv())

model = ChatOpenAI()
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
# 定义了一个检索器 retriever,可以从文本中检索上下文
retriever = vectorstore.as_retriever()  
# template 定义了一个带占位符的prompt模板，所以prompt组件需要一个包含context和question键的字典作为输入。
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

retrieval_chain.invoke("where did harrison work?")
