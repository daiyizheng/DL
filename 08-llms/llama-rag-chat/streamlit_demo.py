import streamlit as st

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.llms import VLLM
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from PyPDF2 import PdfReader
#API_KEY = "xxx"
# embedding_model = OpenAIEmbeddings(openai_api_key=API_KEY)
# llm = ChatOpenAI(openai_api_key=API_KEY)

# 机器人template
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://t4.ftcdn.net/jpg/02/10/96/95/360_F_210969565_cIHkcrIzRpWNZzq8eaQnYotG4pkHh0P9.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# 用户template
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/474x/be/3b/9b/be3b9b983cfea7c8aa64706203174fcf.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# 配置界面
st.set_page_config(page_title="RAG ChatBot")
st.header("ChatBot")
# 初始化Streamlit会话状态
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# 创建一个文本输入框，用于用户输入问题。这个输入框的值绑定到会话状态变量user_input
user_input = st.text_input("请输入您的问题: ", value=st.session_state.user_input, key="input")
# 处理用户输入, 并返回响应结果
if st.button("Submit",key="submit"):
    # 如果会话状态中的conversation对象存在且用户有输入，则处理用户的输入。
    if st.session_state.conversation and user_input:
        # 调用 conversation 函数，将用户输入的问题作为参数传入，获取响应。
        response = st.session_state.conversation({"question": user_input})
        # 将用户的问题和机器人的答案作为字典添加到聊天历史
        st.session_state.chat_history.append({"user": user_input, "bot": response["answer"]})
        # 清空用户输入框的内容
        st.session_state.user_input = ""
        # 如果点击了清除按钮，则清空聊天历史。
        if st.button("clear", key="clear"):
            st.session_state.chat_history = []
        # 遍历聊天历史，显示每一轮的对话
        for chat in st.session_state.chat_history:
            # 显示用户的消息
            st.write(user_template.replace("{{MSG}}", chat["user"]),  unsafe_allow_html=True)
            # 显示机器人的回答
            st.write(bot_template.replace("{{MSG}}", chat["bot"]),  unsafe_allow_html=True)
        # 检查会话状态中是否需要重新运行
        if st.session_state.get("need_rerun"):
            # 设置 need_rerun 状态为 False
            st.session_state["need_rerun"] = False
            # 重新运行 Streamlit 应用
            st.rerun()
 

with st.sidebar:
    # 设置子标题
    st.subheader("PDF files")
    # 上传文档
    files = st.file_uploader("upload your files, then submit", accept_multiple_files=True)
    if files and st.button("Submit", key="submit_pdfs"):
        with st.spinner("wait for moment, processing ... "):
            # 解析PDF文档内容（加载多个文档）
            text = ""
            for file in files:
                file_reader = PdfReader(file)
                # 解析每一页
                for page in file_reader.pages:
                    text += page.extract_text()

            # 将获取到的文档内容进行切分
            splitter = CharacterTextSplitter(separator="\n",
                                             chunk_size=500,
                                             chunk_overlap=80,
                                             length_function=len)
            chunks = splitter.split_text(text)

            # embedding model
            embedding_model = HuggingFaceEmbeddings(model_name="/root/autodl-tmp/embedding_models/models",
                                                    model_kwargs={"device": "cpu"})

            # 向量数据库
            db = FAISS.from_texts(texts=chunks,
                                  embedding=embedding_model)

            # llm model
            llm = VLLM(model="/root/autodl-tmp/llama-7b-chat-hf",
                       trust_remote_code=False,  # mandatory for hf models
                       max_new_tokens=300,
                       top_k=3,
                       top_p=0.95,
                       temperature=0.8,
                       )

            # 存储历史记录
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # 创建对话链
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(),
                memory=memory
            )



               


