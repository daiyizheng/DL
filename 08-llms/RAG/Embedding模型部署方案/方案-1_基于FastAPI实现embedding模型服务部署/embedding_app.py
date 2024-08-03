# FastAPI是一个用于构建API高性能的Web框架，HTTPException用于处理HTTP请求中的异常情况。
from fastapi import FastAPI, HTTPException, Request

# pydantic是一个数据验证和设置管理的库，BaseModel用于创建数据模型，
# field_validator是一个装饰器，用于对模型中的字段进行验证。
from pydantic import BaseModel, field_validator
# 类型注解
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch
# uvicorn是一个基于asyncio开发的一个轻量级高效的web服务器框架
import uvicorn
# 导入asyncio模块，这是Python的标准异步I/O框架，用于编写并发代码。
import asyncio
# ThreadPoolExecutor，这是用于异步执行的线程池。
from concurrent.futures import ThreadPoolExecutor
import logging
# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    # 配置了两个处理器：一个是将日志信息写入名为app.log的文件，另一个是将日志信息输出到控制台。
                    handlers=[logging.FileHandler("app.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = FastAPI()

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/slurm/resources/weights/huggingface/BAAI/bge-large-zh-v1.5"
# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
# 评估模式，在推理阶段使用，以关闭如dropout等在训练阶段会使用的特定层。
model.eval()

class TextData(BaseModel):
    # text可以是一个字符串或字符串列表
    text: Union[str, List[str]]
    # 装饰器，用于为text字段指定一个验证器。
    @field_validator('text')
    def convert_string_to_list(cls, v):
        '''验证器的定义，它将单个字符串转换为字符串列表，如果输入已经是列表，则直接返回。'''
        if isinstance(v, str):
            return [v]
        return v

async def async_compute_embeddings(texts: List[str]):
    '''异步计算文本列表的句子嵌入向量'''
    try:
        # 获取当前运行的事件循环，事件循环负责管理和执行异步任务。
        loop = asyncio.get_running_loop()
        # 线程池允许你在单独的线程中执行函数
        with ThreadPoolExecutor() as pool:
            # 执行两个主要的异步任务：
            # (1) 通过在线程池中执行分词器将文本编码为模型可以处理的格式
            # 使用await和loop.run_in_executor，这个过程是异步执行的，不会阻塞事件循环。
            encoded_input = await loop.run_in_executor(pool, lambda: tokenizer(texts,
                                                                               padding=True,
                                                                               truncation=True,
                                                                               return_tensors="pt",
                                                                                max_length=1024).to(device))
            with torch.no_grad():
                # (2) 在线程池中异步执行模型推理
                outputs = await loop.run_in_executor(pool, lambda: model(**encoded_input))
                # 从模型输出中提取句子嵌入，获取输出的最后一层的隐藏状态并选择每个序列的第一个token（通常是[CLS]）的嵌入
                sentence_embeddings = outputs[0][:, 0]
                # L2标准化句子嵌入
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            # 将句子嵌入从PyTorch张量转换为numpy数组，然后转换为列表，作为函数的返回值。
            return sentence_embeddings.cpu().numpy().tolist()
    except Exception as e:
        # exc_info=True : 告诉日志记录器除了记录错误消息外，还要将异常的信息包括在日志消息中。
        logger.error(f"Error embeddings : {e}", exc_info=True)

# @app.post("/embeddings/")指定这是一个处理POST请求的路由，URL路径为/embeddings/。
@app.post("/embeddings/")
async def get_embeddings(request: Request, text_data: TextData):
    try:
        # 日志记录请求内容
        logger.info(f"请求内容: {await request.json()}")
        # 异步计算文本的嵌入表示，并等待结果。
        embeddings = await async_compute_embeddings(text_data.text)
        # 函数将返回一个包含嵌入向量的字典，这个字典将被自动序列化为JSON响应体。
        return {"embeddings": embeddings}
    except Exception as e:
        # 如果在处理请求或计算嵌入时发生异常，将抛出一个HTTP异常，状态码为500（表示服务器内部错误）
        logger.error(f"Error requests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    # 使用uvicorn作为ASGI服务器来运行我们的FastAPI应用
    # host="0.0.0.0"表示服务器监听所有网络接口，
    # port=8000指定服务器运行在8000端口，应用就可以接收来自网络上任何客户端的请求了。
    uvicorn.run(app, host="0.0.0.0", port=8000)
