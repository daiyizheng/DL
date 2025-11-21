#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :graph.py
@Description  :
@Time         :2025/11/20 09:44:42
@Author       :flow-laic
@Version      :1.0
'''
import os
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import StateGraph
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.constants import START


import dotenv
dotenv.load_dotenv('/Users/a1-6/Documents/projects/DL/08-llms/Agent_/04-langgraph-example/06-HITLs/01-HITL/.env')

client = MongoClient(host=os.getenv("MONGO_HOST"),
                     port=int(os.getenv("MONGO_PORT")),
                     username=os.getenv("MONGO_USER"),
                     password=os.getenv("MONGO_PASSWORD"))
memory = MongoDBSaver(client)

class State(TypedDict):
    messages: Annotated[List, add_messages]
    ask_human: bool

@tool
def get_weather(location: str) -> str:
    """获取某个位置的当前天气。

    Args:
        location (str): 城市名称。

    Returns:
        str: 天气预报。
    """
    return f"{location} 的天气是晴天"
@tool
def request_assistance():
    """将对话升级至专家。如果用户需要的指导超出了助手的能力范围，请使用此功能。"""
    return ""



## 构建模型
model = ChatDeepSeek(model="deepseek-chat", 
                     api_key=os.getenv("DEEPSEEK_API_KEY"),
                     base_url=os.getenv("DEEPSEEK_API_BASE"))
llm_with_tools = model.bind_tools([get_weather, request_assistance])

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if response.tool_calls and response.tool_calls[0]["name"] == "request_assistance":
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


## 构建工具节点
tools_node = ToolNode(tools=[get_weather])


## 构建人机节点

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"]
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        new_messages.append(
            create_response(
                "提前三个月计划您的旅行，避免在巴塞罗那穿着皇马球衣。",
                state["messages"][-1],
            )
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }


## 构建条件节点

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


## 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("human", human_node)
graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
# 入口节点 或者设置一个start节点
graph_builder.set_entry_point("chatbot") 
# graph_builder.add_edge(START, "chatbot")

## 在human节点前中断，等待人工输入
graph = graph_builder.compile(checkpointer=memory, 
                              interrupt_before=["human"])

# from IPython.display import Image, display
# from langchain_core.runnables.graph import MermaidDrawMethod

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )


from langchain_core.messages import HumanMessage

## 上下文配置项
config = {"configurable": {"thread_id": "51"}}
input_message = HumanMessage(
    content="我需要一些专家建议，如何规划去巴塞罗那的旅行"
)
res = graph.invoke({"messages": input_message}, config=config)


print(res)

res = graph.invoke(None, config=config)
print("----------------------")
print(res)
