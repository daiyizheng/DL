#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :client.py
@Description  :
@Time         :2025/04/20 10:25:04
@Author       :daiyizheng
@Version      :1.0
'''

import asyncio # 用于导入异步IO库，用于支持异步编程
import json # 用于导入JSON库，用于处理JSON数据
import sys # 用于处理命令行参数
from typing import Optional # 用于类型提示功能
from contextlib import AsyncExitStack # 异步资源管理器，用于管理多个异步资源

# MCP 客户端相关导入
from mcp import ClientSession, StdioServerParameters # 导入 MCP 客户端会话和标准输入输出服务器参数
from mcp.client.stdio import stdio_client # 导入标准输入输出客户端通信模块

from openai import OpenAI # Openai SDK

# 环境变量加载相关
from dotenv import load_dotenv # 导入环境变量加载工具
import os # 用于获取环境变量值

load_dotenv()  # 加载 .env 文件中的环境变量

# 定义 MCP 客户端类
class DeepSeekMCPClient:
    """
    使用 DeepSeek V3 API 的 MCP 客户端类
    处理 MCP 服务器连接和 DeepSeek V3 API 的交互

    这个类就像是一个翻译官，一方面与MCP服务器进行通信，另一方面与DeepSeek API进行通信，
    帮助用户通过自然语言来使用各种强大的工具
    """
    def __init__(self):
        """ 
        初始化MCP客户端的各项属性
        主要设置了三个重要组件：
        - session: 用于与MCP服务器通信的会话
        - exit_stack: 用于管理异步资源的上下文管理器，确保资源正确释放，那么在与 MCP 服务通信时，它会负责接收和发送通信数据
        - llm_client: DeepSeek API 的客户端，使用 OpenAI 的 SDK
        """
        # MCP 客户端会话，初始值为 None
        self.session: Optional[ClientSession] = None
        # 创建异步资源管理器，用于管理多个异步资源
        self.exit_stack = AsyncExitStack()
        # 初始化 DeepSeek API 客户端
        self.llm_client = OpenAI(
            api_key=os.getenv("api_key"), # 从环境变量中获取 API 密钥
            base_url=os.getenv("base_url") # 从环境变量中获取 API 基础 URL
        )
        # 从环境变量获取模型名称
        self.model = os.getenv("MODEL")
    
    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务
        这个函数就像是拨通电话，建立与 MCP 服务器的连接，它会根据服务脚本的类型(Python 或 JavaScript)选择正确的命令启动服务器，然后与之建立通信。

        参数:
            server_script_path: MCP 服务脚本路径，支持 Python(.py) 或 Node.js(.js) 文件
        异常:
            ValueError: 如果服务器脚本不是.py或.js文件
        """
        # 检查脚本类型
        is_python = server_script_path.endswith('.py') # 判断是否是 Python 脚本
        is_js = server_script_path.endswith('.js') # 判断是否是 JavaScript 脚本

        if not (is_python or is_js): # 如果脚本类型不是 Python 或 JavaScript，则抛出异常
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        # 根据脚本类型选择正确的运行命令
        command = "python" if is_python else "node" # Python 使用 python 运行，JavaScript 使用 node 运行
        # 设置服务器启动参数，那么 server_params 最终会生成类似于 Python xxx.py 这种运行命令
        server_params = StdioServerParameters(
            command=command, # 要执行的命令(python 或 node)
            args=[server_script_path], # 要执行的命令的参数(脚本路径)
            env=None # 环境变量, 使用 None 表示继承当前环境变量
        )
        # 创建标准输入输出通信信道
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        # 解构对象中的读写通信，分别用于向MCP服务接收和发送数据
        self.stdio, self.write = stdio_transport
        # 创建MCP客户端会话
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        # 初始化MCP客户端会话
        await self.session.initialize() # 初始化会话，准备好与MCP服务进行通信
        # 列出可用的工具
        # 获取 MCP 服务提供的工具列表
        response = await self.session.list_tools()
        # 获取工具列表
        tools = response.tools
        # 打印工具列表
        print("\n已连接到MCP服务，可用的工具列表:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        处理用户查询，根据查询参数使用DeepSeek V3和MCP工具

        这个函数就是整个系统的核心，他就像一个指挥官，它接收用户的问题，然后再把问题交给AI模型，
        然后模型决定使用哪些工具，然后告诉 MCP Client 去调用这些工具。
        然后获取到工具结果再返回给模型，模型根据工具结果生成最终的回答。
        整个过程就像是：
        用户提问->模型判断->MCP Client 使用工具->返回工具结果->模型根据工具结果生成回答

        参数：
            query: 用户的问题
        返回：
            str: 处理后的最终响应的文本
        """
        # 创建消息列表，用于存储用户的问题和模型的回答
        messages = [
            {
                "role": "system", # 系统角色，用于设定AI的行为准则
                "content": "你是一个专业的助手，可以通过调用合适的工具来帮助用户解决问题，请根据用户的需求选择最合适的工具。"
            },
            {
                "role": "user", # 用户角色，表示这是用户发送的消息
                "content": query
            }
        ]
        # 请求 MCP 服务获取服务提供的工具列表
        response = await self.session.list_tools()
        # 获取工具列表
        tools = response.tools
        # 构建工具信息数组，我们需要把工具信息转换成 DeepSeek API 需要的格式
        available_tools = [{
            "type": "function", # 工具类型，表示这是一个函数工具
            "function": { # 工具的详细定义
                "name": tool.name, # 工具名称
                "description": tool.description, # 工具描述
                "parameters": tool.inputSchema # 工具参数
            }
        } for tool in tools]
        # 打印可用工具信息，便于调试
        print(f"当前 MCP 服务所有工具列表: {available_tools}\n--------------------------------\n")
        
        # 调用 DeepSeek API，发送用户查询和可用工具信息，告诉 DeepSeek API 根据用户提问你可以使用哪些工具，最终返回可调用的工具
        response = self.llm_client.chat.completions.create(
            model=self.model, # 指定的模型名称
            messages=messages, # 消息历史（系统提示和用户问题）
            tools=available_tools if available_tools else None, # 可用的工具列表
            temperature=0.5, # 温度参数，控制响应的随机性(0.5是中等随机性)
            max_tokens=4096 # 最大生成令牌数，限制响应长度
        )
        # 打印模型响应，便于调试
        print(f"DeepSeek API 响应: {response}\n--------------------------------\n")
        # 获取模型的回复，包含 role(消息发送者) 和 content(消息内容) 以及 tool_calls(工具调用请求)
        reply = response.choices[0].message # 获取模型的回答
        # 打印模型的回答
        print(f"DeepSekk 初始回复: {reply}\n--------------------------------\n")
        
        # 初始化最终文本结果列表
        final_text = []

        # 将模型回复添加到历史消息中，用于维护完整的对话历史
        # 这一步非常重要，确保模型 记得 自己之前决定使用什么工具，即使模型没有请求调用工具，也要保持对话连贯性。
        messages.append(reply)

        # 检查模型响应中是否包含工具调用请求，如果用户的问题涉及到使用工具，那就会包含 tool_calls 字段,否则就没有
        if hasattr(reply, "tool_calls") and reply.tool_calls:
            # 遍历所有工具调用请求
            for tool_call in reply.tool_calls:
                # 获取工具名称
                tool_name = tool_call.function.name
                # 获取工具参数
                try:
                    # 尝试将工具的参数从 JSON 字符串解析为 Python 字典
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                
                # 打印工具调用信息，便于调试
                print(f"准备调用工具: {tool_name} 参数: {tool_args}\n--------------------------------\n")

                # 异步调用 MCP 服务上的工具，传入工具名称和函数参数，返回工具函数执行结果
                result = await self.session.call_tool(tool_name, tool_args)
                # 打印工具执行结果，便于调试
                print(f"工具 {tool_name} 执行结果: {result}\n--------------------------------\n")

                # 将工具调用信息添加到最终输出文本中，便于用户了解执行过程
                final_text.append(f"调用工具: {tool_name}, 参数: {tool_args}\n")

                # 确保工具结果是字符串格式
                tool_result_content = result.content
                if isinstance(tool_result_content, list):
                    # 如果工具结果是列表，则将列表中的每个元素转换为字符串并添加到最终文本中
                    text_content = ""
                    for item in tool_result_content:
                        if hasattr(item, 'text'):
                            text_content += item.text
                    tool_result_content = text_content
                elif not isinstance(tool_result_content, str):
                    # 如果不是字符串，则转换为字符串
                    tool_result_content = str(tool_result_content)
                # 打印工具返回结果
                print(f"工具返回结果(格式化后): {tool_result_content}\n--------------------------------\n")

                # 将工具调用结果添加到历史消息中，保证与模型会话的连贯性
                tool_message = {
                    "role": "tool", # 工具角色，表示这是工具返回的结果
                    "tool_call_id": tool_call.id, # 工具调用ID
                    "content": tool_result_content, # 工具返回的结果
                    
                }
                # 打印消息内容
                print(f"添加到历史消息中的工具消息: {tool_message}\n--------------------------------\n")
                # 添加到历史消息中
                messages.append(tool_message)

                # 尝试解析工具返回的JSON结果，检查是否包含MCP模板结构
                try:
                    # 将工具返回结果 JSON格式 转换为 Python 字典
                    tool_result_json = json.loads(tool_result_content)
                    # 检查是否包含 MCP 模板结构(具有 prompt_template 和 template_args 字段)
                    if(isinstance(tool_result_json, dict) and "prompt_template" in tool_result_json and "template_args" in tool_result_json):
                        raw_data = tool_result_json["raw_data"] # 原始数据
                        prompt_template = tool_result_json["prompt_template"] # 模板函数名称
                        template_args = tool_result_json["template_args"] # 模板参数

                        # 将模板参数转换为字符串类型(MCP规范要求)
                        string_args = {k:str(v) for k,v in template_args.items()}
                        # 打印模板参数
                        print(f"模板名称: {prompt_template}, 模板参数: {string_args}\n--------------------------------\n")

                        # 调用 MCP 服务上的工具，传入工具名称和函数参数，返回工具函数执行结果
                        template_response = await self.session.get_prompt(prompt_template, string_args)
                        # 打印工具执行结果，便于调试
                        print(f"模板响应: {template_response}\n--------------------------------\n")

                        if hasattr(template_response, "messages") and template_response.messages:
                            # 打印模板响应
                            print(f"模板具体的信息: {template_response.messages}\n--------------------------------\n")
                            for msg in template_response.messages:
                                # 提取消息内容
                                content = msg.content.text if hasattr(msg.content, "text") else msg.content
                                # 构建历史信息
                                template_message = {
                                    "role": msg.role, # 保持原始角色
                                    "content": content # 消息内容
                                }
                                print(f"模板消息历史: {template_message}\n--------------------------------\n")
                                # 添加到历史消息中
                                messages.append(template_message)
                        else:
                            print("警告：模板响应中没有包含消息内容。")
                except json.JSONDecodeError:
                    pass
                # 再次调用 DeepSeek API，让模型根据工具结果生成最终的回答
                try:
                    print("正在请求 DeepSeek API 生成最终回答...")
                    # 发送包含工具调用和结果的完整消息历史
                    final_response = self.llm_client.chat.completions.create(
                        model=self.model, # 指定的模型名称
                        messages=messages, # 消息历史（系统提示和用户问题）
                        temperature=0.5, # 温度参数，控制响应的随机性(0.5是中等随机性)
                        max_tokens=4096 # 最大生成令牌数，限制响应长度
                    )
                    # 添加 DeepSeek 对工具结果的解释然后到最终输出
                    final_content = "DeepSeek回答：" + final_response.choices[0].message.content
                    if final_content:
                        # 如果模型生成了对工具结果的解释，就将其添加到最终输出数组中
                        final_text.append(final_content)
                    else:
                        print("警告：DeepSeek API 没有生成任何内容。")
                        # 如果没用内容，直接显示工具结果
                        final_text.append(f"工具调用结果：\n{tool_result_content}")
                except Exception as e:
                    print(f"生成最终回复时出错: {e}")
                    final_text.append(f"工具返回结果：\n{tool_result_content}")
        else:
            # 如果模型没有请求调用工具，那么就直接返回模型的内容
            if reply.content:
                # 将模型的直接回复添加到最终输出数组
                final_text.append(f"{reply.content}")
            else:
                # 如果模型没有生成内容，则添加提示信息
                final_text.append("模型没有生成有效回复。")
        
        # 我们把用户的问题和MCP服务可用工具全部给到 DeepSeek，DeepSeek 判断出具体需要调用哪个工具，然后让 MCP Client 去调用这个工具，
        # 然后我们再把工具函数返回的结果给到 DeepSeek，让 DeepSeek 根据工具结果生成最终的回答
        # 返回最终的回答
        return '\n'.join(final_text)
    async def chat_loop(self):
        """
        运行交互式聊天循环，处理用户输入并显示回复
        
        这个函数就是一个简单的聊天界面，不断接收用户输入，
        处理问题，并显示回答，直到用户输入'quit'退出。
        """
        print("\nDeepSeek MCP 客户端已经启动!")
        print("请输入你的问题，输入'quit'退出。")
        # 循环处理用户输入
        while True:
            try:
                # 获取用户输入
                query = input("\n问题: ").strip()
                # 检查是否要退出
                if query.lower() == 'quit':
                    break
                # 处理用户输入，传入到查询函数中
                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n错误: {str(e)}")

    async def cleanup(self):
        """
        清理资源，关闭所有打开的连接和上下文。
        这个函数就像是收拾房间，确保在程序结束时，所有打开的资源都被正常关闭，防止资源泄露。
        """
        # 关闭所有打开的连接和上下文,释放资源
        await self.exit_stack.aclose()
async def main():
    """
    主函数，处理命令行参数并启动客户端
    这个函数是程序的起点，它解析命令行参数，创建客户端实例，连接服务器，并启动一个聊天循环
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python client.py <服务器脚本路径>")
        sys.exit(1) # 如果参数不足，显示使用说明并退出
    # 创建客户端实例
    client = DeepSeekMCPClient()
    try:
        # 连接到MCP服务器
        await client.connect_to_server(sys.argv[1])
        # 启动聊天循环
        await client.chat_loop()
    finally:
        # 清理资源,确保在任何情况下都清理资源
        await client.cleanup()

# 程序入口点
if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())

# 使用说明
# 激活虚拟环境（如果尚未激活）
# source .venv/Scripts/activate

# 运行 MCP 客户端，连接到天气查询 MCP 服务器（示例）
# uv run client.py D:\\开源MCP项目\\weather\\weather.py

# 示例问题：