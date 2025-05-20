#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :01_fastmcp_base.py
@Description  :
@Time         :2025/05/19 16:45:18
@Author       :flow-laic
@Version      :1.0
'''

import asyncio
from fastmcp import FastMCP, Client

mcp = FastMCP("My MCP Server")

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello, {name}!"

client = Client(mcp)
## 或者单文件引入
# client = Client("my_server.py")

async def call_tool(name: str):
    async with client:
        result = await client.call_tool("greet", {"name": name})
        print(result)

asyncio.run(call_tool("Ford"))


# 使用 FastMCP CLI
## 要让 FastMCP 为我们运行服务器，我们可以使用该fastmcp run命令。
## 这将启动服务器并使其保持运行，直到停止。默认情况下，它将使用stdio传输协议，
## 这是一种基于文本的简单协议，用于与服务器交互。

## fastmcp run my_server.py:mcp
## 请注意，FastMCP不需要__main__服务器文件中的块，如果存在，它将忽略它。相反，它会查找 CLI 命令（此处为 ）中提供的服务器对象。mcp如果没有提供服务器对象，fastmcp run它将自动在文件中搜索名为“mcp”、“app”或“server”的服务器。