#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :09_mcp_package_runing_server.py
@Description  :
@Time         :2025/05/14 15:25:54
@Author       :flow-laic
@Version      :1.0
'''

# 可流式传输的 HTTP 传输
# 注意：在生产部署中，可流式 HTTP 传输正在取代 SSE 传输。

from mcp.server.fastmcp import FastMCP

# Stateful server (maintains session state)
mcp = FastMCP("StatefulServer")

# Stateless server (no session persistence)
mcp = FastMCP("StatelessServer", stateless_http=True)

# Run server with streamable_http transport
mcp.run(transport="streamable-http")




### 您可以在 FastAPI 应用程序中挂载多个 FastMCP 服务器：

# 创建echo.py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP(name="EchoServer", stateless_http=True)
@mcp.tool(description="A simple echo tool")
def echo(message: str) -> str:
    return f"Echo: {message}"

# 创建math.py
from mcp.server.fastmcp import FastMCP
mcp = FastMCP(name="MathServer", stateless_http=True)
@mcp.tool(description="A simple add tool")
def add_two(n: int) -> int:
    return n + 2

# main.py
from fastapi import FastAPI
from .echo import echo
from .math import math


app = FastAPI()

# Use the session manager's lifespan
app = FastAPI(lifespan=lambda app: echo.mcp.session_manager.run())
app.mount("/echo", echo.mcp.streamable_http_app())
app.mount("/math", math.mcp.streamable_http_app())

# 对于具有 Streamable HTTP 实现的低级服务器，请参阅：
# - 有状态服务器：examples/servers/simple-streamablehttp/
# - 无状态服务器：examples/servers/simple-streamablehttp-stateless/

# 可流式传输的 HTTP 传输支持：
# - 有状态和无状态操作模式
# - 事件存储的可恢复性
# - JSON 或 SSE 响应格式
# - 多节点部署的更好可扩展性