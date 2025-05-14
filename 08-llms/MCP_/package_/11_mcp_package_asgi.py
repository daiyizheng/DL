#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :11_mcp_package_asgi.py
@Description  :
@Time         :2025/05/14 15:51:58
@Author       :flow-laic
@Version      :1.0
'''

# 挂载到现有的 ASGI 服务器
# 注意：SSE 传输正在被Streamable HTTP 传输取代。

# 默认情况下，SSE 服务器安装在/sse，Streamable HTTP 服务器安装在/mcp。您可以使用下述方法自定义这些路径。

# 您可以使用该sse_app方法将 SSE 服务器挂载到现有的 ASGI 服务器。这允许您将 SSE 服务器与其他 ASGI 应用程序集成。


from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("My App")

# Mount the SSE server to the existing ASGI server
app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)

# or dynamically mount as host
app.router.routes.append(Host('mcp.acme.corp', app=mcp.sse_app()))