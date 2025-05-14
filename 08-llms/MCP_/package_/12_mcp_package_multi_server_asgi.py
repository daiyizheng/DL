#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :11_mcp_package_asgi.py
@Description  :
@Time         :2025/05/14 15:51:58
@Author       :flow-laic
@Version      :1.0
'''

# 当在不同路径下挂载多个 MCP 服务器时，可以通过以下几种方式配置挂载路径：


from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

# Create multiple MCP servers
github_mcp = FastMCP("GitHub API")
browser_mcp = FastMCP("Browser")
curl_mcp = FastMCP("Curl")
search_mcp = FastMCP("Search")

# 方法 1：通过设置配置挂载路径（建议用于持久配置）
github_mcp.settings.mount_path = "/github"
browser_mcp.settings.mount_path = "/browser"

# 方法 2：直接将挂载路径传递给 sse_app（首选用于临时挂载）
# 这种方法不会永久修改服务器设置

# 创建带有多个安装服务器的 Starlette 应用程序
app = Starlette(
    routes=[
        # Using settings-based configuration
        Mount("/github", app=github_mcp.sse_app()),
        Mount("/browser", app=browser_mcp.sse_app()),
        # Using direct mount path parameter
        Mount("/curl", app=curl_mcp.sse_app("/curl")),
        Mount("/search", app=search_mcp.sse_app("/search")),
    ]
)

# 方法 3：若要直接执行，也可将挂载路径传递给 run()
if __name__ == "__main__":
    search_mcp.run(transport="sse", mount_path="/search")