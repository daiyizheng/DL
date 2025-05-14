#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :09_mcp_package_runing_server.py
@Description  :
@Time         :2025/05/14 15:25:54
@Author       :flow-laic
@Version      :1.0
'''

# 运行您的服务器
# 开发模式
# 测试和调试服务器的最快方法是使用 MCP Inspector：

"""
mcp dev server.py

# 添加依赖
mcp dev server.py --with pandas --with numpy
# 挂宅本地code
mcp dev server.py --with-editable .
"""


# Claude 桌面集成
# 服务器准备就绪后，请在 Claude Desktop 中安装它：
"""
mcp install server.py

# Custom name
mcp install server.py --name "My Analytics Server"

# Environment variables
mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install server.py -f .env
"""

# 直接执行
# 对于自定义部署等高级场景

""" 
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("My App")

if __name__ == "__main__":
    mcp.run()

# 使用以下命令运行它：

python server.py
# or
mcp run server.py
"""