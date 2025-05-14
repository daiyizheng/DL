#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :mcp_package_test.py
@Description  : 对MCP包的源码调试
@Time         :2025/05/14 08:40:14
@Author       :flow-laic
@Version      :1.0
'''
import sys
sys.path.insert(0, "/Users/a1-6/Documents/projects/DL/08-llms/MCP_/package_")



## 基本使用


# 引入依赖
from mcp.server.fastmcp import FastMCP

# 创建 MCP server
mcp = FastMCP(name="demo")


# 在函数上配置工具装饰器， 并且写上写函数注释，
@mcp.tool()
def add(a: int, b: int) -> int:
    """两个数的加法运算

    Args:
        a (int): 变量a, 例如：1
        b (int): 变量b, 例如：2

    Returns:
        int: 对a和b变量进行加发后，返回一个数
    """
    return a + b


# 添加动态问候资源
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

"""
{ contents:[0:{
            uri:
            "greeting://jjj"
            mimeType:
            "text/plain"
            text:
            "Hello, jjj!"}]
}
"""

## 您可以在 Claude Desktop 中安装该服务器，并通过运行 mcp install server.py

## 您也可以使用 MCP 页面检查器进行测试：mcp dev server.py



