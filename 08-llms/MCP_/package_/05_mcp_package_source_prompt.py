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



## 提示 
# 
# 提示是可重复使用的模板，可帮助 LLM 与服务器进行有效互动：

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("My App")


@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]