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



## 上下文

# Context 对象为您的工具和资源提供对 MCP 功能的访问权限

from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("My App")


@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files)) ## 进度条
        data, mime_type = await ctx.read_resource(f"file://{file}") ## 读取资源
    return "Processing complete"