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



## 图像
# 
# FastMCP 提供了一个图像类，可自动处理图像数据：

from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("My App")


@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")