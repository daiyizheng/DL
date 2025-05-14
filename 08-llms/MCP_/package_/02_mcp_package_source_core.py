#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :02_mcp_package_source.py
@Description  :
@Time         :2025/05/14 09:46:48
@Author       :flow-laic
@Version      :1.0
'''

## 核心概念

### 服务器

## FastMCP 服务器是 MCP 协议的核心接口。它处理连接管理、协议合规性和消息路由：
import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import Context, FastMCP


class Database:
    def __init__(self):
        self._data = {"name": "Mock User", "age": 99}  # 模拟存储

    @classmethod
    async def connect(cls):
        """模拟异步连接"""
        await asyncio.sleep(0.1)  # 假装是异步操作
        return cls()
    
    async def disconnect(self):
        """模拟异步关闭"""
        await asyncio.sleep(0.1)
        print("Database disconnected")

    async def query(self, key=None):
        """模拟异步查询"""
        await asyncio.sleep(0.05)
        if key is None:
            return self._data
        return self._data.get(key)

    async def insert(self, key, value):
        """模拟异步插入"""
        await asyncio.sleep(0.05)
        self._data[key] = value
        return True
    
    


# Create a named server
mcp = FastMCP("My App")

# Specify dependencies for deployment and development
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])


@dataclass
class AppContext:
    db: Database


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # Cleanup on shutdown
        await db.disconnect()


# Pass lifespan to server
mcp = FastMCP("My App", lifespan=app_lifespan)


# 在工具中访问lifespan上下文
@mcp.tool()
def query_db(ctx: Context) -> str:
    """Tool that uses initialized resources"""
    db = ctx.request_context.lifespan_context.db
    return db.query()