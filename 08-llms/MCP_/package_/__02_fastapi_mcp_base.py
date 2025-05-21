#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :__01_ fastapi_mcp_base.py
@Description  :
@Time         :2025/05/21 08:36:37
@Author       :flow-laic
@Version      :1.0
'''

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()

mcp = FastApiMCP(app)
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)