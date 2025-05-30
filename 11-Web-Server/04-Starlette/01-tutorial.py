#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :01-tutorial.py
@Description  :
@Time         :2025/05/29 20:09:00
@Author       :flow-laic
@Version      :1.0
'''

# 例子

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def homepage(request):
    return JSONResponse({'hello': 'world'})


app = Starlette(debug=True, routes=[
    Route('/', homepage),
])

## 启动 uvicorn 01-tutorial:app