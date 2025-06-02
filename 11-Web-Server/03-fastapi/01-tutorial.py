#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :01-tutorial.py
@Description  :
@Time         :2025/05/31 17:37:50
@Author       :flow-laic
@Version      :1.0
'''
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}