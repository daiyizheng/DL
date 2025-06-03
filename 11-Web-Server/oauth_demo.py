#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :oauth.py
@Description  :
@Time         :2025/06/03 22:00:05
@Author       :flow-laic
@Version      :1.0
'''

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse
import httpx
import jwt
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# 配置环境变量
CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI")
GITHUB_API_URL = "https://api.github.com/user"

# JWT 配置
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"

# OAuth 2.0 URL
AUTH_URL = f"https://github.com/login/oauth/authorize?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}" # 由用户端发起

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# 路由：重定向到 GitHub 授权页面
@app.get("/login")
async def login():
    return RedirectResponse(url=AUTH_URL)


# 路由：GitHub 回调并获取用户信息
@app.get("/auth/callback")
async def callback(code: str):
    # 获取 GitHub 返回的 `code`，用它请求 `access_token`
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
        )
    
    data = response.json()
    access_token = data.get("access_token")

    if not access_token:
        raise HTTPException(status_code=400, detail="GitHub authorization failed")

    # 使用 Access Token 获取 GitHub 用户信息
    async with httpx.AsyncClient() as client:
        user_info = await client.get(
            GITHUB_API_URL,
            headers={"Authorization": f"Bearer {access_token}"}
        )

    user = user_info.json()
    
    # 生成 JWT 令牌
    jwt_token = jwt.encode({"sub": user["login"]}, SECRET_KEY, algorithm=ALGORITHM)

    return {"access_token": jwt_token, "token_type": "bearer"}


# 路由：受保护的资源，只有认证用户可以访问
@app.get("/protected")
async def protected(token: str = Depends(oauth2_scheme)):
    try:
        # 解码 JWT，验证身份
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = payload.get("sub")
        if user is None:
            raise HTTPException(status_code=403, detail="Could not validate credentials")
        return {"message": f"Welcome, {user}!"}
    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)