import os
import httpx
from getpass import getpass
from langgraph_sdk import get_client

# 从命令行获取电子邮件
email = getpass("请输入你的电子邮件：")
base_email = email.split("@")
password = "secure-password"  # 请更改此密码
email1 = f"{base_email[0]}+1@{base_email[1]}"
email2 = f"{base_email[0]}+2@{base_email[1]}"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
if not SUPABASE_URL:
    SUPABASE_URL = getpass("请输入你的 Supabase 项目 URL：")

# 这是你的公开匿名密钥（可在客户端安全使用）
# 不要将其与秘密服务角色密钥混淆
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
if not SUPABASE_ANON_KEY:
    SUPABASE_ANON_KEY = getpass("请输入你的 Supabase 公开匿名密钥：")

async def sign_up(email: str, password: str):
    """创建新的用户账户。"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            json={"email": email, "password": password},
            headers={"apiKey": SUPABASE_ANON_KEY},
        )
        assert response.status_code == 200
        return response.json()


async def main():
    # 创建两个测试用户
    print(f"创建测试用户：{email1} 和 {email2}")
    await sign_up(email1, password)
    await sign_up(email2, password)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    print("所有用户已创建！")