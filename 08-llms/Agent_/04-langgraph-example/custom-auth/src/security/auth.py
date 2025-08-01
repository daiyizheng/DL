## 修改1##################
from langgraph_sdk import Auth

# 这是我们的测试用户数据库。生产环境中不要这样做
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

# `Auth` 对象是 LangGraph 用于标记认证函数的容器
auth = Auth()

# `authenticate` 装饰器告诉 LangGraph 将此函数作为中间件在每个请求上调用
# 以确定请求是否被允许
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """检查用户的令牌是否有效。"""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"
    # 检查令牌是否有效
    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="无效令牌")

    # 如果有效，返回用户信息
    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }


