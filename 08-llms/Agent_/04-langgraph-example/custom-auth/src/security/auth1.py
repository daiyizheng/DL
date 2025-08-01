######################### 修改二 ##########################
from langgraph_sdk import Auth
# 保留上一教程中的测试用户
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

auth = Auth()

@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """上一教程中的认证处理程序。"""
    print("----------", authorization)
    assert authorization
    
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="无效令牌")

    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }

@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,  # 包含当前用户的信息
    value: dict,  # 被创建或访问的资源
):
    """通过资源元数据使资源对创建者私有。"""
    # 示例：
    # ctx: AuthContext(
    #     permissions=[],
    #     user=ProxyUser(
    #         identity='user1',
    #         is_authenticated=True,
    #         display_name='user1'
    #     ),
    #     resource='threads',
    #     action='create_run'
    # )
    # value: 
    # {
    #     'thread_id': UUID('1e1b2733-303f-4dcd-9620-02d370287d72'),
    #     'assistant_id': UUID('fe096781-5601-53d2-b2f6-0d3403f7e9ca'),
    #     'run_id': UUID('1efbe268-1627-66d4-aa8d-b956b0f02a41'),
    #     'status': 'pending',
    #     'metadata': {},
    #     'prevent_insert_if_inflight': True,
    #     'multitask_strategy': 'reject',
    #     'if_not_exists': 'reject',
    #     'after_seconds': 0,
    #     'kwargs': {
    #         'input': {'messages': [{'role': 'user', 'content': 'Hello!'}]},
    #         'command': None,
    #         'config': {
    #             'configurable': {
    #                 'langgraph_auth_user': ... 你的用户对象 ...
    #                 'langgraph_auth_user_id': 'user1'
    #             }
    #         },
    #         'stream_mode': ['values'],
    #         'interrupt_before': None,
    #         'interrupt_after': None,
    #         'webhook': None,
    #         'feedback_keys': None,
    #         'temporary': False,
    #         'subgraphs': False
    #     }
    # }

    # 执行两件事：
    # 1. 将用户 ID 添加到资源的元数据中。每个 LangGraph 资源都有一个随资源持久化的 `metadata` 字典。
    # 此元数据在读取和更新操作中可用于过滤
    # 2. 返回一个过滤器，仅允许用户看到自己的资源
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)

    # 只允许用户看到自己的资源
    return filters
