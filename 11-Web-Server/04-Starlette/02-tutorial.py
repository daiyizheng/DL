# 框架或工具包
# Starlette 的设计初衷既可以作为一个完整的框架，也可以作为一个 ASGI 工具包。你可以单独使用它的任何组件。

from starlette.responses import PlainTextResponse


async def app(scope, receive, send):
    assert scope['type'] == 'http'
    response = PlainTextResponse('Hello, world!')
    await response(scope, receive, send)


## uvicorn 02-tutorial:app