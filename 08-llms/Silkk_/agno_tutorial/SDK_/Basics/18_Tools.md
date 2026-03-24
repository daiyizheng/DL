# 工具

## 什么是工具？

工具是代理程序调用来与外部系统交互的功能。

工具使代理和团队能够执行实际操作。直接使用 LLM 时，您只能生成文本回复；而配备工具的代理和团队则可以与外部系统交互并执行实际操作。工具可能需要确认。当出现这种情况时，运行将暂停，直到确认问题得到解决。
使用工具可以执行的操作示例包括：搜索网络、运行 SQL、发送电子邮件或调用 API。
Agno 内置 120 多个预构建工具包，可用于赋予您的代理各种功能。您还可以编写自己的工具，进一步增强代理的功能。通用语法如下：

```python
import random

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool

def get_weather(city: str) -> str:
    """Get the weather for the given city.

    Args:
        city (str): The city to get the weather for.
    """

    # In a real implementation, this would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    random_weather = random.choice(weather_conditions)

    return f"The weather in {city} is {random_weather}."

# To equipt our Agent with our tool, we simply pass it with the tools parameter
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_weather],
    markdown=True,
)

# Our Agent will now be able to use our tool, when it deems it relevant
agent.print_response("What is the weather in San Francisco?", stream=True)
```

### 工具是如何工作的？
代理执行的核心是LLM循环。LLM循环的典型执行流程如下：
- 代理将运行上下文（系统消息、用户消息、聊天记录等）和工具定义发送给模型。
- 模型会以消息或工具调用作为响应。
- 如果模型发出工具调用，则执行该工具并将结果返回给模型。
- 该模型处理更新后的上下文，重复此循环，直到生成最终消息而无需任何工具调用。
- 客服人员向来电者返回最终回复。


### 工具定义
Agno 会自动将您的工具函数转换为模型所需的工具定义格式。通常，这是一个 JSON 模式，用于描述工具的参数和返回类型。
例如：

```python

def get_weather(city: str) -> str:
    """
    Get the weather for a given city.

    Args:
        city (str): The city to get the weather for.
    """
    return f"The weather in {city} is sunny."
```

这将转换为以下工具定义：

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for."
                }
            },
            "required": ["city"]
        }
    }
}
```

然后，该工具定义会被发送给模型，以便模型在收到请求时知道如何调用该工具。您还会注意到，该Args部分会自动从定义中移除，经过解析后用于填充各个属性的定义。
当在工具函数的参数中使用 Pydantic 模型时，Agno 会自动将模型转换为所需的工具定义格式。
例如：

```python

from pydantic import BaseModel, Field

class GetWeatherRequest(BaseModel):
    city: str = Field(description="The city to get the weather for")

def get_weather(request: GetWeatherRequest) -> str:
    """
    Get the weather for a given city.

    Args:
        request (GetWeatherRequest): The request object containing the city to get the weather for.

    """
    return f"The weather in {request.city} is sunny."
```

这将转换为以下工具定义：

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
              "request": {
                "type": "object",
                "properties": {
                  "city": {
                    "type": "string",
                    "description": "The city to get the weather for."
                  }
                },
                "required": ["city"]
              }
            },
            "required": ["request"]
        }
    }
}
```

> 务必为工具函数创建文档字符串。确保包含文档说明Args部分，并涵盖函数的每个参数。
> 为工具函数使用合理的名称。请记住，模型在需要时会直接使用这些名称来调用工具。


### 工具执行
当模型请求工具调用时，工具将被执行，并将结果返回给模型。
- 模型可以在单个响应中请求多个工具调用。
- 当使用arun该工具执行代理或团队时，如果模型请求多个工具调用，则这些工具将同时执行。

Agno Agents 可以同时执行多个工具，从而高效地处理模型发出的函数调用。当函数涉及耗时操作时，这一点尤为重要。它能够提高响应速度并缩短整体执行时间。

当你调用`arun`or`aprint_response` 时，你的工具将并发执行。如果你提供的是同步函数作为工具，它们将在不同的线程上并发执行。

```python
import asyncio
import time

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.utils.log import logger

async def atask1(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 1 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 1 has slept for 1s")
    logger.info("Task 1 has completed")
    return f"Task 1 completed in {delay:.2f}s"


async def atask2(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 2 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 2 has slept for 1s")
    logger.info("Task 2 has completed")
    return f"Task 2 completed in {delay:.2f}s"


async def atask3(delay: int):
    """Simulate a task that takes a random amount of time to complete
    Args:
        delay (int): The amount of time to delay the task
    """
    logger.info("Task 3 has started")
    for _ in range(delay):
        await asyncio.sleep(1)
        logger.info("Task 3 has slept for 1s")
    logger.info("Task 3 has completed")
    return f"Task 3 completed in {delay:.2f}s"


async_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[atask2, atask1, atask3],
    markdown=True,
)

asyncio.run(
    async_agent.aprint_response("Please run all tasks with a delay of 3s", stream=True)
)
```

<img src="https://mintcdn.com/agno-v2/Y7twezR0wF2re1xh/images/async-tools.png?w=2500&fit=max&auto=format&n=Y7twezR0wF2re1xh&q=85&s=512ee4e5ccf4e4bfa33edac026b3a472">
在此示例中，gpt-5-mini 同时对 atask1、atask2 和 atask3 进行三个工具调用。 通常这些工具调用会按顺序执行，但使用 aprint_response 函数，它们可以并发运行，从而缩短执行时间。



### 使用工具包
Agno Toolkit 提供了一种管理多个工具的方法，并能更好地控制它们的执行。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[
        HackerNewsTools(),
    ],
)

agent.print_response("What are the top stories on HackerNews?", markdown=True)

```


在这个例子中，HackerNewsTools工具包被添加到代理程序中。该工具包使代理程序能够从 HackerNews 获取新闻报道。

### 工具内置参数
Agno 会自动为您的工具提供特殊参数，以便访问代理的参数、状态和其他变量。这些参数会自动注入，代理无需了解它们。
​


#### 使用运行上下文
你可以通过`run_context`参数访问当前运行中的值：`run_context.session_state`、`run_context.dependencies`、`run_context.knowledge_filters`、`run_context.metadata`。更多信息请参见 RunContext 模式。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.run import RunContext


def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}

    run_context.session_state["shopping_list"].append(item)  # type: ignore
    return f"The shopping list is now {run_context.session_state['shopping_list']}"  # type: ignore


# Create an Agent that maintains state
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    # Initialize the session state with a counter starting at 0 (this is the default session state for all users)
    session_state={"shopping_list": []},
    db=SqliteDb(db_file="tmp/agents.db"),
    tools=[add_item],
    # You can use variables from the session state in the instructions
    instructions="Current state (shopping list) is: {shopping_list}",
    markdown=True,
)

# Example usage
agent.print_response("Add milk, eggs, and bread to the shopping list", stream=True)
print(f"Final session state: {agent.get_session_state()}")
```

更多信息请参见“代理状态”。


#### 使用代理或团队


您可以通过添加代理或团队作为参数，直接在工具功能中访问代理或团队实例。 这使您可以完全访问代理或团队的属性，例如模型、说明等。


```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

def get_agent_instructions(agent: Agent) -> str:
    """Get the model of the agent."""
    return agent.instructions

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are a helpful assistant that can answer questions about the model.",
    tools=[get_agent_instructions],
)

agent.print_response("What is the instructions of the agent?", stream=True)
```

> 使用团队作为参数时，确保工具是在团队上下文中使用的。同样，当工具与代理一起使用时，请使用代理。


#### 媒体参数
内置参数图像、视频、音频和文件允许工具访问和修改代理的输入媒体。

> 利用send_media_to_model参数，你可以控制媒体是否发送到模型;用store_media参数，你可以控制媒体是否存储在运行输出中。

有关使用媒体的高级示例，请参阅图像输入示例和文件输入示例。
​
### 工具结果
根据工具的复杂程度以及需要向代理传达的信息，工具可以返回不同类型的结果。
​
#### 简单返回类型
大多数工具可以直接返回简单的 Python类型，例如：str int float dict list




```python

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75°F"

@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

@tool
def get_user_info(user_id: str) -> dict:
    """Get user information."""
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "status": "active"
    }

@tool
def search_products(query: str) -> list:
    """Search for products."""
    return [
        {"id": 1, "name": "Product A", "price": 29.99},
        {"id": 2, "name": "Product B", "price": 39.99}
    ]
```

#### ToolResult媒体内容
当您的工具需要返回媒体文件（图像、视频、音频）时，您必须使用ToolResult：

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| content | str | 必需的 | 工具的主要文本内容／输出 |
| images | Optional［List［Image］］ | None | 生成的图像伪影 |
| videos | Optional［List［Video］］ | None | 生成的视频伪影 |
| audios | Optional［List［Audio］］ | None | 生成的音频瑕疪 |

```python

from agno.tools.function import ToolResult
from agno.media import Image

@tool
def generate_image(prompt: str) -> ToolResult:
    """Generate an image from a prompt."""

    # Create your image (example)
    image_artifact = Image(
        id="img_123",
        url="https://example.com/generated-image.jpg",
        original_prompt=prompt
    )

    return ToolResult(
        content=f"Generated image for: {prompt}",
        images=[image]
    )
```
这将使生成的媒体可供LLM模型使用。


### 可调用工厂
Agno 工具支持可调用工厂模式，用于动态配置。使用可调用对象实现动态工具工厂，以进行运行时依赖注入和上下文感知工具集定制。
- 运行时解析：工具在每次运行开始时通过基于签名的注入进行延迟初始化。
- 上下文注入：自动将agent/ team、run_context 和 session_state 映射到工厂参数。
- 精细范围：支持专门针对user_id或 session_id量身定制的独特工具集。
- 性能优化：包含内置缓存（cache_callables）以根据唯一的会话键缓存工具集。

```python
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunContext

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


def search_internal_docs(query: str) -> str:
    """Search internal documentation (admin only)."""
    return f"Internal doc results for: {query}"


def get_account_balance(account_id: str) -> str:
    """Get account balance (finance only)."""
    return f"Balance for {account_id}: $42,000"


# ---------------------------------------------------------------------------
# Callable Factory
# ---------------------------------------------------------------------------


def tools_for_user(run_context: RunContext):
    """Return different tools based on the user's role stored in session_state."""
    role = (run_context.session_state or {}).get("role", "viewer")
    print(f"--> Resolving tools for role: {role}")

    base_tools = [search_web]
    if role == "admin":
        base_tools.append(search_internal_docs)
    if role in ("admin", "finance"):
        base_tools.append(get_account_balance)

    return base_tools


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5-mini"),
    tools=tools_for_user,
    instructions=[
        "You are a helpful assistant.",
        "Use the tools available to you to answer the user's question.",
    ],
)


# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run 1: viewer role - only search_web available
    # Each user_id gets its own cached toolset
    print("=== Run as viewer ===")
    agent.print_response(
        "Search for recent news about AI agents",
        user_id="viewer_user",
        session_state={"role": "viewer"},
        stream=True,
    )

    # Run 2: admin role - all tools available
    # Different user_id means the factory is called again with new context
    print("\n=== Run as admin ===")
    agent.print_response(
        "Search internal docs for the deployment guide and check account balance for ACC-001",
        user_id="admin_user",
        session_state={"role": "admin"},
        stream=True,
    )
```

## 带有代理的工具

### 代理工具

为代理人配备执行外部操作的功能和工具包。

智能体使用工具来采取行动并与外部系统交互。
工具是代理程序可以运行以完成任务的功能。例如：搜索网络、运行 SQL、发送电子邮件或调用 API。您可以使用任何 Python 函数作为工具，也可以使用预构建的 Agno工具包。

```python
from agno.agent import Agent

agent = Agent(
    # Add functions or Toolkits
    tools=[...],
)
```

工具可以是可调用的工厂，并在运行时解析。

### 使用工具包
Agno 提供了许多预构建的工具包，您可以将其添加到您的代理中。例如，我们可以使用 HackerNews 工具包来获取科技新闻。

1. 创建新闻代理

创建文件news_agent.py

```python
from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools

agent = Agent(tools=[HackerNewsTools()], markdown=True)
agent.print_response("What are the top stories on HackerNews?", stream=True)
```

2. 运行代理

安装依赖项

```bash
uv pip install openai agno
```

3. 运行代理
`python news_agent.py`



### 编写自己的工具
为了获得更精细的控制，您可以编写自己的 Python 函数并将其作为工具添加到代理中。例如，以下是如何向代理添加get_top_hackernews_stories工具的方法。



```python
import json
import httpx

from agno.agent import Agent

def get_top_hackernews_stories(num_stories: int = 10) -> str:
    """Use this function to get top stories from Hacker News.

    Args:
        num_stories (int): Number of stories to return. Defaults to 10.
    """

    # Fetch top story IDs
    response = httpx.get('https://hacker-news.firebaseio.com/v0/topstories.json')
    story_ids = response.json()

    # Fetch story details
    stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json')
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        stories.append(story)
    return json.dumps(stories)

agent = Agent(tools=[get_top_hackernews_stories], markdown=True)
agent.print_response("Summarize the top 5 stories on hackernews?", stream=True)
```

## 团队使用定制工具

本示例演示了如何创建具有自定义工具的团队，将自定义工具与代理工具结合使用，以回答知识库中的问题，并在需要时回退到网络搜索。

### 代码

```python
from agno.agent import Agent
from agno.team.team import Team
from agno.tools import tool
from agno.tools.hackernews import HackerNewsTools


@tool()
def answer_from_known_questions(question: str) -> str:
    """Answer a question from a list of known questions

    Args:
        question: The question to answer

    Returns:
        The answer to the question
    """

    # FAQ knowledge base
    faq = {
        "What is the capital of France?": "Paris",
        "What is the capital of Germany?": "Berlin",
        "What is the capital of Italy?": "Rome",
        "What is the capital of Spain?": "Madrid",
        "What is the capital of Portugal?": "Lisbon",
        "What is the capital of Greece?": "Athens",
        "What is the capital of Turkey?": "Ankara",
    }

    # Check if question is in FAQ
    if question in faq:
        return f"From my knowledge base: {faq[question]}"
    else:
        return "I don't have that information in my knowledge base. Try asking the news agent."


# Create news agent for fallback
news_agent = Agent(
    name="News Agent",
    role="Search HackerNews for information",
    tools=[HackerNewsTools()],
    markdown=True,
)

# Create team with custom tool and agent members
team = Team(name="Q & A team", members=[news_agent], tools=[answer_from_known_questions])

# Test the team
team.print_response("What is the capital of France?", stream=True)

# Check if team has session state and display information
print("\nTeam Session Info:")
session = team.get_session()
print(f"   Session ID: {session.session_id}")
print(f"   Session State: {session.session_data['session_state']}")

# Show team capabilities
print("\nTeam Tools Available:")
for t in team.tools:
    print(f"   - {t.name}: {t.description}")

print("\nTeam Members:")
for member in team.members:
    print(f"   - {member.name}: {member.role}")
```

## 团队与工具钩

本示例演示了如何使用工具钩子与团队和代理来拦截和监控工具函数调用，提供日志记录、计时和其他可观测性功能。

```python
import time
from typing import Any, Callable, Dict

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.utils.log import logger


def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """
    Tool hook that logs function calls and measures execution time.

    Args:
        function_name: Name of the function being called
        function_call: The actual function to call
        arguments: Arguments passed to the function

    Returns:
        The result of the function call
    """
    if function_name == "delegate_task_to_member":
        member_id = arguments.get("member_id")
        logger.info(f"Delegating task to member {member_id}")

    # Start timer
    start_time = time.time()
    result = function_call(**arguments)
    # End timer
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Function {function_name} took {duration:.2f} seconds to execute")
    return result


# News agent with tool hooks
news_agent = Agent(
    name="News Agent",
    id="news-agent",
    role="Search HackerNews for information",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools(cache_results=True)],
    instructions=[
        "Find information about the company on HackerNews",
    ],
    tool_hooks=[logger_hook],
)

# Finance agent with tool hooks
finance_agent = Agent(
    name="Finance Agent",
    id="finance-agent",
    role="Get stock prices and financial data",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools(cache_results=True)],
    instructions=[
        "Get stock prices and financial information",
    ],
    tool_hooks=[logger_hook],
)

# Create team with tool hooks
research_team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[
        news_agent,
        finance_agent,
    ],
    markdown=True,
    instructions=[
        "You are a team that researches companies.",
        "Use the news agent for HackerNews discussions and finance agent for stock data.",
    ],
    show_members_responses=True,
    tool_hooks=[logger_hook],
)

if __name__ == "__main__":
    research_team.print_response(
        "Research NVIDIA - get the stock price and find any HackerNews discussions.",
        stream=True,
    )
```


## 异步团队及其工具


本示例演示了如何使用多个代理和不同的工具创建异步团队，利用各种信息收集工具异步收集全面的信息。
​
### 代码

```python
import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

# HackerNews agent
news_agent = Agent(
    name="News Agent",
    role="Search HackerNews for tech news",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    instructions=[
        "Find the latest tech news and discussions",
    ],
)

# Finance agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get stock prices and financial data",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    instructions=[
        "Get stock prices and financial information",
    ],
)

# Create the research team
research_team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[
        news_agent,
        finance_agent,
    ],
    markdown=True,
    instructions=[
        "You are a team that researches companies.",
        "Use the news agent to find recent discussions and the finance agent for stock data.",
    ],
    show_members_responses=True,
)

if __name__ == "__main__":
    asyncio.run(
        research_team.aprint_response(
            "Research NVIDIA - get the current stock price and find any recent HackerNews discussions about the company.",
            stream=True,
        )
    )
```


## 创建你自己的工具

编写自定义工具函数并使用@tool装饰器来修改工具行为。

在大多数生产环境中，您都需要编写自己的工具。因此，我们致力于在 Agno 中提供最佳的工具使用体验。

规则很简单：   
- 任何 Python 函数都可以被代理用作工具。
- 使用@tool装饰器来修改调用此工具之前和之后发生的情况。


在 Agno 中创建工具主要有两种方法：

- 创建一个Python函数（可选使用@tool装饰器）
- 创建工具包
​

### Python 函数作为工具
将任何 Python 函数变成代理工具。

任何 Python 函数都可以被代理用作工具。

例如，以下是如何将get_top_hackernews_stories函数用作工具的方法：

```python
import json
import httpx

from agno.agent import Agent

def get_top_hackernews_stories(num_stories: int = 10) -> str:
    """
    Use this function to get top stories from Hacker News.

    Args:
        num_stories (int): Number of stories to return. Defaults to 10.

    Returns:
        str: JSON string of top stories.
    """

    # Fetch top story IDs
    response = httpx.get('https://hacker-news.firebaseio.com/v0/topstories.json')
    story_ids = response.json()

    # Fetch story details
    stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json')
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        stories.append(story)
    return json.dumps(stories)

agent = Agent(tools=[get_top_hackernews_stories], markdown=True)
agent.print_response("Summarize the top 5 stories on hackernews?", stream=True)
```

### 访问工具中的内置参数
Agno 会自动将一些内置参数注入到您的工具函数中，以便您可以轻松访问工具中的重要信息和对象。
这些内置参数包括：

- run_context：运行上下文对象，您可以从中访问会话状态、依赖项、元数据等。
- agent代理对象。
- team团队对象。
- images图像对象。
- videos视频对象。
- audio音频对象。
- files文件对象。

例如，要在工具中访问代理，您可以执行以下操作：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

def get_agent_model(agent: Agent) -> str:
    """Get the model of the agent."""
    return agent.model.id

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_agent_model],
)

agent.print_response("What is the model of the agent?", stream=True)
```

有关run_context、代理、团队和媒体参数的更多细节，请参见工具内置参数。


## 自定义工具包

将相关的工具功能打包成可重用的工具包类。

许多高级用例需要编写自定义工具包。工具包是一组可以添加到代理中的函数集合。工具包中的函数旨在协同工作、共享内部状态，并提供更佳的开发体验。
大致流程如下：
- 创建一个继承agno.tools.Toolkit类的类。
- 将你的函数添加到类中。
- 将所有函数包含在 Toolkit 构造函数的 tools 参数中。

例如：

```python
from typing import List

from agno.agent import Agent
from agno.tools import Toolkit
from agno.utils.log import logger

class ShellTools(Toolkit):
    def __init__(self, working_directory: str = "/", **kwargs):
        self.working_directory = working_directory

        tools = [
            self.run_shell_command,
        ]

        super().__init__(name="shell_tools", tools=tools, **kwargs)
    
    def list_files(self, directory: str):
        """
        List the files in the given directory.

        Args:
            directory (str): The directory to list the files from.
        Returns:
            str: The list of files in the directory.
        """
        import os

        # List files relative to the toolkit's working_directory
        path = os.path.join(self.working_directory, directory)
        try:
            files = os.listdir(path)
            return "\n".join(files)
        except Exception as e:
            logger.warning(f"Failed to list files in {path}: {e}")
            return f"Error: {e}"
        return os.listdir(directory)

    def run_shell_command(self, args: List[str], tail: int = 100) -> str:
        """
        Runs a shell command and returns the output or error.

        Args:
            args (List[str]): The command to run as a list of strings.
            tail (int): The number of lines to return from the output.
        Returns:
            str: The output of the command.
        """
        import subprocess

        logger.info(f"Running shell command: {args}")
        try:
            logger.info(f"Running shell command: {args}")
            result = subprocess.run(args, capture_output=True, text=True, cwd=self.working_directory)
            logger.debug(f"Result: {result}")
            logger.debug(f"Return code: {result.returncode}")
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            # return only the last n lines of the output
            return "\n".join(result.stdout.split("\n")[-tail:])
        except Exception as e:
            logger.warning(f"Failed to run shell command: {e}")
            return f"Error: {e}"

agent = Agent(tools=[ShellTools()], markdown=True)
agent.print_response("List all the files in my home directory.")
```


### 添加异步方法
任何工具包都可以同时包含同步方法和异步方法。对于受益于异步执行的操作（例如 HTTP 请求、数据库查询或浏览器自动化），您可以提供工具的同步和异步版本。框架会根据执行上下文自动使用合适的版本：
- agent.run()/ agent.print_response()→ 使用同步工具
- agent.arun()/ agent.aprint_response()→ 如果可用，则使用异步工具；否则，回退到同步工具。


要向工具包中添加异步工具，请使用以下async_tools参数：

```python
from typing import Any, Dict

from agno.agent import Agent
from agno.tools import Toolkit

try:
    import httpx
except ImportError:
    raise ImportError("`httpx` not installed. Run `uv pip install httpx`")


class APITools(Toolkit):
    def __init__(self, base_url: str, timeout: float = 30.0, **kwargs):
        self.base_url = base_url
        self.timeout = timeout

        # Sync tools for agent.run() and agent.print_response()
        tools = [
            self.fetch_data,
            self.post_data,
        ]

        # Async tools for agent.arun() and agent.aprint_response()
        # Format: (async_method, "tool_name")
        async_tools = [
            (self.afetch_data, "fetch_data"),
            (self.apost_data, "post_data"),
        ]

        super().__init__(name="api_tools", tools=tools, async_tools=async_tools, **kwargs)

    # Sync methods
    def fetch_data(self, endpoint: str) -> Dict[str, Any]:
        """
        Fetch data from an API endpoint.

        Args:
            endpoint: The API endpoint to fetch data from (e.g., "/users/123")
        Returns:
            The JSON response from the API
        """
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def post_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post data to an API endpoint.

        Args:
            endpoint: The API endpoint to post data to
            data: The data to post as JSON
        Returns:
            The JSON response from the API
        """
        url = f"{self.base_url}{endpoint}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            return response.json()

    # Async methods (used automatically in async contexts)
    async def afetch_data(self, endpoint: str) -> Dict[str, Any]:
        """
        Fetch data from an API endpoint asynchronously.

        Args:
            endpoint: The API endpoint to fetch data from (e.g., "/users/123")
        Returns:
            The JSON response from the API
        """
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    async def apost_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post data to an API endpoint asynchronously.

        Args:
            endpoint: The API endpoint to post data to
            data: The data to post as JSON
        Returns:
            The JSON response from the API
        """
        url = f"{self.base_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            return response.json()

# Create the agent with the toolkit (using JSONPlaceholder - a free fake API for testing)
agent = Agent(tools=[APITools(base_url="https://jsonplaceholder.typicode.com")], markdown=True)

# Sync usage - uses fetch_data
agent.print_response("Fetch the user with ID 1")

# Async usage - uses afetch_data automatically
import asyncio
asyncio.run(agent.aprint_response("Fetch the post with ID 1"))
```
async_tools参数包含一组元组，每个元组包含：

- 异步方法引用
- 工具名称（应与同步工具名称一致，以便自动切换）

异步工具的功能名称不同，但我们用与LLM所识别的同步功能同名注册。示例：在上述代码块中，异步工具是afetch_data的，但大型语言模型（LLM）将其视为fetch_data。

重要提示：

- 请为每个函数填写文档字符串，详细描述函数及其参数。
- 请记住，此函数是提供给 LLM 的，并且不会在代码的其他地方使用，因此文档字符串应该对 LLM 有意义，并且函数名称需要具有描述性。


## 工具钩

使用前置和后置钩子来修改工具行为。

您可以使用工具钩子在调用工具之前或之后执行验证、日志记录或任​​何其他逻辑。
工具钩子是一个函数，它接受一个函数名、函数调用和参数。你也可以选择访问 ` Agent`or`Team`对象。在工具钩子内部，你必须调用该函数并返回结果。

> 定义工具钩子时，务必使用准确的参数名称。agent、team、run_context、function_name、function_call和arguments都是可用的参数。

例如：

```python
def logger_hook(
    function_name: str, function_call: Callable, arguments: Dict[str, Any]
):
    """Log the duration of the function call"""
    start_time = time.time()

    # Call the function
    result = function_call(**arguments)

    end_time = time.time()
    duration = end_time - start_time

    logger.info(f"Function {function_name} took {duration:.2f} seconds to execute")

    # Return the result
    return result
```


或者

```python
def confirmation_hook(
    function_name: str, function_call: Callable, arguments: Dict[str, Any]
):
    """Confirm the function call"""
    if function_name != "get_top_hackernews_stories":
        raise ValueError("This tool is not allowed to be called")
    return function_call(**arguments)
```

您可以为客服人员和团队分配工具钩子。这些工具钩子将应用于客服人员或团队发起的所有工具调用。

例如：

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    tool_hooks=[logger_hook],
)
```

您也可以通过工具钩子访问该RunContext对象。在运行上下文中，您可以找到会话状态、依赖项和元数据。


```python
from agno.run import RunContext

def grab_customer_profile_hook(
    run_context: RunContext, function_name: str, function_call: Callable, arguments: Dict[str, Any]
):
    if not run_context.session_state:
        run_context.session_state = {}

    cust_id = arguments.get("customer")
    if cust_id not in run_context.session_state["customer_profiles"]:
        raise ValueError(f"Customer profile for {cust_id} not found")
    customer_profile = run_context.session_state["customer_profiles"][cust_id]

    # Replace the customer with the customer_profile for the function call
    arguments["customer"] = json.dumps(customer_profile)
    # Call the function with the updated arguments
    result = function_call(**arguments)

    return result
```

### 多用途工具钩
您还可以一次性分配多个工具钩子。它们将按照分配的顺序应用。

```python
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    tool_hooks=[logger_hook, confirmation_hook],  # The logger_hook will run on the outer layer, and the confirmation_hook will run on the inner layer
)

```

您还可以将工具钩子分配给特定的自定义工具。

```python
@tool(tool_hooks=[logger_hook, confirmation_hook])
def get_top_hackernews_stories(num_stories: int) -> Iterator[str]:
    """Fetch top stories from Hacker News.

    Args:
        num_stories (int): Number of stories to retrieve
    """
    # Fetch top story IDs
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # Yield story details
    final_stories = []
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        final_stories.append(story)

    return json.dumps(final_stories)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[get_top_hackernews_stories],
)

```

### 前置钩和后置钩
前置钩子和后置钩子允许你修改工具调用前后的操作。它是工具钩子的替代方案。

- pre_hook在装饰器中设置参数@tool，以便在调用工具之前运行一个函数。
- post_hook在装饰器中设置@tool，以便在工具调用后运行一个函数。

这里有一个使用pre_hook的演示示例，post_hook配合Agent Context。


```python
import json
from typing import Iterator

import httpx
from agno.agent import Agent
from agno.tools import FunctionCall, tool


def pre_hook(fc: FunctionCall):
    print(f"Pre-hook: {fc.function.name}")
    print(f"Arguments: {fc.arguments}")
    print(f"Result: {fc.result}")


def post_hook(fc: FunctionCall):
    print(f"Post-hook: {fc.function.name}")
    print(f"Arguments: {fc.arguments}")
    print(f"Result: {fc.result}")


@tool(pre_hook=pre_hook, post_hook=post_hook)
def get_top_hackernews_stories(agent: Agent) -> Iterator[str]:
    num_stories = agent.context.get("num_stories", 5) if agent.context else 5

    # Fetch top story IDs
    response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
    story_ids = response.json()

    # Yield story details
    for story_id in story_ids[:num_stories]:
        story_response = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
        )
        story = story_response.json()
        if "text" in story:
            story.pop("text", None)
        yield json.dumps(story)


agent = Agent(
    dependencies={
        "num_stories": 2,
    },
    tools=[get_top_hackernews_stories],
    markdown=True,
)
agent.print_response("What are the top hackernews stories?", stream=True)

```

## 模型上下文协议（MCP）

通过标准化的 MCP 接口将代理连接到外部系统。

模型上下文协议 (MCP)使代理能够通过标准化接口与外部系统交互。您可以使用 Agno 的 MCP 集成将代理连接到任何 MCP 服务器。
下面这个简单示例展示了如何将代理连接到 Agno MCP 服务器：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools

# Create the Agent
agno_agent = Agent(
    name="Agno Agent",
    model=Claude(id="claude-sonnet-4-0"),
    # Add the Agno MCP server to the Agent
    tools=[MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp")],
)
```

### 基本流程
1. 找到您要使用的 MCP 服务器

您可以使用任何可用的MCP服务器。要查看一些示例，您可以查看MCP维护者自己提供的这个GitHub仓库。

2. 初始化 MCP 集成

初始化MCPTools类并连接到MCP服务器。定义MCP服务器的推荐方法是使用命令或URL参数。通过命令，你可以传递运行你想要的MCP服务器的命令。通过 url，你可以传递你想使用的运行中的 MCP 服务器的 URL。

例如，要连接到 Agno 文档 MCP 服务器，您可以执行以下操作：

```python
from agno.tools.mcp import MCPTools

# Initialize and connect to the MCP server
mcp_tools = MCPTools(transport="streamable-http", url="https://docs.agno.com/mcp"))
await mcp_tools.connect()
```

3. 向代理提供 MCPTools

初始化代理时，请将MCPTools实例作为tools参数传递。完成后，请记得关闭连接。
代理现在可以开始使用MCP服务器了：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools

# Initialize and connect to the MCP server
mcp_tools = MCPTools(url="https://docs.agno.com/mcp")
await mcp_tools.connect()

try:
    # Setup and run the agent
    agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
    await agent.aprint_response("Tell me more about MCP support in Agno", stream=True)
finally:
    # Always close the connection when done
    await mcp_tools.close()

```

### 示例：文件系统代理
这是一个使用文件系统 MCP 服务器来浏览和分析文件的文件系统代理：

```python
import asyncio
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools


async def run_agent(message: str) -> None:
    """Run the filesystem agent with the given message."""

    file_path = "<path to the directory you want to explore>"

    # Initialize and connect to the MCP server to access the filesystem
    mcp_tools = MCPTools(command=f"npx -y @modelcontextprotocol/server-filesystem {file_path}")
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
            instructions=dedent("""\
                You are a filesystem assistant. Help users explore files and directories.

                - Navigate the filesystem to answer questions
                - Use the list_allowed_directories tool to find directories that you can access
                - Provide clear context about files you examine
                - Use headings to organize your responses
                - Be concise and focus on relevant information\
            """),
            markdown=True,
        )

        # Run the agent
        await agent.aprint_response(message, stream=True)
    finally:
        # Always close the connection when done
        await mcp_tools.close()


# Example usage
if __name__ == "__main__":
    # Basic example - exploring project license
    asyncio.run(run_agent("What is the license for this project?"))
```

### 连接您的 MCP 服务器
​
使用connect()和close()

建议使用`connect()`and`close()`方法来管理 MCP 服务器的连接生命周期。

```python

mcp_tools = MCPTools(command="uvx mcp-server-git")
await mcp_tools.connect()
```

完成后，您应该关闭与 MCP 服务器的连接。

```python

await mcp_tools.close()
```

### 自动连接管理
如果您在未先调用 `connect()` 的情况下将 `MCPTools` 实例传递给 `Agent` 或 `Team` 实例，则将自动管理连接。

例如：

```python
mcp_tools = MCPTools(command="uvx mcp-server-git")
agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
await agent.aprint_response("What is the license for this project?", stream=True)  # The connection is established and closed on each run.
```
> 此处，每次运行都会建立和关闭与 MCP 服务器（如果是托管 MCP 服务器）的连接。此外，每次运行都会刷新可用工具列表。
> 这会影响性能，不建议在生产中使用。


### 使用异步上下文管理器
MCPTools如果您愿意，也可以使用MultiMCPTools异步上下文管理器进行自动资源清理：

```python
async with MCPTools(command="uvx mcp-server-git") as mcp_tools:
    agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
    await agent.aprint_response("What is the license for this project?", stream=True)
```

这种模式会自动处理连接和清理，但显式.connect()的.close()方法可以更好地控制连接生命周期。

### AgentOS 中的自动连接管理
在 AgentOS 中使用 MCPTools 时，生命周期会自动管理。无需手动连接或断开MCPTools实例。但连接不会自动刷新，您需要手动刷新连接refresh_connection。
更多详情请参见AgentOS + MCPTools页面。



### 连接刷新
您可以设置refresh_connection实例MCPTools，MultiMCPTools以便在每次运行时刷新与 MCP 服务器的连接。


```python
mcp_tools = MCPTools(command="uvx mcp-server-git", refresh_connection=True)
await mcp_tools.connect()

agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
await agent.aprint_response("What is the license for this project?", stream=True)  # The connection will be refreshed on each run.

await mcp_tools.close()
```


### 工作原理
- 调用该connect()方法时，会与 MCP 服务器建立一个新的会话。如果该服务器不可用，则该连接将被关闭，并且必须建立一个新的连接。
- 如果设置refresh_connection为True，则每次运行代理时都会检查与 MCP 服务器的连接，并在需要时重新建立连接，然后刷新可用工具列表。
- 这对于容易重启或经常更改其架构或工具列表的托管 MCP 服务器尤其有用。
- 建议仅在手动管理 MCP 服务器的连接生命周期，或在MCPTools中AgentOS使用代理/团队时使用此方法。
​

### 运输
模型上下文协议 (MCP) 中的传输方式定义了消息的发送和接收方式。Agno 集成支持以下三种现有类型：
- 标准输入输出 (stdio) -> 请参阅标准输入输出传输文档
- Streamable HTTP -> 请参阅Streamable HTTP 传输文档
- SSE -> 请参阅SSE 传输文档

> stdio（标准输入/输出）传输是 Agno 的默认传输MCPTools方式MultiMCPTools。


### 最佳实践
1. 资源清理：完成后务必关闭 MCP 连接，以防止资源泄漏：

```python
mcp_tools = MCPTools(command="uvx mcp-server-git")
await mcp_tools.connect()

try:
    # Your agent code here
    pass
finally:
    await mcp_tools.close()
```

2. 错误处理：始终为 MCP 服务器连接和操作包含适当的错误处理。

3. 明确指示：向您的代理人提供清晰明确的指示：
```python
instructions = """
You are a filesystem assistant. Help users explore files and directories.
- Navigate the filesystem to answer questions
- Use the list_allowed_directories tool to find accessible directories
- Provide clear context about files you examine
- Be concise and focus on relevant information
"""
```



## MCP 工具箱

连接到具有工具筛选功能的 MCP 数据库工具箱。

MCPToolbox使代理能够连接到 Google 的MCP Toolbox for Databases，并提供高级筛选功能。它扩展了 Agno 的MCPTools功能，允许按工具集或工具名称筛选工具，从而使代理能够仅加载所需的特定数据库工具。

### 先决条件
要使用 MCPToolbox，您需要以下工具：

```python
uv pip install toolbox-core
```

我们的默认设置还要求您安装 Docker 或 Podman，以便运行示例的 MCP Toolbox 服务器和数据库。


### 快速入门
立即使用我们功能齐全的演示版开始体验 MCPToolbox。

```python
# Clone the repo and navigate to the demo folder
git clone https://github.com/agno-agi/agno.git
cd agno/cookbook/14_tools/mcp/mcp_toolbox_demo

# Start the database and MCP Toolbox servers

# With Docker and Docker Compose
docker-compose up -d

# With Podman
podman compose up -d

# Install dependencies
uv sync

# Set your API key and run the basic agent
export OPENAI_API_KEY="your_openai_api_key"
uv run agent.py
```

这会启动一个包含示例酒店数据的 PostgreSQL 数据库和一个 MCP Toolbox 服务器，该服务器将数据库操作公开为经过筛选的工具。


### 确认
要验证您的 docker/podman 设置是否正常工作，您可以检查数据库连接：


```python
# Using Docker Compose
docker-compose exec db psql -U toolbox_user -d toolbox_db -c "SELECT COUNT(*) FROM hotels;"

# Using Podman
podman exec db psql -U toolbox_user -d toolbox_db -c "SELECT COUNT(*) FROM hotels;"
```

### 基本示例
以下是使用 MCPToolbox 的最简单方法（在运行快速入门设置之后）：

```python
import asyncio
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp_toolbox import MCPToolbox

async def main():
    # Connect to the running MCP Toolbox server and filter to hotel tools only
    async with MCPToolbox(
        url="http://127.0.0.1:5001",
        toolsets=["hotel-management"]  # Only load hotel search tools
    ) as toolbox:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[toolbox],
            instructions="You help users find hotels. Always mention hotel ID, name, location, and price tier."
        )

        # Ask the agent to find hotels
        await agent.aprint_response("Find luxury hotels in Zurich")

# Run the example
asyncio.run(main())
```

### MCPToolbox 的工作原理
MCPToolbox 解决了工具过载的问题。如果没有过滤，您的代理程序会被过多的数据库工具淹没：

不含 MCPToolbox（50 多种工具）：

```python
# Agent gets ALL database tools - overwhelming!
tools = MCPTools(url="http://127.0.0.1:5001")  # 50+ tools
```

使用 MCPToolbox（3 个相关工具）：

```python
# Agent gets only hotel management tools - focused!
tools = MCPToolbox(url="http://127.0.0.1:5001", toolsets=["hotel-management"])  # 3 tools
```
流程：
- MCP Toolbox Server 提供 50 多种数据库工具
- MCPToolbox 连接并加载内部所有工具
- 筛选出仅包含hotel-management工具集（3 个工具）
- 代理只看到 3 个相关工具，并保持专注。


### 高级用法
​
多套工具集

从多个相关工具集中加载工具：

```python
import asyncio
from textwrap import dedent
from agno.agent import Agent
from agno.tools.mcp_toolbox import MCPToolbox

url = "http://127.0.0.1:5001"

async def run_agent(message: str = None) -> None:
    """Run an interactive CLI for the Hotel agent with the given message."""

    async with MCPToolbox(
        url=url, toolsets=["hotel-management", "booking-system"]
    ) as db_tools:
        print(db_tools.functions)  # Print available tools for debugging
        agent = Agent(
            tools=[db_tools],
            instructions=dedent(
                """ \
                You're a helpful hotel assistant. You handle hotel searching, booking and
                cancellations. When the user searches for a hotel, mention it's name, id,
                location and price tier. Always mention hotel ids while performing any
                searches. This is very important for any operations. For any bookings or
                cancellations, please provide the appropriate confirmation. Be sure to
                update checkin or checkout dates if mentioned by the user.
                Don't ask for confirmations from the user.
            """
            ),
            markdown=True,
            show_tool_calls=True,
            add_history_to_messages=True,
            debug_mode=True,
        )

        await agent.acli_app(message=message, stream=True)

if __name__ == "__main__":
    asyncio.run(run_agent(message=None))
```


### 自定义身份验证和参数
对于需要身份验证的生产环境：

```python
async def production_example():
    async with MCPToolbox(url=url) as toolbox:
        # Load with authentication and bound parameters
        hotel_tools = await toolbox.load_toolset(
            "hotel-management",
            auth_token_getters={"hotel_api": lambda: "your-hotel-api-key"},
            bound_params={"region": "us-east-1"},
        )

        booking_tools = await toolbox.load_toolset(
            "booking-system",
            auth_token_getters={"booking_api": lambda: "your-booking-api-key"},
            bound_params={"environment": "production"},
        )

        # Use individual tools instead of the toolbox
        all_tools = hotel_tools + booking_tools[:2]  # First 2 booking tools only

        agent = Agent(tools=all_tools, instructions="Hotel management with auth.")
        await agent.aprint_response("Book a hotel for tonight")
```

### 手动连接管理
为了对连接进行显式控制：


```python
async def manual_connection_example():
    # Initialize without auto-connection
    toolbox = MCPToolbox(url=url, toolsets=["hotel-management"])

    try:
        await toolbox.connect()
        agent = Agent(
            tools=[toolbox],
            instructions="Hotel search assistant.",
            markdown=True
        )
        await agent.aprint_response("Show me hotels in Basel")
    finally:
        await toolbox.close()  # Always clean up
```

### 工具包参数

只能指定工具组或tool_name中的一种。实现会验证这一点，如果两者都提供，则会触发 ValueError。
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| url | str | － | 工具箱服务的基本 URL（如果缺少 ＂／mcp＂，则自动附加） |
| toolsets | Optional［List［str］］ | None | 用于筛选工具的工具集名称列表。不能与．一起使用 tool＿name 。 |
| tool＿name | Optional［str］ | None | 仅加载单个工具名称。不能与 toolsets ． |
| headers | Optional［Dict［str， Any］］ | None | 工具箱客户端请求的 HTTP 标头 |
| transport | str | ＂streamable－ http＂ | MCP传输协议。选项：＂stdio＂，．＂sse＂＂streamab le－http＂ |


### 工具包功能

| 功能 | 描述 |
| :--- | :--- |
| async connect（） | 初始化并连接到 MCP 服务器和工具箱客户端 |
| async load＿tool（tool＿name，auth＿token＿getters＝\｛}, bound_params= \｛}) | 按名称加载单个工具，并可选择进行身份验证。 |
| async load＿toolset（toolset＿name，auth＿token＿getters＝\｛}, bound＿params＝\｛}, strict=False) | 从特定工具集中加载所有工具 |
| async load＿multiple＿toolsets（toolset＿names，auth＿token＿getters＝ \｛}, bound_params={}, strict=False) | 从多个工具集中加载工具 |
| async load＿toolset＿safe（toolset＿name） | 安全加载工具集井返回工具名称以进行错误处理 |
| get＿client（） | 获取底层 ToolboxClient 实例 |
| async close（） | 关闭工具箱客户端和 MCP 客户端的连接 |



## 多台MCP服务器

了解如何使用 Agno 连接到多个 MCP 服务器

Agno 的 MCP 集成还支持处理与多个服务器的连接、指定服务器参数以及使用您自己的 MCP 服务器。
对此有两种方法：
使用多个MCPTools实例
使用单个MultiMCPTools实例

### 使用多个MCPTools实例

```python
import asyncio
import os

from agno.agent import Agent
from agno.tools.mcp import MCPTools


async def run_agent(message: str) -> None:
    """Run the Airbnb and Google Maps agent with the given message."""

    env = {
        **os.environ,
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
    }

    # Initialize and connect to multiple MCP servers
    airbnb_tools = MCPTools(command="npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
    google_maps_tools = MCPTools(command="npx -y @modelcontextprotocol/server-google-maps", env=env)
    await airbnb_tools.connect()
    await google_maps_tools.connect()

    try:
        agent = Agent(
            tools=[airbnb_tools, google_maps_tools],
            markdown=True,
        )

        await agent.aprint_response(message, stream=True)
    finally:
        await airbnb_tools.close()
        await google_maps_tools.close()


# Example usage
if __name__ == "__main__":
    # Pull request example
    asyncio.run(
        run_agent(
            "What listings are available in Cape Town for 2 people for 3 nights from 1 to 4 August 2025?"
        )
    )
```

### 使用单个MultiMCPTools实例
```python
import asyncio
import os

from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools


async def run_agent(message: str) -> None:
    """Run the Airbnb and Google Maps agent with the given message."""

    env = {
        **os.environ,
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
    }

    # Initialize and connect to multiple MCP servers
    mcp_tools = MultiMCPTools(
        commands=[
            "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
            "npx -y @modelcontextprotocol/server-google-maps",
        ],
        env=env,
    )
    await mcp_tools.connect()

    try:
        agent = Agent(
            tools=[mcp_tools],
            markdown=True,
        )

        await agent.aprint_response(message, stream=True)
    finally:
        # Always close the connection when done
        await mcp_tools.close()


# Example usage
if __name__ == "__main__":
    # Pull request example
    asyncio.run(
        run_agent(
            "What listings are available in Cape Town for 2 people for 3 nights from 1 to 4 August 2025?"
        )
    )
```

#### 允许部分失败MultiMCPTools
如果使用该类连接到多个 MCP 服务器MultiMCPTools，则默认情况下，如果连接到任何 MCP 服务器失败，都会引发错误。

如果不想在这种情况下引发异常，可以将allow_partial_failures参数设置为True。

如果您要连接到并非始终可用的 MCP 服务器，并且不希望在其中一个服务器不可用时退出程序，这将非常有用。


```python
import asyncio
from os import getenv

from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools


async def run_agent(message: str) -> None:
    # Initialize the MCP tools
    mcp_tools = MultiMCPTools(
        [
            "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
            "npx -y @modelcontextprotocol/server-brave-search",
        ],
        env={
            "BRAVE_API_KEY": getenv("BRAVE_API_KEY"),
        },
        timeout_seconds=30,
        # Set the allow_partial_failure to True to allow for partial failure connecting to the MCP servers
        allow_partial_failure=True,
    )

    # Connect to the MCP servers
    await mcp_tools.connect()

    # Use the MCP tools with an Agent
    agent = Agent(
        tools=[mcp_tools],
        markdown=True,
    )
    await agent.aprint_response(message)

    # Close the MCP connection
    await mcp_tools.close()


# Example usage
if __name__ == "__main__":
    asyncio.run(run_agent("What listings are available in Barcelona tonight?"))
    asyncio.run(run_agent("What's the fastest way to get to Barcelona from London?"))

```

### 避免工具名称冲突
使用多个 MCP 服务器时，可能会遇到工具名称冲突的情况。这种情况通常发生在多个服务器上都存在同一个工具时。

为避免这种情况，您可以使用该tool_name_prefix参数。这将为来自 MCPTools 实例的所有工具名称添加指定的前缀。

```python
import asyncio

from agno.agent import Agent
from agno.tools.mcp import MCPTools


async def run_agent():
    # Development environment tools
    dev_tools = MCPTools(
        transport="streamable-http",
        url="https://docs.agno.com/mcp",
        # By providing this tool_name_prefix, all the tool names will be prefixed with "dev_"
        tool_name_prefix="dev",
    )
    await dev_tools.connect()

    agent = Agent(tools=[dev_tools])
    await agent.aprint_response("Which tools do you have access to? List them all.")

    await dev_tools.close()


if __name__ == "__main__":
    asyncio.run(run_agent())
```
## 动态标头

使用 Agno MCP 工具设置动态标头

使用 MCP 工具时，您通常需要通过 HTTP 标头向 MCP 服务器发送信息。例如，您可能需要传递有关当前用户的信息或授权相关数据。


要动态实现这一点，你可以在初始化MCPTools类时使用header_provider参数。

你还可以访问 RunContext 对象，该对象包含当前运行的上下文，包括用户 ID、会话 ID 和元数据，以及上下文中的代理或团队。只需将run_context、代理和团队参数添加到你的header_provider函数中即可。

```python
from agno.tools.mcp import MCPTools

# Our header_provider function, which will generate the headers per run.
def header_provider(
    run_context: RunContext, # The current run's context
    agent: Optional["Agent"] = None, # The contextual Agent instance
    team: Optional["Team"] = None, # The contextual Team instance
) -> dict:
    headers = {
        "X-User-ID": run_context.user_id or "unknown",
        "X-Session-ID": run_context.session_id or "unknown",
        "X-Run-ID": run_context.run_id,
        "X-Agent-Name": agent.name if agent else None,
        "X-Team-Name": team.name if team else None,
    }
    return headers

# When these MCP tools are used, the header_provider function will be used to update the headers for each run.
mcp_tools = MCPTools(
    url="http://localhost:8000/mcp",
    header_provider=header_provider,
)
```

### 该header_provider函数
传递给 MCPTools 实例的函数header_provider将用于每次运行更新标头。
该函数预期返回一个dict包含头部名称-值对的数组。
以下参数将自动注入到函数中，可用于生成标头：

| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| run＿context | RunContext | 包含特定于运行的数据，例如 run＿id，，，和 user＿id session＿id metadata |
| agent | Agent | 发出工具调用的代理实例（如果适用） |
| team | Team | 发起工具调用的团队实例（如果适用） |

您可以在RunContext 参考文档中阅读更多有关该RunContext对象及其字段的信息。

### 完整示例
运行示例 MCP 服务器：

```python
from fastmcp import FastMCP
from fastmcp.server import Context
from fastmcp.server.dependencies import get_http_request

mcp = FastMCP("My Server")


@mcp.tool
async def greet(name: str, ctx: Context) -> str:
    """Greet a user with personalized information from headers."""
    # Get the HTTP request object
    request = get_http_request()

    # Access headers (lowercase!)
    user_id = request.headers.get("x-user-id", "unknown")
    tenant_id = request.headers.get("x-tenant-id", "unknown")
    agent_name = request.headers.get("x-agent-name", "unknown")

    print("=" * 60)
    print(f"Headers -> Agent: {agent_name}, User: {user_id}, Tenant: {tenant_id}")
    print("=" * 60)

    return f"Hello, {name}! (User: {user_id}, Tenant: {tenant_id})"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
```

### 该header_provider函数
传递给 MCPTools 实例的函数header_provider将用于每次运行更新标头。

该函数预期返回一个dict包含头部名称-值对的数组。

以下参数将自动注入到函数中，可用于生成标头：

| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| run＿context | RunContext | 包含特定于运行的数据，例如 run＿id，，，和 user＿id session＿id metadata |
| agent | Agent | 发出工具调用的代理实例（如果适用） |
| team | Team | 发起工具调用的团队实例（如果适用） |

您可以在RunContext 参考文档中阅读更多有关该RunContext对象及其字段的信息。

### 完整示例
运行示例 MCP 服务器：


```python
from fastmcp import FastMCP
from fastmcp.server import Context
from fastmcp.server.dependencies import get_http_request

mcp = FastMCP("My Server")


@mcp.tool
async def greet(name: str, ctx: Context) -> str:
    """Greet a user with personalized information from headers."""
    # Get the HTTP request object
    request = get_http_request()

    # Access headers (lowercase!)
    user_id = request.headers.get("x-user-id", "unknown")
    tenant_id = request.headers.get("x-tenant-id", "unknown")
    agent_name = request.headers.get("x-agent-name", "unknown")

    print("=" * 60)
    print(f"Headers -> Agent: {agent_name}, User: {user_id}, Tenant: {tenant_id}")
    print("=" * 60)

    return f"Hello, {name}! (User: {user_id}, Tenant: {tenant_id})"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
```

运行示例客户端：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.run import RunContext
from agno.tools.mcp import MCPTools


def header_provider(run_context: RunContext) -> dict:
    """Generate headers from the current run context."""
    return {
        "X-User-ID": run_context.user_id or "anonymous",
        "X-Session-ID": run_context.session_id or "no-session",
        "X-Run-ID": run_context.run_id,
    }


async def main():
    # Create MCPTools with dynamic headers
    mcp_tools = MCPTools(
        url="http://localhost:8000/mcp",
        transport="streamable-http",
        header_provider=header_provider,  # Enable dynamic headers
    )
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
        )

        # The header_provider receives context from these parameters
        await agent.arun(
            "Hello, my name is Bob!",
            user_id="user-123",
            session_id="session-456",
        )
    finally:
        await mcp_tools.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 与 MultiMCPTools 配合使用
动态标头的工作方式相同MultiMCPTools：

```python
from agno.tools.mcp import MultiMCPTools

mcp_tools = MultiMCPTools(
    urls=[
        "http://server1.example.com/mcp",
        "http://server2.example.com/mcp",
    ],
    transport="streamable-http",
    header_provider=header_provider,  # Applied to all servers
)
```


## 传输
### Stdio 传输

在 Agno 的集成中，stdio（标准输入/输出）传输是默认的传输方式。它最适合本地集成。

使用方法很简单，只需MCPTools用参数初始化类即可command。您要传递的命令是用于运行代理将要访问的 MCP 服务器的命令。

例如uvx mcp-server-git，运行git MCP 服务器的程序：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools

# Initialize and connect to the MCP server
# Can also use custom binaries: command="./my-mcp-server"
mcp_tools = MCPTools(command="uvx mcp-server-git")
await mcp_tools.connect()

try:
    agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
    await agent.aprint_response("What is the license for this project?", stream=True)
finally:
    # Always close the connection when done
    await mcp_tools.close()

```

您还可以通过该类同时使用多个 MCP 服务器MultiMCPTools。例如：

```python
import asyncio
import os

from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools


async def run_agent(message: str) -> None:
    """Run the Airbnb and Google Maps agent with the given message."""

    env = {
        **os.environ,
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
    }

    # Initialize and connect to multiple MCP servers
    mcp_tools = MultiMCPTools(
        commands=[
            "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
            "npx -y @modelcontextprotocol/server-google-maps",
        ],
        env=env,
    )
    await mcp_tools.connect()

    try:
        agent = Agent(
            tools=[mcp_tools],
            markdown=True,
        )

        await agent.aprint_response(message, stream=True)
    finally:
        # Always close the connection when done
        await mcp_tools.close()


# Example usage
if __name__ == "__main__":
    # Pull request example
    asyncio.run(
        run_agent(
            "What listings are available in Cape Town for 2 people for 3 nights from 1 to 4 August 2025?"
        )
    )

```



### Streamable HTTP 传输

新的Streamable HTTP 传输协议取代了协议版本中的 HTTP+SSE 传输协议2024-11-05。

该传输方式使 MCP 服务器能够处理多个客户端连接，并且还可以使用 SSE 进行服务器到客户端的流式传输。

要使用它，请初始化并MCPTools传入 MCP 服务器的 URL，并将传输方式设置为streamable-http：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools

# Initialize and connect to the Streamable HTTP MCP server
mcp_tools = MCPTools(url="https://docs.agno.com/mcp", transport="streamable-http")
await mcp_tools.connect()

try:
    agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
    await agent.aprint_response("What can you tell me about MCP support in Agno?", stream=True)
finally:
    # Always close the connection when done
    await mcp_tools.close()
```

您还可以使用该server_params参数来定义 MCP 连接。这样，您可以指定每次请求时要发送到 MCP 服务器的标头以及超时值：

```python
from agno.tools.mcp import MCPTools, StreamableHTTPClientParams

server_params = StreamableHTTPClientParams(
    url=...,
    headers=...,
    timeout=...,
    sse_read_timeout=...,
    terminate_on_close=...,
)

# Initialize and connect using server parameters
mcp_tools = MCPTools(server_params=server_params, transport="streamable-http")
await mcp_tools.connect()

try:
    # Use mcp_tools with your agent
    pass
finally:
    await mcp_tools.close()
```

### 完整示例
让我们搭建一个简单的本地服务器，并使用 Streamable HTTP 传输协议连接到它：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calendar_assistant")


@mcp.tool()
def get_events(day: str) -> str:
    return f"There are no events scheduled for {day}."


@mcp.tool()
def get_birthdays_this_week() -> str:
    return "It is your mom's birthday tomorrow"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```


```python
import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools, MultiMCPTools

# This is the URL of the MCP server we want to use.
server_url = "http://localhost:8000/mcp"


async def run_agent(message: str) -> None:
    # Initialize and connect to the Streamable HTTP MCP server
    mcp_tools = MCPTools(transport="streamable-http", url=server_url)
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)
    finally:
        await mcp_tools.close()


# Using MultiMCPTools, we can connect to multiple MCP servers at once, even if they use different transports.
# In this example we connect to both our example server (Streamable HTTP transport), and a different server (stdio transport).
async def run_agent_with_multimcp(message: str) -> None:
    # Initialize and connect to multiple MCP servers with different transports
    mcp_tools = MultiMCPTools(
        commands=["npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt"],
        urls=[server_url],
        urls_transports=["streamable-http"],
    )
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)
    finally:
        await mcp_tools.close()


if __name__ == "__main__":
    asyncio.run(run_agent("Do I have any birthdays this week?"))
    asyncio.run(
        run_agent_with_multimcp(
            "Can you check when is my mom's birthday, and if there are any AirBnb listings in SF for two people for that day?"
        )
    )
```

### SSE传输
Agno 的 MCP 集成支持SSE 传输。这种传输方式支持服务器到客户端的流媒体传输，在网络受限的情况下，它比标准 I/O更有用。

MCP 协议不再推荐使用此传输方式。请改用Streamable HTTP 传输方式。

要使用它，请初始化并MCPTools传入 MCP 服务器的 URL，并将传输方式设置为sse：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools

server_url = "http://localhost:8000/sse"

# Initialize and connect to the SSE MCP server
mcp_tools = MCPTools(url=server_url, transport="sse")
await mcp_tools.connect()

try:
    agent = Agent(model=OpenAIResponses(id="gpt-5.2"), tools=[mcp_tools])
    await agent.aprint_response("What is the license for this project?", stream=True)
finally:
    # Always close the connection when done
    await mcp_tools.close()
```

您还可以使用该server_params参数来定义 MCP 连接。这样，您可以指定每次请求时要发送到 MCP 服务器的标头以及超时值：

```python
from agno.tools.mcp import MCPTools, SSEClientParams

server_params = SSEClientParams(
    url=...,
    headers=...,
    timeout=...,
    sse_read_timeout=...,
)

# Initialize and connect using server parameters
mcp_tools = MCPTools(server_params=server_params, transport="sse")
await mcp_tools.connect()

try:
    # Use mcp_tools with your agent
    pass
finally:
    await mcp_tools.close()

```


### 完整示例
让我们搭建一个简单的本地服务器，并使用 SSE 传输协议连接到它：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calendar_assistant")


@mcp.tool()
def get_events(day: str) -> str:
    return f"There are no events scheduled for {day}."


@mcp.tool()
def get_birthdays_this_week() -> str:
    return "It is your mom's birthday tomorrow"


if __name__ == "__main__":
    mcp.run(transport="sse")
```


```python
import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.mcp import MCPTools, MultiMCPTools

# This is the URL of the MCP server we want to use.
server_url = "http://localhost:8000/sse"


async def run_agent(message: str) -> None:
    # Initialize and connect to the SSE MCP server
    mcp_tools = MCPTools(transport="sse", url=server_url)
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)
    finally:
        await mcp_tools.close()


# Using MultiMCPTools, we can connect to multiple MCP servers at once, even if they use different transports.
# In this example we connect to both our example server (SSE transport), and a different server (stdio transport).
async def run_agent_with_multimcp(message: str) -> None:
    # Initialize and connect to multiple MCP servers with different transports
    mcp_tools = MultiMCPTools(
        commands=["npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt"],
        urls=[server_url],
        urls_transports=["sse"],
    )
    await mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIResponses(id="gpt-5.2"),
            tools=[mcp_tools],
            markdown=True,
        )
        await agent.aprint_response(message=message, stream=True, markdown=True)
    finally:
        await mcp_tools.close()


if __name__ == "__main__":
    asyncio.run(run_agent("Do I have any birthdays this week?"))
    asyncio.run(
        run_agent_with_multimcp(
            "Can you check when is my mom's birthday, and if there are any AirBnb listings in SF for two people for that day?"
        )
    )
```

## 了解服务器参数

了解如何配置 MCPTools 和 MultiMCPTools 类的服务器参数

推荐的配置方法MCPTools是使用commandorurl参数。

或者，您可以使用server_params参数MCPTools来更详细地配置与 MCP 服务器的连接。

使用stdio传输时，该server_params参数应为 . 的一个实例StdioServerParameters。它包含以下键：

- command：运行 MCP 服务器的命令。
    - 适用npx于可通过 npm 安装的 mcp 服务器（或node在 Windows 上运行）。
    - 适用uvx于可通过 uvx 安装的 mcp 服务器。
    - 使用自定义二进制可执行文件（例如，.bashrc ./my-server、../usr/local/bin/my-server.bashrc 或 PATH 中的二进制文件）。
- args要传递给 MCP 服务器的参数。
- env：可选的环境变量，用于传递给 MCP 服务器。请务必将所有当前环境变量包含在env字典中。如果env未提供，则将使用当前环境变量。例如：

```python
{
    **os.environ,
    "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
}
```

使用SSE传输时，该server_params参数应为 的实例SSEClientParams。它包含以下字段：    
- url MCP 服务器的 URL。
- headers:要传递给 MCP 服务器的标头（可选）。
- timeout：与 MCP 服务器连接的超时时间（可选）。
- sse_read_timeout：SSE 连接本身的超时时间（可选）。

使用Streamable HTTP传输时，该server_params参数应为 的实例StreamableHTTPClientParams。它包含以下字段：   
- url MCP 服务器的 URL。
- headers:要传递给 MCP 服务器的标头（可选）。
- timeout：与 MCP 服务器连接的超时时间（可选）。
- sse_read_timeout：客户端在断开连接前等待新事件的时间（以秒为单位）。所有其他 HTTP 操作均由timeout（可选）控制。
- terminate_on_close：客户端关闭时是否终止连接（可选）。


## 并行MCP代理

使用Parallel MCP 服务器创建一个代理，该代理可以使用 Parallel 的 AI 优化搜索功能搜索网络：

```python
"""MCP Parallel Agent - Search for Parallel

This example shows how to create an agent that uses Parallel to search for information using the Parallel MCP server.

Run: `uv pip install anthropic mcp agno` to install the dependencies

Prerequisites:
- Set the environment variable "PARALLEL_API_KEY" with your Parallel API key.
- Set the environment variable "ANTHROPIC_API_KEY" with your Anthropic API key.
- You can get the Parallel API key from: https://platform.parallel.ai/
- You can get the Anthropic API key from: https://console.anthropic.com/

Usage:
  python cookbook/14_tools/mcp/parallel.py
"""

import asyncio
from os import getenv

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools
from agno.tools.mcp.params import StreamableHTTPClientParams
from agno.utils.pprint import apprint_run_response

server_params = StreamableHTTPClientParams(
    url="https://search-mcp.parallel.ai/mcp",
    headers={
        "authorization": f"Bearer {getenv('PARALLEL_API_KEY')}",
    },
)


async def run_agent(message: str) -> None:
    async with MCPTools(
        transport="streamable-http", server_params=server_params
    ) as parallel_mcp_server:
        agent = Agent(
            model=Claude(id="claude-sonnet-4-20250514"),
            tools=[parallel_mcp_server],
            markdown=True,
        )
        response_stream = await agent.arun(message)
        await apprint_run_response(response_stream)


if __name__ == "__main__":
    asyncio.run(run_agent("What is the weather in Tokyo?"))
```


## 推理工具


该ReasoningTools工具包允许智能体在执行过程中的任何阶段像使用其他工具一样进行推理。与传统方法在开始时进行一次推理以创建固定计划不同，该工具包使智能体能够在每一步之后进行反思，调整其思路，并实时更新其行动。

我们发现，这种方法显著提高了智能体解决原本无法处理的复杂问题的能力。通过给予智能体“思考”其行为的空间，它可以更深入地审视自身的反应，质疑其假设，并从不同的角度解决问题。

该工具包包含以下工具：

think：该工具被智能体用作草稿本，用于思考问题并逐步解决问题。它有助于将复杂问题分解成更小、更易于处理的部分，并跟踪推理过程。

analyze该工具用于分析推理步骤的结果，并确定下一步行动。

### 例子
以下是如何使用该工具包的示例ReasoningTools：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

thinking_agent = Agent(
    model=Claude(id="claude-3-7-sonnet-latest"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        ),
    ],
    instructions="Use tables where possible",
    markdown=True,
)

thinking_agent.print_response("Write a report comparing NVDA to TSLA", stream=True)
```

该工具包包含默认说明和简短示例，旨在帮助代理有效使用该工具。以下是启用这些说明和示例的方法：

```python
reasoning_agent = Agent(
    model=Claude(id="claude-3-7-sonnet-latest"),
    tools=[
        ReasoningTools(
            think=True,
            analyze=True,
            add_instructions=True,
            add_few_shot=True,
        ),
    ],
)
```

ReasoningTools可以与任何支持函数调用的模型提供程序一起使用。以下是一个使用推理代理的OpenAIResponses示例：

```python
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.reasoning import ReasoningTools

reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[ReasoningTools(add_instructions=True)],
    instructions=dedent("""\
        You are an expert problem-solving assistant with strong analytical skills! 🧠

        Your approach to problems:
        1. First, break down complex questions into component parts
        2. Clearly state your assumptions
        3. Develop a structured reasoning path
        4. Consider multiple perspectives
        5. Evaluate evidence and counter-arguments
        6. Draw well-justified conclusions

        When solving problems:
        - Use explicit step-by-step reasoning
        - Identify key variables and constraints
        - Explore alternative scenarios
        - Highlight areas of uncertainty
        - Explain your thought process clearly
        - Consider both short and long-term implications
        - Evaluate trade-offs explicitly

        For quantitative problems:
        - Show your calculations
        - Explain the significance of numbers
        - Consider confidence intervals when appropriate
        - Identify source data reliability

        For qualitative reasoning:
        - Assess how different factors interact
        - Consider psychological and social dynamics
        - Evaluate practical constraints
        - Address value considerations
        \
    """),
    add_datetime_to_context=True,
    stream_events=True,
    markdown=True,
)
```

该代理可用于提出引发深入分析的问题，例如：
```python
reasoning_agent.print_response(
    "A startup has $500,000 in funding and needs to decide between spending it on marketing or "
    "product development. They want to maximize growth and user acquisition within 12 months. "
    "What factors should they consider and how should they analyze this decision?",
    stream=True
)
```
或者，
```python
reasoning_agent.print_response(
    "Solve this logic puzzle: A man has to take a fox, a chicken, and a sack of grain across a river. "
    "The boat is only big enough for the man and one item. If left unattended together, the fox will "
    "eat the chicken, and the chicken will eat the grain. How can the man get everything across safely?",
    stream=True,
)
```

## 工作流工具

该WorkflowTools工具包使代理能够执行、分析和推理工作流操作。它与现有工作流集成Workflow，并提供了一种结构化的方法来运行工作流并评估其结果。
该工具包实现了“思考→运行→分析”的循环，使智能体能够：
1. 仔细思考问题，并规划工作流程输入和执行策略。
2. 使用适当的输入和参数执行工作流程
3. 分析结果，以确定结果是否足够，或者是否需要运行额外的工作流程。
4. 这种方法通过赋予代理规划、执行和评估工作流操作的工具，显著提高了代理成功执行复杂工作流的能力。

该工具包包含以下工具：
1. think：一个用于规划工作流程执行、集思广益和完善方案的草稿本。这些想法仅供代理内部使用，不会向用户显示。
2. 3. run_workflow：使用指定的输入和其他参数执行工作流。
analyze评估工作流执行结果是否正确和充分，确定是否需要进一步运行工作流。


### 例子
以下是如何使用该工具包的示例WorkflowTools：

```python
import asyncio
from textwrap import dedent

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.workflow import WorkflowTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

FEW_SHOT_EXAMPLES = dedent("""\
    You can refer to the examples below as guidance for how to use each tool.
    ### Examples
    #### Example: Blog Post Workflow
    User: Please create a blog post on the topic: AI Trends in 2024
    Run: input_data="AI trends in 2024", additional_data={"topic": "AI, AI agents, AI workflows", "style": "The blog post should be written in a style that is easy to understand and follow."}
    Final Answer: I've created a blog post on the topic: AI trends in 2024 through the workflow. The blog post shows...
    
    You HAVE TO USE additional_data to pass the topic and style to the workflow.
""")


# Define agents
web_agent = Agent(
    name="Web Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[DuckDuckGoTools()],
    role="Search the web for the latest news and trends",
)
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)

writer_agent = Agent(
    name="Writer Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="Write a blog post on the topic",
)


def prepare_input_for_web_search(step_input: StepInput) -> StepOutput:
    title = step_input.input
    topic = step_input.additional_data.get("topic")
    return StepOutput(
        content=dedent(f"""\
	I'm writing a blog post with the title: {title}
	<topic>
	{topic}
	</topic>
	Search the web for atleast 10 articles\
	""")
    )


def prepare_input_for_writer(step_input: StepInput) -> StepOutput:
    title = step_input.additional_data.get("title")
    topic = step_input.additional_data.get("topic")
    style = step_input.additional_data.get("style")

    research_team_output = step_input.previous_step_content

    return StepOutput(
        content=dedent(f"""\
	I'm writing a blog post with the title: {title}
	<required_style>
	{style}
	</required_style>
	<topic>
	{topic}
	</topic>
	Here is information from the web:
	<research_results>
	{research_team_output}
	<research_results>\
	""")
    )


# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)


content_creation_workflow = Workflow(
    name="Blog Post Workflow",
    description="Automated blog post creation from Hackernews and the web",
    db=SqliteDb(
        session_table="workflow_session",
        db_file="tmp/workflow.db",
    ),
    steps=[
        prepare_input_for_web_search,
        research_team,
        prepare_input_for_writer,
        writer_agent,
    ],
)

workflow_tools = WorkflowTools(
    workflow=content_creation_workflow,
    add_few_shot=True,
    few_shot_examples=FEW_SHOT_EXAMPLES,
    async_mode=True,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[workflow_tools],
    markdown=True,
)

asyncio.run(agent.aprint_response(
    "Create a blog post with the following title: Quantum Computing in 2025",
    instructions="When you run the workflow using the `run_workflow` tool, remember to pass `additional_data` as a dictionary of key-value pairs.",
    stream=True,
    debug_mode=True,
))
```

## 知识工具


该KnowledgeTools工具包使客服人员能够搜索、检索和分析知识库中的信息。它集成了Knowledge一套结构化的工作流程，用于在响应用户之前查找和评估相关信息。
该工具包实现了“思考→搜索→分析”的循环，使智能体能够：
1. 仔细思考问题并规划搜索查询
2. 在知识库中搜索相关信息
3. 分析结果，以确定其是否足够，或者是否需要进行其他搜索。
这种方法通过赋予智能体查找、评估和综合知识的工具，显著提高了智能体提供准确信息的能力。

该工具包包含以下工具：
1. think：一个用于规划、头脑风暴关键词和完善方案的草稿本。这些想法仅供智能体内部使用，不会向用户显示。
2. search：对知识库执行查询以检索相关文档。
3. analyze评估返回的文件是否正确和充分，确定是否需要进一步搜索。


### 例子
以下是如何使用该工具包的示例KnowledgeTools：

```python
from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.tools.knowledge import KnowledgeTools
from agno.vectordb.lancedb import LanceDb, SearchType

# Create a knowledge base containing information from a URL
agno_docs = Knowledge(
    # Use LanceDB as the vector database and store embeddings in the `agno_docs` table
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="agno_docs",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)
agno_docs.insert(
    url="https://docs.agno.com/llms-full.txt"
)

knowledge_tools = KnowledgeTools(
    knowledge=agno_docs,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=True,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[knowledge_tools],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response("How do I build multi-agent teams with Agno?", stream=True)
```

该工具包附带默认说明和少量示例，帮助代理有效使用这些工具。以下是配置方法：

```python
from agno.tools.knowledge import KnowledgeTools

knowledge_tools = KnowledgeTools(
    knowledge=my_knowledge_base,
    think=True,                # Enable the think tool
    search=True,               # Enable the search tool
    analyze=True,              # Enable the analyze tool
    add_instructions=True,     # Add default instructions
    add_few_shot=True,         # Add few-shot examples
    few_shot_examples=None,    # Optional custom few-shot examples
)
```


## 内存工具

该MemoryTools工具包使代理能够通过创建、更新和删除操作来管理用户记忆。该工具包与提供的数据库集成，记忆信息存储在该数据库中。

该工具包实现了一个“思考→操作→分析”的循环，使智能体能够：
- 仔细考虑内存管理需求并规划操作。
- 对数据库执行内存操作（添加、更新、删除）
- 分析结果，确保操作顺利完成并满足要求。
- 这种方法使代理能够跨对话持久地存储、检索和管理用户信息、偏好和上下文。


该工具包包含以下工具：
- think：一个用于规划记忆操作、集思广益构思内容和完善方案的草稿本。这些想法仅供智能体内部使用，不会向用户显示。
- get_memories从数据库中获取当前用户的内存列表。
- add_memory：在数据库中创建具有指定内容和可选主题的新记忆。
- update_memory：通过内存 ID 修改现有内存，允许更新内容和主题。
- delete_memory：根据内存 ID 从数据库中删除内存。
- analyze：评估内存操作是否成功完成并产生了预期结果。


### 例子
以下是如何使用该工具包的MemoryTools示例：
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.memory import MemoryTools

# Create a database connection
db = SqliteDb(
    db_file="tmp/memory.db"
)

memory_tools = MemoryTools(
    db=db,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[memory_tools],
    markdown=True,
)

agent.print_response(
    "My name is John Doe and I like to hike in the mountains on weekends. "
    "I like to travel to new places and experience different cultures. "
    "I am planning to travel to Africa in December. ",
    user_id="john_doe@example.com",
    stream=True
)

# This won't use the session history, but instead will use the memory tools to get the memories
agent.print_response("What have you remembered about me?", stream=True, user_id="john_doe@example.com")
```

以下是如何配置工具包的方法：
```python
from agno.tools.memory import MemoryTools

memory_tools = MemoryTools(
    db=my_database,
    enable_think=True,            # Enable the think tool (true by default)
    enable_get_memories=True,     # Enable the get_memories tool (true by default)
    enable_add_memory=True,       # Enable the add_memory tool (true by default)
    enable_update_memory=True,    # Enable the update_memory tool (true by default)
    enable_delete_memory=True,    # Enable the delete_memory tool (true by default)
    enable_analyze=True,          # Enable the analyze tool (true by default)
    add_instructions=True,        # Add default instructions
    instructions=None,            # Optional custom instructions
    add_few_shot=True,           # Add few-shot examples
    few_shot_examples=None,      # Optional custom few-shot examples
)
```

## 工具调用限制


限制客服人员可以进行的工具调用次数。

限制代理可以调用​​工具的次数有助于防止循环，并更好地控制成本和性能。

使用 Agno 实现这一点非常简单。您只需tool_call_limit在初始化代理或团队时传递参数即可。

例如：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools(company_news=True, cache_results=True)],
    tool_call_limit=1, # The Agent will not perform more than one tool call.
)

# The first tool call will be performed. The second one will fail gracefully.
agent.print_response(
    "Find me the current price of TSLA, then after that find me the latest news about Tesla.",
    stream=True,
)
```

## 包括和排除工具

工具包中是否包含特定工具。

Toolkit您可以使用 `include` 和 `exclude` 参数指定要从代理include_tools程序中包含或排除哪些工具exclude_tools。这对于限制代理程序可用的工具数量非常有用。

例如，以下是如何仅将某个get_latest_emails工具包含在GmailTools工具包中的方法：

```python
agent = Agent(
    tools=[GmailTools(include_tools=["get_latest_emails"])],
)
```
同样，以下是如何从GmailTools工具包中排除create_draft_email该工具的方法：

```python
agent = Agent(
    tools=[GmailTools(exclude_tools=["create_draft_email"])],
)

```

### 例子
以下示例展示了如何使用 ` include_toolsand`exclude_tools参数来限制代理可用的工具数量：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.calculator import CalculatorTools
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[
        CalculatorTools(
            exclude_tools=["exponentiate", "factorial", "is_prime", "square_root"],
        ),
        YFinanceTools(include_tools=["get_stock_price"]),
    ],
    markdown=True,
)

agent.print_response(
    "Get the stock price of AAPL and NVDA, then calculate the sum of both prices.",
)
```


### 工具结果缓存

缓存工具结果以减少重复的 API 调用并提高性能。

工具结果缓存旨在通过将函数调用结果存储在磁盘上来避免不必要的重复计算。这在开发和测试过程中非常有用，可以加快开发速度、避免速率限制并降低成本。



### 具包
将此参数传递cache_results=True给 Toolkit 构造函数，以启用该 Toolkit 的缓存功能。

```python
import asyncio

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools(cache_results=True), YFinanceTools(cache_results=True)],
)

asyncio.run(
    agent.aprint_response(
        "What is the current stock price of AAPL and top stories on HackerNews?",
        markdown=True,
    )
)
```

### 在 @tool 上
将此参数传递cache_results=True给@tool装饰器以启用该工具的缓存。

```python
from agno.tools import tool

@tool(cache_results=True)
def get_stock_price(ticker: str) -> str:
    """Get the current stock price of a given ticker"""

    # ... Long running operation

    return f"The current stock price of {ticker} is 100"

```

## 更新经纪人的工具

初始化后，在代理和团队上添加或更新工具。

工具可以在创建后添加到代理和团队中。这让你在初始化后灵活地向现有的代理或团队实例添加工具，这对动态工具管理或需要根据运行时需求有条件添加工具非常有用。通过使用set_tools通话，也可以更新所有可供代理或团队使用的工具。请注意，这会移除已分配给你的代理或团队的其他工具，并用提供的工具列表覆盖set_tools。

### 代理示例
例如get_weather，您可以创建自己的工具，然后调用add_tool该工具将其附加到您的代理。

```python
import random

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool


@tool(stop_after_tool_call=True)
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    # In a real implementation, this would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    random_weather = random.choice(weather_conditions)

    return f"The weather in {city} is {random_weather}."


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    markdown=True,
)

agent.print_response("What can you do?", stream=True)

agent.add_tool(get_weather)

agent.print_response("What is the weather in San Francisco?", stream=True)
```

### 团队示例

创建工具列表，并set_tools分配给你的团队

```python
import random

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team
from agno.tools import tool
from agno.tools.calculator import CalculatorTools


agent1 = Agent(
    name="Stock Searcher",
    model=OpenAIResponses(id="gpt-5.2"),
)

agent2 = Agent(
    name="Company Info Searcher",
    model=OpenAIResponses(id="gpt-5.2"),
)

team = Team(
    name="Stock Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[agent1, agent2],
    tools=[CalculatorTools()],
    markdown=True,
    show_members_responses=True,
)


@tool
def get_stock_price(stock_symbol: str) -> str:
    """Get the current stock price of a stock."""
    return f"The current stock price of {stock_symbol} is {random.randint(100, 1000)}."

@tool
def get_stock_availability(stock_symbol: str) -> str:
    """Get the current availability of a stock."""
    return f"The current stock available of {stock_symbol} is {random.randint(100, 1000)}."


team.set_tools([get_stock_price, get_stock_availability])

team.print_response("What is the current stock price of NVDA?", stream=True)
team.print_response("How much stock NVDA stock is available?", stream=True)
```

## 异常和重试

通过异常处理和自动重试来处理工具错误。

如果在工具调用后，您需要向模型提供反馈以改变其行为或退出工具调用循环，您可以引发以下异常之一：

- RetryAgentRun当您希望向模型提供指令，使其更改行为并重试工具调用时，请使用此异常。异常消息将作为工具调用错误传递给模型，从而允许模型在 LLM 循环的下一次迭代中重试或调整其方法。
- StopAgentRun当您想要退出模型执行循环并结束代理运行时，请使用此异常。当此异常由工具函数引发时，代理将退出工具调用循环，并且运行状态将被设置为“已完成” COMPLETED。此时为止的所有会话状态、消息、工具调用和工具结果都将存储在数据库中。


### 使用 RetryAgentRun
此示例展示了如何使用RetryAgentRun异常向模型提供反馈，使其能够调整自身行为：

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.exceptions import RetryAgentRun
from agno.models.openai import OpenAIResponses
from agno.utils.log import logger
from agno.run import RunContext


def add_item(run_context: RunContext, item: str) -> str:
    """Add an item to the shopping list."""
    if not run_context.session_state:
        run_context.session_state = {}
    
    if "shopping_list" not in run_context.session_state:
        run_context.session_state["shopping_list"] = []
    
    run_context.session_state["shopping_list"].append(item)
    len_shopping_list = len(run_context.session_state["shopping_list"])
    
    if len_shopping_list < 3:
        raise RetryAgentRun(
            f"Shopping list is: {run_context.session_state['shopping_list']}. Minimum 3 items in the shopping list. "
            + f"Add {3 - len_shopping_list} more items.",
        )
    
    logger.info(f"The shopping list is now: {run_context.session_state.get('shopping_list')}")
    return f"The shopping list is now: {run_context.session_state.get('shopping_list')}"


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    session_id="retry_example_session",
    db=SqliteDb(
        session_table="retry_example_session",
        db_file="tmp/retry_example.db",
    ),
    # Initialize the session state with empty shopping list
    session_state={"shopping_list": []},
    tools=[add_item],
    markdown=True,
)
agent.print_response("Add milk", stream=True)
print(f"Final session state: {agent.get_session_state(session_id='retry_example_session')}")

```

在这个例子中，当add_item调用函数时传入的元素少于 3 个，会抛出异常RetryAgentRun并附带说明。模型会将此视为工具调用错误，并可以add_item再次调用函数并传入更多元素以满足要求。

### 使用 StopAgentRun
此示例展示了如何使用StopAgentRun异常来退出工具调用循环：

```python
from agno.agent import Agent
from agno.exceptions import StopAgentRun
from agno.models.openai import OpenAIResponses
from agno.run import RunContext


def check_condition(run_context: RunContext, value: int) -> str:
    """Check a condition and stop tool calls if met."""
    if value > 100:
        raise StopAgentRun(
            f"Value {value} exceeds threshold. Stopping tool call execution."
        )
    return f"Value {value} is acceptable."


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[check_condition],
    markdown=True,
)

# When the model calls check_condition with value > 100,
# the tool call loop will exit and the run will complete
agent.print_response("Use the check_condition tool to check if 150 is acceptable", stream=True)
```

在这个例子中，当check_condition调用该函数时传入的值大于 100 时，会引发异常StopAgentRun。工具调用循环立即退出，运行以状态码结束COMPLETED。此时为止的所有会话状态、消息和工具调用都会存储在数据库中。

## 工具包索引


Agno支持的所有工具包索引。

工具包是一组可以添加到代理中的函数集合。工具包中的函数旨在协同工作、共享内部状态，并提供更佳的开发体验。

以下工具包可供使用。

### 搜索
![](https://i-blog.csdnimg.cn/direct/09a7f19f4f86482fafa35c0cba5c2972.png)

### 社会的
![](https://i-blog.csdnimg.cn/direct/578a6b6124f04f4fa4dcf922af3c786d.png)


### 网络爬虫
![](https://i-blog.csdnimg.cn/direct/898c8887487e4c3cbcf2423c5e7e7f73.png)

### 数据
![](https://i-blog.csdnimg.cn/direct/413d534d1c274f49b6fccafe4482b4cf.png)

### 本地
![](https://i-blog.csdnimg.cn/direct/a31b3138374b4eaebc67023cc735b83c.png)

### 原生模型工具包
![](https://i-blog.csdnimg.cn/direct/84f973ab84f442bb87097460d23e06ae.png)

### 其他工具包
![](https://i-blog.csdnimg.cn/direct/709da762bca540bf884e3ef4562c511b.png)


## PubMed
PubmedTools使代理能够搜索 Pubmed 中的文章。
​
以下代理将在 Pubmed 中搜索与“溃疡性结肠炎”相关的文章。

```python
from agno.agent import Agent
from agno.tools.pubmed import PubmedTools

agent = Agent(tools=[PubmedTools()])
agent.print_response("Tell me about ulcerative colitis.")
```

### 工具包参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| email | str | ＂your＿email＠example．com＂ | 指定要使用的电子邮件地址。 |
| max＿results | int | None | 可选参数，用于指定要返回的最大结果数。 |
| enable＿search＿pubmed | bool | True | 启用 PubMed 搜索功能。 |
| all | bool | False | 启用所有功能。 |

### 工具包功能

| 功能 | 描述 |
| :--- | :--- |
| search＿pubmed | 根据指定的查询条件在 PubMed 中搜索文章。参数包括 query 搜索词和 max＿results 要返回的最大结果数（默认值为 10）。返回包含搜索 <br> 结果的 JSON 字符串，其中包括发表日期、标题和摘要。 |