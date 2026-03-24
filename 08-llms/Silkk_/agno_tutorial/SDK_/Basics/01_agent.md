# Agent
## 什么是代理人？

智能体是利用工具完成任务的人工智能程序。

智能体是围绕无状态模型构建的有状态控制回路。模型在指令的引导下，循环进行推理并调用工具。根据需要添加内存、知识、存储、人机交互和防护机制。


## 建立单一代理
| 抽象                         | 它的作用                                                   |
| ----------------------------------- | -------------------------------------------------------------- |
| [**团队**](/teams/overview)         | 协同工作的代理人                                    |
| [**工作流程**](/workflows/overview) | 通过既定步骤协调代理人、团队和职能部门。 |


## 建立代理商
从简单的开始：一个模型、工具和说明书。

要构建有效的智能体，首先要从简单的入手：一个模型、一些工具和一些指令。一旦这些功能运行正常，就可以根据需要逐步添加更多功能。例如，以下是一个最简单的智能体，它拥有以下访问权限HackerNews：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)
agent.print_response("Trending startups and products.", stream=True)
```

## 运行您的代理
用于Agent.print_response()开发。它会将响应以易于阅读的格式打印到终端中。
用于生产、使用Agent.run()或Agent.arun()：

```python
from typing import Iterator
from agno.agent import Agent, RunOutputEvent, RunEvent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)

# Stream the response
stream: Iterator[RunOutputEvent] = agent.run("Trending products", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content)
```

## 动态代理配置
在 Agno 中，可调用工厂是动态运行时配置的一流模式。对于代理而言，可调用对象用于根据实时运行上下文构建工具和知识，而不是使用固定配置。
团队可以利用可调用工厂，根据用户或任务动态地组合成员、工具和知识。工具仅公开当前相关的信息，以确保执行的专注性。对于知识，可调用工厂会根据请求将检索路由到最佳来源，从而获得更精确、更及时的响应。

## 后续步骤
熟悉基本操作后，根据需要添加功能：
| 任务 | 指导 |
| :--- | :--- |
| 运行代理 | 运行代理 |
| 调试代理 | 调试代理 |
| 管理会话 | 代理会话 |
| 处理输入／输出 | 输入和输出 |
| 添加工具 | 工具 |
| 管理上下文 | 上下文工程 |
| 增加知识 | 知识 |
| 处理图像、音频、视频和文件 | 多模态 |
| 加装护栏 | 护栏 |
| 开发过程中缓存响应 | 响应缓存 |

## 运行代理
运行代理并处理其输出。

通过调用Agent.run()或Agent.arun()来运行您的代理。执行流程：
- 代理构建上下文以发送给模型（系统消息、用户消息、聊天记录、用户记忆、会话状态和其他相关输入）。
- 代理将此上下文发送给模型。
- 该模型会以消息或工具调用作为响应。
- 如果模型发出工具调用，代理将执行该调用并将结果返回给模型。
- 该模型处理更新后的上下文，重复此循环，直到生成最终消息而无需调用工具。
- 客服人员向来电者返回最终回复。

### 基本执行
Agent.run()返回一个RunOutput对象，或者在stream=True以下情况下返回一个RunOutputEvent对象流：

```python
from agno.agent import Agent, RunOutput
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)

# Run agent and return the response as a variable
response: RunOutput = agent.run("Trending startups and products.")

# Print the response in markdown format
pprint_run_response(response, markdown=True)
```

### 运行输入
参数input可以是字符串、列表、字典、消息、Pydantic 模型或消息列表：

```python
from agno.agent import Agent, RunOutput
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)

# Run agent with input="Trending startups and products."
response: RunOutput = agent.run(input="Trending startups and products.")
# Print the response in markdown format
pprint_run_response(response, markdown=True)
```

### 运行输出
Agent.run()非流式传输时返回RunOutput对象。核心属性：
- run_id：运行的 ID。
- agent_id代理人的ID。
- agent_name代理人的姓名。
- session_id会话 ID。
- user_id用户ID。
- content回复内容。
- content_type内容类型。对于结构化输出，这是 Pydantic 模型的类名。
- reasoning_content：推理内容。
- messages：发送给模型的消息列表。
- metrics：本次运行的各项指标。参见“指标”部分。
- model：运行中使用的模型。

请参阅[RunOutput](https://docs.agno.com/reference/agents/run-response) 参考文档以获取完整文档。

### 流媒体
设置stream=True为返回对象迭代器RunOutputEvent：

```python
from typing import Iterator
from agno.agent import Agent, RunOutputEvent, RunEvent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
)

# Run agent and return the response as a stream
stream: Iterator[RunOutputEvent] = agent.run("Trending products", stream=True)
for chunk in stream:
    if chunk.event == RunEvent.run_content:
        print(chunk.content)

```

默认情况下，只有RunContent事件（模型响应）会被流式传输。

要流式传输所有事件（工具调用、推理、内存更新等），设 stream_events=True：

```python
response_stream: Iterator[RunOutputEvent] = agent.run(
    "Trending products",
    stream=True,
    stream_events=True
)
```


### 事件类型
根据代理配置，由Agent.run()和产生的事件：Agent.arun()

#### 核心事件
| 事件类型 | 描述 |
| :--- | :--- |
| RunStarted | 表示运行的开始 |
| RunContent | 包含模型响应文本的各个部分 |
| RunContentCompleted | 表示内容流传输已完成 |
| RunIntermediateConten t | 包含模型的中间响应文本，以单独的文本块形式呈现。仅 output＿model 在设置时使用。 |
| RunCompleted | 表示运行成功完成 |
| RunError | 表示运行过程中发生错误 |
| RunCancelled | 表示运行已取消 |

#### 控制流事件
| 事件类型 | 描述 |
| :--- | :--- |
| RunPaused | 表示运行已暂停。 |
| RunContinued | 表示暂停的运行已恢复 |

#### 工具事件
| 事件类型 | 描述 |
| :--- | :--- |
|ToolCallStarted |	表示工具调用的开始|
|ToolCallCompleted	| 表示工具调用完成，包括工具调用结果|


#### 推理事件
| 事件类型 | 描述 |
| :--- | :--- |
| ReasoningStarted | 表示智能体推理过程的开始 |
| ReasoningStep | 包含推理过程中的一个步骤 |
| ReasoningCompleted | 表示推理过程已完成 |

#### 记忆事件
| 事件类型 | 描述 |
| :--- | :--- |
| MemoryUpdateStarted | 表示代理正在更新其内存 |
| MemoryUpdateCompleted | 表示内存更新已完成 |

#### 会议总结活动
| 事件类型 | 描述 |
| :--- | :--- |
| SessionSummaryStarted | 表示会话摘要生成开始 |
| SessionSummaryCompleted | 发出会话摘要生成完成的信号 |

#### 钩子前事件
| 事件类型 | 描述 |
| :--- | :--- |
| PreHookStarted | 表示预跑钩的开始 |
| PreHookCompleted | 表示预运行钩子执行完成 |

#### 钩子事件后
| 事件类型 | 描述 |
| :--- | :--- |
| PostHookStarted | 表示跑动后钩子的开始 |
| PostHookCompleted | 表示运行后钩子执行已完成 |


#### 解析器模型事件
| 事件类型 | 描述 |
| :--- | :--- |
| ParserModeLResponseStarted | 表示解析器模型响应的开始 |
| ParserModelResponseCompleted | 表示解析器模型响应已完成 |

#### 输出模型事件
| 事件类型 | 描述 |
| :--- | :--- |
| OutputModeLResponseStarted | 表示输出模型响应的开始 |
| OutputModelResponseCompleted | 输出模型响应完成的信号 |

#### 自定义事件
通过CustomEvent扩展创建自定义事件：

```python
from dataclasses import dataclass
from agno.run.agent import CustomEvent
from typing import Optional

@dataclass
class CustomerProfileEvent(CustomEvent):
    """CustomEvent for customer profile."""

    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
```

从您的工具中生成自定义事件：

```python
from agno.tools import tool

@tool()
async def get_customer_profile():
    """Example custom tool that simply yields a custom event."""

    yield CustomerProfileEvent(
        customer_name="John Doe",
        customer_email="john.doe@example.com",
        customer_phone="1234567890",
    )
```

### 指定运行用户和会话
传递此参数user_id以session_id将运行与特定用户和会话关联：
```python
agent.run("Tell me a 5 second short story about a robot", 
        user_id="john@example.com", 
        session_id="session_123")
```

### 传递图像/音频/视频/文件
通过images, audio, video, 或 files参数传递媒体：
```python
agent.run("Tell me a 5 second short story about this image", 
images=[Image(url="https://example.com/image.jpg")])
```
### 传递输出模式
传递结构化输出的输出模式：

```python
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class TVShow(BaseModel):
    title: str
    episodes: int

agent = Agent(model=OpenAIResponses(id="gpt-5.2"))
agent.run("Create a TV show", output_schema=TVShow)
```

### 暂停和继续跑步
对于人机交互流程，代理运行可以暂停。继续执行请使用Agent.continue_run().
更多详情请参见“人机交互” 。
​
### 取消跑步
Agent.cancel_run()使用.取消运行
请参阅“取消运行”了解更多详情。


## 调试代理

复制页面

检查执行流程、工具调用和中间步骤。

调试模式有助于了解执行流程和中间步骤：
- 在你的代理上设置debug_mode=True，让所有运行都启用它。
- 在运行方法上设置 debug_mode=True，以启用单次运行。
- 设置 AGNO_DEBUG=True 环境变量以全局启用调试模式。


```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic. Output only the report.",
    markdown=True,
    debug_mode=True,
    # debug_level=2, # Uncomment for more detailed logs
)

# Run agent and print response to the terminal
agent.print_response("Trending startups and products.")
```

### 交互式命令行界面
Agno 包含一个预构建的交互式 CLI，可将您的代理作为命令行应用程序运行。使用它来测试多轮对话：
```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,
    markdown=True,
)

# Run agent as an interactive CLI app
agent.cli_app(stream=True)
```

## 带有工具的代理

为您的代理提供与外部服务交互的工具。

赋予代理程序与外部服务交互的工具。代理程序使用这些工具HackerNewsTools来获取热门新闻和用户详情。

1. 创建一个 Python 文件

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[HackerNewsTools()],
    instructions="Write a report on the topic.",
    markdown=True,
)

agent.print_response("Trending AI startups on Hacker News", stream=True)
```

2. 设置虚拟环境

```python
uv venv --python 3.12
source .venv/bin/activate
```

3. 安装依赖项
```python
uv pip install -U agno anthropic
```

4. 导出您的 Anthropic API 密钥
```python
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

5. 运行代理

```python
python tools.py
```

## 具有结构化输出的代理

复制页面

获取 Pydantic 的格式化回复，而不是自由文本。

使用此output_schema功能可获得结构化、可信赖的响应。代理会返回 Pydantic 模型，而不是自由文本。

```python
from typing import List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from pydantic import BaseModel, Field


class StockAnalysis(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    current_price: float = Field(..., description="Current price in USD")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")
    summary: str = Field(..., description="One-line summary")
    key_drivers: List[str] = Field(..., description="2-3 key growth drivers")
    key_risks: List[str] = Field(..., description="2-3 key risks")


agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    output_schema=StockAnalysis,
)

response = agent.run("Analyze NVIDIA stock")

# Access typed data directly
analysis: StockAnalysis = response.content
print(f"{analysis.company_name} ({analysis.ticker})")
print(f"Price: ${analysis.current_price}")
print(f"P/E Ratio: {analysis.pe_ratio or 'N/A'}")
print(f"Summary: {analysis.summary}")
print("Key Drivers:")
for driver in analysis.key_drivers:
    print(f"  - {driver}")
print("Key Risks:")
for risk in analysis.key_risks:
    print(f"  - {risk}")
```

### 带存储的代理

复制页面

跨运行保留对话历史记录。

存储功能让您的代理能够记住对话。有了它session_id，即使重新启动，它也能从上次中断的地方继续对话。

```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

db = SqliteDb(db_file="tmp/agents.db")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    db=db,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
)

session_id = "finance-session"

# Turn 1: Analyze a stock
agent.print_response(
    "Give me a quick analysis of NVIDIA",
    session_id=session_id,
    stream=True,
)

# Turn 2: The agent remembers NVDA from turn 1
agent.print_response(
    "Compare that to AMD",
    session_id=session_id,
    stream=True,
)

# Turn 3: Ask based on full conversation
agent.print_response(
    "Which looks like the better investment?",
    session_id=session_id,
    stream=True,
)
```

### 具有记忆的代理
存储用户偏好设置，使其在对话中保持不变。

内存功能使您的代理能够记住用户在对话过程中的各种信息。与存储（用于保存对话历史记录）不同，内存存储用户级别的信息，例如偏好和上下文。


```python
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools
from rich.pretty import pprint

db = SqliteDb(db_file="tmp/agents.db")

memory_manager = MemoryManager(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    db=db,
    memory_manager=memory_manager,
    enable_agentic_memory=True,
    markdown=True,
)

user_id = "investor@example.com"

# Tell the agent about yourself
agent.print_response(
    "I'm interested in AI and semiconductor stocks. My risk tolerance is moderate.",
    user_id=user_id,
    stream=True,
)

# The agent now knows your preferences
agent.print_response(
    "What stocks would you recommend for me?",
    user_id=user_id,
    stream=True,
)

# View stored memories
memories = agent.get_user_memories(user_id=user_id)
print("\nStored Memories:")
pprint(memories)
```

### 工作原理
- 知识库：文档被分块、嵌入并存储在矢量数据库中。
- 搜索：代理使用混合搜索（语义搜索+关键词搜索）来搜索知识库。
- 上下文：在生成响应之前，会将相关数据块添加到上下文中。


### 添加不同类型的内容
```python
# From a URL
knowledge.insert(url="https://example.com/document.pdf")

# From a local file
knowledge.insert(path="./documents/guide.pdf")

# From text
knowledge.insert(text="Your content here...")
```


