# TEAM
## 什么是团队？

协同解决复杂任务的智能体群体。

团队是由若干代理人（或子团队）组成的协作团队。团队领导根据成员的角色分配任务。
<img src="https://i-blog.csdnimg.cn/direct/16581df0b37340d49a30085b25a4af4b.png)" width="500" />


```python
from agno.team import Team
from agno.agent import Agent

team = Team(members=[
    Agent(name="English Agent", role="You answer questions in English"),
    Agent(name="Chinese Agent", role="You answer questions in Chinese"),
    Team(
        name="Germanic Team",
        role="You coordinate the team members to answer questions in German and Dutch",
        members=[
            Agent(name="German Agent", role="You answer questions in German"),
            Agent(name="Dutch Agent", role="You answer questions in Dutch"),
        ],
    ),
])
```

### 为什么选择团队？
单个代理很快就会达到极限。上下文窗口会被填满，决策变得混乱，调试也变得不可能。
团队将工作分配给各个专业人员：

| 益处 | 描述 |
| :--- | :--- |
| 专业化 | 每个特工都精通一个领域，而不是样样平庸。 |
| 并行处理 | 多个智能体同时处理独立的子任务 |
| 可维护性 | 当出现故障时，您确切地知道该由哪个代理人来修理。 |
| 可扩展性 | 通过添加代理来增加功能，而不是重写所有内容。 |

权衡之下：协调开销。代理需要通信和共享状态。如果这方面出错，就会造成代价更高的故障模式。

### 何时使用团队
在以下情况下需要使用团队：
- 一项任务需要多个拥有不同工具或专业知识的专业人员。
- 单个代理的上下文窗口超出限制
- 你希望每个代理人专注于狭窄的范围。

在以下情况下使用单个代理：
- 这项任务属于某一专业领域。
- 降低代币成本至关重要
- 你还不确定（先从简单的开始，达到上限后再添加代理）
​
### 团队能力
​
#### 模块化执行
Agno代理和团队采用模块化设计。消息构建、会话处理、存储和后台管理器被分离到专用组件中，从而将协调逻辑与API分离。
​
#### 可调用工厂
可调用工厂允许您为代理、团队成员、工具或知识库提供可调用对象。Agno 在运行时通过访问运行上下文来解析这些可调用对象，并将其返回值用于该次运行。这支持按运行或按会话进行配置、延迟设置和动态团队组成。

| 行为 | 细节 |
| :--- | :--- |
| 注入参数 | team ，，当出现在工厂签名中 run＿context 时 session＿state |
| 缓存 | cache＿callables＝True 然后，按自定义键缓存，user＿id 然后 session＿id ，请参阅缓存设置 |
| 返回类型 | 工具和成员返回列表或元组。知识返回一个 KnowledgeProtocol 实例。 |
| 异步工厂 | 异步工厂需要 arun（）或 aprint＿response（） |


`Team.members`可以是一个返回成员（代理/团队）列表的可调用工厂。`Team.tools` 和 `Team.knowledge` 也支持可调用工厂。Team还支持可调用的缓存设置，如callable_members_cache_key、tool/knowledge缓存键。

### 团队模式

Team 2.0引入了TeamMode，使协作风格变得明确。优先选择模式=，而不是直接切换respond_directly或delegate_to_all_members。

```python
from agno.team import Team, TeamMode

team = Team(
    name="Research Team",
    members=[...],
    mode=TeamMode.broadcast,
)
```

模式选择控制领导者如何与成员协作。模式覆盖遗留标志。如果没有设置模式，respond_directly=True映射到TeamMode.route，delegate_to_all_members=True映射到TeamMode.broadcast，否则为TeamMode.coordinate。

| 模式 | 配置 |  | 用例 |
| :--- | :--- | :--- | :--- |
| 协调 | mode＝TeamMode．coordinate（默认） |  | 分解工作，委派给成员，综合结果 |
| 路线 | mode＝TeamMode．route |  | 直接联系一位专家并回复其意见。 |
| 播送 | mode＝TeamMode．broadcast |  | 将同一任务分配给所有成员并进行综合分析。 |
| 任务 | mode＝TeamMode．tasks |  | 运行任务列表循环，直到目标完成。 |


模式是明确的编排模式。使用模式可以在不改变代理逻辑的情况下命名和交换协调拓扑。团队运行可以根据需要暂停，并在获得批准后继续。

## 团队建设

定义多智能体协调的团队成员、角色和结构。

先从简单的开始：一个模型、团队成员和操作说明。根据需要添加功能

### 最小示例
```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

news_agent = Agent(
    name="News Agent",
    role="Get trending tech news from HackerNews",
    tools=[HackerNewsTools()]
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get stock prices and financial data",
    tools=[YFinanceTools()]
)

team = Team(
    name="Research Team",
    members=[news_agent, finance_agent],
    model=OpenAIResponses(id="gpt-4o"),
    instructions="Delegate to the appropriate agent based on the request."
)

team.print_response("What are the trending AI stories and how is NVDA stock doing?", stream=True)
```

### 团队模式
团队默认采用协调模式（领导者委派任务并进行综合）。设置此项mode可更改领导者与成员的协作方式。

```python
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses

team = Team(
    name="Language Router",
    members=[...],
    model=OpenAIResponses(id="gpt-4o"),
    mode=TeamMode.route
)
```

| 模式 | 配置 |  | 用例 |
| :--- | :--- | :--- | :--- |
| 协调 | mode＝TeamMode．coordinate | （默认） | 分解工作，委派给成员，综合结果 |
| 路线 | mode＝TeamMode．route |  | 直接联系一位专家并回复其意见。 |
| 播送 | mode＝TeamMode．broadcast |  | 将同一任务分配给所有成员并进行综合分析。 |
| 任务 | mode＝TeamMode．tasks |  | 运行任务列表循环，直到目标完成。 |

任务模式运行一个迭代任务循环。用于max_iterations限制领导者可以运行的循环次数。
```python
from agno.models.openai import OpenAIResponses

team = Team(
    name="Ops Team",
    members=[...],
    model=OpenAIResponses(id="gpt-4o"),
    mode=TeamMode.tasks,
    max_iterations=6
)
```

### 团队成员
每个成员都应该有一个职责name和责任role。团队领导会根据这些职责来决定谁负责什么。
```python
news_agent = Agent(
    name="News Agent",                              # Identifies the agent
    role="Get trending tech news from HackerNews",  # Tells the leader what this agent does
    tools=[HackerNewsTools()]
)
```

为了更好地追踪，还可以设置一个id：

```python
news_agent = Agent(
    id="news-agent",
    name="News Agent",
    role="Get trending tech news from HackerNews",
    tools=[HackerNewsTools()]
)
```

当成员上同时设置了 id 和 name 时，团队委派使用 id 作为成员标识符。

### 嵌套团队
团队可以包含其他团队。最高领导将任务委派给子团队领导，子团队领导再将任务委派给各自的成员。

```python
from agno.team import Team
from agno.agent import Agent

team = Team(
    name="Language Team",
    members=[
        Agent(name="English Agent", role="Answer in English"),
        Agent(name="Chinese Agent", role="Answer in Chinese"),
        Team(
            name="Germanic Team",
            role="Handle German and Dutch questions",
            members=[
                Agent(name="German Agent", role="Answer in German"),
                Agent(name="Dutch Agent", role="Answer in Dutch"),
            ],
        ),
    ],
)
```

### 模型继承
如果未明确设置，团队成员将继承model其父团队的属性。

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.models.anthropic import Claude

# This agent uses its own model (Claude)
agent_with_model = Agent(
    name="Claude Agent",
    model=Claude(id="claude-sonnet-4-5"),
    role="Research with Claude"
)

# This agent inherits gpt-4o from the team
agent_without_model = Agent(
    name="Inherited Agent",
    role="Research with inherited model"
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-4o"),  # Default for team and members without a model
    members=[agent_with_model, agent_without_model]
)
```

### 可调用工厂

利用可调用工厂构建动态可配置的团队。

Agno 框架支持团队配置的可调用工厂模式，就像它支持单个代理的配置一样。这允许您根据特定的会话或用户上下文，在运行时动态解析团队成员、工具和知识。

您可以将函数传递给团队的members, tools或knowledge参数。 该框架使用签名检查来注入团队实例和 run_context。

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIChat

def get_team_members(team: Team, run_context: any):
    """
    Dynamic member factory: 
    Only include the 'Analyst' if the user is a premium user.
    """
    members = [Agent(name="Generalist", model=OpenAIChat(id="gpt-5"))]
    
    # Logic based on session/user metadata
    if run_context.user_id == "premium_user_123":
        members.append(Agent(name="Analyst", role="High-level data insights"))
        
    return members

# Initialize team with the factory function
agent_team = Team(
    name="Dynamic Team",
    members=get_team_members,  # Callable factory
    show_tool_calls=True,
    cache_callables=True       # Optional: Cache the result for the session
)
```

### 可调用缓存设置
可调用工厂可以按用户或会话进行缓存，也可以按自定义键进行缓存。
| 环境 | 目的 |
| :--- | :--- |
| cache＿callables | 启用或禁用可调用工厂的缓存 |
| callable＿tools＿cache＿key | 工具工厂的自定义缓存键 |
| callable＿knowledge＿cache＿key | 知识工厂的自定义缓存键 |
| callable＿members＿cache＿key | 成员工厂的自定义缓存键 |

当您需要强制重新解析时，请清除缓存结果：

```python
team.clear_callable_cache(kind="tools")
```
`aclear_callable_cache()在异步代码中使用。`

### 团队特色
团队支持与代理相同的功能：

| 特征 | 描述 |  |  |
| :--- | :--- | :--- | :--- |
| 指示 | 指导团队领导如何协调 |  |  |
| 模式 | 选择协调策略（ coordinate ，broadcast，tasks，route ） |  |   |
| 数据库 | 持久化会话历史记录和状态 |  |  |
| 推理 | 让领导者在授权前做好计划。 |  |  |
| 知识 | 赋予领导者访问知识库的权限 |  |  |
| 记忆 | 跨会话存储和调用信息 |  |  |
| 工具 | 赋予领导者直接使用的工具。 |  |  |

请参阅以下指南以添加这些功能。


### 后续步骤

| 任务 | 指导 |
| :--- | :--- |
| 运行团队 | 跑步队 |
| 控制权委托 | 代表团 |
| 添加聊天记录 | 聊天记录 |
| 管理会话 | 会议 |
| 处理输入／输出 | 输入和输出 |
| 增加知识 | 知识 |
| 加装护栏 | 护栏 |

## Running Teams
使用 Team.run() 执行团队并处理其输出。

使用Team.run()（同步）或Team.arun()（异步）方式管理您的团队。


### 基本执行
```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response

news_agent = Agent(name="News Agent", role="Get tech news", tools=[HackerNewsTools()])
finance_agent = Agent(name="Finance Agent", role="Get stock data", tools=[YFinanceTools()])

team = Team(
    name="Research Team",
    members=[news_agent, finance_agent],
    model=OpenAIResponses(id="gpt-4o")
)

# Run and get response
response = team.run("What are the trending AI stories?")
print(response.content)

# Run with streaming
stream = team.run("What are the trending AI stories?", stream=True)
for chunk in stream:
    print(chunk.content, end="", flush=True)
```


### 执行流程
当你打电话时run()：
- 预钩子执行（如果已配置）
- 推理运行（如果已启用）用于规划任务
- 上下文由系统消息、历史记录、记忆和会话状态构建而成。
- 模型决定是直接回应、使用工具还是委托给成员。
- 成员们执行各自的任务（以异步模式并发执行）
- 领导者将成员的成果综合起来，形成最终回应。
- 后置钩子执行（如果已配置）
- 会话和指标将被存储（如果已配置数据库）。
- 可调用工厂在会话状态加载后解析，因此工厂可以访问run_context和session_state。异步工厂需要arun()或aprint_response()。

| 模式 | 执行风格 |
| :--- | :--- |
| coordinate | 领导者分解工作，委派任务给成员，并综合分析结果。 |
| route | 领导者将路由指向一个成员并返回该成员的响应 |
| broadcast | 领导者将同一任务分配给所有成员，然后进行综合分析。 |
| tasks | 领导者运行任务列表循环，直到目标完成。 |

在TeamMode.tasks，领导者使用任务管理工具来构建和执行共享任务列表，循环执行直到目标完成或max_iterations达到为止。
团队可以暂停运行以满足人为因素（例如，审批或用户输入）。当运行需要确认时，运行会返回并带有待处理的需求，以便您在继续之前收集输入或解决审批问题。暂停的运行会返回并status=RunStatus.paused继续requirements运行TeamRunOutput。人工监督是一种控制途径。运行可以暂停以进行确认或外部执行，并在需求解决后恢复运行。

### 流媒体
启用流式传输stream=True。这将返回一个事件迭代器，而不是单个响应。

```python
stream = team.run("What are the top AI stories?", stream=True)
for chunk in stream:
    print(chunk.content, end="", flush=True)
```

### 所有事件
默认情况下，仅传输内容。stream_events=True要获取工具调用、推理步骤和其他内部事件，请进行相应设置：

```python
stream = team.run(
    "What are the trending AI stories?",
    stream=True,
    stream_events=True
)

for event in stream:
    if event.event == TeamRunEvent.run_content:
        print(event.content, end="", flush=True)
    elif event.event == TeamRunEvent.run_paused:
        print("Run paused")
    elif event.event == TeamRunEvent.run_continued:
        print("Run continued")
    elif event.event == TeamRunEvent.tool_call_started:
        print(f"Tool call started")
    elif event.event == TeamRunEvent.tool_call_completed:
        print(f"Tool call completed")
```

### 流式成员事件

当使用 arun（） 与多个成员时，它们会同时执行。会员活动按时发生，不是按顺序排列的。

禁用成员事件流，stream_member_events=False：

```python
team = Team(
    name="Research Team",
    members=[news_agent, finance_agent],
    model=OpenAIResponses(id="gpt-4o"),
    stream_member_events=False
)
```

### 运行输出
Team.run()返回一个TeamRunOutput包含以下内容的对象：

| 场地 | 描述 |
| :--- | :--- |
| content | 最终回复文本 |
| messages | 发送给模型的所有消息 |
| metrics | 令牌使用情况、执行时间等。 |
| member＿responses | 受委托成员的回应 |


请参阅TeamRunOutput 参考文档以获取完整架构。

### 异步执行
用于arun()异步执行。当领导者同时将任务委托给多个成员时，成员将并发运行。

```python
import asyncio

async def main():
    response = await team.arun("Research AI trends and stock performance")
    print(response.content)

asyncio.run(main())
```

### 异步流媒体

用于arun()异步执行。当领导者同时将任务委托给多个成员时，成员将并发运行。
```python
import asyncio

async def main():
    response = await team.arun("Research AI trends and stock performance")
    print(response.content)

asyncio.run(main())
```

### 任务模式
任务模式运行一个迭代循环，该循环创建、执行和更新任务，直到目标完成。
```python
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses

team = Team(
    name="Ops Team",
    members=[news_agent, finance_agent],
    model=OpenAIResponses(id="gpt-4o"),
    mode=TeamMode.tasks,
    max_iterations=6
)

response = team.run("Compile a short report on recent AI agent frameworks.")
print(response.content)
```

### 指定用户和会话
关联程序会与用户和会话一起运行，以进行历史记录跟踪：

```python
team.run(
    "Get my monthly report",
    user_id="john@example.com",
    session_id="session_123"
)
```

### 传递文件
向团队传递图片、音频、视频或文件：
```python
from agno.media import Image

team.run(
    "Analyze this image",
    images=[Image(url="https://example.com/image.jpg")]
)
```

### 传递文件
向团队传递图片、音频、视频或文件：
```python
from agno.media import Image

team.run(
    "Analyze this image",
    images=[Image(url="https://example.com/image.jpg")]
)
```

### 结构化输出
传递输出模式以获取结构化响应：
```python
from pydantic import BaseModel

class Report(BaseModel):
    overview: str
    findings: list[str]

response = team.run("Analyze the market", output_schema=Report)
```
详情请参见“输入与输出”部分。
​
### 取消运行
取消跑步队伍`Team.cancel_run()`。请参阅“跑步取消”。
​
### 打印回复
用于开发，print_response()显示格式化输出：
```python
team.print_response("What are the top AI stories?", stream=True)

# Show member responses too
team.print_response("What are the top AI stories?", show_members_responses=True)

```

## 代表团
控制团队领导如何向团队成员分配任务。

当你run()向团队提出请求时，领导者决定如何处理该请求：直接回应、使用工具或委派给团队成员。

![](https://i-blog.csdnimg.cn/direct/b045baa3c9364dd1b47e3a2d0d9735ae.png)

默认流程：
- 团队收到用户输入
- 领导者分析意见，并决定将任务委派给哪些成员。
- 领导者为每个选定的成员制定任务。
- 成员执行并返回结果（以异步模式并发执行）
- 领导者综合分析结果，形成最终回应

模式定义了领导者是进行委托、路由到单个成员、广播到所有成员还是运行任务循环。模式是显式的编排模式，您可以在不更改成员逻辑的情况下进行切换。成员也可以由可调用工厂提供并在运行时解析。请参阅可调用工厂。

您可以使用团队模式自定义此流程：

| 模式 | 配置 |  | 行为 |
| :--- | :--- | :--- | :--- |
| 坐标（默认） | mode＝TeamMode．coordinate（或省略 mode ） |  | 领导者挑选成员、制定任务、总结结果 |
| 路线 | mode＝TeamMode．route |  | 领导者直接将路由发送给一名成员并返回其回复。 |
| 播送 | mode＝TeamMode．broadcast |  | 领导者同时将同一任务委派给所有成员。 |
| 任务 | mode＝TeamMode．tasks |  | 领导者制定并执行共享任务清单，直至目标完成。 |


用agno.team.mode里的TeamMode来明确设置模式。旧标志依然有效，但mode是推荐的做法。

成员选择和运行跟踪使用成员ID。为成员设置明确的id值以实现稳定委托身份。

### 坐标模式（默认）
领导者掌控一切：使用哪些成员，给他们分配什么任务，以及如何整合他们的成果。
```python
from agno.team import Team
from agno.agent import Agent
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-4o"),
    members=[
        Agent(name="News Agent", role="Get tech news", tools=[HackerNewsTools()]),
        Agent(name="Finance Agent", role="Get stock data", tools=[YFinanceTools()])
    ],
    mode=TeamMode.coordinate,
    instructions="Research the topic thoroughly, then synthesize findings into a clear report."
)

team.print_response("What's happening with AI companies and their stock prices?")
```

适用情况：
- 任务需要分解成子任务。
- 您希望对最终输出进行质量控制
- 领导者应该为成员的发言添加背景信息或理由。


### 路线模式
领导者选择由哪个成员处理请求，并直接返回该成员的响应。默认情况下，领导者仍然可以创建任务；设置`determine_input_for_members=False`为直接传递用户输入，不做任何修改。


```python
from agno.team import Team
from agno.agent import Agent
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses

team = Team(
    name="Language Router",
    model=OpenAIResponses(id="gpt-4o"),
    members=[
        Agent(name="English Agent", role="Answer questions in English"),
        Agent(name="Japanese Agent", role="Answer questions in Japanese"),
    ],
    mode=TeamMode.route,
    determine_input_for_members=False # Pass user input unchanged to member
)

team.print_response("How are you?")        # Routes to English Agent
team.print_response("お元気ですか?")        # Routes to Japanese Agent
```

适用情况：

- 您拥有专业代理，并希望实现自动路由。
- 领导者不应修改请求或回复。
- 你想要更低的延迟（无需合成步骤）

### 广播模式
领导者一次性将任务分配给所有成员。这有助于收集多方观点或进行并行研究
![](https://i-blog.csdnimg.cn/direct/7b9cbb6335f342ca81258ba05b8fb060.png)

```python
import asyncio
from agno.team import Team
from agno.agent import Agent
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools
from agno.tools.arxiv import ArxivTools
from agno.tools.duckduckgo import DuckDuckGoTools

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-4o"),
    members=[
        Agent(name="HackerNews Researcher", role="Find discussions on HackerNews", tools=[HackerNewsTools()]),
        Agent(name="Academic Researcher", role="Find academic papers", tools=[ArxivTools()]),
        Agent(name="Web Researcher", role="Search the web", tools=[DuckDuckGoTools()]),
    ],
    mode=TeamMode.broadcast,
    instructions="Synthesize findings from all researchers into a comprehensive report."
)

# Use async for concurrent execution
asyncio.run(team.aprint_response("Research the current state of AI agents"))
```

适用情况：
- 你想了解同一主题的多个观点
- 成员可以独立工作
- 并行执行可降低延迟

### 任务模式
任务模式是一个自主循环，领导者将目标分解为任务，执行任务，并将目标标记为已完成。

```python
from agno.team import Team
from agno.agent import Agent
from agno.team.mode import TeamMode
from agno.models.openai import OpenAIResponses

team = Team(
    name="Ops Team",
    model=OpenAIResponses(id="gpt-4o"),
    members=[
        Agent(name="Research Agent", role="Collect findings"),
        Agent(name="Writer Agent", role="Draft the final report"),
    ],
    mode=TeamMode.tasks,
    max_iterations=6
)

team.print_response("Compile a short report on recent AI agent frameworks.")
```

### 结构化输入
使用时determine_input_for_members=False，您可以将结构化的 Pydantic 模型直接传递给成员：

```python
from pydantic import BaseModel, Field
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

class ResearchRequest(BaseModel):
    topic: str
    num_sources: int = Field(default=5)

research_agent = Agent(
    name="Research Agent",
    role="Research topics on HackerNews",
    tools=[HackerNewsTools()]
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-4o"),
    members=[research_agent],
    determine_input_for_members=False  # Pass input directly to member
)

request = ResearchRequest(topic="AI Agents", num_sources=10)
team.print_response(input=request)
```

### 生产方面的考虑
​
代币成本
每种模式的令牌开销都不同：

| 模式 | 协调成本 | 何时使用 |
| :--- | :--- | :--- |
| 协调 | 高（分解 + 合成） | 质量比成本更重要 |
| 路线 | 低（仅限部分） | 简单的路由，对成本敏感 |
| 播送 | 中等（仅限合成） | 平行研究，多视角 |
| 任务 | 高（计划 + 迭代循环） | 具有依赖关系的多步骤目标 |

#### 延迟
- 协调方式：顺序式（领导者思考→成员执行→领导者综合）
- 路由：快速（领导者选择 → 成员执行）
- 异步广播：成员并行执行，但合成会增加延迟
- 任务：迭代式；多次循环直至任务完成或`max_iterations`达到目标。


#### 错误处理
如果成员出现故障会发生什么？
- 协调：领导者可以接受其他成员提供的部分成果。
- 路由：失败直接返回给调用者
- 广播：领导者综合现有结果，并可能指出缺失数据。
- 任务：任务列表跟踪失败/阻塞的任务；负责人可以重试或重新分配任务，直到完成。


## 调试团队

使用调试模式、跟踪和常见故障模式来排查和检查团队行为。

团队会增加协调的复杂性。一旦出现问题，你需要追溯领导者和所有成员的执行过程。
​
### 调试模式
启用调试模式以查看发送到模型的消息、工具调用、委托模式和指标。

```python
from agno.team import Team
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

news_agent = Agent(name="News Agent", role="Get the latest news")
weather_agent = Agent(name="Weather Agent", role="Get weather forecasts")

team = Team(
    name="Research Team",
    members=[news_agent, weather_agent],
    model=OpenAIResponses(id="gpt-4o"),
    debug_mode=True
)

team.print_response("What is the weather in Tokyo?", show_members_responses=True)
```
启用调试模式的三种方法：

| 方法 | 范围 |
| :--- | :--- |
| debug＿mode＝True 在团队中 | 该队所有跑动 |
| debug＿mode＝True 在 run（） | 单次运行 |
| AGNO＿DEBUG＝True 环境变量 | 全球所有团队 |

设置`debug_level=2`查看更详细的日志：

```python

team = Team(
    name="Research Team",
    members=[news_agent, weather_agent],
    model=OpenAIResponses(id="gpt-4o"),
    debug_mode=True,
    debug_level=2
)
```

### 需要注意什么
调试时，请检查：
```python

| 问题 | 检查什么 |
| :--- | :--- |
| 选错了成员 | 领导者的推理，成员的角色 |
| 成员未回复 | 成员工具调用，错误 |
| 执行缓慢 | 令牌计数，顺序执行与并行执行 |
| 意外输出 | 领导者综合步骤，成员回应 |
| 高代币使用率 | 协调开销，上下文大小 |
```

### 常见故障模式
​
领导委派给错误的成员
领导者根据成员的特质挑选成员role。如果授权不当：
- 检查角色描述是否清晰地说明了每个成员的职责。
- 明确角色分工（避免重叠）
- 向团队负责人添加指示

```python
# Bad: Roles are vague
agent1 = Agent(name="Agent 1", role="Research things")
agent2 = Agent(name="Agent 2", role="Look stuff up")

# Good: Roles are specific and distinct
news_agent = Agent(name="News Agent", role="Get tech news from HackerNews")
finance_agent = Agent(name="Finance Agent", role="Get stock prices from Yahoo Finance")
```

### 常见故障模式
​
#### 领导委派给错误的成员
领导者根据成员的特质挑选成员role。如果授权不当：
- 检查角色描述是否清晰地说明了每个成员的职责。
- 明确角色分工（避免重叠）
- 向团队负责人添加指示

```python
# Bad: Roles are vague
agent1 = Agent(name="Agent 1", role="Research things")
agent2 = Agent(name="Agent 2", role="Look stuff up")

# Good: Roles are specific and distinct
news_agent = Agent(name="News Agent", role="Get tech news from HackerNews")
finance_agent = Agent(name="Finance Agent", role="Get stock prices from Yahoo Finance")
```

### 成员默默失败
如果某个成员失败，领导者可以不使用该成员的输出来合成响应。启用此功能可show_members_responses=True查看每个成员的返回结果：

```python
team.print_response("Research AI trends", show_members_responses=True)
```

### 无限委托循环
领导者不断下达任务，却始终没有给出最终答复。这通常意味着：
- 说明书没有明确说明何时停止。
- 成员返回的结果不完整
- 在指令中添加明确的停止条件

```python
team = Team(
    name="Research Team",
    members=[...],
    instructions=[
        "Delegate to members to gather information.",
        "Once you have enough information, synthesize and respond directly.",
        "Do not delegate more than 3 times per request."
    ]
)
```

### 高代币使用率
多智能体协作会消耗代币。一个四人团队消耗的代币数量很容易达到单个智能体消耗量的十倍。
检查调试输出或指标中的令牌使用情况：

```python
response = team.run("Research AI trends")
print(f"Total tokens: {response.metrics.total_tokens}")
```


#### 减少代币：
- 减少成员数量
- 成员回复应简洁明了。
- 使用mode=TeamMode.route（或respond_directly=True）跳过合成

### 交互式命令行界面
使用内置命令行界面测试多轮对话：
```python
from agno.team import Team
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses

news_agent = Agent(name="News Agent", role="Get the latest news")
weather_agent = Agent(name="Weather Agent", role="Get weather forecasts")

team = Team(
    name="Research Team",
    members=[news_agent, weather_agent],
    model=OpenAIResponses(id="gpt-4o"),
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3
)

team.cli_app(stream=True)
```

用于`await team.acli_app()`异步操作。

### 使用 AgentOS 进行追踪
为了进行生产环境调试，请将您的团队连接到AgentOS以获取：
- 所有委托和工具调用的可视化跟踪
- 成员代币使用情况细分
- 会话历史记录和回放
- 错误跟踪

## 基础团队
一组人工智能代理协同工作，共同研究课题。

一个由两名专业特工组成的基础团队：
- HackerNews 研究员- 获取来自 HackerNews 的热门新闻
- 财务代理- 获取股票价格和财务数据

团队负责人根据用户请求，将任务委派给合适的代理人进行协调。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets trending stories from HackerNews.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets stock prices and financial data.",
    tools=[YFinanceTools()],
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[hn_researcher, finance_agent],
    instructions=[
        "Delegate to the HackerNews Researcher for tech news and trends.",
        "Delegate to the Finance Agent for stock prices and financial data.",
        "Synthesize the results into a clear summary.",
    ],
    markdown=True,
    show_members_responses=True,
)

team.print_response(
    input="What are the top AI stories on HackerNews and how is NVDA doing?",
    stream=True
)
```

### 团队流

实时流式传输团队的回复。

实时流式传输团队的响应，以便stream=True在团队工作时实现实时输出。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

news_agent = Agent(
    name="News Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets trending tech stories from HackerNews.",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gets stock prices and financial data.",
    tools=[YFinanceTools()],
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    markdown=True,
    show_members_responses=True,
)

# Stream the response
team.print_response(
    "What are the trending AI stories and how is NVDA stock doing?",
    stream=True,
)
```


### 直接响应模式

将请求转交给直接响应的专业人员。

用于`mode=TeamMode.route`将请求路由到相应的代理并直接返回成员响应。旧版respond_directly=True标志仍然有效，但mode建议使用新版。

此示例创建了一个包含三个代理的语言路由器：
- 英语客服人员- 用英语回复
- 日本特工- 用日语回复
- 西班牙客服人员- 用西班牙语回复

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team
from agno.team.mode import TeamMode

english_agent = Agent(
    name="English Agent",
    role="You only answer in English",
    model=OpenAIResponses(id="gpt-5.2"),
)
japanese_agent = Agent(
    name="Japanese Agent",
    role="You only answer in Japanese",
    model=OpenAIResponses(id="gpt-5.2"),
)
spanish_agent = Agent(
    name="Spanish Agent",
    role="You only answer in Spanish",
    model=OpenAIResponses(id="gpt-5.2"),
)

language_router = Team(
    name="Language Router",
    model=OpenAIResponses(id="gpt-5.2"),
    mode=TeamMode.route,
    members=[english_agent, japanese_agent, spanish_agent],
    instructions=[
        "Route questions to the appropriate language agent.",
        "If the language is not supported, respond in English.",
    ],
    markdown=True,
    show_members_responses=True,
)

# English
language_router.print_response("How are you?", stream=True)

# Japanese
language_router.print_response("お元気ですか?", stream=True)

# Spanish
language_router.print_response("¿Cómo estás?", stream=True)
```


