# 上下文压缩

压缩工具调用结果以节省上下文空间，同时保留关键信息。

上下文压缩允许您在代理运行时管理其上下文，帮助代理保持在上下文窗口内，避免速率限制或响应质量下降。

你可以把它想象成一个研究助理，他阅读冗长的报告，然后把关键要点总结出来，而不是把整份文件都给你。

问题：工具结果过于冗长

如果您使用响应数据量很大的工具，且未进行压缩，则工具结果会迅速占据您的上下文窗口：
| 成分 | 累计令牌计数 | 笔记 |
| :--- | :--- | :--- |
| 系统提示 | 1200 个代币 |  |
| 用户消息 | 1300个代币 |  |
| LLM 响应 | 1500个代币 |  |
| 工具调用 1 | 2500 个代币 |  |
| 工具调用 2 | 5700 个代币 | $2,500+3,200$ 新增 |
| 工具调用 3 | 8，500 个代币 | $5700+2800$ 新增 |
| 工具调用 4 | 12，000 个代币 | $8,500+3,500$ 新增 |

这很快就会变得成本高昂，并且在复杂的工作流程中会遇到上下文限制。

### 解决方案：自动压缩
上下文压缩会在达到阈值后对工具结果进行汇总：

```python
Tool Call 1: 2,500 tokens
Tool Call 2: 5,700 tokens
Tool Call 3: 8,500 tokens
[Compression triggered]
Tool Call 4: 1,300 tokens (800 compressed + 500 new)
```

好处：
- 大幅降低代币成本
- 保持在上下文窗口限制范围内
- 保存关键事实和数据
- 自动压缩

### 工作原理
上下文压缩遵循一个简单的模式：
1. 启用压缩
设置compress_tool_results=True您的代理或团队，或提供联系方式CompressionManager。系统会监控工具调用结果的实时更新。

2. 已达到阈值
达到阈值后，将触发压缩。每个未压缩的工具调用结果都会被单独汇总。

3. 智能摘要
压缩模型保留了关键事实（数字、日期、实体、URL），同时删除了样板文字、冗余信息和填充文本。

4. LLM循环仍在继续

压缩后的工具结果将用于下一次 LLM 执行，从而减少令牌使用量并延长上下文窗口的生命周期。

### 启用压缩
启用compress_tool_results=True此功能可自动压缩刀具结果。默认阈值为 3 次刀具调用。
例如：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools

web_agent = Agent(
    name="HackerNews Researcher",
    tools=[HackerNewsTools()],
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[web_agent],
    compress_tool_results=True,
)

team.print_response("Get the top stories on HackerNews about AI, ML, startups, and tech trends")

```

### 自定义压缩
提供一个CompressionManager选项来自定义压缩行为：

```python
from agno.agent import Agent
from agno.compression.manager import CompressionManager
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

compression_manager = CompressionManager(
    model=OpenAIResponses(id="gpt-5.2"),  # Use a faster model for compression
    compress_tool_results_limit=2,  # Compress after 2 tool calls (default: 3)
    compress_tool_call_instructions="Your custom compression prompt here...",
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    compression_manager=compression_manager,
)

agent.print_response("Find stories about AI startup funding on HackerNews")
```

```python
from agno.agent import Agent
from agno.compression.manager import CompressionManager
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools

compression_manager = CompressionManager(
    model=OpenAIResponses(id="gpt-5.2"),  # Use a faster model for compression
    compress_tool_results_limit=2,  # Compress after 2 tool calls (default: 3)
    compress_tool_call_instructions="Your custom compression prompt here...",
)

web_agent = Agent(
    name="HackerNews Researcher",
    tools=[HackerNewsTools()],
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[web_agent],
    compression_manager=compression_manager,
)

team.print_response("Find stories about AI startup funding on HackerNews")
```

### 压缩触发器
它CompressionManager支持两种触发压缩的阈值：
| 模式 | 范围 | 用例 |
| :--- | :--- | :--- |
| 基于计数的 | compress＿tool＿results＿lim it | 可预测的工具调用模式。在 $N$ 次未压缩的工具结果后触发。 |
| 基于令牌的 | compress＿token＿limit | 结果大小可变或上下文限制严格。当上下文超出令牌阈值时触发。 |


### 基于工具的压缩
compress_tool_results_limit当您的工具调用模式可预测，并且希望在固定数量的工具调用结果后触发压缩时，请进行设置。

### 基于令牌的压缩
compress_token_limit当您需要精确控制上下文大小时，尤其是在工具结果大小差异很大的情况下，请使用此方法：

```python
from agno.agent import Agent
from agno.compression.manager import CompressionManager
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

compression_manager = CompressionManager(
    model=OpenAIResponses(id="gpt-5.2"),
    compress_tool_results=True,
    compress_token_limit=5000,  # or compress_tool_results_limit
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    compression_manager=compression_manager,
)

agent.print_response("Find HackerNews discussions about OpenAI, Anthropic, Google DeepMind, and Meta AI")
```

```python
from agno.agent import Agent
from agno.compression.manager import CompressionManager
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools

compression_manager = CompressionManager(
    model=OpenAIResponses(id="gpt-5.2"),
    compress_tool_results=True,
    compress_token_limit=5000,  # or compress_tool_results_limit
)

web_agent = Agent(
    name="HackerNews Researcher",
    tools=[HackerNewsTools()],
)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[web_agent],
    compression_manager=compression_manager,
)

team.print_response("Find HackerNews discussions about OpenAI, Anthropic, Google DeepMind, and Meta AI")
```

### 何时使用上下文压缩
非常适合：
- 拥有能够返回详细结果的工具（网络搜索、API）的代理
- 包含多个工具调用的多步骤工作流程
- 长时间的会话，其中会积累上下文信息
- 成本至关重要的生产系统
​

### 令牌计数

用于上下文规划和压缩的令牌估计。

令牌计数有助于估算代理运行期间的上下文令牌数量。令牌计数可用于基于令牌的上下文压缩和内存优化等功能。

背景信息可以包括：

- 消息
    - 消息内容 - 包括系统消息、用户消息和助手消息内容。
    - 工具调用参数和结果
    - 可选的推理内容
    - 多模态内容块
- 工具
    - 工具定义可能是总标记计数的重要组成部分，尤其是在参数模式较大或描述较长的情况下。
    - 输出方案
    - 如果使用输出模式，则该模式包含在令牌计数中。
- 多模态附件
    - 邮件中的图像、音频、视频和附件数量均采用保守估计进行统计。

### 可选依赖项（推荐）
为了获得更准确的本地词元计数估算结果，请安装分词器：
```bash
uv pip install -U tiktoken tokenizers
```
- tiktoken：当支持 OpenAI 式分词时使用。
- tokenizers：用于某些可用的开源分词器。
- 如果给定模型没有上述两种方法，我们就采用启发式估计。

### 例如：计数令牌
```python
from pydantic import BaseModel

from agno.models.message import Message
from agno.models.openai import OpenAIResponses


class Answer(BaseModel):
    answer: str


model = OpenAIResponses(id="gpt-5.2")

messages = [
    Message(role="system", content="You are a concise assistant."),
    Message(role="user", content="Summarize context compression in 2 sentences."),
]

# Tool definitions can be passed as OpenAI-style tool dicts
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

tokens = model.count_tokens(messages=messages, tools=tools, output_schema=Answer)
print(f"Estimated tokens: {tokens}")
```

### 基于词元的上下文压缩中的词元计数
设置后compress_token_limit，Agno 会在运行循环期间检查估计的令牌计数，并在达到阈值时触发压缩。

因为令牌计数可以包含消息历史记录、工具定义和输出模式/响应格式，所以它比仅计算消息文本更接近“真实”请求大小。

### 多模态估计
Agno采用保守估计方法来处理多模式输入，以支持情境规划：
- 图像：通过基于图块的方法（视觉式计数）进行估计
- 音频：按每秒令牌数估算
- 视频：帧数估算方式与图像类似（如果帧率/尺寸未知，则采用保守的默认值）
- 文件数量：根据文件类型/大小估算