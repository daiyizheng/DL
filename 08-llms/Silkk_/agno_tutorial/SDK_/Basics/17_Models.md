# 模型
## 什么是模型？

语言模型是经过训练以理解自然语言和代码的机器学习程序。

当我们讨论模型时，我们通常指的是大型语言模型（LLM）。

这些模型就像智能体的大脑，使它们能够推理、行动并响应用户。模型越好，智能体就越智能。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    description="Share 15 minute healthy recipes.",
    markdown=True,
)
agent.print_response("Share a breakfast recipe.", stream=True)
```

### 错误处理
您可以配置模型，使其在请求失败时重试。这对于处理临时故障或模型提供程序的速率限制错误非常有用。

```python
model = OpenAIResponses(
  id="gpt-5.2",
  retries=2, # Number of retries to attempt before raising a ModelProviderError
  retry_delay=1, # Delay between retries, in seconds
  exponential_backoff=True, # If True, the delay between retries is doubled each time
)
```


## 模型作为字符串

使用便捷的 provider:model_id 字符串格式来指定模型，而无需导入模型类。

v2.2.6

Agno 提供了一种便捷的字符串语法，用于指定模型provider:model_id格式。这种方法无需导入模型类，从而减少了代码的冗余度，同时又保持了完整的功能。
传统的对象语法和字符串语法都同样有效，功能也完全相同。选择最符合您的编码风格和需求的方法即可。


### 格式
字符串格式遵循以下模式：

`"provider:model_id"`

- provider: 字符串，模型提供商的名称。
- model_id: 字符串，模型的唯一标识符。

例如：
- "openai:gpt-4o"
- "anthropic:claude-sonnet-4-20250514"
- "google:gemini-2.0-flash-exp"
- "groq:llama-3.3-70b-versatile"

### 基本用法
​
具有字符串语法的代理

```python
from agno.agent import Agent

agent = Agent(
    model="openai:gpt-4o",
    instructions="You are a helpful assistant.",
    markdown=True,
)

agent.print_response("Share a 2 sentence horror story.")
```

### 使用字符串语法的团队
在 Teams 中使用模型字符串，实现多代理协同工作流：
```python
from agno.agent import Agent
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

news_agent = Agent(
    name="News Agent",
    role="Search for tech news",
    model="openai:gpt-4o",
    tools=[HackerNewsTools()],
)

finance_agent = Agent(
    name="Finance Agent",
    role="Analyze financial data",
    model="openai:gpt-4o-mini",
    tools=[YFinanceTools()],
)

agent_team = Team(
    members=[news_agent, finance_agent],
    model="openai:gpt-4o",
    instructions="Coordinate research and provide comprehensive reports.",
)

agent_team.print_response("Research Tesla's latest developments")
```


### 多种模型类型
代理程序支持不同的模型以满足各种目的：

```python
from agno.agent import Agent

agent = Agent(
    # Main model for general responses
    model="openai:gpt-4o",
    # Reasoning model for complex thinking
    reasoning_model="anthropic:claude-sonnet-4-20250514",
    # Parser model for structured outputs
    parser_model="openai:gpt-4o-mini",
    # Output model for final formatting
    output_model="openai:gpt-4o",
)
```

### 普通提供者
| Provider | String Format | Example |
| :--- | :--- | :--- |
| OpenAl | openai:model_id | "openai:gpt-40" |
| Anthropic | anthropic:model_id | "anthropic:claude-sonnet-4-20250514" |
| Google | google:model_id | "google:gemini-2.0-flash-exp" |
| Groq | groq:model_id | "groq:llama-3.3-70b-versatile" |
| Ollama | ollama:model_id | "ollama:llama3.2" |
| Azure AI Foundry | azure-ai-foundry:model_id | "azure-ai-foundry:gpt-40" |
| Mistral | mistral:model_id | "mistral:mistral-large-latest" |
| LiteLLM | litellm:model_id | "litellm:gpt-4o" |
| OpenRouter | openrouter:model_id | "openrouter:anthropic/claude-3.5-sonnet" |
| Together | together:model_id | "together:meta-llama/Llama-3-70b-chat-hf" |

有关完整列表和特定提供商的文档，请参阅模型概述。

## 兼容性概述
了解 Agno 中不同模型提供商支持哪些功能。

Agno 为所有主流模型提供商提供全面支持，确保您无论选择哪种模型都能获得一致的功能。本页面概述了各提供商支持的功能和特性。
​

### 核心功能
Agno 上的所有型号均支持：
- 流媒体响应
- 工具调用
- 结构化输出
- 异步执行

| Agno Supparted Models | Image Input | Audio Input | Audio Responses | Video Input |
| :--- | :--- | :--- | :--- | :--- |
| AMLAPI |  |  |  |  |
| Anthropic Claudie |  |  |  |  |
| AWS Bedrock |  |  |  |  |
| AWS Bedrock Claude |  |  |  |  |
| Arure Al Foundry |  |  |  |  |
| Azzere OpenAl |  |  |  |  |
| Cerebras |  |  |  |  |
| Cerebras OpenAl |  |  |  |  |
| Cohere |  |  |  |  |
| Comentapi |  |  |  |  |
| DashScope |  |  |  |  |
| Decpinfra |  |  |  |  |
| DeepSuck |  |  |  |  |
| Fireworks |  |  |  |  |
| Geemini |  |  |  |  |
| Graq |  |  |  |  |
| HuggingFace |  |  |  |  |
| IBM WatsonX |  |  |  |  |
| InternLM |  |  |  |  |
| Langce |  |  |  |  |
| LiteLLLM |  |  |  |  |
| LtteLLMOpenAl |  |  |  |  |
| LiamoCpp |  |  |  |  |
| LM Studio |  |  |  |  |
| Ltamia |  |  |  |  |
| LlamaOpenAl |  |  |  |  |
| Mistral |  |  |  |  |
| Neblus |  |  |  |  |
| Neosantara |  |  |  |  |
| Nexus |  |  |  |  |
| Nvdia |  |  |  |  |
| Ollama |  |  |  |  |
| OpenAIChat | □ | □ |  |  |
| OpenAlRespanses | □ |  |  |  |
| Openfouter |  |  |  |  |
| Perplexity |  |  |  |  |
| Portikey |  |  |  |  |
| Requesty |  |  |  |  |
| Sambanova |  |  |  |  |
| Sticanfiow |  |  |  |  |
| Tugether |  |  |  |  |
| Vercel VO |  |  |  |  |
| VLLM |  |  |  |  |
| Vertex AI Claude |  |  |  |  |
| xal |  |  |  |  |

### 响应缓存
在开发和测试期间，将模型响应缓存到本地以降低成本。

在开发或测试新功能时，通常会多次使用相同的查询语句访问模型。在这种情况下，通常不需要模型生成相同的答案，可以将响应缓存起来以节省令牌。

响应缓存允许您在本地缓存模型响应，以避免重复 API 调用，并在多次执行相同查询时降低成本。

### 为什么要使用响应缓存？
响应缓存具有以下几个优点：
- 加快开发速度：避免在迭代开发过程中等待 API 响应
- 降低成本：消除相同查询的冗余 API 调用
- 一致性测试：确保测试用例在每次运行中都得到相同的响应。
- 离线开发：在 API 访问受限时使用缓存响应。
- 速率限制管理：减少 API 调用次数，以确保在速率限制范围内。

### 工作原理
启用响应缓存后：

- 缓存键生成：根据请求参数（消息、响应格式、工具等）生成唯一键。
- 缓存查找：在进行 API 调用之前，Agno 会检查是否存在与该键对应的缓存响应。
- 缓存命中：如果找到缓存，则立即返回缓存的响应。
- 缓存未命中：如果未找到缓存，则调用 API 并将响应缓存以供将来使用。
- TTL过期：缓存的响应会遵循配置的生存时间 (TTL) 并自动过期。

缓存默认存储在磁盘上，跨会话和程序重启持久存在。

### 基本用法
cache_response=True在初始化模型时启用响应缓存：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.2",
        cache_response=True  # Enable response caching
    )
)

# First call - cache miss, calls the API
response = agent.run("What is the capital of France?")

# Second identical call - cache hit, returns cached response instantly
response = agent.run("What is the capital of France?")

```

### 配置选项
​
缓存生存时间 (TTL)

控制响应缓存保留时间cache_ttl（以秒为单位）：

```python
agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.2",
        cache_response=True,
        cache_ttl=3600  # Cache expires after 1 hour
    )
)
```

如果cache_ttl未指定（或设置为None），则缓存的响应永不过期。

### 自定义缓存目录
使用以下方法将缓存的响应存储在特定位置cache_dir：

```python
agent = Agent(
    model=OpenAIResponses(
        id="gpt-5.2",
        cache_response=True,
        cache_dir="./path/to/custom/cache"
    )
)
```

如果未指定，Agno 使用默认缓存位置，即~/.agno/cache/model_responses您的主目录中的缓存位置。

### 与代理商一起使用
响应缓存是在模型级别配置的，并且会自动与代理配合使用：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create agent with cached responses
agent = Agent(
    model=Claude(
        id="claude-sonnet-4-20250514",
        cache_response=True,
        cache_ttl=3600
    ),
    tools=[...],  # Your tools
    instructions="Your instructions here"
)

# All agent runs will use caching
agent.run("Your query")
```

### 与 Teams 配合使用
响应缓存也适用Team。您可以为单个团队成员和团队领导模型启用此功能：

```python
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIResponses

# Create team members with cached responses
researcher = Agent(
    model=OpenAIResponses(id="gpt-5.2", cache_response=True),
    name="Researcher",
    role="Research information"
)

writer = Agent(
    model=OpenAIResponses(id="gpt-5.2", cache_response=True),
    name="Writer",
    role="Write content"
)

team = Team(members=[researcher, writer], model=OpenAIResponses(id="gpt-5.2", cache_response=True))
```

每个团队成员都根据自己的特定查询维护自己的缓存。

### 流式缓存
使用流式传输时，响应数据也可以被缓存。如果缓存命中，则会将整个响应作为一个数据块返回。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(model=OpenAIResponses(id="gpt-5.2", cache_response=True))

for i in range(1, 3):
    print(f"\n{'=' * 60}")
    print(
        f"Run {i}"
    )
    print(f"{'=' * 60}\n")
    agent.print_response("Write me a short story about a cat that can talk and solve problems.", stream=True)
```

## 模型指数

Agno支持的所有型号索引。

Agno 支持以下按类别划分的模型提供商：

### 原生模型提供商
![](https://i-blog.csdnimg.cn/direct/5b0bf390670742e086cfa390a4c4be96.png)

### 本地模型提供商
![](https://i-blog.csdnimg.cn/direct/f80bdcfa25534653a4fdeb4db5416e48.png)

### 云模型提供商
![](https://i-blog.csdnimg.cn/direct/43fd3ccaeb9a47b0a2af9433e65a7290.png)

### 模型网关和聚合器

![](https://i-blog.csdnimg.cn/direct/ae6a444ae7ad4a2eaafdf5bdce595977.png)

## 原始模型供应商
### 人格克劳德

使用具有 Agno 代理的 Anthropic Claude 模型。

Claude是Anthropic公司开发的一系列基础人工智能模型，可用于多种应用场景。点击此处查看模型对比。
我们建议您进行试验，找到最适合您使用场景的模型。以下是一些通用建议：
- claude-sonnet-4-20250514该模型适用于大多数使用场景，并支持图像输入。
- claude-opus-4-1-20250805该型号是他们最好的型号。
- claude-3-5-haiku-20241022这款是他们速度最快的型号。
- Anthropic 对其 API 设置了速率限制。详情请参阅文档。

### 验证
设置你的ANTHROPIC_API_KEY环境。你可以在这里从Anthropic购买一个。

`export ANTHROPIC_API_KEY=***`

### 例子
与Claude您的Agent：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    model=Claude(id="claude-3-5-sonnet-20240620"),
    markdown=True
)

# Print the response on the terminal
agent.print_response("Share a 2 sentence horror story.")
```
### 测试版功能
您可以通过设置以下betas参数，将 Anthropic 的测试版功能与 Agno 结合使用：
```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    model=Claude(
        betas=["context-management-2025-06-27"],
    ),
)
```

点击此处了解更多关于AgnoClaude模型测试版功能的信息。

### 提示缓存
cache_system_prompt您可以通过设置以下选项启用系统提示缓存True：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

agent = Agent(
    model=Claude(
        id="claude-3-5-sonnet-20241022",
        cache_system_prompt=True,
    ),
)
```

### 结构化输出
结构化输出用于确保模型的响应与定义的模式相匹配。
这有助于消除诸如字段缺失或值无效等问题。适用于需要以特定格式提供可靠、一致响应的生产系统。
Agno 使用 Claude 对结构化输出的原生支持。此功能适用于claude-sonnet-4-5-20250929所有较新的型号。有关更多详细信息，请参阅 Anthropic 的结构化输出文档。

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

agent = Agent(
    model=Claude(id="claude-sonnet-4-5-20250929"),
    description="Extract user information.",
    output_schema=User,
)
```
了解更多关于AgnoClaude模型中结构化输出的信息：
- 基本结构化输出
- 流式结构化输出
- 使用严格工具生成结构化输出


## 本地模型提供商
### Ollama
在 Agno 代理中使用 Ollama 运行本地模型。

使用 Ollama 运行大型语言模型，既可以在本地运行，也可以通过 Ollama Cloud 运行。

Ollama是一个非常棒的工具，既可以在本地运行模型，也可以在云端运行模型。

本地使用：使用 Ollama 客户端在您自己的硬件上运行模型。

云端使用：通过Ollama Cloud使用 API 密钥访问云端托管模型。

Ollama支持多种开源模型。请点击此处查看库文件。

尝试不同的模型，找到最适合您使用场景的模型。以下是一些通用建议：
- gpt-oss:120b-cloud对于大多数任务而言，它是一个优秀的通用云模型。
- llama3.3模型适用于大多数基本使用场景。
- qwen模型在使用工具时表现尤为出色。
- deepseek-r1模型具有强大的推理能力。
- phi4这些模型功能强大，体积却非常小巧。

`export OLLAMA_API_KEY=***`

使用 Ollama Cloud 时，主机会自动设置https://ollama.com。本地使用时，无需 API 密钥。

### 建立模型
​
本地使用情况
安装ollama并运行模型：

```python
ollama run llama3.1
```

这将开始与模型进行互动。
要下载该模型以便在 Agno 代理中使用：

`ollama pull llama3.1`

### 云使用情况
对于 Ollama Cloud，无需在本地安装 Ollama 服务器。只需安装 Ollama 库，按照上文“身份验证”部分所述设置 API 密钥，即可直接访问云端托管的模型。

### 示例
​
本地使用情况
模型在本地可用后，可以使用Ollama模型类来访问它：

```python
from agno.agent import Agent
from agno.models.ollama import Ollama

agent = Agent(
    model=Ollama(id="llama3.1"),
    markdown=True
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")
```

### 云使用情况

```python
from agno.agent import Agent
from agno.models.ollama import Ollama

agent = Agent(
    model=Ollama(id="gpt-oss:120b-cloud"),
    markdown=True
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")
```
### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | ＂llama3．2＂ | 要使用的 Ollama 模型名称 |
| name | str | ＂Ołlama＂ | 型号名称 |
| provider | str | ＂Ollama＂ | 模型提供者 |
| host | str | ＂http：／／localhost：11 434＂ | Ollama 服务器的主机 URL |
| timeout | Optional［int］ | None | 请求超时时间（秒） |
| format | Optional［str］ | None | 返回响应的格式（例如，＂json＂） |
| options | Optional［Dict［str， Any］］ | None | 其他模型选项（温度、峰值压力等） |
| keep＿alive | Optional［Union［float， str］］ | None | 模型加载保持时间（例如， ＂ 5 m ＂，即 3600 秒） |
| template | Optional［str］ | None | 要使用的提示模板 |
| system | Optional［str］ | None | 要使用的系统消息 |
| raw | Optional［bool］ | None | 是否返回未经格式化的原始响应 |
| stream | bool | True | 是否直播响应 |

Ollama是Model类的子类，可以访问相同的参数。

### Responses API
Ollama v0.13.3+ 通过 /v1/responses 端点支持 OpenAI Responses API。使用此界面使用OllamaResponses：

```python
from agno.agent import Agent
from agno.models.ollama import OllamaResponses

agent = Agent(
    model=OllamaResponses(id="gpt-oss:20b"),
    markdown=True,
)

agent.print_response("Share a 2 sentence horror story")
```

响应 API 是无状态的。每个请求都是独立的，不存在previous_response_id链式调用。

有关完整参数，请参阅OllamaResponses 参考文档。


### vLLM

vLLM是一个快速易用的 LLM 推理和服务库，专为高吞吐量和内存高效的 LLM 服务而设计。

### 先决条件
安装 vLLM 并开始运行模型：
```python
uv pip install vllm
```

启动 vLLM 服务器

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```
这将启动具有 OpenAI 兼容 API 的 vLLM 服务器。
### 例子
基础代理

```python
from agno.agent import Agent
from agno.models.vllm import VLLM

agent = Agent(
    model=VLLM(
        id="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/",
    ),
    markdown=True
)

agent.print_response("Share a 2 sentence horror story.")
```
### 高级用法
​
使用工具

vLLM模型与Agno工具无缝协作：
```python
from agno.agent import Agent
from agno.models.vllm import VLLM
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=VLLM(id="meta-llama/Llama-3.1-8B-Instruct"),
    tools=[HackerNewsTools()],
    markdown=True
)

agent.print_response("What's the latest news about AI?")
```
### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | ＂microsoft／DialoGPT－ medium＂ | 要与 vLLM 一起使用的模型 ID |
| name | str | ＂vLLM＂ | 型号名称 |
| provider | str | ＂vLLM＂ | 模型提供者 |
| api＿key | Optional［str］ | None | API密钥（通常本地vLLM不需要） |
| base＿url | str | ＂http：／／localhost：8000／v1 ＂ | vLLM 服务器的基本 URL |





## 云模型提供商
### AWS Bedrock

将 AWS Bedrock 基础模型与 Agno 代理结合使用。

使用 AWS Bedrock 访问 AWS 上的各种基础模型。在门户网站上管理您对模型的访问权限。

查看所有AWS Bedrock 基础模型。并非所有 Bedrock 模型都支持所有功能。请查看每个模型支持的功能。

我们建议您进行试验，找到最适合您使用场景的模型。以下是一些通用建议：
- 对于性能总体良好的Mistral型号，请查看mistral.mistral-large-2402-v1:0。
- 您可以试用亚马逊 Nova 系列模型。可amazon.nova-pro-v1:0用于一般用途。
- 对于 Claude 模型，请参阅我们的Claude 集成。

### 验证
#### AWS Bedrock 支持三种身份验证方法：
​
方法一：访问密钥和私钥（推荐）

设置您的AWS_ACCESS_KEY_ID、AWS_SECRET_ACCESS_KEY和AWS_REGION环境变量。

从这里领取钥匙。
```bash
export AWS_ACCESS_KEY_ID=***
export AWS_SECRET_ACCESS_KEY=***
export AWS_REGION=***

```

或者直接将它们传递给模型：

```python
from agno.agent import Agent
from agno.models.aws import AwsBedrock

agent = Agent(
    model=AwsBedrock(
        id="mistral.mistral-large-2402-v1:0",
        aws_access_key_id="your-access-key",
        aws_secret_access_key="your-secret-key",
        aws_region="us-east-1"
    )
)
```
### 方法二：单点登录身份验证
利用您当前 AWS 配置文件的身份验证，使用 SSO 身份验证：

```python
from agno.agent import Agent
from agno.models.aws import AwsBedrock

agent = Agent(
    model=AwsBedrock(
        id="mistral.mistral-large-2402-v1:0",
        aws_sso_auth=True,
        aws_region="us-east-1"
    )
)
```

### 方法三：Boto3会话
使用预配置的 boto3 会话进行高级身份验证场景（包括 SSO、角色承担等）：
```python
from boto3.session import Session
from agno.agent import Agent
from agno.models.aws import AwsBedrock

# Create a boto3 session with your preferred authentication
session = Session(
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    region_name="us-east-1"
)

agent = Agent(
    model=AwsBedrock(
        id="mistral.mistral-large-2402-v1:0",
        session=session
    )
)
```

### 例子
与AwsBedrock您的Agent：

```python
from agno.agent import Agent
from agno.models.aws import AwsBedrock

agent = Agent(
    model=AwsBedrock(id="mistral.mistral-large-2402-v1:0"),
    markdown=True
)

# Print the response on the terminal
agent.print_response("Share a 2 sentence horror story.")
```
### 参数
| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | ＂mistral．mistral－ small－2402－v1：0＂ | 用于生成响应的特定模型 ID。 |
| name | str | ＂AwsBedrock＂ | AWS Bedrock 代理的名称标识符。 |
| provider | str | ＂AwsBedrock＂ | 模型提供者。 |
| aws＿access＿key＿id | Optional［str］ | None | 用于身份验证的 AWS 访问密钥 ID。也可以通过 AWS＿ACCESS＿KEY＿ID 环境变量设置。 |
| aws＿secret＿access＿k ey | Optional［str］ | None | 用于身份验证的 AWS 秘密访问密钥。也可以通过 <br> AWS＿SECRET＿ACCESS＿KEY 环境变量设置。 |
| aws＿region | Optional［str］ | None | 用于 API 请求的 AWS 区域。也可以通过 AWS＿REGION 环境变量设置。 |
| session | Optional［Session］ | None | 用于高级身份验证场景（SSO、角色承担等）的 boto3 Session 对象。 |
| aws＿sso＿auth | Optional［bool］ | False | 利用当前配置文件的身份验证，无需访问密钥和秘密访问密钥。 |
| max＿tokens | Optional［int］ | None | 响应中要生成的最大令牌数。 |
| temperature | Optional［float］ | None | 要使用的采样温度，介于 0 和 2 之间。较高的值（例如 0.8 ）会使输出更随机，而较低的值（例如 0.2 ）会使输出更集中和确定。 |
| top＿p | Optional［float］ | None | 细胞核采样参数。该模型考虑了概率质量为 top＿p 的标记的结果。 |
| stop＿sequences | Optional［List［str］ ］ | None | API 将停止生成更多令牌的序列列表。 |
| request＿params | Optional［Dict［str， Any］］ | None | 请求的附加参数，以字典形式提供。 |
| client＿params | Optional［Dict［str， Any］］ | None | 用于初始化客户端的其他客户端参数 AwsBedrock，以字典形式提供。 |
| client | Optional［AwsClient ］ | None | 预配置的 AWS 客户端实例。 |

AwsBedrock是Model类的子类，可以访问相同的参数。

## 模型网关和聚合器
### LiteLLM
将 LiteLLM 与 Agno 集成，打造统一的 LLM 体验。

LiteLLM为各种 LLM 提供商提供统一的接口，允许您使用相同的代码使用不同的模型。
Agno 通过两种方式与 LiteLLM 集成：
- 直接 SDK 集成- 使用 LiteLLM Python SDK
- 代理服务器集成- 使用 LiteLLM 作为 OpenAI 兼容的代理

### 先决条件
两种集成方法都需要
```python
# Install required packages
uv pip install agno litellm
```

设置您的 API 密钥：无论使用哪个模型（OpenAI、Hugging Face 或 XAI），API 密钥都引用为LITELLM_API_KEY。

`export LITELLM_API_KEY=your_api_key_here`

### SDK集成
该类LiteLLM提供了与 LiteLLM Python SDK 的直接集成。

### 基本用法

```python
from agno.agent import Agent
from agno.models.litellm import LiteLLM

# Create an agent with GPT-4o
agent = Agent(
    model=LiteLLM(
        id="gpt-5-mini",  # Model ID to use
        name="LiteLLM",  # Optional display name
    ),
    markdown=True,
)

# Get a response
agent.print_response("Share a 2 sentence horror story")
```

### 使用拥抱脸模型
LiteLLM 还可以与 Hugging Face 模型配合使用：
```python
from agno.agent import Agent
from agno.models.litellm import LiteLLM

agent = Agent(
    model=LiteLLM(
        id="huggingface/mistralai/Mistral-7B-Instruct-v0.2",
        top_p=0.95,
    ),
    markdown=True,
)

agent.print_response("What's happening in France?")
```


### 使用拥抱脸模型
LiteLLM 还可以与 Hugging Face 模型配合使用：

```python
from agno.agent import Agent
from agno.models.litellm import LiteLLM

agent = Agent(
    model=LiteLLM(
        id="huggingface/mistralai/Mistral-7B-Instruct-v0.2",
        top_p=0.95,
    ),
    markdown=True,
)

agent.print_response("What's happening in France?")
```

### 配置选项
该类LiteLLM接受以下参数：

| 范围 | 类型 | 描述 | 默认 |
| :--- | :--- | :--- | :--- |
| id | 斯特 | 模型标识符（例如，＂gpt－5－mini＂或 ＂huggingface／mistralai／Mistral－7B－ Instruct－v0．2＂） | ＂gpt－5－mini＂ |
| name | 斯特 | 模型显示名称 | ＂LiteLLM＂ |
| provider | 斯特 | 提供商名称 | ＂LiteLLM＂ |
| api＿key | 可选［字符串］ | API密钥（回退到LITELLM＿API＿KEY环境变量） | 没有任何 |
| api＿base | 可选［字符串］ | API 请求的基本 URL | 没有任何 |
| max＿tokens | 可选［int］ | 响应中的最大令牌数 | 没有任何 |
| temperature | 漂浮 | 采样温度 | 0.7 |
| top＿p | 漂浮 | Top－p 采样值 | 1.0 |
| request＿params | 可选［字典［字符串，任意类型］ | 其他请求参数 | 没有任何 |

## 兼容 OpenAI 的模型
使用任何与 OpenAI 兼容的端点配合 Agno 代理。

许多提供商支持 OpenAI API 格式。使用OpenAILike模型替换base_url来访问这些文件。

### 例子
```python
from os import getenv
from agno.agent import Agent
from agno.models.openai.like import OpenAILike

agent = Agent(
    model=OpenAILike(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key=getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")
```

### 参数

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | ＂not－provided＂ | 要使用的模型的 ID |
| name | str | ＂OpenAILike＂ | 型号名称 |
| provider | str | ＂OpenAILike＂ | 模型提供者 |
| api＿key | Optional［str］ | ＂not－provided＂ | 用于身份验证的 API 密钥 |
| base＿url | Optional［str］ | None | API 服务的基本 URL |
| collect＿metrics＿on＿co mpletion | bool | False | 仅从最后一个流式传输数据块收集令牌指标（适用于具有累计令牌计数的提供商） |

OpenAILike扩展了与 OpenAI 兼容的接口，并支持OpenAIChat的所有参数。

只需更改 ` base_url`and`api_key`即可指向您首选的 OpenAI 兼容服务。

### 响应 API
对于实现了Open Responses API 规范的提供商，请使用OpenResponses：

```python
from agno.agent import Agent
from agno.models.openai import OpenResponses

agent = Agent(
    model=OpenResponses(
        id="your-model-id",
        base_url="https://your-provider.com/v1",
        api_key="your-api-key",
    ),
)

agent.print_response("Share a 2 sentence horror story.")
```

