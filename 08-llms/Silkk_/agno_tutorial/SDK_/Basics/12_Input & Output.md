# 输入和输出

学习如何将数据传递给代理并处理他们的响应。

代理程序可以接受多种格式的输入并生成输出，从简单的字符串到经过验证的 Pydantic 模型。您可以先从字符串开始，然后在需要验证时添加结构。


## 格式类型用法

| 用例 | 格式 |
| :--- | :--- |
| 原型设计、聊天界面 | 字符串工作正常 |
| 数据提取、分类 | 结构化输出 |
| API响应、管道 | 结构化输入和输出 |

## 字符串 I/O
输入字符串，输出字符串

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

agent = Agent(model=OpenAIResponses(id="gpt-5.2"))

response = agent.run("What's the capital of France?")
print(response.content)  # "The capital of France is Paris."

```

## 结构化 I/O
使用 Pydantic 模型来验证输入和返回的结果：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class ReviewInput(BaseModel):
    text: str
    product_id: str

class SentimentResult(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(description="One sentence summary")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_schema=SentimentResult,
)

response = agent.run(
    input=ReviewInput(text="Love this product!", product_id="SKU-123")
)

result: SentimentResult = response.content
print(result.sentiment)   # "positive"
print(result.confidence)  # 0.95
```

## 智能体的结构化输入
使用 Pydantic 模型验证代理的输入数据。

使用 Pydantic 模型将结构化数据传递给代理。您可以直接传递模型实例，也可以设置input_schema为自动验证字典。


### 输入格式类型
| 用例 | 输入格式 |
| :--- | :--- |
| 您正在用代码构建输入。 | 使用 Pydantic 模型实例 |
| 输入来自外部来源（API、文件、用户输入） | 使用 input＿schema |


### 使用 Pydantic 模型
将 Pydantic 模型实例传递给input：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class ResearchRequest(BaseModel):
    topic: str
    max_sources: int = Field(ge=1, le=20, default=5)
    focus_areas: list[str] = Field(default_factory=list)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    input_schema=ResearchRequest,
)

# Pass a dict - validated against ResearchRequest
response = agent.run(
    input={
        "topic": "AI Agents",
        "max_sources": 10,
        "focus_areas": ["multi-agent systems", "tool use"]
    }
)

```

当输入来自外部来源（例如 API 请求或配置文件）时，这非常有用。
​
### 处理输入ValidationError
输入无效会引发 Pydantic 异常ValidationError：

```python
from pydantic import BaseModel, Field, ValidationError
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class OrderRequest(BaseModel):
    product_id: str
    quantity: int = Field(gt=0)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    input_schema=OrderRequest,
)

try:
    agent.run(input={"product_id": "SKU-123", "quantity": -5})
except ValidationError as e:
    print(e)
    # quantity: Input should be greater than 0
```

### 常见模式
​API 请求处理程序

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class SummaryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    max_length: int = Field(ge=50, le=500, default=200)
    style: str = Field(default="concise")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    input_schema=SummaryRequest,
)

# In your API endpoint
def summarize(request_data: dict):
    response = agent.run(input=request_data)  # Auto-validated
    return {"summary": response.content}
```

## 配置驱动任务

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

class ResearchConfig(BaseModel):
    topic: str
    depth: int = Field(ge=1, le=10, default=5)
    include_sources: bool = True
    output_format: str = Field(default="markdown")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    input_schema=ResearchConfig,
)

# Load config from file or environment
config = {
    "topic": "LLM frameworks",
    "depth": 7,
    "include_sources": True
}

response = agent.run(input=config)
```

## 嵌套模型
```python
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class Author(BaseModel):
    name: str
    email: str

class ArticleRequest(BaseModel):
    title: str
    author: Author
    tags: list[str]

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    input_schema=ArticleRequest,
)

response = agent.run(
    input={
        "title": "Getting Started with Agno",
        "author": {"name": "Jane Doe", "email": "jane@example.com"},
        "tags": ["tutorial", "agents"]
    }
)

```
## 团队结构化输入
使用 Pydantic 模型验证团队的输入数据。

使用 Pydantic 模型向团队传递结构化数据。您可以直接传递模型实例，也可以设置input_schema为自动验证字典。

### 输入格式类型

| 用例 | 输入格式 |
| :--- | :--- |
| 您正在用代码构建输入。 | 使用 Pydantic 模型实例 |
| 输入来自外部来源（API、文件、用户输入） | 使用 input＿schema |

### 使用 Pydantic 模型
将 Pydantic 模型实例传递给input：


```python

from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

class ResearchProject(BaseModel):
    topic: str
    focus_areas: list[str] = Field(min_length=1)
    max_sources: int = Field(ge=1, le=20, default=10)

news_agent = Agent(
    name="News Researcher",
    role="Research tech news and trends",
    tools=[HackerNewsTools()]
)

finance_agent = Agent(
    name="Finance Researcher",
    role="Research financial data",
    tools=[YFinanceTools()]
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
)

# Pass the model instance directly
project = ResearchProject(
    topic="AI Agents",
    focus_areas=["multi-agent systems", "tool use"],
    max_sources=15
)

response = team.run(input=project)

```

### 使用input_schema
设置input_schema团队自动验证字典：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools

class ResearchProject(BaseModel):
    topic: str
    focus_areas: list[str] = Field(min_length=1)
    max_sources: int = Field(ge=1, le=20, default=10)

news_agent = Agent(
    name="News Researcher",
    role="Research tech news and trends",
    tools=[HackerNewsTools()]
)

finance_agent = Agent(
    name="Finance Researcher",
    role="Research financial data",
    tools=[YFinanceTools()]
)

team = Team(
    name="Research Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    input_schema=ResearchProject,
)

# Pass a dict - validated against ResearchProject
response = team.run(
    input={
        "topic": "AI Agents",
        "focus_areas": ["multi-agent systems", "tool use"],
        "max_sources": 15
    }
)

```

当输入来自外部来源（例如 API 请求或配置文件）时，这非常有用。
​
###  处理输入ValidationError
输入无效会引发 Pydantic 异常ValidationError：

```python
from pydantic import BaseModel, Field, ValidationError

class ResearchProject(BaseModel):
    topic: str
    focus_areas: list[str] = Field(min_length=1)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    input_schema=ResearchProject,
)

try:
    team.run(input={"topic": "AI", "focus_areas": []})  # Empty list
except ValidationError as e:
    print(e)
    # focus_areas: List should have at least 1 item

```

## 常见模式
​
### 多主题研究
```python
from pydantic import BaseModel, Field

class ComparisonProject(BaseModel):
    title: str
    items_to_compare: list[str] = Field(min_length=2, max_length=5)
    comparison_criteria: list[str]
    output_format: str = Field(default="table")

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    input_schema=ComparisonProject,
)

response = team.run(
    input={
        "title": "AI Framework Comparison",
        "items_to_compare": ["Agno", "LangChain", "CrewAI"],
        "comparison_criteria": ["performance", "ease of use", "documentation"],
        "output_format": "table"
    }
)

```

### 范围分析

```python
from pydantic import BaseModel, Field
from datetime import date

class AnalysisScope(BaseModel):
    company: str
    analysis_type: str = Field(description="financial, competitive, or market")
    start_date: date | None = None
    end_date: date | None = None
    include_competitors: bool = True

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    input_schema=AnalysisScope,
)

response = team.run(
    input={
        "company": "NVIDIA",
        "analysis_type": "competitive",
        "include_competitors": True
    }
)

```

### 嵌套配置

```python
from pydantic import BaseModel, Field

class Source(BaseModel):
    name: str
    priority: int = Field(ge=1, le=3, default=2)

class ResearchConfig(BaseModel):
    topic: str
    sources: list[Source]
    depth: int = Field(ge=1, le=10, default=5)

team = Team(
    model=OpenAIResponses(id="gpt-5.2"),
    members=[news_agent, finance_agent],
    input_schema=ResearchConfig,
)

response = team.run(
    input={
        "topic": "Quantum Computing",
        "sources": [
            {"name": "HackerNews", "priority": 1},
            {"name": "Financial Reports", "priority": 2}
        ],
        "depth": 7
    }
)

```

## 面向代理的结构化输出

从代理获取已验证的 Pydantic 对象，而不是原始文本。

结构化输出将代理的响应限制为与 Pydantic 模式相匹配。它不再解析自由格式的文本，而是生成一个包含类型化字段的已验证对象。

### 基本用法
定义一个 Pydantic 模型并将其作为参数传递output_schema：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

class MovieScript(BaseModel):
    setting: str = Field(description="Where the movie takes place")
    genre: str = Field(description="Movie genre")
    storyline: str = Field(description="Brief plot summary")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_schema=MovieScript,
)

response = agent.run("Write a movie script about a heist in Tokyo")

# response.content is a MovieScript object, not a string
print(response.content.setting)    # "Tokyo, Japan - 2024"
print(response.content.genre)      # "Action/Thriller"
print(response.content.storyline)  # "A retired thief is pulled back..."
```


​
## 工作原理
当您设置时output_schema，Agno：
- 将您的 Pydantic 模型转换为 JSON 模式
- 将此模式传递给模型的结构化输出 API（如果支持）。
- 根据你的架构验证响应。
- 返回一个类型化的 Pydantic 对象response.content


## output_schema每次运行的控制
在运行时覆盖或设置架构：

```python
agent = Agent(model=OpenAIResponses(id="gpt-5.2"))

# Different schemas for different calls
sentiment = agent.run("Analyze sentiment: 'Great product!'", output_schema=SentimentResult)
entities = agent.run("Extract entities from this text...", output_schema=EntityList)
```

当一个代理处理多个具有不同输出格式的任务时，这非常有用。


## 使用工具
结构化输出与工具协同工作。代理在执行过程中调用工具，然后根据您的模式格式化最终响应：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

class StockAnalysis(BaseModel):
    symbol: str
    current_price: float
    change_percent: float
    recommendation: str = Field(description="buy, hold, or sell")
    reasoning: str

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[YFinanceTools()],
    output_schema=StockAnalysis,
)

# Agent calls YFinanceTools to get live data, then returns structured StockAnalysis
response = agent.run("Analyze NVDA and give me a recommendation")
analysis: StockAnalysis = response.content

print(analysis.symbol)          # "NVDA"
print(analysis.current_price)   # 142.50
print(analysis.recommendation)  # "buy"

```

## 模式设计技巧
​
#### 使用字段描述
描述指导模型生成什么内容：
```python
class Review(BaseModel):
    # Good: clear guidance
    sentiment: str = Field(description="Must be 'positive', 'negative', or 'neutral'")
    confidence: float = Field(ge=0, le=1, description="Confidence score from 0.0 to 1.0")

    # Less effective: no guidance
    rating: int
```


#### 使用约束
Pydantic 验证器确保输出有效：
```python
from pydantic import BaseModel, Field

class Rating(BaseModel):
    score: int = Field(ge=1, le=5, description="Rating from 1 to 5")
    tags: list[str] = Field(min_length=1, max_length=5)
```

#### 对于不确定字段，请使用可选参数。
当数据可能不可用时，将字段标记为可选：

```python
class CompanyInfo(BaseModel):
    name: str
    ticker: str
    market_cap: float | None = Field(None, description="Market cap if publicly traded")
    founded_year: int | None = None

```

#### 常见模式
​数据提取

```python
from pydantic import BaseModel, Field

class ExtractedData(BaseModel):
    emails: list[str] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list)
    addresses: list[str] = Field(default_factory=list)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_schema=ExtractedData,
)

response = agent.run(f"Extract contact info from: {document_text}")
```

#### 分类
```python

from typing import Literal
from pydantic import BaseModel, Field

class Classification(BaseModel):
    category: Literal["spam", "not_spam"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_schema=Classification,
)
```

#### 多项目生成
```python
from pydantic import BaseModel

class BlogPost(BaseModel):
    title: str
    summary: str
    sections: list[str]

class BlogPostList(BaseModel):
    posts: list[BlogPost]

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_schema=BlogPostList,
)

response = agent.run("Generate 3 blog post ideas about AI trends")
for post in response.content.posts:
    print(f"- {post.title}")
```

### 备用方案use_json_mode
对于本身不支持结构化输出的模型，启用 JSON 模式：

```python
agent = Agent(
    model=SomeModel(),
    output_schema=MySchema,
    use_json_mode=True,
)
```


## 多模态输入/输出
向代理人传递图像、音频、视频和文件。

代理可以处理图像、音频、视频和文件作为输入，并生成图像和音频作为输出。

本节介绍多模态 I/O。更多详情请参阅完整指南。

### 媒体课程
| 班级 | 参数 | 
| :--- | :--- | 
| Image | url，filepath，content（字节） |  
| Audio | url，filepath，content（字节）, format |
| Video | url，filepath，content（字节） |
| File | url，filepath，content（字节） |

### 快速入门
选择媒体类型：

通过 URL、文件路径或 base64 内容传递图像：


```python
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIResponses

agent = Agent(model=OpenAIResponses(id="gpt-5.2"))


# From URL
agent.run(

    "What's in this image?",

    images=[Image(url="https://example.com/photo.jpg")]

)

# From file
agent.run(
    "Describe this image",
    images=[Image(filepath="./photo.jpg")]
)

# Multiple images
agent.run(
    "Compare these two images",
    images=[
        Image(url="https://example.com/photo1.jpg"),
        Image(url="https://example.com/photo2.jpg")
    ]
)
```

### 了解更多
更多多模态输入输出示例，请参阅多模态文档：

## 输出模型
使用辅助模型和自定义样式来优化最终输出。

Agno 支持模块化输出模型管道，以便在需要时优化响应。如果需要优化响应或验证来自主模型的响应，请使用辅助模型。

### 工作原理
- 主模型生成响应（可选择将此响应作为中间响应传递给次级模型）
- 可选的辅助模型处理中间响应，格式化并返回最终响应。
- 最终响应还可以选择性地进行自定义样式设置和验证。



### 模型选择
根据三个维度设计您的代理：推理（逻辑）、表现（风格）和结构（模式）。
- 单一模型（model）：根据推理能力选择“执行者”或“管道大脑” （例如，复杂任务选择 GPT-4o，简单任务选择 Claude-3）。
- 具有输出改进的单模型（output_model）：在单模型流程中，根据格式化能力选择“格式化程序”或“流程样式器” （例如，Claude Opus 4.5 用于散文，GPT-5-mini 用于成本优化）。
- 多模型（parser_model）：如果主模型较弱或输出不结构化或需要改进，则添加辅助模型。（例如，对于 OpenAI/Anthropic，可parser_model与智能模型（例如 Claude 4.5 或更高版本）一起使用来“修复”或“提取”所需的输出）。


### 使用以下参数选择所需的输出模型流程：
- model必填，指定生成响应的主要模型
- output_schema用于验证响应的模式
- output_model当使用单个模型时，如果该模型不支持结构化输出，请使用此方法验证响应。
- output_model_prompt：可选，与参数结合使用output_model以指定其他自定义格式来优化输出。
- parser_model：用于进一步优化中间响应以获得更好结果的二级模型
- parser_model_prompt：可选，与输出一起使用parser_model以指定控制parser_model输出的附加指令

### 用例
| 模型设计 | 用例 | 配置 |
| :--- | :--- | :--- |
| 单模型 | 需要提高散文质量 | output＿model＝克劳德作品 4.5 |
| 单模型 | 需要降低成本 | output＿model＝GPT－5－mini |
| 单模型 | 主要模型缺乏结构化输出或缺少所需的格式 | parser＿model＝GPT－4o，output＿schema＝您期望的模式，output＿model＝Claude Opus 4.5 |
| 单模型 | 需要自定义格式样式 | output＿model＝Claude Opus 4．5， output＿model＿prompt＝您所需的格式说明 |
| 单模型 | 简单结构化数据提取 | output＿schema＝您所需的模式 |
| 多模型 | 标准代理 | model＝OpenAlResponses（id＝＂gpt－5．2＂） |
| 多模型 | 严格的 JSON | model＝OpenAIResponses（id＝＂gpt－5．2＂）和 output＿schema＝您所需的模式 |
| 多模型 | 严格 JSON（弱模型） | model＝OpenAIResponses（id＝＂gpt－5．2＂）and parser＿model＝GPT－4o and output＿schema＝您所需的模式 |
| 多模型 | 推理＋优美的散文 | model＝OpenAIResponses（id＝＂gpt－5．2＂）和 output＿model＝Claude Opus 4.5 |
| 多模型 | 成本效益高 | model＝OpenAIResponses（id＝＂gpt－5．2＂）和 output＿model＝GPT－5－mini |


#### output_model_prompt
每当您使用`output_model `设置最终输出的样式、语气和格式时，可以选择使用`output_model_prompt`。 它替换指定`output_model` 的默认系统提示符。

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

# Executive summary style
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_model=Claude(id="claude-sonnet-4-5"),
    output_model_prompt="Format as a concise executive summary. No fluff, just insights.",
    tools=[HackerNewsTools()],
)

# Technical documentation style
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    output_model=Claude(id="claude-sonnet-4-5"),
    output_model_prompt="Format as technical documentation with code examples where relevant.",
    tools=[HackerNewsTools()],
)

```

#### parser_model_prompt
`parser_model_prompt`是可选的。在大多数情况下，默认系统提示对次要模型来说效果良好。每当你使用二级`parser_model`来设置最终输出的风格、语气和格式时，可以选择性地使用它。

```python
from typing import List
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from pydantic import BaseModel, Field
class MovieScript(BaseModel):
    setting: str = Field(..., description="Provide a nice setting for a blockbuster movie.")
    ending: str = Field(..., description="Ending of the movie. If not available, provide the best ending you can think of.")
    genre: str = Field(..., description="Genre of the movie. If not available, select the best genre you can think of.")
    name: str = Field(..., description="Name of the movie")
    characters: List[str] = Field(..., description="Name of characters for this movie.")
    storyline: str = Field(..., description="3 sentence storyline for the movie. Make it exciting!")
# Agent with a parser model + custom parser prompt
agent = Agent(
    model=Ollama(id="llama3.1"),
    description="You are a movie script writer.",
    output_schema=MovieScript,
    parser_model=OpenAIChat(id="gpt-4o"),
    parser_model_prompt="Extract the movie details from the input. Ensure the JSON is valid and matches the MovieScript schema exactly."
)
agent.print_response("New york")

```


### 示例
| 用例 | 例子 |
| :--- | :--- |
| 更好的写作 | 使用 GPT－5．2 进行研究，使用 Claude Opus 4.5 进行写作 |
| 成本优化 | 使用 DeepSeek 进行推理，使用 GPT－5－mini 进行格式化 |
| 结构化输出 | 使用不支持原生支持的模型，并使用支持原生支持的模型进行格式化。 |

### 更好的写作
GPT-5.2 擅长研究和工具使用，但 Claude Opus 4.5 的散文写作能力更强。将它们结合起来：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),       # Research and tool calls
    output_model=Claude(id="claude-opus-4-5"), # Creative writing
    output_model_prompt="Write an engaging, well-structured article based on these findings.",
    tools=[HackerNewsTools()],
)

agent.print_response("Write an article about the latest AI breakthroughs", stream=True)

```

### 更好的写作
GPT-5.2 擅长研究和工具使用，但 Claude Opus 4.5 的散文写作能力更强。将它们结合起来：

```python

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),       # Research and tool calls
    output_model=Claude(id="claude-opus-4-5"), # Creative writing
    output_model_prompt="Write an engaging, well-structured article based on these findings.",
    tools=[HackerNewsTools()],
)

agent.print_response("Write an article about the latest AI breakthroughs", stream=True)
```
主要模型从 HackerNews 收集信息。Claude Opus 4.5 将这些发现转化为润色过的文字。

### 成本优化
对于复杂的推理，使用功能强大但价格昂贵的模型；对于格式化，使用价格较低的模型：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),       # Expensive: complex analysis + tools
    output_model=OpenAIResponses(id="gpt-5-mini"),  # Cheap: just formatting
    output_model_prompt="Summarize the analysis in 3 bullet points.",
    tools=[YFinanceTools()],
)

agent.print_response("Deep analysis of NVDA financials", stream=True)
```

或者使用成本更低、格式更规范的推理模型：

```python
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=DeepSeek(id="deepseek-chat"),        # Cheap: reasoning + tools
    output_model=OpenAIResponses(id="gpt-5.2"), # Better formatting
    tools=[YFinanceTools()],
)

agent.print_response("Analyze AAPL stock performance", stream=True)
```

或者使用成本更低、格式更规范的推理模型：

```python
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIResponses
from agno.tools.yfinance import YFinanceTools

agent = Agent(
    model=DeepSeek(id="deepseek-chat"),        # Cheap: reasoning + tools
    output_model=OpenAIResponses(id="gpt-5.2"), # Better formatting
    tools=[YFinanceTools()],
)

agent.print_response("Analyze AAPL stock performance", stream=True)

```


## 结构化输出支持
某些模型本身不具备结构化输出功能。请使用支持结构化输出的输出模型：

```python
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

class ArticleSummary(BaseModel):
    title: str
    key_points: list[str] = Field(description="3-5 main takeaways")
    sentiment: str = Field(description="positive, negative, or neutral")

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),            # Primary reasoning
    output_model=OpenAIResponses(id="gpt-5.2"),    # Structured output
    output_schema=ArticleSummary,
    tools=[HackerNewsTools()],
)

response = agent.run("Summarize the top AI story on HackerNews")
summary: ArticleSummary = response.content
print(summary.key_points)

```