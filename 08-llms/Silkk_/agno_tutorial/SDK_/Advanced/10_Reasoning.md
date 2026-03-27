# 推理
## 什么是推理

推理使智能体能够在做出反应之前“思考”，并“分析”其行动（即工具调用）的结果，从而大大提高智能体解决需要顺序工具调用的问题的能力。

想象一下，你让一个普通的AI智能体去解决一个复杂的数学问题、分析一篇科学论文，或者规划一个多步骤的旅行路线。它往往会在没有充分思考问题的情况下匆忙给出答案。结果呢？计算错误、分析不完整，或者计划不合逻辑。

现在想象一下，一个智能体会停下来，一步一步地思考问题，验证自己的推理，发现自身的错误，然后才给出答案。这就是推理的实践，它能将智能体从快速反应者转变为谨慎的问题解决者。

## 为什么推理很重要
缺乏推理能力的智能体难以完成以下任务：

- 多步骤思考——将复杂问题分解为逻辑步骤
- 自我验证——在回复之前检查自己的工作
- 纠错——在过程中发现并纠正错误
- 战略规划——未雨绸缪而非亡羊补牢


例如：问一个普通智能体“9.11 和 9.9 哪个更大？”，它可能会错误地回答 9.11（因为它是逐位比较数字，而不是比较小数部分）。而一个推理智能体会先进行小数比较，从而得出正确的答案。

## 推理是如何运作的
思维链（CoT）：该模型在内部逐步思考问题，将复杂的推理过程分解为逻辑步骤，最终得出答案。推理模型和推理智能体都采用这种方法。

ReAct（推理与行动）：智能体在推理和行动之间交替进行的迭代循环：
- 理由——仔细思考问题，计划下一步行动
- 行动- 采取行动（调用工具、执行计算）
- 观察并分析结果
- 重复——根据新信息继续推理，直到问题解决。

这种模式对于推理工具尤其有用，当智能体需要通过现实世界的反馈来验证假设时，这种模式也很有用。

## 三种推理方法
Agno 提供了三种为智能体添加推理功能的方法，每种方法都适用于不同的使用场景：

1. 推理模型

内容：预先训练的、在回答问题之前能够进行思考的模型（例如 OpenAI gpt-5、Claude 4.5 Sonnet、Gemini 2.0 Flash Thinking、DeepSeek-R1）。
工作原理：模型在生成最终结果之前，会先生成一个内部逻辑推理过程。这一过程发生在模型层：您只需使用模型，推理过程就会自动进行。

最适合：
- 单次解决复杂问题（数学、编程、物理）
- 当你信任模型能够内部处理推理时，就会出现问题。
- 无需控制推理过程的应用场景

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

# Setup your Agent using a reasoning model
agent = Agent(model=OpenAIResponses(id="gpt-5.2"))

# Run the Agent
agent.print_response(
    "Solve the trolley problem. Evaluate multiple ethical frameworks. Include an ASCII diagram of your solution.",
    stream=True,
    show_full_reasoning=True,
)
```

### 推理模型 + 响应模型
这里有一个强大的模式：使用一个模型进行推理（例如 DeepSeek-R1），另一个模型用于生成最终答案（例如 GPT-4o）。为什么呢？推理模型虽然擅长解决问题，但往往会产生机械或过于技术化的回答。通过将推理模型与听起来自然的回答模型相结合，就能获得既准确又流畅的思考结果。

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.groq import Groq

# Setup your Agent using Claude as main model, and DeepSeek as reasoning model
claude_with_deepseek_reasoner = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    reasoning_model=Groq(
        id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
    ),
)

# Run the Agent
claude_with_deepseek_reasoner.print_response(
    "9.11 and 9.9 -- which is bigger?",
    stream=True,
    show_full_reasoning=True,
)
```


2. 推理工具
内容：为任何模型提供明确的思考工具（如草稿纸或记事本），以便逐步解决问题。

工作原理：你提供了像think（）和analyze（）这样的工具，让智能体能够明确地构建其推理过程。代理人调用这些工具来整理思绪，然后再回应。

最适合：
- 为非推理模型（例如常规的 GPT-4o 或 Claude 3.5 Sonnet）添加推理功能
- 当你想了解推理过程时
- 需要结构化思维（研究、分析、计划）的任务

例子：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools

# Setup our Agent with the reasoning tools
reasoning_agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[
        ReasoningTools(add_instructions=True),
    ],
    instructions="Use tables where possible",
    markdown=True,
)

# Run the Agent
reasoning_agent.print_response(
    "Write a report on NVDA. Only the report, no other text.",
    stream=True,
    show_full_reasoning=True,
)
```
欲了解更多有关推理工具的信息，请参阅《推理工具指南》。

### 3. 推理智能体
内容：通过提示工程，利用结构化的思维链处理，将任何常规模型转换为推理系统。

工作原理：可reasoning=True应用于任何智能体。Agno 会创建一个独立的推理智能体，该智能体使用相同的模型（而非不同的模型），但会提供专门的提示，以强制其进行逐步思考、使用工具和进行自我验证。它最适用于非推理模型，例如 gpt-4o 或 Claude Sonnet。对于像 gpt-5-mini 这样的推理模型，通常最好直接使用它们。

最适合：
- 将常规模型转换为推理系统
- 需要多次顺序调用工具的复杂任务
- 当您需要具有迭代和自我纠错功能的自动化思维链时

例子：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

# Transform a regular model into a reasoning system
reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),  # Regular model, not a reasoning model
    reasoning=True,  # Enables structured chain-of-thought
    markdown=True,
)

# The agent will now think step-by-step before responding
reasoning_agent.print_response(
    "Solve the trolley problem. Evaluate multiple ethical frameworks. Include an ASCII diagram of your solution.",
    stream=True,
    show_full_reasoning=True,
)
```

## 选择正确的方法
以下是三种方法的比较：

| 方法 | 透明度 | 最佳用例 | 模型要求 |
| :--- | :--- | :--- | :--- |
| 推理模型 | 连续（完整推理轨迹） | 单次复杂问题 | 需要具备推理能力的模型 |
| 推理工具 | 结构化的（明确的逐步） | 结构化研究与分析 | 适用于任何型号 |
| 推理智能体 | 迭代（代理交互） | 多步骤工具任务 | 适用于任何型号 |

## 推理模型

推理模型是一类大型语言模型，经过预训练后会在回答问题前进行思考。它们在做出反应前会产生一个较长的内部思维链。

推理模型示例包括：
- OpenAI o1-pro 和 gpt-5-mini
- 克劳德 3.7 十四行诗（扩展思维模式）
- 双子座2.0闪思
- DeepSeek-R1

推理模型在采取行动之前会进行深入思考和周密计划。关键在于模型在生成响应之前所做的一切。推理模型擅长处理单次应用场景。它们非常适合解决无需多次迭代或顺序调用工具的难题（例如编程、数学、物理）。

## 示例

​```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

# Setup your Agent using a reasoning model
agent = Agent(model=OpenAIResponses(id="gpt-5.2"))

# Run the Agent
agent.print_response(
    "Solve the trolley problem. Evaluate multiple ethical frameworks. Include an ASCII diagram of your solution.",
    stream=True,
    show_full_reasoning=True,
)
```

### 带有工具的 gpt-5-mini
```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

# Setup your Agent using a reasoning model
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    markdown=True,
)

# Run the Agent
agent.print_response("What is the best basketball team in the NBA this year?", stream=True)
```
### 带有推理努力的 gpt-5-mini

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

# Setup your Agent using a reasoning model with high reasoning effort
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2", reasoning_effort="high"),
    tools=[HackerNewsTools()],
    markdown=True,
)

# Run the Agent
agent.print_response("What is the best basketball team in the NBA this year?", stream=True)
```

### 使用 Groq 的 DeepSeek-R1

```python

from agno.agent import Agent
from agno.models.groq import Groq

# Setup your Agent using a reasoning model
agent = Agent(
    model=Groq(
        id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
    ),
    markdown=True,
)

# Run the Agent
agent.print_response("9.11 and 9.9 -- which is bigger?", stream=True)
```

### 推理模型 + 响应模型
运行上面的 DeepSeek-R1 代理后，你会发现它的响应并不理想。这是因为 DeepSeek-R1 非常擅长解决问题，但在做出自然流畅的响应方面（例如 Claude Sonnet 或 GPT-4.5）却不尽如人意。
为了解决这个问题，Agno 支持使用独立的模型进行推理和生成响应。这种方法利用推理模型解决问题，同时使用针对自然语言响应优化的另一个模型，从而结合了两者的优势。

### DeepSeek-R1 + 克劳德·索内特

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.groq import Groq

# Setup your Agent using an extra reasoning model
deepseek_plus_claude = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    reasoning_model=Groq(
        id="deepseek-r1-distill-llama-70b", temperature=0.6, max_tokens=1024, top_p=0.95
    ),
)

# Run the Agent
deepseek_plus_claude.print_response("9.11 and 9.9 -- which is bigger?", stream=True)
```

### 流式推理内容
使用时reasoning_model，您可以实时流式传输推理内容。这使您可以实时查看模型的思考过程。

为了启用流式推理，运行代理时请设置 stream=True 和 stream_events=True：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude

# Create an agent with a reasoning model
agent = Agent(
    reasoning_model=Claude(
        id="claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 1024},
    ),
    reasoning=True,
    instructions="Think step by step about the problem.",
)

# Stream the response with reasoning events
agent.print_response(
    "What is 25 * 37? Show your reasoning.",
    stream=True,
    stream_events=True,
)
```



### 捕获推理事件
您还可以捕获单个推理事件。这使您可以对推理内容的显示方式进行精细控制：

```python
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.run.agent import RunEvent

agent = Agent(
    reasoning_model=Claude(
        id="claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 1024},
    ),
    reasoning=True,
    instructions="Think step by step about the problem.",
)

for run_output_event in agent.run(
    "What is 25 * 37? Show your reasoning.",
    stream=True,
    stream_events=True,
):
    if run_output_event.event == RunEvent.run_started:
        print(f"EVENT: {run_output_event.event}")
    elif run_output_event.event == RunEvent.reasoning_started:
        print(f"EVENT: {run_output_event.event}")
        print("Reasoning started...\n")
    elif run_output_event.event == RunEvent.reasoning_content_delta:
        # Stream reasoning content as it's being generated
        print(run_output_event.reasoning_content, end="", flush=True)
    elif run_output_event.event == RunEvent.run_content:
        if run_output_event.content:
            print(run_output_event.content, end="", flush=True)
    elif run_output_event.event == RunEvent.run_completed:
        print(f"EVENT: {run_output_event.event}")
```

流式推理的关键事件包括：

| 事件 | 描述 |
| :--- | :--- |
| RunEvent．reasoning＿started | 推理开始时发出 |
| RunEvent．reasoning＿content＿delta | 每当推理内容流式传输时，都会发出该信号。 |
| RunEvent．run＿content | 为最终响应内容发出 |


## 推理工具

为任何模型提供明确的结构化思维工具，通过深思熟虑的推理步骤，将普通模型转变为严谨的问题解决者。

问题：推理智能体强制对每个请求进行系统性思考。推理模型需要专门的模型。如果您只想在需要时进行推理，并针对特定上下文进行调整，该怎么办？ 解决方案：推理工具为您的智能体提供明确的think()工具analyze()，并让智能体决定何时使用这些工具。智能体选择何时进行推理、何时采取行动以及何时拥有足够的信息来做出响应。
Agno 提供四款专门的推理工具包，每款工具包都针对不同的领域进行了优化：

| 工具包 | 目的 | 核心工具 |
| :--- | :--- | :--- |
| 推理工具 | 通用思维和分析 | think（），analyze（） |
| 知识工具 | 利用知识库搜索进行推理 | think（），，search＿knowledge（）analyze（） |
| 内存工具 | 对用户内存操作的推理 | think（），，get／add／update／delete＿memory（） analyze（） |
| 工作流程工具 | 关于工作流执行的推理 | think（），，run＿workflow（）analyze（） |


> 注意：所有推理工具包都以相同名称注册其 think（）/analyze（） 函数。当你合并工具包时，代理只保留每个函数名称的第一个实现，并且会悄无声息地丢弃重复的。如果你仍希望它们在不与 scratchpad 工具冲突的情况下，可以在后期工具包中禁用 enable_think/enable_analyze（或重命名/自定义功能）。


这四个工具包都遵循相同的“思考→行动→分析”模式，但提供了针对其用例量身定制的特定领域操作。

这种方法最初是由 Anthropic 在其“扩展思维”博客文章中推广的，尽管许多人工智能工程师（包括我们的团队）早在很久以前就使用了类似的模式。

## 为什么需要推理工具？
推理工具能让你兼得两者之长：
- 适用于任何模型——即使是那些本身不具备推理能力的模型。
- 显式控制——智能体决定何时思考，何时行动。
- 完全透明——您可以清楚地看到经纪人的想法。
- 灵活的工作流程——代理可以在思考和工具调用之间交替进行。
- 针对特定领域优化——每个工具包都针对其特定用例进行了专门设计。
- 自然推理——感觉更像是人类解决问题的方式（思考、行动、分析、重复）

主要区别在于：推理代理会在结构化的循环中自动进行推理。而推理工具则由代理显式地选择何时使用哪些think()工具analyze()，从而赋予您更大的控制权和更清晰的可见性。
​

### 四种推理工具包
​
1. 推理工具 - 通用思维
用于解决一般问题，无需特定领域的工具。

它提供的功能：
- think()- 计划并分析问题
- analyze()- 评估结果并确定后续步骤

何时使用：
- 数学或逻辑问题
- 战略规划
- 不需要外部数据的分析任务
- 任何需要结构化推理的场景

### 例子：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.reasoning import ReasoningTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[ReasoningTools(add_instructions=True)],
)

agent.print_response(
    "Which is bigger: 9.11 or 9.9? Explain your reasoning.",
    stream=True,
)
```

### 2. KnowledgeTools - 基于知识库的推理
用于从知识库（RAG）中搜索和分析信息。

它提供的功能：
- think()- 制定搜索策略并完善方法
- search_knowledge()- 查询知识库
- analyze()- 评估搜索结果的相关性和完整性

何时使用：
- 文档检索与分析
- RAG（检索增强生成）工作流程
- 需要多次搜索迭代的研究任务
- 当您需要验证知识库中的信息时

例子：

```python
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.openai import OpenAIResponses
from agno.tools.knowledge import KnowledgeTools
from agno.vectordb.pgvector import PgVector

# Create knowledge base
knowledge = PDFKnowledgeBase(
    path="data/research_papers/",
    vector_db=PgVector(
        table_name="research_papers",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[KnowledgeTools(knowledge=knowledge, add_instructions=True)],
    instructions="Search thoroughly and cite your sources",
)

agent.print_response(
    "What are the latest findings on quantum entanglement in our research papers?",
    stream=True,
)
```

工作原理：
- 代理人来电think()：“我需要搜索量子纠缠。让我尝试几个搜索词。”
- 代理人致电search_knowledge("quantum entanglement")
- 经纪人来电analyze()：“搜索结果范围太广，需要更具体的搜索条件。”
- 代理人致电search_knowledge("quantum entanglement recent findings")
- 代理人来电analyze()：“我现在有足够且相关的结果了。”
- 代理人提供最终答案

### 3. MemoryTools - 用户记忆推理
用于管理和推理具有 CRUD 操作的用户内存。
它提供的功能：
- think()- 计划内存操作
- get_memories()- 检索用户记忆
- add_memory()- 存储新的记忆
- update_memory()- 修改现有内存
- delete_memory()- 删除记忆
- analyze()- 评估内存操作

何时使用：
- 个性化代理互动
- 用户偏好管理
- 在不同会话中保持对话上下文
- 随着时间推移构建用户画像

例子：

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.tools.memory import MemoryTools

db = PostgresDb(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[MemoryTools(db=db, add_instructions=True)],
    db=db,
)

agent.print_response(
    "I prefer vegetarian recipes and I'm allergic to nuts.",
    user_id="user_123",
)
```

#### 工作原理：
- 代理呼叫think()：“用户正在分享饮食偏好。我应该存储此信息。”
- 代理人致电add_memory(memory="User prefers vegetarian recipes and is allergic to nuts", topics=["dietary_preferences", "allergies"])
- 代理呼叫analyze()：“内存已成功存储，并包含相应的主题。”
- 代理回复用户，确认信息已保存。


### 4. WorkflowTools - 工作流执行推理
用于执行和分析复杂的工作流程。
它提供的功能：
- think()- 规划工作流程输入和策略
- run_workflow()- 使用特定输入执行工作流程
- analyze()- 评估工作流程结果

何时使用：
- 多步骤自动化流程
- 复杂任务编排
- 当工作流程需要根据上下文进行不同的输入时
- A/B 测试不同的工作流程配置

例子：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.workflow import WorkflowTools
from agno.workflow import Workflow
from agno.workflow.step import Step

# Define a research workflow
research_workflow = Workflow(
    name="research-workflow",
    steps=[
        Step(name="search", agent=search_agent),
        Step(name="summarize", agent=summary_agent),
        Step(name="fact-check", agent=fact_check_agent),
    ],
)

# Create agent with workflow tools
orchestrator = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[WorkflowTools(workflow=research_workflow, add_instructions=True)],
)

orchestrator.print_response(
    "Research climate change impacts on agriculture",
    stream=True,
)
```

### 常见模式：思考→行动→分析
这四款工具包都遵循相同的推理循环：

1. 思考——计划要做什么，完善方案，集思广益。    
2. ACT（领域特定）    
- 推理工具：直接推理
- 知识工具：search_knowledge()
- 内存工具：get/add/update/delete_memory()
- 工作流程工具：run_workflow()

3. 分析- 评估结果，决定下一步行动   
4. 重复- 如有需要，返回“思考”阶段，或提供答案   

这与人类解决复杂问题的方式类似：我们先思考再行动，评估结果，并根据所学到的知识调整我们的方法。

### 选择合适的推理工具包
| 如果你需要．．．．．． | 使用 | 例子 |
| :--- | :--- | :--- |
| 解决逻辑谜题或数学问题 | ReasoningTools | ＂解：如果 $x^2+5 x+6=0$ ，那么 $x$ 是多少？＂ |
| 搜索文档 | KnowledgeTools | ＂在我们的文档中查找所有提及用户身份验证的内容＂ |
| 记住用户偏好 | MemoryTools | ＂记住，我对贝类过敏。＂ |
| 协调复杂的多步骤任务 | WorkflowTools | ＂研究、撰写并核实文章的事实＂ |
| 合并多个领域 | 使用多个工具包 | 请参阅示例以了解更多模式 |

### 整合多种推理工具包
您可以同时使用多个推理工具包，实现强大的多领域推理。但请记住，工具名称必须保持唯一，因此请禁用重叠的think/analyze条目（或重命名后面的条目）以防止静默覆盖：

```python
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.openai import OpenAIResponses
from agno.tools.knowledge import KnowledgeTools
from agno.tools.memory import MemoryTools
from agno.tools.reasoning import ReasoningTools

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[
        ReasoningTools(add_instructions=True),
        KnowledgeTools(
            knowledge=my_knowledge,
            enable_think=False,
            enable_analyze=False,
            add_instructions=False,
        ),
        MemoryTools(
            db=my_db,
            enable_think=False,
            enable_analyze=False,
            add_instructions=False,
        ),
    ],
    instructions="Use reasoning for planning, knowledge for facts, and memory for personalization",
)

```


在这种配置下：
- ReasoningTools提供共享think/analyze草稿纸。
- KnowledgeTools仍然公开search_knowledge()（以及任何其他独特方法），而不尝试注册重复的临时函数。
- MemoryTools贡献了 CRUD 内存工具，同时继承了相同的核心思维循环。
如果您需要每个域的单独临时文件，请在think()/周围创建自定义包装器analyze()，以便每个工具包注册唯一命名的函数（例如knowledge_think，，memory_analyze）。

### 配置选项
​
启用/禁用特定工具

您可以控制哪些推理工具可用：

```python
# Only thinking, no analysis
ReasoningTools(enable_think=True, enable_analyze=False)

# Only analysis, no thinking
ReasoningTools(enable_think=False, enable_analyze=True)

# Both (default)
ReasoningTools(enable_think=True, enable_analyze=True)

# Shorthand for both
ReasoningTools()

```

### 自动添加指令
许多工具包都附带预先编写好的指南，解释如何使用其工具。设置add_instructions=True会将这些说明注入到代理提示符中（如果工具包确实包含这些说明的话）：

```python
ReasoningTools(add_instructions=True)
```

- ReasoningTools、KnowledgeTools、MemoryTools 和 WorkflowTools 都包含 Agno 编写的说明（以及可选的少数示例），描述其“思考 → 行动 → 分析”工作流程。
- 其他工具包可能没有定义默认指令； 在这种情况下 add_instructions=True 是无操作的，除非您提供自己的 instructions=....

内置说明涵盖了何时使用 think（） 与 analyze（）、如何迭代以及每个领域的最佳实践。除非你打算提供定制指导，否则就开启它们。


### 添加少量示例


想向您的代理人展示一些良好推理的例子吗？ 一些工具包附带了预先编写的一些示例，可以演示实际的工作流程。 使用 `add_few_shot=True` 打开它们：

```python
ReasoningTools(add_instructions=True, add_few_shot=True)
```

这些例子向智能体展示了如何迭代解决问题、决定下一步行动，以及如何将思考与实际工具调用结合起来。

什么时候应该使用它们？

- 你使用的是较小或较便宜的型号，需要额外的指导。
- 您的推理工作流程包含多个阶段或非常复杂。
- 您希望不同运行结果之间具有更一致的行为。

### 自定义说明
请提供您自己的自定义指令以进行特殊推理：
```python
custom_instructions = """
Use the think and analyze tools for rigorous scientific reasoning:
- Always think before making claims
- Cite evidence in your analysis
- Acknowledge uncertainty
- Consider alternative hypotheses
"""

ReasoningTools(
    instructions=custom_instructions,
    add_instructions=False  # Don't include default instructions
)
```

### 自定义少镜头示例
您还可以编写针对您所在领域的自定义示例：

```python
medical_examples = """
Example: Medical Diagnosis

User: Patient has fever and cough for 3 days.

Agent thinks:
think(
    title="Gather Symptoms",
    thought="Need to collect all symptoms and their duration. Fever and cough suggest respiratory infection. Should check for other symptoms.",
    action="Ask about additional symptoms",
    confidence=0.9
)
"""

ReasoningTools(
    add_instructions=True,
    add_few_shot=True,
    few_shot_examples=medical_examples  # Your custom examples
)
```

### 监控你的经纪人的想法
使用`show_full_reasoning=True`此`stream_events=True`功能可实时显示推理步骤。有关详细信息，请参阅“推理代理中的显示选项”；有关以编程方式访问推理步骤的信息，请参阅“推理参考” 。


### 推理工具与推理智能体
两种方法都能为任何模型增加推理能力，但它们在控制和自动化方面有所不同：
| 方面 | 推理工具 | 推理智能体 |
| :--- | :--- | :--- |
| 激活 | 代理人决定何时使用 think（） | 每次请求都会自动执行 |
| 控制 | 显式工具调用 | 自动循环 |
| 透明度 | 看看每一个 think（），然后 analyze（）打电话 | 参见结构化推理步骤 |
| 工作流程 | 代理驱动（灵活） | 框架驱动（结构化） |
| 最适合 | 研究、分析、探索性任务 | 具有明确结构的复杂多步骤问题 |

经验法则：
- 当您希望智能体控制自身的推理过程时，请使用推理工具。
- 当您希望确保对每个请求都进行系统性思考时，请使用推理代理。

## 推理智能体

通过结构化的思维链处理，将任何模型转化为推理系统，非常适合需要多个步骤、工具使用和自我验证的复杂问题。

- 问题在于：常规模型在处理复杂问题时常常急于给出答案，忽略步骤或犯逻辑错误。
- 解决方案：启用reasoning=True并观察您的模型如何分解问题、探索多种方法、验证结果并提供经过全面验证的解决方案。
- 它的优点在于：它适用于任何模型，从 GPT-4o 到 Claude，再到通过 Ollama 连接的本地模型。你不再局限于特定的推理模型。
​

### 工作原理
通过设置以下reasoning=True选项启用任何代理的推理功能：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),  # Any model works
    reasoning=True,
)
```

在后台，Agno 会创建一个独立的推理代理实例，该实例使用相同的模型，但会提供专门的提示，引导其完成严格的 6 步推理框架：


## 推理框架
1. 问题分析
- 重述任务以确保完全理解
- 确定所需信息和必要工具
2. 分解与策略制定
- 将问题分解成子任务
- 制定多种不同的方法
3. 意图澄清与规划
- 阐明用户意图
- 选择最佳策略并给出明确理由。
- 制定详细的行动计划
4. 执行行动计划
- 每个步骤包含以下内容：文档标题、操作、结果、理由、下一步操作和置信度评分
- 根据需要调用工具收集信息
- 检测到错误时自动纠正
5. 验证（必填）
- 与其他方法进行交叉验证
- 使用其他工具来确认准确性
- 如果验证失败，则重置并修改。
6. 最终答案
- 交付经过全面验证的解决方案
- 解释它如何解决最初的任务


推理代理会迭代地执行这些步骤（默认最多 10 个步骤），在前期结果的基础上不断迭代，调用工具并进行自我纠错，直到找到一个可靠的解决方案。完成后，它会将完整的推理过程返回给主代理，由其给出最终响应。

### 不同型号之间的区别
使用常规模型（gpt-4o、Claude Sonnet、Gemini）：
- 通过六步框架构建思维链的力量
- 创建带有置信度评分的详细推理步骤
- 这就是推理智能体的优势所在：将任何模型转化为推理系统。

使用原生推理模型（gpt-5-mini、DeepSeek-R1、o3-mini）：
- 利用模型内置的推理能力
- 添加来自您主代理的验证通行证
- 对于关键任务很有用，但对于更简单的问题来说，往往是不必要的开销。

### 基本示例
让我们把一个普通的 GPT-4o 模型转换成一个推理系统：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

# Transform a regular model into a reasoning system
reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    reasoning=True,
    markdown=True,
)

reasoning_agent.print_response(
    "Solve the trolley problem. Evaluate multiple ethical frameworks. Include an ASCII diagram of your solution.",
    stream=True,
    show_full_reasoning=True,  # Shows the complete reasoning process
)
```


你会看到什么
有了它show_full_reasoning=True，你就会看到：

- 每个推理步骤及其标题、操作和结果
- 代理人的思考过程，包括它选择每种方法的原因。
- 推理过程中调用的工具（如果提供了工具）
- 执行验证检查以确认解决方案
- 每一步的置信度得分（0.0–1.0）
- 如果代理检测到错误，则会进行自我纠正。
- 您的主要代理人最终给出的正式回复

### 运用工具进行推理
推理智能体的真正优势在于：将多步骤推理与工具使用相结合。推理智能体可以迭代地调用工具，分析结果，并逐步构建出全面的解决方案。

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.hackernews import HackerNewsTools

reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    instructions=["Use tables to display data"],
    reasoning=True,
    markdown=True,
)

reasoning_agent.print_response(
    "Compare the market performance of NVDA, AMD, and INTC over the past quarter. What are the key drivers?",
    stream=True,
    show_full_reasoning=True,
)
```

推理代理将：
- 将任务分解（需要3家公司的股票数据）
- 使用 HackerNews 搜索最新科技新闻
- 分析各公司的业绩
- 搜索有关关键驱动因素的新闻
- 从多个来源验证研究结果
- 创建包含表格的全面对比。
- 提供最终答案，并阐明观点。
​
### 配置选项
​
显示选项

想一窥究竟吗？控制推理过程中看到的内容：

```python
agent.print_response(
    "Your question",
    show_full_reasoning=True,  # Display complete reasoning process (default: False)
)
```

### 捕获推理事件
要构建自定义用户界面或以编程方式跟踪推理进度，您可以在流式传输过程中捕获推理事件（ReasoningStarted、ReasoningCompleted、ReasoningStep ）。有关事件属性和完整的代码示例，请参阅推理参考文档。

### 迭代控制
调整智能体执行的推理步骤数：

```python
reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    reasoning=True,
    reasoning_min_steps=2,  # Minimum reasoning steps (default: 1)
    reasoning_max_steps=15,  # Maximum reasoning steps (default: 10)
)
```

### 自定义推理代理
对于高级用例，您可以提供自己的推理代理：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

# Create a custom reasoning agent with specific instructions
custom_reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=[
        "Focus heavily on mathematical rigor",
        "Always provide step-by-step proofs",
    ],
)

main_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    reasoning=True,
    reasoning_agent=custom_reasoning_agent,  # Use your custom agent
)
```

### 示例用例

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses

task = (
    "Three missionaries and three cannibals need to cross a river. "
    "They have a boat that can carry up to two people at a time. "
    "If, at any time, the cannibals outnumber the missionaries on either side of the river, the cannibals will eat the missionaries. "
    "How can all six people get across the river safely? Provide a step-by-step solution and show the solution as an ASCII diagram."
)

reasoning_agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    reasoning=True,
    markdown=True,
)

reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)
```

### 何时使用推理智能体
在以下情况下使用推理智能体：
- 您的任务需要多个连续步骤。
- 你需要代理人反复调用各种工具，并根据结果不断改进。
- 你想要的是无需手动调用推理工具就能自动生成思路链。
- 你需要自我验证和纠错。
- 在最终确定解决方案之前，探索多种方法对解决这个问题大有裨益。

在以下情况下考虑其他方案：
- 你使用的是原生推理模型（gpt-5-mini，DeepSeek-R1）来处理简单的任务：直接使用该模型即可。
- 如果您希望明确控制智能体何时思考何时行动：请使用推理工具。
- 这项任务很简单，不需要多步骤思考。
