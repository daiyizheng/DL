# 对话式工作流程
在 Agno 中构建多轮对话工作流程。

如果您的用户直接与工作流交互，通常将其设置为对话式工作流会很有用。这样，用户就可以像与客服人员或团队互动一样与工作流进行对话。

WorkflowAgent此功能允许您在工作流程中添加一个智能组件，该组件可以自动决定是否执行以下操作：

1. 直接根据当前输入和以往工作流程结果作答。    
2. 当输入内容无法根据以往结果得到解答时，运行工作流。   


> 什么是WorkflowAgent？
> Workflow Agent这是一个专为工作流编排而设计的受限版本。

## 快速入门

这就是你可以将 WorkflowAgent 添加到工作流程的方式：

```python
from agno.workflow import WorkflowAgent
from agno.workflow.workflow import Workflow
from agno.models.openai import OpenAIResponses

workflow_agent = WorkflowAgent(
    model=OpenAIResponses(id="gpt-5.2"),  # Set the model that should be used
    num_history_runs=4  # How many of the previous runs should it take into account
)

workflow = Workflow(
    name="Story Generation Workflow",
    description="A workflow that generates stories, formats them, and adds references",
    agent=workflow_agent,
)
```

## 结构
<img src="https://mintcdn.com/agno-v2/zKmlURgt8K26VBJI/images/workflow-agent-flow-light.png?w=2500&fit=max&auto=format&n=zKmlURgt8K26VBJI&q=85&s=56702e0fdd18e1f8946c2f216b24e1e2" width="500">


## 对话式工作流的工作流历史记录：
与步骤的工作流历史记录类似，用户WorkflowAgent可以访问当前会话中所有工作流运行的完整历史记录。这使得回答有关先前结果的问题、比较多次运行的输出以及保持对话的连续性成为可能。


如何控制工作流代理可以看到的先前运行次数？

该num_history_runs参数控制代理在做出决策时可以看到多少个先前的工作流运行结果。这一点至关重要：
- 上下文感知：智能体需要查看过去的运行记录才能回答后续问题。
- 内存限制：运行次数过多可能会超出模型上下文窗口。
- 性能：运行次数越少，处理速度越快，输入标记越少。


## WorkflowAgent 使用说明：
您可以提供自定义指令来WorkflowAgent控制其行为。虽然系统提供了默认指令，指示代理直接从历史记录中响应或在需要新处理时运行工作流，但您可以通过提供自己的指令来覆盖这些默认指令。
```python
workflow_agent = WorkflowAgent(
    model=OpenAIResponses(id="gpt-5.2"),
    num_history_runs=4,
    instructions="You are a helpful assistant that can answer questions and run workflows when new processing is needed.",
)
```

## 使用示例

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.workflow import WorkflowAgent
from agno.workflow.types import StepInput
from agno.workflow.workflow import Workflow

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


story_writer = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are tasked with writing a 100 word story based on a given topic",
)

story_formatter = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    instructions="You are tasked with breaking down a short story in prelogues, body and epilogue",
)


def add_references(step_input: StepInput):
    """Add references to the story"""

    previous_output = step_input.previous_step_content

    if isinstance(previous_output, str):
        return previous_output + "\n\nReferences: https://www.agno.com"


# Create a WorkflowAgent that will decide when to run the workflow
workflow_agent = WorkflowAgent(model=OpenAIResponses(id="gpt-5.2"), num_history_runs=4)

# Create workflow with the WorkflowAgent
workflow = Workflow(
    name="Story Generation Workflow",
    description="A workflow that generates stories, formats them, and adds references",
    agent=workflow_agent,
    steps=[story_writer, story_formatter, add_references],
    db=PostgresDb(db_url),
)

# First call - will run the workflow (new topic)
workflow.print_response(
    "Tell me a story about a dog named Rocky", stream=True
)

# Second call - will answer directly from history
workflow.print_response(
    "What was Rocky's personality?", stream=True
)

# Third call - will run the workflow (new topic)
workflow.print_response(
    "Now tell me a story about a cat named Luna", stream=True
)

# Fourth call - will answer directly from history
workflow.print_response(
    "Compare Rocky and Luna", stream=True
)
```
