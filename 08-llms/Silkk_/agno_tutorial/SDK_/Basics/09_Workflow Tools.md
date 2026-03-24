# 工作流工具

如何在代理或团队中执行工作流

## 例子
您可以使用以下方法将工作流程分配给代理或团队执行WorkflowTools。

```python

from agno.agent import Agent    
from agno.models.openai import OpenAIResponses
from agno.tools.workflow import WorkflowTools

# Create your workflows...

workflow_tools = WorkflowTools(
    workflow=blog_post_workflow,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[workflow_tools],
)

agent.print_response("Create a blog post on the topic: AI trends in 2024", stream=True)
```