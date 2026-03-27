# 前钩和后钩

使用钩子在代理运行前后执行自定义逻辑。

你可以使用代理和团队的钩子，在运行的主要执行之前或之后执行工作。

钩子的用途包括：
- 安全防护措施（例如个人身份信息检测、快速注入防御）
- 输入验证
- 输出验证
- 数据预处理（例如，对输入数据进行归一化）
- 数据后处理（例如，向输出添加额外上下文）
- 日志记录（例如记录运行持续时间）
- 调试（例如调试运行过程）

## 当钩子被触发时
钩子会在代理/团队运行生命周期的特定节点执行：

- 预钩子：在当前会话加载完成后立即执行，在任何处理开始之前。它们在模型上下文准备完成之前以及任何 LLM 执行开始之前运行，也就是说，对输入、会话状态或依赖项的任何修改都将在 LLM 执行之前应用。
- 后置钩子：在代理/团队生成响应并准备好输出之后，但在将响应返回给用户之前执行。在流式响应中，它们在生成响应的每个数据块之后运行。

### 预钩
预钩子会在 Agent 运行之初执行，使您可以完全控制到达 LLM 的内容。

它们非常适合对代理接收到的输入进行输入验证、安全检查或任何数据预处理。


### 常见用例
#### 安全护栏
- 检测并阻止个人身份信息 (PII) 到达 LLM。
- 防御快速注入和越狱尝试。
- 过滤不适宜或不雅内容。
- 更多详情请参阅Guardrails文档。
#### 输入验证
- 验证输入文件的格式、长度、内容或任何其他属性。
- 删除或屏蔽敏感信息。
- 对输入数据进行归一化处理。
#### 数据预处理
- 转换输入格式或结构。
- 添加更多上下文信息，丰富输入内容。
- 在将输入发送到 LLM 之前，应用任何其他业务逻辑。

### 基本示例
让我们创建一个简单的预钩子，用于验证输入长度，如果输入过长则引发错误：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.exceptions import CheckTrigger, InputCheckError
from agno.run.agent import RunInput

# Simple function we will use as a pre-hook
def validate_input_length(
    run_input: RunInput,
) -> None:
    """Pre-hook to validate input length."""
    max_length = 1000
    if len(run_input.input_content) > max_length:
        raise InputCheckError(
            f"Input too long. Max {max_length} characters allowed",
            check_trigger=CheckTrigger.INPUT_NOT_ALLOWED,
        )

agent = Agent(
    name="My Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    # Provide the pre-hook to the Agent using the pre_hooks parameter
    pre_hooks=[validate_input_length],
)
```

### 前置参数
预钩子在代理运行期间自动运行，并接收以下参数：
- run_input：代理运行的输入，可以进行验证或修改
- agent：对 Agent 实例的引用
- session当前代理会话
- run_context：当前运行上下文。请参阅运行上下文参考。
- debug_mode是否启用调试模式（可选）

该框架会自动注入你的钩子函数接受的参数，因此你可以只定义需要的参数的钩子。

您可以在“预钩子”参考文档中了解有关参数的更多信息。

### 柱钩
后置钩子会在代理生成响应后执行，允许您在输出到达用户之前对其进行验证、转换或丰富。

它们非常适合用于输出过滤、合规性检查、响应增强或您需要的任何其他输出转换。

### 常见用例
#### 输出验证
- 验证回复格式、长度和内容质量。
- 请从回复中删除敏感或不当信息。
- 确保遵守业务规则和规章制度。
#### 输出转换
- 为回复添加元数据或其他上下文信息。
- 针对不同的客户或使用场景转换输出格式。
- 使用更多数据或格式丰富回复。

### 基本示例
让我们创建一个简单的后置钩子，用于验证输出长度，如果输出过长则引发错误：

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.exceptions import CheckTrigger, OutputCheckError
from agno.run.agent import RunOutput

# Simple function we will use as a post-hook
def validate_output_length(
    run_output: RunOutput,
) -> None:
    """Post-hook to validate output length."""
    max_length = 1000
    if len(run_output.content) > max_length:
        raise OutputCheckError(
            f"Output too long. Max {max_length} characters allowed",
            check_trigger=CheckTrigger.OUTPUT_NOT_ALLOWED,
        )

agent = Agent(
    name="My Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    # Provide the post-hook to the Agent using the post_hooks parameter
    post_hooks=[validate_output_length],
)
```

### 钩子后参数
代理运行时，后置钩子会自动运行，并接收以下参数：
- run_output代理运行的输出，可以进行验证或修改
- agent：对 Agent 实例的引用
- session当前代理会话
- run_context：当前运行上下文。请参阅运行上下文参考。
- user_id运行的用户 ID（可选）
- debug_mode是否启用调试模式（可选）

该框架会自动注入你的钩子函数接受的参数，因此你可以只定义需要的参数的钩子。

您可以在Post-hooks参考文档中了解更多有关参数的信息。

### 护栏
hooks 的一个常见用途是护栏：为您的代理提供内置的安全保障。

您可以在“护栏”部分了解更多相关信息。
​

### @hook装饰师
装饰器允许您配置各个钩子的行为。目前，它支持在与AgentOS@hook一起使用时，将钩子标记为在后台运行

### 后台执行
默认情况下，钩子函数要么同步执行，要么异步执行；在 API 上下文中，它们都会阻塞响应，直到执行完毕。对于执行非关键任务（日志记录、分析、通知）的钩子函数，您可以将其标记为在后台运行：

```python
from agno.hooks import hook

@hook(run_in_background=True)
async def send_notification(run_output, agent):
    """This hook will run in the background without blocking the response."""
    await send_email_notification(run_output.content)
```

### 何时使用背景钩子
背景挂钩非常适合用于：
- 日志记录和分析：记录指标而不影响响应时间
- 通知：发送电子邮件、Slack 消息或 Webhook
- 异步数据存储：写入外部数据库或 API
- 非关键性后处理：不影响响应的任务