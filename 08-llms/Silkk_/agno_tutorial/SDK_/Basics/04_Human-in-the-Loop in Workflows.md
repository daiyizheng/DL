# 人机交互
## 工作流中的人机交互

暂停工作流程执行，以便用户在步骤级别进行确认、输入或决策。

工作流中的人机交互（HITL）功能允许您在任何步骤暂停执行，以便收集用户的确认、输入或决策。工作流状态会被持久化，从而允许您在用户响应后恢复执行。

> 目前支持用户输入，用于 Step（收集参数）和 Router（选择路由）。其他原语（Condition、Loop、Steps）仅支持确认。
> 代理工具级的HITL（例如，@tool（requires_confirmation=True））不会传播到工作流程中。如果某个步骤中的代理有工具级HITL，工作流程会继续，但暂停的工具可能无法执行。改用工作流程层面的HITL（Step.requires_confirmation）。

```python
from agno.workflow import Workflow, OnReject
from agno.workflow.step import Step
from agno.db.sqlite import SqliteDb

workflow = Workflow(
    name="data_pipeline",
    db=SqliteDb(db_file="workflow.db"),  # Required for HITL
    steps=[
        Step(name="fetch_data", agent=fetch_agent),
        Step(
            name="process_data",
            agent=process_agent,
            requires_confirmation=True,
            confirmation_message="Process sensitive data?",
            on_reject=OnReject.skip,
        ),
        Step(name="save_results", agent=save_agent),
    ],
)

run_output = workflow.run("Process user data")

if run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        req.confirm()  # or req.reject()
    run_output = workflow.continue_run(run_output)
```

#### 要求
HITL 工作流需要一个数据库来保存暂停期间的状态：

```python
from agno.db.sqlite import SqliteDb
from agno.db.postgres import PostgresDb

# SQLite for development
workflow = Workflow(db=SqliteDb(db_file="workflow.db"), ...)

# PostgreSQL for production
workflow = Workflow(db=PostgresDb(db_url="postgresql://..."), ...)
```

#### HITL 类型

| 类型 | 用例 | 旗帜 |
| :--- | :--- | :--- |
| 确认 | 步骤执行前批准／拒绝 | requires＿confirmation＝True |
| 用户输入 | 步骤执行前收集参数 | requires＿user＿input＝True |
| 路线选择 | 用户选择要执行的路径。 | 带路由器的 requires＿user＿input＝True |
| 错误处理 | 重试或跳过失败的步骤 | on＿error＝OnError．pause |

#### 支持的基本元素
所有工作流原语都支持 HITL：

| 原始 | 确认 | 用户输入 | 路线选择 |
| :--- | :--- | :--- | :--- |
| 步 | ✓ | ✓ | － |
| 步骤 | ✓ | － | － |
| 健康）状况 | ✓ | － | － |
| 环形 | ✓ | － | － |
| 路由器 | ✓ | － | ✓ |

#### 运行输出属性
当工作流暂停时，请检查以下属性WorkflowRunOutput：

| 财产 | 描述 |
| :--- | :--- |
| is＿paused | True 如果工作流正在等待用户操作 |
| steps＿requiring＿confirmation | 需要确认／拒绝的步骤 |
| steps＿requiring＿user＿input | 需要用户输入值的步骤 |
| steps＿requiring＿route | 需要进行路由选择的路由器 |
| steps＿with＿errors | 失败的步骤 on＿error＝＂pause＂ |

#### 确认
执行步骤之前暂停。 用户确认继续或拒绝跳过/取消。

```python
Step(
    name="delete_records",
    agent=delete_agent,
    requires_confirmation=True,
    confirmation_message="Delete 1000 records?",
    on_reject=OnReject.skip,  # cancel | skip (default)
)
```


在你的代码中处理：


```python
for req in run_output.steps_requiring_confirmation:
    print(req.confirmation_message)
    if user_approves():
        req.confirm()
    else:
        req.reject()
```

#### 用户输入
步骤执行前，从用户处收集参数。

```python
from agno.workflow.types import UserInputField

Step(
    name="generate_report",
    agent=report_agent,
    requires_user_input=True,
    user_input_message="Configure report settings:",
    user_input_schema=[
        UserInputField(name="format", field_type="str", required=True),
        UserInputField(name="include_charts", field_type="bool", required=False),
    ],
)
```

在你的代码中处理：

```python
for req in run_output.steps_requiring_user_input:
    print(req.user_input_message)
    for field in req.user_input_schema:
        value = get_user_value(field.name, field.field_type)
        req.set_user_input(**{field.name: value})
```

#### 路线选择
允许用户选择路由器执行的路径。

```python
from agno.workflow.router import Router

Router(
    name="analysis_router",
    choices=[
        Step(name="quick_analysis", ...),
        Step(name="deep_analysis", ...),
    ],
    requires_user_input=True,
    user_input_message="Select analysis type:",
    allow_multiple_selections=False,
)
```

在你的代码中处理：
```python   
for req in run_output.steps_requiring_route:
    print(req.available_choices)  # ["quick_analysis", "deep_analysis"]
    req.select("deep_analysis")
    # or req.select_multiple(["quick_analysis", "deep_analysis"])
```

#### 错误处理
当某个步骤失败时暂停，允许用户重试或跳过。 This is only at the Step level.
```python
from agno.workflow import OnError

Step(
    name="api_call",
    executor=unreliable_function,
    on_error=OnError.pause,  # fail | skip(default) | pause
)

```

在你的代码中处理：
```python
for req in run_output.steps_with_errors:
    print(f"Error: {req.error_message}")
    if should_retry():
        req.retry()
    else:
        req.skip()

```

#### 拒绝行为
该on_reject参数控制用户拒绝某个步骤时会发生什么：

| 价值 | 行为 |
| :--- | :--- |
| OnReject．skip | 跳过此步骤，继续执行下一步（大多数基本运算的默认行为） |
| OnReject．cancel | 取消整个工作流程 |
| OnReject．else＿branch | 仅适用于条件：执行 else＿steps（条件的默认值） |


#### 流媒体
HITL 适用于流式工作流。请检查事件流中是否存在暂停：

```python
for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        # Handle pause
        pass

session = workflow.get_session()
run_output = session.runs[-1]

if run_output.is_paused:
    # Handle requirements
    workflow.continue_run(run_output, stream=True, stream_events=True)
```

#### @pause 装饰器
使用@pause装饰器标记带有 HITL 配置的自定义函数步骤：

```python
from agno.workflow.decorators import pause
from agno.workflow.types import UserInputField

@pause(
    requires_user_input=True,
    user_input_message="Enter parameters:",
    user_input_schema=[
        UserInputField(name="threshold", field_type="float", required=True),
    ],
)
def process_data(step_input: StepInput) -> StepOutput:
    threshold = step_input.additional_data["user_input"]["threshold"]
    return StepOutput(content=f"Processed with threshold {threshold}")

# The decorator config is auto-detected when used in a custom function step
Step(name="process", executor=process_data)
```

## 步骤 HITL

在执行之前，暂停各个步骤以供确认或用户输入。

步骤支持两种 HITL 模式：确认（批准/拒绝）和用户输入（收集参数）。

### 确认
执行步骤前暂停。用户确认继续或拒绝跳过/取消。

```python
from agno.workflow import Workflow, OnReject
from agno.workflow.step import Step
from agno.db.sqlite import SqliteDb

workflow = Workflow(
    name="data_pipeline",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="fetch_data", agent=fetch_agent),
        Step(
            name="process_data",
            agent=process_agent,
            requires_confirmation=True,
            confirmation_message="Process sensitive data?",
            on_reject=OnReject.skip,
        ),
        Step(name="save_results", agent=save_agent),
    ],
)

run_output = workflow.run("Process user data")

if run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        print(f"Step: {req.step_name}")
        print(f"Message: {req.confirmation_message}")
        
        if input("Confirm? (y/n): ").lower() == "y":
            req.confirm()
        else:
            req.reject()
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```


#### 参数

| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿confirmation | bool | 执行前暂停等待用户确认 |
| confirmation＿message | str | 向用户显示的消息 |
| on＿reject | OnReject | 被拒绝时的操作：（ skip 默认），cancel |

### 拒绝选项

| 价值 | 行为 |
| :--- | :--- |
| OnReject．skip | 跳过此步骤，继续执行下一步（默认）。 |
| OnReject．cancel | 取消整个工作流程 |


### 用户输入
步骤执行前，从用户收集参数。输入值通过以下step_input.additional_data["user_input"]方式传递给步骤

```python
from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput, UserInputField
from agno.db.sqlite import SqliteDb

def process_with_params(step_input: StepInput) -> StepOutput:
    user_input = step_input.additional_data.get("user_input", {})
    threshold = user_input.get("threshold", 0.5)
    mode = user_input.get("mode", "fast")
    
    return StepOutput(content=f"Processed with threshold={threshold}, mode={mode}")

workflow = Workflow(
    name="configurable_pipeline",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="analyze", agent=analyze_agent),
        Step(
            name="process",
            executor=process_with_params,
            requires_user_input=True,
            user_input_message="Configure processing:",
            user_input_schema=[
                UserInputField(
                    name="threshold",
                    field_type="float",
                    description="Processing threshold (0.0-1.0)",
                    required=True,
                ),
                UserInputField(
                    name="mode",
                    field_type="str",
                    description="Mode: 'fast' or 'accurate'",
                    required=True,
                ),
                UserInputField(
                    name="batch_size",
                    field_type="int",
                    description="Records per batch",
                    required=False,
                ),
            ],
        ),
        Step(name="report", agent=report_agent),
    ],
)

run_output = workflow.run("Process Q4 data")

if run_output.is_paused:
    for req in run_output.steps_requiring_user_input:
        print(f"Step: {req.step_name}")
        print(f"Message: {req.user_input_message}")
        
        values = {}
        for field in req.user_input_schema:
            marker = "*" if field.required else ""
            prompt = f"{field.name}{marker} ({field.field_type}): "
            value = input(prompt)
            
            # Convert to appropriate type
            if value:
                if field.field_type == "int":
                    values[field.name] = int(value)
                elif field.field_type == "float":
                    values[field.name] = float(value)
                elif field.field_type == "bool":
                    values[field.name] = value.lower() in ("true", "yes", "1")
                else:
                    values[field.name] = value
        
        req.set_user_input(**values)
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```



#### 参数
| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿user＿input | bool | 暂停以收集用户输入，然后再执行 |
| user＿input＿message | str | 向用户显示的消息 |
| user＿input＿schema | List［UserInputField］ | 定义预期输入字段的模式 |

#### 用户输入字段

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| name | str | 字段名称（用户输入字典中的键） |
| field＿type | str | 类型：str ，int ，float ，bool |
| description | str | 向用户显示的描述 |
| required | bool | 是否必填字段（默认值 True ：） |
| allowed＿values | List［Any］ | 可选的有效值列表 |


### 访问用户输入
用户可通过以下方式step_input.additional_data["user_input"]在步进函数中输入：

```python
def my_step(step_input: StepInput) -> StepOutput:
    user_input = step_input.additional_data.get("user_input", {})
    
    threshold = user_input.get("threshold")
    mode = user_input.get("mode")
    
    # Process with user-provided values
    return StepOutput(content=f"Done with {threshold}, {mode}")
```
对于基于代理的步骤，用户输入会自动附加到消息中。
​
### @pause 装饰器
使用@pause装饰器直接在函数上配置 HITL：

```python
from agno.workflow.decorators import pause
from agno.workflow.types import StepInput, StepOutput, UserInputField

@pause(
    requires_confirmation=True,
    confirmation_message="Execute this step?",
)
def step_with_confirmation(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Executed after confirmation")

@pause(
    requires_user_input=True,
    user_input_message="Enter parameters:",
    user_input_schema=[
        UserInputField(name="value", field_type="str", required=True),
    ],
)
def step_with_input(step_input: StepInput) -> StepOutput:
    value = step_input.additional_data["user_input"]["value"]
    return StepOutput(content=f"Received: {value}")

# Decorator config is auto-detected when used in a Step
workflow = Workflow(
    steps=[
        Step(name="confirm_step", executor=step_with_confirmation),
        Step(name="input_step", executor=step_with_input),
    ],
    ...
)
```

### 流媒体
在流媒体工作流程中处理 HITL：

```python
from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        req.confirm()
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]
```

## 路由器 HITL
允许用户选择路线或确认自动路线选择。

路由器支持两种 HITL 模式：用户选择（用户选择路线）和确认（用户批准自动路由）。


### 用户选择
允许用户选择要执行的路由。路由器暂停并显示可用的选项。

```python
from agno.workflow import Workflow
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.db.sqlite import SqliteDb

def quick_analysis(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Quick analysis: Basic metrics computed")

def deep_analysis(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Deep analysis: Full statistical analysis")

def custom_analysis(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Custom analysis: User-defined parameters")

workflow = Workflow(
    name="analysis_workflow",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="prepare", executor=prepare_data),
        Router(
            name="analysis_router",
            choices=[
                Step(name="quick", description="Fast analysis (2 min)", executor=quick_analysis),
                Step(name="deep", description="Full analysis (10 min)", executor=deep_analysis),
                Step(name="custom", description="Custom parameters", executor=custom_analysis),
            ],
            requires_user_input=True,
            user_input_message="Select analysis type:",
            allow_multiple_selections=False,
        ),
        Step(name="report", executor=generate_report),
    ],
)

run_output = workflow.run("Analyze Q4 data")

if run_output.is_paused:
    for req in run_output.steps_requiring_route:
        print(f"Router: {req.step_name}")
        print(f"Message: {req.user_input_message}")
        print(f"Options: {req.available_choices}")
        
        choice = input("Select: ")
        req.select(choice)
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```

#### 参数

| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿user＿input | bool | 暂停片刻，等待用户选择路线 |
| user＿input＿message | str | 向用户显示的消息 |
| allow＿multiple＿selections | bool | 允许选择多条路线（默认值 False ：） |

#### 选择方法

| 方法 | 描述 |
| :--- | :--- |
| req．select（＂route＿name＂） | 选择一条路线 |
| req．select＿single（＂route＿name＂） | 选择一条路线 |
| req．select＿multiple（［＂a＂，＂b＂］） | 选择多条路线（必需 allow＿multiple＿selections＝True ） |


### 多项选择
允许用户选择多条路线。选定的路线按顺序执行。

```python
Router(
    name="processing_pipeline",
    choices=[
        Step(name="clean", description="Clean data", executor=clean_data),
        Step(name="validate", description="Validate data", executor=validate_data),
        Step(name="enrich", description="Enrich data", executor=enrich_data),
        Step(name="transform", description="Transform data", executor=transform_data),
    ],
    requires_user_input=True,
    user_input_message="Select processing steps:",
    allow_multiple_selections=True,
)
```

处理多项选择：

```python
for req in run_output.steps_requiring_route:
    print(f"Available: {req.available_choices}")
    
    # User selects: "clean, validate, transform"
    selections = input("Select (comma-separated): ").split(",")
    selections = [s.strip() for s in selections]
    
    req.select_multiple(selections)

```

### 确认模式
确认自动路由决策。选择器函数会确定路由，但用户必须先批准才能执行。

```python
def route_by_priority(step_input: StepInput) -> str:
    content = step_input.previous_step_content or ""
    if "urgent" in content.lower():
        return "urgent_handler"
    elif "billing" in content.lower():
        return "billing_handler"
    return "general_handler"

Router(
    name="request_router",
    choices=[
        Step(name="urgent_handler", executor=handle_urgent),
        Step(name="billing_handler", executor=handle_billing),
        Step(name="general_handler", executor=handle_general),
    ],
    selector=route_by_priority,
    requires_confirmation=True,
    confirmation_message="Proceed with the selected route?",
)
```

#### 处理确认：
```python
for req in run_output.steps_requiring_confirmation:
    print(f"Router: {req.step_name}")
    print(f"Message: {req.confirmation_message}")
    
    if input("Confirm? (y/n): ").lower() == "y":
        req.confirm()
    else:
        req.reject()

```

#### 确认参数
| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿confirmation | bool | 暂停等待用户确认路由决策 |
| confirmation＿message | str | 向用户显示的消息 |
| on＿reject | OnReject | 被拒绝时的操作：（ skip 默认），cancel |


#### 用户选择与确认
| 模式 | 选择器 | 用户操作 | 用例 |
| :--- | :--- | :--- | :--- |
| 用户选择 | 没有任何 | 选择路线 | 交互式向导，用户驱动的工作流程 |
| 确认 | 功能 | 批准／拒绝 | 对自动化决策的监督 |

当需要用户决定路径时，使用用户选择；当系统做出决定但需要人工确认时，使用确认机制。


### 流媒体
处理流媒体工作流程中的路由器 HITL：
```python

from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at router: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_requiring_route:
        req.select(req.available_choices[0])
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]

```

## HITL 条件

允许用户在条件工作流中决定执行哪个分支。

条件支持确认 HITL，允许用户在运行时决定要执行哪个分支。

### 用户控制分支
当requires_confirmation=True用户做出决定时，条件判断会暂停：
- 确认：执行steps分支（如果存在分支）
- 拒绝：行为取决于on_reject设置

```python
from agno.workflow import Workflow, OnReject
from agno.workflow.condition import Condition
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.db.sqlite import SqliteDb

def detailed_analysis(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Detailed analysis: Full review completed")

def quick_summary(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Quick summary: Key highlights identified")

workflow = Workflow(
    name="analysis_workflow",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="analyze", executor=analyze_data),
        Condition(
            name="analysis_depth",
            steps=[Step(name="detailed", executor=detailed_analysis)],
            else_steps=[Step(name="quick", executor=quick_summary)],
            requires_confirmation=True,
            confirmation_message="Perform detailed analysis?",
            on_reject=OnReject.else_branch,
        ),
        Step(name="report", executor=generate_report),
    ],
)

run_output = workflow.run("Analyze Q4 data")

if run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        print(f"Decision: {req.step_name}")
        print(f"Message: {req.confirmation_message}")
        
        if input("Confirm? (y/n): ").lower() == "y":
            req.confirm()
            print("Executing 'if' branch")
        else:
            req.reject()
            print("Executing 'else' branch")
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```

#### 参数
| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿confirmation | bool | 暂停等待用户决定 |
| confirmation＿message | str | 向用户显示的消息 |
| on＿reject | OnReject | 被拒绝时的操作 |

#### 拒绝选项
| 价值 | 行为 |
| :--- | :--- |
| OnReject．else＿branch | 执行 else＿steps（默认） |
| OnReject．skip | 跳过整个条件 |
| OnReject．cancel | 取消工作流程 |


### 分支执行
| 用户操作 | on＿reject | 结果 |
| :--- | :--- | :--- |
| 确认 | 任何 | 执行 steps |
| 拒绝 | else＿branch | 执行 else＿steps |
| 拒绝 | skip | 跳过条件，继续工作流程 |
| 拒绝 | cancel | 取消工作流程 |


### 如果没有 else_steps
如果没有else_steps定义任何条件，且用户拒绝on_reject=OnReject.else_branch，则跳过该条件：

```python
Condition(
    name="optional_processing",
    steps=[Step(name="process", executor=process)],
    # No else_steps defined
    requires_confirmation=True,
    confirmation_message="Run optional processing?",
    on_reject=OnReject.else_branch,  # Will skip if rejected
)
```


### 与评估器结合
当未指定参数requires_confirmation=True时，该参数evaluator将被忽略。用户的选择优先：
```python

Condition(
    name="user_controlled",
    # evaluator is ignored when requires_confirmation=True
    steps=[Step(name="if_branch", ...)],
    else_steps=[Step(name="else_branch", ...)],
    requires_confirmation=True,
    confirmation_message="Proceed with if branch?",
)
```


### 流媒体
处理流式工作流中的 HITL 条件：
```python
from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        req.confirm()  # or req.reject()
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]

```


## Loop  HITL

在工作流中开始迭代执行之前，请进行确认。

循环支持确认 HITL，在第一次迭代之前暂停，让用户决定是否开始循环。
​


### 开始确认
当requires_confirmation=True循环执行完毕后，循环会暂停一段时间：
- 确认：执行循环迭代
- 拒绝：跳过整个循环

```python
from agno.workflow import Workflow
from agno.workflow.loop import Loop
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.db.sqlite import SqliteDb

def refine_analysis(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Iteration complete: Quality improved")

workflow = Workflow(
    name="refinement_workflow",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="prepare", executor=prepare_data),
        Loop(
            name="refinement_loop",
            steps=[Step(name="refine", executor=refine_analysis)],
            max_iterations=5,
            requires_confirmation=True,
            confirmation_message="Start refinement loop? (up to 5 iterations)",
        ),
        Step(name="finalize", executor=finalize_results),
    ],
)

run_output = workflow.run("Process data")

if run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        print(f"Loop: {req.step_name}")
        print(f"Message: {req.confirmation_message}")
        
        if input("Start loop? (y/n): ").lower() == "y":
            req.confirm()
            print("Starting loop")
        else:
            req.reject()
            print("Skipping loop")
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```


#### 参数

| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿confirmation | bool | 第一次迭代前暂停 |
| confirmation＿message | str | 向用户显示的消息 |
| on＿reject | OnReject | 被拒绝时的操作：（ skip 默认），cancel |

### 循环行为
确认操作在循环开始前进行一次。每次迭代不会因确认而暂停。
| 用户操作 | 结果 |  
| :--- | :--- | 
| 确认 | 执行所有迭代（直到max_iterations或直到should_continue返回False）|
| 拒绝 | 完全跳过循环 |  

### 使用 should_continue
该should_continue函数控制迭代。每次迭代之前都会进行确认：

```python
def check_quality(step_input: StepInput, iteration: int) -> bool:
    # Continue if quality threshold not met
    return iteration < 3  # Example: max 3 iterations

Loop(
    name="quality_loop",
    steps=[Step(name="improve", executor=improve_quality)],
    should_continue=check_quality,
    requires_confirmation=True,
    confirmation_message="Start quality improvement loop?",
)
```

### 流媒体
处理流媒体工作流中的循环 HITL：

```python
from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        req.confirm()
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]
```


## 步骤 HITL


在执行一组步骤的流程之前，请先进行确认。

该Steps组件将多个步骤组合成一个流水线。它支持确认 HITL，即在整个流水线执行之前暂停。
​

### 管道确认
当出现这种情况时requires_confirmation=True，管道会在执行任何步骤之前暂停：
- 确认：执行流程中的所有步骤
- 拒绝：跳过整个流程

```python

from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.workflow.steps import Steps
from agno.workflow.types import StepInput, StepOutput
from agno.db.sqlite import SqliteDb

def validate_data(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Validation: Schema verified")

def transform_data(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Transform: Data normalized")

def enrich_data(step_input: StepInput) -> StepOutput:
    return StepOutput(content="Enrichment: External data merged")

workflow = Workflow(
    name="data_pipeline",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(name="collect", executor=collect_data),
        Steps(
            name="advanced_processing",
            steps=[
                Step(name="validate", executor=validate_data),
                Step(name="transform", executor=transform_data),
                Step(name="enrich", executor=enrich_data),
            ],
            requires_confirmation=True,
            confirmation_message="Run advanced processing pipeline?",
        ),
        Step(name="report", executor=generate_report),
    ],
)

run_output = workflow.run("Process data")

if run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        print(f"Pipeline: {req.step_name}")
        print(f"Message: {req.confirmation_message}")
        
        if input("Run pipeline? (y/n): ").lower() == "y":
            req.confirm()
            print("Executing pipeline")
        else:
            req.reject()
            print("Skipping pipeline")
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)

```

#### 参数
| 范围 | 类型 | 描述 |
| :--- | :--- | :--- |
| requires＿confirmation | bool | 执行管道前暂停 |
| confirmation＿message | str | 向用户显示的消息 |
| on＿reject | OnReject | 被拒绝时的操作：（ skip 默认），cancel |

### 管道行为
确认操作在流水线启动前执行一次。流水线中的各个步骤不会因确认而暂停（除非它们有自己的 HITL 配置）。

| 用户操作 | 结果 |
| :--- | :--- |
| 确认 | 按顺序执行所有步骤 |
| 拒绝 | 跳过流程中的所有步骤。 |


### 流媒体
处理流式工作流中的管道 HITL：

```python
from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_requiring_confirmation:
        req.confirm()
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]
```


## 错误处理 HITL

步骤失败时暂停，允许用户重试或跳过。

当遇到错误时，操作步骤可以暂停，让用户决定是重试还是跳过失败的步骤。

### 错误暂停模式
当某个步骤失败时设置`on_error=OnError.pause`暂停：

```python
from agno.workflow import Workflow, OnError
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.db.sqlite import SqliteDb
import random

def unreliable_api_call(step_input: StepInput) -> StepOutput:
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("API call failed: Connection timeout")
    return StepOutput(content="API call succeeded")

def process_data(step_input: StepInput) -> StepOutput:
    return StepOutput(content=f"Processed: {step_input.previous_step_content}")

workflow = Workflow(
    name="api_workflow",
    db=SqliteDb(db_file="workflow.db"),
    steps=[
        Step(
            name="fetch_data",
            executor=unreliable_api_call,
            on_error=OnError.pause,
        ),
        Step(name="process", executor=process_data),
    ],
)

run_output = workflow.run("Fetch and process")

while run_output.is_paused:
    for req in run_output.steps_with_errors:
        print(f"Step '{req.step_name}' failed")
        print(f"Error: {req.error_message}")
        print(f"Retry count: {req.retry_count}")
        
        choice = input("Retry or skip? (r/s): ").lower()
        if choice == "r":
            req.retry()
        else:
            req.skip()
    
    run_output = workflow.continue_run(run_output)

print(run_output.content)
```

### 错误处理选项

| 价值 | 行为 |
| :--- | :--- |
| OnError．fail | 立即使工作流程失败（默认） |
| OnError．skip | 跳过此步骤并继续 |
| OnError．pause | 暂停等待用户决定（重试或跳过） |


### 错误要求属性

当步骤失败且 on_error=OnError.pause 时，会创建一个 ErrorRequirement：

| 财产 | 类型 | 描述 |
| :--- | :--- | :--- |
| step＿name | str | 失败步骤的名称 |
| error＿message | str | 异常消息 |
| error＿type | str | 异常类名（例如，＂ValueError＂） |
| retry＿count | int | 迄今为止的重试次数 |

### 错误需求方法

| 方法 | 描述 |
| :--- | :--- |
| req．retry（） | 重试失败的步骤 |
| req．skip（） | 跳过此步骤并继续 |


### 重试行为

当你打电话时req.retry()：
- 该步骤使用相同的输入再次执行。
- retry_count增量
- 如果再次失败，工作流程将再次暂停。
- 您可以无限次重试，也可以尝试几次后跳过。

```python
for req in run_output.steps_with_errors:
    if req.retry_count < 3:
        print(f"Retrying (attempt {req.retry_count + 1}/3)")
        req.retry()
    else:
        print("Max retries reached, skipping")
        req.skip()
```

### 跳过行为
当你打电话时req.skip()：
- 该步骤被标记为已跳过（而非失败）。
- 工作流程继续进行下一步
- step_input.previous_step_content 在下一步将变成 None

### 结合确认
一个步骤可以同时包含错误处理和确认：
```python

Step(
    name="risky_operation",
    executor=risky_function,
    requires_confirmation=True,
    confirmation_message="Execute risky operation?",
    on_error=OnError.pause,
)
```

首先进行确认。如果确认成功但步骤失败，则会触发错误暂停。

### 流媒体
处理流式工作流中的 HITL 错误：

```python
from agno.run.workflow import StepPausedEvent

for event in workflow.run("input", stream=True, stream_events=True):
    if isinstance(event, StepPausedEvent):
        print(f"Paused at: {event.step_name}")

session = workflow.get_session()
run_output = session.runs[-1]

while run_output.is_paused:
    for req in run_output.steps_with_errors:
        print(f"Error: {req.error_message}")
        req.retry()  # or req.skip()
    
    for event in workflow.continue_run(run_output, stream=True, stream_events=True):
        pass
    
    session = workflow.get_session()
    run_output = session.runs[-1]
```

### 错误类型
常见错误场景及处理方法：

| 设想 | 建议采取的措施 |
| :--- | :--- |
| Network timeout | 多试几次，然后跳过 |
| Rate limit | 延迟后重试 |
| Invalid input | 跳过（重试无效） |
| Resource unavailable | 根据严重程度重试或跳过 |

```python
for req in run_output.steps_with_errors:
    if "timeout" in req.error_message.lower():
        if req.retry_count < 3:
            req.retry()
        else:
            req.skip()
    elif "rate limit" in req.error_message.lower():
        import time
        time.sleep(5)  # Wait before retry
        req.retry()
    else:
        req.skip()  # Unknown error, skip
```


