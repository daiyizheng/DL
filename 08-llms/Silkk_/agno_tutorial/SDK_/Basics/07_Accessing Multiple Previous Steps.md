# 访问多个先前步骤

如何访问多个先前的步骤

高级工作流程通常需要来自多个先前步骤的数据，而不仅仅是前一个步骤的数据。该StepInput对象提供了强大的方法，可以按名称访问任何先前步骤的输出，或检索所有累积的内容。

## 例子
```python
def create_comprehensive_report(step_input: StepInput) -> StepOutput:
    """
    Custom function that creates a report using data from multiple previous steps.
    This function has access to ALL previous step outputs and the original workflow message.
    """

    # Access original workflow input
    original_topic = step_input.workflow_message or ""

    # Access specific step outputs by name
    hackernews_data = step_input.get_step_content("research_hackernews") or ""
    web_data = step_input.get_step_content("research_web") or ""

    # Or access ALL previous content
    all_research = step_input.get_all_previous_content()

    # Create a comprehensive report combining all sources
    report = f"""
        # Comprehensive Research Report: {original_topic}

        ## Executive Summary
        Based on research from HackerNews and web sources, here's a comprehensive analysis of {original_topic}.

        ## HackerNews Insights
        {hackernews_data[:500]}...

        ## Web Research Findings  
        {web_data[:500]}...
    """

    return StepOutput(
        step_name="comprehensive_report", 
        content=report.strip(), 
        success=True
    )

# Use in workflow
workflow = Workflow(
    name="Enhanced Research Workflow",
    steps=[
        Step(name="research_hackernews", agent=hackernews_agent),
        Step(name="research_web", agent=web_agent),
        Step(name="comprehensive_report", executor=create_comprehensive_report),  # Accesses both previous steps
        Step(name="final_reasoning", agent=reasoning_agent),
    ],
)

```


可用方法
- `step_input.get_step_content("step_name")`- 按名称从特定步骤获取内容
- `step_input.get_step_output("step_name")`-从特定步骤中获取StepOutput完整对象
- `step_input.get_all_previous_content()`- 获取之前所有步骤的内容
- `step_input.workflow_message`- 访问原始工作流输入消息
- `step_input.previous_step_content`- 获取上一步的内容

## 访问嵌套步骤
你可以直接访问嵌套在Parallel、Condition、Router、Loop和Steps组内的步骤，使用它们的步骤名称。查找通过递归搜索寻找任意深度的嵌套步骤。

### 直接访问并行组中的步骤
```python

def aggregator(step_input: StepInput) -> StepOutput:
    # Access the entire Parallel group output (returns a dict)
    parallel_data = step_input.get_step_content("parallel_processing")
    
    # Direct access to individual nested steps (recursive search)
    step_a_output = step_input.get_step_output("step_a")  # Returns StepOutput object
    step_a_content = step_input.get_step_content("step_a")  # Returns content string
    
    return StepOutput(
        step_name="aggregator",
        content=f"Step A: {step_a_content}",
        success=True
    )

workflow = Workflow(
    name="Parallel Access Example",
    steps=[
        Parallel(
            step_a,  # Can be accessed directly by name
            step_b,
            step_c,
            name="parallel_processing",
        ),
        Step(name="aggregator", executor=aggregator),
    ],
)
```
### 嵌套很深的步骤
递归搜索适用于任何深度级别：

```python
def deep_nested_step(step_input: StepInput) -> StepOutput:
    """Step nested deep inside Parallel -> Condition -> Steps."""
    return StepOutput(step_name="deep_nested_step", content="Deep nested content", success=True)

def verify_nested_access(step_input: StepInput) -> StepOutput:
    """Verify we can access deeply nested step."""
    nested_output = step_input.get_step_output("deep_nested_step")
    nested_content = step_input.get_step_content("deep_nested_step")
    
    return StepOutput(
        step_name="verifier",
        content=f"Nested output found: {nested_output is not None}, Content: {nested_content}",
        success=True
    )

def condition_evaluator(step_input: StepInput) -> bool:
    """Evaluator for the condition."""
    return True

workflow = Workflow(
    name="Multiple Depth Access",
    steps=[
        Parallel(
            Condition(
                name="condition_layer",
                evaluator=condition_evaluator,
                steps=[
                    Steps(
                        name="steps_layer",
                        steps=[deep_nested_step],
                    )
                ],
            ),
            name="Parallel Processing",
        ),
        Step(name="verifier", executor=verify_nested_access),
    ],
)
```

### 嵌套很深的步骤
递归搜索适用于任何深度级别：
```python
def deep_nested_step(step_input: StepInput) -> StepOutput:
    """Step nested deep inside Parallel -> Condition -> Steps."""
    return StepOutput(step_name="deep_nested_step", content="Deep nested content", success=True)

def verify_nested_access(step_input: StepInput) -> StepOutput:
    """Verify we can access deeply nested step."""
    nested_output = step_input.get_step_output("deep_nested_step")
    nested_content = step_input.get_step_content("deep_nested_step")
    
    return StepOutput(
        step_name="verifier",
        content=f"Nested output found: {nested_output is not None}, Content: {nested_content}",
        success=True
    )

def condition_evaluator(step_input: StepInput) -> bool:
    """Evaluator for the condition."""
    return True

workflow = Workflow(
    name="Multiple Depth Access",
    steps=[
        Parallel(
            Condition(
                name="condition_layer",
                evaluator=condition_evaluator,
                steps=[
                    Steps(
                        name="steps_layer",
                        steps=[deep_nested_step],
                    )
                ],
            ),
            name="Parallel Processing",
        ),
        Step(name="verifier", executor=verify_nested_access),
    ],
)

```

### 并行组输出格式
Parallel当按名称访问组时，输出结果是一个字典，其中每个嵌套步骤的名称作为键：
```python
parallel_output = step_input.get_step_content("parallel_processing")
# Returns:
# {
#     "step_a": "output from step_a",
#     "step_b": "output from step_b",
#     "step_c": "output from step_c",
# }
```