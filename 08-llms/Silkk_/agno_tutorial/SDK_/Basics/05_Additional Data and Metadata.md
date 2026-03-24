# 附加数据和元数据

如何向工作流传递额外数据

何时使用： 将元数据、配置或上下文信息传递给特定步骤，而不会使主工作流消息流变得混乱。

主要优势
- 关注点分离：将工作流逻辑与元数据分离
- 步骤特定上下文：访问自定义函数中的更多信息
- 清晰的信息流：主要信息始终聚焦于内容。
- 灵活配置：传递用户信息、优先级、设置等

访问模式 用于step_input.additional_data对传递给工作流的所有其他数据进行字典访问。

## 例子

```python
from agno.workflow import Step, Workflow, StepInput, StepOutput

def custom_content_planning_function(step_input: StepInput) -> StepOutput:
    """Custom function that uses additional_data for enhanced context"""
    
    # Access the main workflow message
    message = step_input.input
    previous_content = step_input.previous_step_content
    
    # Access additional_data that was passed with the workflow
    additional_data = step_input.additional_data or {}
    user_email = additional_data.get("user_email", "No email provided")
    priority = additional_data.get("priority", "normal")
    client_type = additional_data.get("client_type", "standard")
    
    # Create enhanced planning prompt with context
    planning_prompt = f"""
        STRATEGIC CONTENT PLANNING REQUEST:
        
        Core Topic: {message}
        Research Results: {previous_content[:500] if previous_content else "No research results"}
        
        Additional Context:
        - Client Type: {client_type}
        - Priority Level: {priority}
        - Contact Email: {user_email}
        
        {"HIGH PRIORITY - Expedited delivery required" if priority == "high" else "Standard delivery timeline"}
        
        Please create a detailed, actionable content plan.
    """
    
    response = content_planner.run(planning_prompt)
    
    enhanced_content = f"""
        ## Strategic Content Plan
        
        **Planning Topic:** {message}
        **Client Details:** {client_type} | {priority.upper()} priority | {user_email}
        
        **Content Strategy:**
        {response.content}
    """
    
    return StepOutput(content=enhanced_content)

# Define workflow with steps
workflow = Workflow(
    name="Content Creation Workflow",
    steps=[
        Step(name="Research Step", team=research_team),
        Step(name="Content Planning Step", executor=custom_content_planning_function),
    ]
)

# Run workflow with additional_data
workflow.print_response(
    input="AI trends in 2024",
    additional_data={
        "user_email": "michael@dundermifflin.com",
        "priority": "high",
        "client_type": "enterprise",
        "budget": "$50000",
        "deadline": "2024-12-15"
    },
    markdown=True,
    stream=True
)
```