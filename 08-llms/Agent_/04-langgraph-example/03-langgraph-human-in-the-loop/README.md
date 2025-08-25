# 主要内容
- 如何在LangGraph提供的ReAct架构的Agent中使用Human in the loop（HIL 人工审查）
    - 在工具被调用时，系统会暂停执行，等待用户的反馈（接受、编辑或直接提供响应），然后根据用户的反馈决定如何继续执行工具，这个功能在需要人工干预的工作流中非常有用：

1．**验证工具输入：**确保工具接收到的输入参数是正确的
2．**调整工具行为：** 允许用户在工具执行前修改参数
3．**直接提供结果：**在某些情况下，用户可能希望跳过工具的执行，直接返回自定义结果
- 自定义工則
    - 使用python实现的一个模拟酒店预订的工具`book＿hotel`
    - 其需传入的参数为：`{hotel＿name}`

## 项目依赖
```shell
pip install langgraph
pip install langchain
pip install langchain－deepseek
pip install langchain－mcp－adapters
```



## human in the loop 的信息反馈按键
<img src="https://i-blog.csdnimg.cn/direct/b566dae6f0144c37ae0c3764442168b5.png">


## 前端框架
- Agent-Chat-UI
