# 技能介绍

技能通过指令、脚本和参考文档为代理人提供结构化的领域专业知识。

Agno Skills 集成基于 Anthropic 的Agent Skills 规范，是一种通过提供特定技能来扩展代理能力的方法。

通过技能，您的代理人将能够逐步发现、获取和利用专业知识和能力。
​

## 什么是技能？
技能是一个独立的软件包，特工可以使用它来扩展其在特定领域的能力，或者获得新的特定能力。

所有技能均包含：
- 说明：关于何时以及如何应用该技能的详细指导（在SKILL.md）
- 脚本：代理可以运行的可选可执行代码模板
- 参考资料：可选的辅助文档（指南、速查表、示例）


### 为什么要运用技能？
​
按需提供领域专业知识
技能不是将系统消息填充为涵盖所有用例的指令，而是将领域知识组织成重点突出的软件包。
代理程序只加载它需要的内容，从而节省代币并最终降低成本。


### 可重用知识包
一次创建技能，即可在多个代理中使用。例如，代码审查技能可以在调试代理、PR 审查代理和代码生成代理之间共享。

### 渐进式发现
技能使用延迟加载来保持上下文窗口的高效运行：
- 浏览：代理在其系统提示中看到技能摘要
- 加载：当任务与技能匹配时，代理会加载完整的指令。
- 参考资料：代理人可根据需要查阅详细文档。
- 执行：代理可以运行技能中的脚本。

### 快速示例

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.skills import Skills, LocalSkills

# Load skills from a directory
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    skills=Skills(loaders=[LocalSkills("/path/to/skills")])
)

# The agent now has access to skill tools:
# - get_skill_instructions(skill_name)
# - get_skill_reference(skill_name, reference_path)
# - get_skill_script(skill_name, script_path)
```

### 技能结构
```bash
my-skill/
├── SKILL.md           # Instructions with YAML frontmatter
├── scripts/           # Optional executable scripts
│   └── helper.py
└── references/        # Optional reference documentation
    └── guide.md
```

## 培养技能

创建包含说明、脚本和参考文档的技能。

技能是一个目录，其中包含一个SKILL.md文件scripts/，以及可选的references/子目录。
​
### 目录结构

```bash
my-skill/
├── SKILL.md           # Required: Instructions with YAML frontmatter
├── scripts/           # Optional: Executable scripts
│   └── helper.py
└── references/        # Optional: Reference documentation
    └── guide.md
```

### SKILL.md 文件
该SKILL.md文件是技能的核心。它包含带有元数据的 YAML 前置信息，以及 Markdown 指令。
​

#### 必填字段

```yaml
---
name: my-skill
description: Short description of what this skill does
---

```
- 名称：必须为小写字母，仅包含字母和数字，且只能使用连字符（最多 64 个字符）
- 描述：显示在代理系统提示中的简要摘要（最多 1024 个字符）
​

### 可选字段
```yaml
---
name: code-review
description: Code review assistance with style checking and best practices
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: your-name
  tags: ["python", "code-quality"]
---
```


### 完整示例

```yaml
---
name: code-review
description: Code review assistance with style checking and best practices
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: your-name
  tags: ["python", "code-quality"]
---

# Code Review Skill

Use this skill when reviewing code for quality, style, and best practices.

## When to Use

- User asks for code review or feedback
- User wants to improve code quality
- User needs help with refactoring

## Process

1. **Analyze Structure**: Review overall code organization
2. **Check Style**: Look for style guide violations
3. **Identify Issues**: Find bugs, security issues, performance problems
4. **Suggest Improvements**: Provide actionable recommendations

## Best Practices

- Focus on the most impactful issues first
- Explain the "why" behind suggestions
- Provide code examples for fixes
```
### 添加脚本
脚本是代理程序可以运行的可执行文件。它们必须包含 shebang 行。
​
#### Python脚本示例
创造scripts/check_style.py：
```python
#!/usr/bin/env python3
"""Check code style and return results."""

import sys

def check_style(code: str) -> dict:
    issues = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        if len(line) > 100:
            issues.append(f"Line {i}: exceeds 100 characters")
        if line.endswith(' '):
            issues.append(f"Line {i}: trailing whitespace")

    return {"issues": issues, "count": len(issues)}

if __name__ == "__main__":
    # Read code from stdin or argument
    code = sys.stdin.read() if not sys.argv[1:] else sys.argv[1]
    result = check_style(code)
    print(result)
```


#### Shell脚本示例
创造scripts/lint.sh：

```bash
#!/bin/bash
# Run linting on provided file

if [ -z "$1" ]; then
    echo "Usage: lint.sh <file>"
    exit 1
fi

ruff check "$1" 2>&1
```

### 添加参考文献
参考文档是代理可以按需加载的文档文件。

#### 示例参考
创造references/style-guide.md：

```markdown
# Python Style Guide

## Naming Conventions

- **Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

## Line Length

- Maximum 100 characters per line
- Break long lines at logical points

## Imports

- Standard library imports first
- Third-party imports second
- Local imports third
- Alphabetize within each group
```

### 验证规则
技能加载后会进行验证。规则如下：
​
名称要求
- 最多 64 个字符
- 仅限小写字母、数字和连字符
- 不能以连字符开头或结尾
- 没有连续的连字符（--）
- 必须与目录名称匹配

### 场地限制
| 场地 | 最大长度 |
| :--- | :--- |
| 姓名 | 64 个字符 |
| 描述 | 1024个字符 |

### 有效许可证
接受常见的 SPDX 标识符：MIT，Apache-2.0，GPL-3.0，BSD-3-Clause等等。

### 组织多种技能
创建一个包含多个技能文件夹的目录：

```bash
skills/
├── code-review/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
├── git-workflow/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
└── testing/
    ├── SKILL.md
    └── references/
```

一次性加载所有技能：
```python
from agno.skills import Skills, LocalSkills

skills = Skills(loaders=[LocalSkills("/path/to/skills")])
```

## 装填技巧


使用 LocalSkills 和技能协调器将技能加载到代理中。

技能是通过Skills类加载的，其中一个或多个SkillLoader实例被设置为加载器。

目前LocalSkills可以从文件系统中加载技能。
​
### 基本用法

```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.skills import Skills, LocalSkills

# Load skills from a directory
agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    skills=Skills(loaders=[LocalSkills("/path/to/skills")])
)
```

### 本地技能加载器
加载器LocalSkills从本地文件系统读取技能。
​

#### 从技能目录加载
如果您在子目录中拥有多个技能：
```python
skills/
├── code-review/
│   └── SKILL.md
├── git-workflow/
│   └── SKILL.md
└── testing/
    └── SKILL.md
```

```python
from agno.skills import Skills, LocalSkills

# Load all skills from the directory
skills = Skills(loaders=[LocalSkills("/path/to/skills")])
```

#### 加载单个技能
如果您只想加载一项技能：
```python
from agno.skills import Skills, LocalSkills

# Load a single skill directory
skills = Skills(loaders=[LocalSkills("/path/to/skills/code-review")])
```

### 多装载机
你可以组合多个装载机，从不同位置装载技能：

```python
from agno.skills import Skills, LocalSkills

skills = Skills(loaders=[
    LocalSkills("/path/to/shared-skills"),
    LocalSkills("/path/to/project-skills"),
])
```

### 代理工具
当您为代理添加技能时，它将自动获得以下工具的访问权限：

| 工具 | 描述 |
| :--- | :--- |
| get＿skill＿instructions（skill＿name） | 加载技能的完整说明 |
| get＿skill＿reference（skill＿name，reference＿path） | 加载参考文档 |
| get＿skill＿script（skill＿name，script＿path，execute，args，timeout） | 读取或执行脚本 |


### 示例：使用技能工具
```python
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.skills import Skills, LocalSkills

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    skills=Skills(loaders=[LocalSkills("/path/to/skills")]),
    instructions=[
        "You have access to specialized skills.",
        "Use get_skill_instructions to load full guidance when needed.",
    ],
)

# The agent will automatically use skills when relevant
agent.print_response("Review this code for best practices: def foo(): pass")
```


### 系统提示集成
技能元数据会自动添加到代理的系统提示中。代理可以看到：
- 技能名称和描述
- 可用脚本和参考资料
- 如何加载完整技能详情的说明
- 这样一来，智能体无需预先加载所有内容即可发现和使用技能。
​
### 装弹技巧
如果你的技能在游戏过程中发生变化，你可以重新加载它们：

```python
from agno.skills import Skills, LocalSkills

skills = Skills(loaders=[LocalSkills("/path/to/skills")])

# ... skills are modified on disk ...

# Reload to pick up changes
skills.reload()
```
### 错误处理
技能加载时会进行验证。如果验证失败，SkillValidationError则会引发异常：

```python
from agno.skills import Skills, LocalSkills, SkillValidationError

try:
    skills = Skills(loaders=[LocalSkills("/path/to/skills")])
except SkillValidationError as e:
    print(f"Skill validation failed: {e}")
    print(f"Errors: {e.errors}")
```

### 完整示例

```python
from pathlib import Path
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.skills import Skills, LocalSkills

# Get skills directory relative to this file
skills_dir = Path(__file__).parent / "skills"

# Create agent with skills
agent = Agent(
    name="Code Assistant",
    model=OpenAIResponses(id="gpt-5.2"),
    skills=Skills(loaders=[LocalSkills(str(skills_dir))]),
    instructions=[
        "You are a helpful coding assistant with access to specialized skills."
    ],
    markdown=True,
)

if __name__ == "__main__":
    agent.print_response(
        "Review this Python function:\n\n"
        "def calc(x,y): return x+y"
    )
```