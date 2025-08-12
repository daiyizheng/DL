# pyproject.toml 详细解析
```markdown
[project]
name = "agents_from_scratch"
version = "0.1.0"
description = "Build an e-mail assistant from scratch"
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3.9",
    "langchain-core>=0.3.59",
    "langchain-openai",
    "langgraph>=0.4.2",
    "langsmith[pytest]>=0.3.4",
    "pandas",
    "matplotlib",
    "pytest",
    "pytest-xdist",
    "jupyter",
    "langgraph-cli[inmem]",
    "google-api-python-client>=2.128.0",
    "google-auth-oauthlib",
    "google-auth-httplib2",
    "python-dotenv",
    "pyppeteer",
    "html2text",
    "rich",
]

[project.optional-dependencies]
dev = ["mypy>=1.11.1", "ruff>=0.6.1"]

[build-system]
requires = ["setuptools>=73.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["email_assistant"]

[tool.setuptools.package-dir]
"email_assistant" = "src/email_assistant"

[tool.setuptools.package-data]
"*" = ["py.typed"]

[tool.ruff]
lint.select = [
    "E",    # pycodestyle
    "F",    # pyflakes
    "I",    # isort
    "D",    # pydocstyle
    "D401", # First line should be in imperative mood
    "T201",
    "UP",
]
lint.ignore = [
    "UP006",
    "UP007",
    "UP035",
    "D417",
    "E501",
]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "UP"]

[tool.ruff.lint.pydocstyle]
convention = "google"

```

##  [project] 部分
定义项目的基本信息和依赖关系。

### 项目元数据
```markdown
name = "agents_from_scratch"  # 项目名称（必须）
version = "0.1.0"            # 项目版本（必须）
description = "Build an e-mail assistant from scratch"  # 项目描述
requires-python = ">=3.11"    # 要求 Python 3.11 或更高版本
name 和 version 是必填字段，用于 PyPI 发布。

requires-python 指定 Python 版本兼容性。
```


### 依赖项 (dependencies)
```markdown
dependencies = [
    "langchain>=0.3.9",            # LangChain 核心库（构建 AI 代理）
    "langchain-core>=0.3.59",      # LangChain 核心组件
    "langchain-openai",            # OpenAI 集成
    "langgraph>=0.4.2",            # LangGraph（用于构建 AI 工作流）
    "langsmith[pytest]>=0.3.4",    # LangSmith（AI 实验跟踪 + pytest 插件）
    "pandas",                      # 数据处理
    "matplotlib",                  # 数据可视化
    "pytest",                      # 单元测试
    "pytest-xdist",                # 并行测试
    "jupyter",                     # Jupyter Notebook 支持
    "langgraph-cli[inmem]",        # LangGraph CLI（内存模式）
    "google-api-python-client>=2.128.0",  # Google API 客户端
    "google-auth-oauthlib",        # Google OAuth 认证
    "google-auth-httplib2",        # Google HTTP 认证
    "python-dotenv",               # 环境变量管理
    "pyppeteer",                   # 无头浏览器（用于网页抓取）
    "html2text",                   # HTML 转纯文本
    "rich",                        # 终端富文本输出
]
```
所有依赖项在 pip install 时自动安装。

部分依赖指定了最低版本（如 >=0.3.9）。

langsmith[pytest] 表示额外安装 pytest 插件。

### 可选依赖项 (optional-dependencies)
toml
```markdown
[project.optional-dependencies]
dev = ["mypy>=1.11.1", "ruff>=0.6.1"]  # 开发环境依赖
```
dev 组包含开发工具：

mypy（静态类型检查）

ruff（代码格式化 + linting）

可通过 pip install -e ".[dev]" 安装。

## [build-system] 部分
定义构建项目所需的工具。

```markdown
requires = ["setuptools>=73.0.0", "wheel"]  # 构建依赖
build-backend = "setuptools.build_meta"     # 使用 setuptools 构建

```
[build-system]

requires：构建时需要的依赖（setuptools + wheel）。

build-backend：指定构建后端（setuptools 是传统 Python 打包工具）。

##  [tool.setuptools] 部分
配置 setuptools 打包行为。

### 包发现 (packages)
```
[tool.setuptools]
packages = ["email_assistant"]  # 指定要打包的 Python 包
```


只打包 email_assistant 目录。

###  包目录映射 (package-dir)
```shell
[tool.setuptools.package-dir]
"email_assistant" = "src/email_assistant"  # 包路径映射
email_assistant 包的源码位于 src/email_assistant 目录（推荐结构）。
```

3.3 包数据 (package-data)
```shell

[tool.setuptools.package-data]
"*" = ["py.typed"]  # 包含类型声明文件
```

py.typed 文件表示该项目支持类型注解（PEP 561）。

##  [tool.ruff] 部分
配置 ruff（现代 Python linter + formatter）。

###  启用的检查规则 (lint.select)

[tool.ruff]
```shell
lint.select = [
    "E",    # pycodestyle（基本代码风格）
    "F",    # pyflakes（代码错误检查）
    "I",    # isort（导入排序）
    "D",    # pydocstyle（文档字符串检查）
    "D401", # 强制文档字符串首行为祈使语气（如 "Return" 而非 "Returns"）
    "T201", # 禁止 print 语句
    "UP",   # pyupgrade（自动升级 Python 语法）
]

```
E, F 是基础规则（类似 flake8）。

D 检查文档字符串。

UP 用于自动升级 Python 语法（如 typing.List → list）。

###  忽略的规则 (lint.ignore)
```shell
lint.ignore = [
    "UP006",  # 忽略 `typing.List` → `list` 建议
    "UP007",  # 忽略 `typing.Dict` → `dict` 建议
    "UP035",  # 忽略 Python 3.8+ 类型注解建议
    "D417",   # 忽略 "文档字符串缺少参数说明"
    "E501",   # 忽略行长度限制
]
```
放宽部分规则（如 E501 不强制 80 字符换行）。

###  文件特定规则 (per-file-ignores)
```shell
[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D", "UP"]  # 测试文件忽略文档和类型升级检查
```

测试文件不需要严格的文档字符串和类型升级检查。

###  文档风格 (pydocstyle.convention)
```shell
[tool.ruff.lint.pydocstyle]
convention = "google"  # 使用 Google 风格的文档字符串
```

要求函数/类的文档字符串遵循 Google 风格。

