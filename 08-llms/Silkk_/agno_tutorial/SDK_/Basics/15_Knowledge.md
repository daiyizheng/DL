# 知识
## 概述

赋予代理商访问文档、数据库和领域专业知识的权限。

知识库使智能体能够获取训练数据之外的信息。加载文件、URL 或原始文本后，智能体可以搜索这些知识，从而提供准确且符合上下文的响应。


```python

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb

# Create a knowledge base
knowledge = Knowledge(
    vector_db=ChromaDb(collection="docs", path="tmp/chromadb"),
)

# Load content
knowledge.insert(url="https://docs.agno.com/introduction.md")

# Create an agent that searches the knowledge base
agent = Agent(knowledge=knowledge, search_knowledge=True)
agent.print_response("What is Agno?")
```

智能体在其知识库中搜索，并根据内容做出响应。

### 工作原理
知识由三个部分组成：
- 内容导入：读取文件、URL、云存储或纯文本中的文档。Agno 内置了对 PDF、DOCX、CSV、Markdown 等多种格式的阅读器。
- 分块和嵌入：将文档分割成可搜索的块，并转换为能够捕捉语义含义的向量嵌入。
- 搜索和检索：当代理需要信息时，它会在向量数据库中搜索相关数据块，并将其包含在其上下文中。

您可以使用代理 RAG（代理决定何时搜索）或传统 RAG（始终注入上下文）。代理 RAG 是默认设置，适用于大多数使用场景。

### 知识为何重要
语言模型拥有广泛的通用知识，但缺乏关于您特定领域的上下文信息。知识库通过在运行时提供相关信息来弥补这一差距。

首先准备相关内容。上传公司文档、数据库架构、产品规格、支持常见问题解答或研究论文。客服人员可以利用这些信息准确回答问题，而不是靠猜测。

然后让智能体学习。知识并非只读的。智能体可以保存它们发现的见解，并在以后检索，从而在对话中积累专业知识。

```python
def save_learning(title: str, insight: str) -> str:
    """Save a reusable insight to the knowledge base."""
    knowledge.insert(name=title, text_content=insight)
    return f"Saved: {title}"

agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    tools=[save_learning],  # Agent can write to knowledge
)
```

这使得智能体从静态系统转变为随时间学习的系统。

## 快速入门
5分钟内构建一个知识驱动型智能体。

创建一个代理程序来回答有关您文件的问题。
​

### 创建具有知识的代理
```python
from agno.agent import Agent
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini
from agno.vectordb.chroma import ChromaDb
from agno.vectordb.search import SearchType

# Create a knowledge base with ChromaDB
knowledge = Knowledge(
    vector_db=ChromaDb(
        collection="docs",
        path="tmp/chromadb",
        persistent_client=True,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(id="gemini-embedding-001"),
    ),
)

# Load content into the knowledge base
knowledge.insert(url="https://docs.agno.com/introduction.md", skip_if_exists=True)

# Create an agent that searches the knowledge base
agent = Agent(
    model=Gemini(id="gemini-3-flash-preview"),
    knowledge=knowledge,
    search_knowledge=True,
    markdown=True,
)

agent.print_response("What is Agno?", stream=True)
```

代理程序搜索知识库，找到相关内容，并根据找到的内容给出答案。
​
加载不同类型的内容

```python
knowledge.insert(path="docs/product-guide.pdf")
knowledge.insert(path="data/")  # Entire directory
knowledge.insert(url="https://example.com/docs.pdf")
knowledge.insert(text_content="Your content here...")
```

Agno 会自动检测文件类型，并使用合适的阅读器来阅读 PDF、DOCX、CSV、Markdown 等文件。
​
发生了什么事
- 插入：内容被分块，嵌入 Gemini，并存储在 ChromaDB 中。
- 查询search_knowledge_base：代理收到您的问题后，决定使用该工具搜索知识库。
- 响应：代理使用检索到的内容进行回答，其响应基于您的数据。

这就是智能体 RAG。智能体会决定何时进行搜索，而不是盲目地在每个查询中注入上下文。

## 概念
### 向量数据库
存储嵌入内容并搜索类似内容。

向量数据库以嵌入向量的形式存储内容，并支持相似性搜索。当智能体搜索知识库时，查询会被转换成嵌入向量，并与存储的向量进行匹配，从而找到相关内容。


### 支持数据库
![](https://i-blog.csdnimg.cn/direct/60ad6d5807a54def90399620b428a291.png)

### 工作原理
1. 为了更精确地检索，文档被分割成更小的部分。
2. 每个数据块都被转换成向量嵌入并存储在数据库中。
3. 查询语句被嵌入并与存储的向量进行匹配，以查找相似内容。

### 混合搜索
许多向量数据库支持混合搜索，将向量相似度与关键词匹配相结合。这可以提高那些既需要语义理解又需要精确词项匹配的查询的搜索结果。
混合搜索的工作原理如下：
- 通过向量搜索查找语义相似的内容
- 通过全文搜索查找关键词匹配项
- 使用排序融合法合并结果

### 异步支持
支持异步操作的向量数据库能够实现非阻塞操作，从而提升生产环境的性能。构建异步代理时应注意使用ainsert\asearch相关方法。

```python
# Async insert
await knowledge.ainsert(url="https://example.com/docs.pdf")

# Async search
results = await knowledge.asearch(query="How do I configure X?")
```

## 内容数据库
跟踪和管理您添加到知识库中的内容。

内容数据库是一个可选组件，用于跟踪您添加到知识库中的内容。向量数据库存储用于搜索的嵌入向量，而内容数据库则存储每条内容的元数据：内容是什么、添加时间以及处理状态。

```python
from agno.knowledge.knowledge import Knowledge
from agno.db.postgres import PostgresDb
from agno.vectordb.pgvector import PgVector

knowledge = Knowledge(
    vector_db=PgVector(table_name="vectors", db_url=db_url),
    contents_db=PostgresDb(db_url=db_url),  # Enables content tracking
)
```

### 为什么要使用内容数据库
如果没有内容数据库，您可以搜索知识库，但无法查看其中的内容或管理单个内容。
使用内容数据库，您可以获得：
- 可见性：查看所有已添加的内容、跟踪处理状态、查看元数据
- 管理：删除特定内容并自动清理相关矢量图
- 更新：无需重建知识库即可编辑名称、描述和元数据
- 过滤：使用代理过滤按元数据过滤搜索结果。

设置

Agno 支持多种数据库后端：
```python
from agno.db.postgres import PostgresDb

contents_db = PostgresDb(
    db_url="postgresql+psycopg://user:pass@localhost:5432/db",
    knowledge_table="knowledge_contents"  # Optional custom table name
)


from agno.db.sqlite import SqliteDb

contents_db = SqliteDb(db_file="knowledge.db")


from agno.db.mongo import MongoDb

contents_db = MongoDb(
    uri="mongodb://localhost:27017",
    database="agno_db"
)


from agno.db.in_memory import InMemoryDb

contents_db = InMemoryDb()  # For testing only

```
其他支持的后端：PostgreSQL（推荐用于生产环境）、SQLite（开发环境）、MySQL、MongoDB、Redis、DynamoDB、Firestore。


### 内容管理
​
添加带有元数据的内容

```python
knowledge.insert(
    name="Product Manual",
    path="docs/manual.pdf",
    metadata={"department": "engineering", "version": "2.1"}
)
```


#### 列表内容

```python
contents, total_count = knowledge.get_content(
    limit=20,
    page=1,
    sort_by="created_at",
    sort_order="desc"
)

for content in contents:
    print(content.name, content.status, content.created_at)
```

#### 按 ID 获取内容

```python
content = knowledge.get_content_by_id(content_id)

print(content.name)         # Content name
print(content.description)  # Description
print(content.metadata)     # Custom metadata
print(content.file_type)    # File type (.pdf, .txt, etc.)
print(content.size)         # File size in bytes
print(content.status)       # Processing status
print(content.created_at)   # When it was added
print(content.updated_at)   # Last modification
```


#### 删除内容
自动删除内容：
1. 从内容数据库中移除内容元数据
2. 从向量数据库中删除关联的向量
3. 保持两个数据库之间的一致性。

```python
# Delete specific content
knowledge.remove_content_by_id(content_id)

# Delete all content
knowledge.remove_all_content()
```

#### 按元数据筛选
```python

# Get available filter keys
valid_filters = knowledge.get_filters()

# Search with filters
results = knowledge.search(
    query="technical documentation",
    filters={"department": "engineering"}
)
```

#### 模式
内容数据库为每条内容存储以下字段：

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| id | str | 唯一标识符 |
| name | str | 内容名称 |
| description | str | 内容描述 |
| metadata | dict | 自定义元数据 |
| type | str | 内容类型 |
| size | int | 文件大小（字节） |
| linked＿to | str | 链接内容的 ID |
| access＿count | int | 访问次数 |
| status | str | 处理状态 |
| status＿message | str | 状态详情 |
| created＿at | int | 创建时间戳 |
| updated＿at | int | 更新时间戳 |
| external＿id | str | 用于 LightRAG 等集成的外部 ID |

#### AgentOS 集成

AgentOS Knowledge UI 需要内容数据库。有了内容数据库，Web 界面可以实现以下功能：
- 内容浏览器：查看所有已上传的内容及其元数据
- 上传界面：通过网页界面添加新内容
- 状态监控：实时处理状态更新
- 元数据编辑器：通过表单更新内容元数据
- 搜索和筛选：按元数据属性查找内容
- 批量操作：一次性管理多个内容项

```python
from agno.os import AgentOS
from agno.agent import Agent

knowledge = Knowledge(
    vector_db=PgVector(table_name="vectors", db_url=db_url),
    contents_db=PostgresDb(db_url=db_url),
)

agent = Agent(name="Knowledge Agent", knowledge=knowledge)

app = AgentOS(
    id="knowledge-demo",
    agents=[agent],
)

```

## 搜索与检索
智能体如何搜索知识库以查找相关信息。

当代理需要信息时，它会搜索相关信息片段，而不是将所有内容都加载到提示中。这样可以保持响应的重点突出和高效。

```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.vectordb.search import SearchType

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="embeddings",
        db_url=db_url,
        search_type=SearchType.hybrid,
    ),
    max_results=5,
)

results = knowledge.search("What's our return policy?")
```

#### 搜索工作原理
1. 查询分析
代理会分析用户的问题，以了解哪些信息会有所帮助。

2. 搜索执行
系统根据配置运行向量搜索、关键字搜索或混合搜索。

3. 检索
知识库返回最相关的内容片段。

4. 响应生成
将检索到的信息与问题结合起来生成答案。


### 搜索类型
​
#### 向量搜索
它通过含义而非确切词语来查找内容。例如，当您搜索“如何重置密码？”时，即使没有直接出现“更改凭据”这几个字，它也能找到相关的文档。
```python
vector_db = PgVector(
    table_name="embeddings",
    db_url=db_url,
    search_type=SearchType.vector,
)
```


最适合：用户用与文档不同的方式表达概念性问题。

#### 关键词搜索
经典的文本搜索，可匹配精确的词语和短语。利用数据库的全文搜索或关键词匹配功能。

```python
vector_db = PgVector(
    table_name="embeddings",
    db_url=db_url,
    search_type=SearchType.keyword,
)
```

最适合用于：特定术语、产品名称、错误代码、技术标识符。
​
#### 混合搜索
结合了向量相似度和关键词匹配。通常是生产环境的最佳选择。

```python
from agno.knowledge.reranker.cohere import CohereReranker

vector_db = PgVector(
    table_name="embeddings",
    db_url=db_url,
    search_type=SearchType.hybrid,
    reranker=CohereReranker(),  # Optional: improves result ordering
)
```

最适合：大多数需要语义理解和精确匹配的实际应用。

### 主动型与传统 RAG
传统 RAG总是使用精确的用户查询进行搜索，并将结果注入到提示符中。

Agentic RAG允许代理决定何时搜索、重新制定查询，并在需要时运行后续搜索。


#### Traditional RAG
```python
# Always searches, always injects results
results = knowledge.search(user_query)
context = "\n\n".join([d.content for d in results])
response = llm.generate(user_query + "\n" + context)
```

#### Agentic RAG

```python

from agno.agent import Agent

# Agent decides when to search
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,  # Agent calls search_knowledge_base tool when needed
)

agent.print_response("What's our return policy?")
```


借助 Agentic RAG，代理可以：
- 如果系统已经知道答案，则无需搜索。
- 重新编写查询语句以获得更好的结果
- 进行多次搜索以收集完整信息
- 合并不同搜索结果



#### 筛选结果
通过元数据筛选搜索结果，以定位特定内容：


```python
# Add content with metadata
knowledge.insert(
    path="policies/",
    metadata={"department": "hr", "type": "policy", "year": 2024}
)

# Search with filters
results = knowledge.search(
    query="vacation policy",
    filters={"department": "hr", "type": "policy"}
)

# Use filters with agents
agent.print_response(
    "What's our vacation policy?",
    knowledge_filters={"department": "hr"}
)

```

对于使用 OR、NOT 和比较的复杂过滤，请参阅过滤。

### 自定义检索逻辑
使用自定义检索器覆盖默认搜索行为：
```python
async def my_retriever(query: str, num_documents: int = 5, filters: dict = None, **kwargs):
    # Reformulate query
    expanded_query = query.replace("vacation", "paid time off PTO")

    # Run search
    docs = await knowledge.asearch(expanded_query, max_results=num_documents, filters=filters)

    return [d.to_dict() for d in docs]

agent = Agent(
    knowledge=knowledge,
    knowledge_retriever=my_retriever,
)

```


### 提高搜索质量
​
#### 块大小
内容拆分方式会影响检索精度：
| 块大小 | 权衡 |
| :--- | :--- |
| 短篇（1000－3000 个字符） | 更精确，但可能忽略上下文 |
| 默认值（5000 个字符） | 平衡的精确性和语境 |
| 大型（8000＋字符） | 背景信息更多，但针对性较弱。 |
| 语义组块 | 在自然主题边界处分裂 |


#### 嵌入模型
您的嵌入器会将文本转换为能够表达含义的向量。正确的选择取决于您的内容：
| 类型 | 用例 |
| :--- | :--- |
| 通用（OpenAI、Gemini） | 适用于大多数内容 |
| 领域特定 | 更适合医疗或法律等专业领域。 |
| 多种语言 | 非英语或混合语言内容必须填写 |

#### 元数据
丰富的元数据能够实现更好的筛选：
```python

# Good: specific, consistent, filterable
metadata = {
    "department": "engineering",
    "document_type": "runbook",
    "service": "payments",
    "last_updated": "2024-01-15",
}

# Bad: vague, inconsistent
metadata = {"type": "doc", "id": "12345"}
```

#### 内容结构
组织良好的内容更容易被搜索到：
- 使用清晰的标题和章节
- 自然地包含相关术语
- 在长文档顶部添加摘要
- 使用描述性文件名（hr_vacation_policy_2024.pdf不要使用空字符串document1.pdf）


### 测试
使用真实查询进行测试，以验证搜索质量：
```python
test_queries = [
    "What's our vacation policy?",
    "How do I submit expenses?",
    "Remote work guidelines",
]

for query in test_queries:
    results = knowledge.search(query)
    print(f"{query} -> {results[0].content[:100]}..." if results else "No results")
```

### 混合搜索
将向量相似度与关键词匹配相结合，以提高检索准确率。

混合搜索结合了向量相似度（语义含义）和关键词匹配（精确匹配），兼具两种方法的优势。它是大多数生产环境推荐的搜索类型。

```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="docs",
        db_url=db_url,
        search_type=SearchType.hybrid,
    ),
)
```

### 工作原理
混合搜索同时运行两个搜索：
- 向量搜索查找语义相似的内容（基于含义）
- 关键词搜索查找完全匹配的词语（基于文本）
- 融合采用倒数排序融合（RRF）方法合并结果

RRF算法将排名与以下公式相结合：RRF(d) = Σ 1/(k + rank)

这样可以确保在两种搜索方法中排名靠前的文档出现在顶部，而只匹配一种方法的文档仍然会显示出来。


#### 何时使用混合搜索
| 设想 | 为什么混合型混合动力系统有帮助？ |
| :--- | :--- |
| 用户查询的措辞各不相同 | 向量捕捉含义，关键词捕捉精确术语 |
| 包含特定术语的技术内容 | 关键词与错误代码和产品名称完全匹配 |
| 混合内容类型 | 平衡概念和精确匹配 |
| 生产系统 | 针对各种查询的最佳总体准确率 |

如果您的查询始终是概念性的，不涉及任何具体术语，请使用仅向量搜索。如果您需要精确匹配（例如，按 ID 或代码搜索），请使用仅关键字搜索。

#### 配置
​
基本设置

```python
from agno.vectordb.pgvector import PgVector, SearchType

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.hybrid,
)

```

#### 重新排名
添加重排序器以改善融合后的结果排序：

```python

from agno.knowledge.reranker.cohere import CohereReranker

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.hybrid,
    reranker=CohereReranker(),
)
```
#### RRF常数
RRF 中的常数k控制着排名较低的结果所占的权重。较高的值（例如 60）会使排名更加平滑；较低的值则会使排名靠前的结果更加突出。

```python
from agno.vectordb.chroma import ChromaDb, SearchType

vector_db = ChromaDb(
    collection="docs",
    path="tmp/chromadb",
    search_type=SearchType.hybrid,
    hybrid_rrf_k=60,  # Default is 60
)

```

#### 例子
```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid,
    ),
)

# Load content
knowledge.insert(
    url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
)

# Search combines semantic similarity + keyword matching
results = knowledge.search("chicken coconut soup", max_results=5)
for doc in results:
    print(doc.content[:200])
```

## 向量搜索

利用向量相似度，根据语义含义查找内容。

向量搜索通过语义而非精确的词语匹配来查找内容。例如，当您搜索“如何重置密码？”时，即使“更改凭据”这两个词没有完全匹配，它也能找到有关“更改凭据”的文档。


```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="docs",
        db_url=db_url,
        search_type=SearchType.vector,
    ),
)

```


#### 工作原理
- 查询嵌入：您的搜索查询将被转换为一个向量（包含数字以表达含义的列表）。
- 相似度匹配：系统查找与查询向量最接近的已存储向量。
- 排名：结果按余弦相似度（含义接近程度）排序。

嵌入模型决定了语义关系捕捉的准确程度。像 OpenAI 这样的通用模型text-embedding-3-small适用于大多数内容。


####  何时使用矢量搜索
| 设想 | 为什么向量搜索有效 |
| :--- | :--- |
| 概念性问题 | 含义相符，而不仅仅是字面意思相符。 |
| 用户表达方式不同 | 无论术语如何，都能找到相关内容 |
| 自然语言查询 | 理解问题背后的意图 |
| 内容包含丰富的词汇 | 连接同义词和相关概念 |

如果还需要精确匹配词语（例如产品名称、错误代码），请使用混合搜索。如果只需要精确匹配文本，请使用关键词搜索。
​
#### 配置
​
基本设置

```python
from agno.knowledge.embedder.openai import OpenAIEmbedder

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.vector,
    embedder=OpenAIEmbedder(id="text-embedding-3-small"),
)
```

#### 重新排名
添加重新排序器以改善结果排序：

```python
from agno.knowledge.reranker.cohere import CohereReranker

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.vector,
    reranker=CohereReranker(),
)
```


### 例子

```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load content
knowledge.insert(
    url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
)

# Search by semantic meaning
results = knowledge.search("chicken coconut soup", max_results=5)
for doc in results:
    print(doc.content[:200])

```

## 关键词搜索

复制页面

使用精确的词语和短语匹配查找内容。

关键词搜索通过匹配精确的词语和短语来查找内容。它利用数据库的全文搜索功能来查找包含特定词语的文档。

```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="docs",
        db_url=db_url,
        search_type=SearchType.keyword,
    ),
)
​

```

#### 工作原理
- 文本解析：您的查询已被拆分为可搜索的词项
- 索引查找：系统查找包含这些术语的文档
- 排名：结果按相关性（词频、文档长度等）排序。
使用 PgVector 时，它利用了 PostgreSQL 内置的全文搜索功能。其他数据库则使用其自身的文本搜索功能。

#### 何时使用关键词搜索

| 设想 | 为什么关键词搜索有效 |
| :--- | :--- |
| 搜索特定词条 | 产品名称、代码、ID 完全匹配 |
| 错误代码和标识符 | 无需语义解释的精确匹配 |
| 技术术语 | 用户知道要搜索的确切关键词 |
| 结构化数据查询 | 匹配特定字段值 |
如果用户的措辞与文档有所不同，请使用向量搜索。如果您既需要精确匹配又需要语义理解，请使用混合搜索。
​

#### 配置
​
基本设置

```python
from agno.vectordb.pgvector import PgVector, SearchType

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.keyword,
)
```


#### 重新排名
添加重新排序器以改善结果排序：

```python
from agno.knowledge.reranker.cohere import CohereReranker

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.keyword,
    reranker=CohereReranker(),
)
```


### 例子
```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector, SearchType

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.keyword,
    ),
)

# Load content
knowledge.insert(
    url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
)

# Search by exact terms
results = knowledge.search("chicken coconut soup", max_results=5)
for doc in results:
    print(doc.content[:200])
```


## 带有重新排序的代理 RAG
结合智能搜索、混合检索和重排序，以获得高质量的搜索结果。

本示例结合了三种技术以实现最佳检索效果：
- 代理 RAG：代理决定何时搜索知识库
- 混合搜索：结合向量相似度和关键词匹配
- 重新排名：使用专门的排名模型对结果进行重新排序

```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.cohere import CohereEmbedder
from agno.knowledge.reranker.cohere import CohereReranker
from agno.models.anthropic import Claude
from agno.vectordb.lancedb import LanceDb, SearchType

knowledge = Knowledge(
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="docs",
        search_type=SearchType.hybrid,
        embedder=CohereEmbedder(id="embed-v4.0"),
        reranker=CohereReranker(model="rerank-v3.5"),
    ),
)

agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    knowledge=knowledge,
    search_knowledge=True,
)

```
### 为什么要结合使用这些技术？
| 技术 | 它的功能 |
| :--- | :--- |
| 代理 RAG | 代理仅在需要时进行搜索，并可重新表述查询 |
| 混合搜索 | 既能匹配语义词，也能匹配精确词。 |
| 重新排名 | 使用专用模型按相关性重新排序结果 |

这些方法结合起来，比任何单一技术都能提供更高的检索准确率。

### 重新排名是如何运作的
混合搜索返回初始结果后，重新排名器：
1. 接收查询和候选文档
2. 使用交叉编码器模型对每个文档的相关性进行评分
3. 重新排序结果，使最相关的结果优先显示。
4. Coherererank-v3.5专门针对这项任务进行了训练，并显著提高了结果质量。



### 示例

```python
import asyncio

from agno.agent import Agent
from agno.knowledge.embedder.cohere import CohereEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reranker.cohere import CohereReranker
from agno.models.anthropic import Claude
from agno.vectordb.lancedb import LanceDb, SearchType

# Create knowledge base with hybrid search and reranking
knowledge = Knowledge(
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="agno_docs",
        search_type=SearchType.hybrid,
        embedder=CohereEmbedder(id="embed-v4.0"),
        reranker=CohereReranker(model="rerank-v3.5"),
    ),
)

# Load content
asyncio.run(
    knowledge.ainsert(url="https://docs.agno.com/introduction/agents.md")
)

# Create agent with knowledge
agent = Agent(
    model=Claude(id="claude-sonnet-4-20250514"),
    knowledge=knowledge,
    search_knowledge=True,
    instructions=[
        "Search your knowledge before answering.",
        "Include sources in your response.",
    ],
    markdown=True,
)

agent.print_response("What are Agents?", stream=True)
```




### 配置选项
​
不同的重排者

```python
# Cohere
from agno.knowledge.reranker.cohere import CohereReranker
reranker = CohereReranker(model="rerank-v3.5")

# Add to vector database
vector_db = LanceDb(
    uri="tmp/lancedb",
    table_name="docs",
    search_type=SearchType.hybrid,
    reranker=reranker,
)

```
调整结果

```python
knowledge = Knowledge(
    vector_db=vector_db,
    max_results=10,  # Number of results to return after reranking
)
```
## Custom Retriever

实现自定义检索逻辑，以完全控制代理搜索知识的方式。

自定义检索器允许您实现自己的搜索逻辑，而不是使用默认的知识库搜索。当您需要以下功能时，这非常有用：
- 直接查询外部 API 或数据库
- 实现自定义排名或筛选
- 搜索前请重新构建查询语句
- 整合多个数据源

```python
from agno.agent import Agent

def knowledge_retriever(query: str, num_documents: int = 5, **kwargs) -> list[dict]:
    # Your custom retrieval logic here
    return [{"content": "..."}]

agent = Agent(
    knowledge_retriever=knowledge_retriever,
    search_knowledge=True,
)
```


#### 工作原理
当代理人决定搜索信息时：
- 代理会knowledge_retriever使用查询调用你的函数。
- 您的函数可以按照您想要的方式检索文档。
- 结果以字典列表的形式返回给代理。
- 代理使用检索到的内容生成响应


#### 检索函数签名

```python
from typing import Optional
from agno.agent import Agent

def knowledge_retriever(
    query: str,
    agent: Optional[Agent] = None,
    num_documents: int = 5,
    **kwargs
) -> Optional[list[dict]]:
    """
    Args:
        query: The search query from the agent
        agent: The agent instance (optional, for accessing agent state)
        num_documents: Number of documents to retrieve
        **kwargs: Additional arguments passed from the agent

    Returns:
        List of documents as dictionaries, or None if search fails
    """
    # Your logic here
    return [{"content": "..."}]
```

### 示例：直接向量数据库查询
此示例绕过了知​​识抽象，直接查询 Qdrant：

```python
from typing import Optional

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from qdrant_client import QdrantClient

embedder = OpenAIEmbedder(id="text-embedding-3-small")
qdrant_client = QdrantClient(url="http://localhost:6333")

def knowledge_retriever(
    query: str, num_documents: int = 5, **kwargs
) -> Optional[list[dict]]:
    try:
        # Generate embedding for the query
        query_embedding = embedder.get_embedding(query)

        # Search Qdrant directly
        results = qdrant_client.query_points(
            collection_name="recipes",
            query=query_embedding,
            limit=num_documents,
        )

        return results.model_dump().get("points")
    except Exception as e:
        print(f"Search error: {e}")
        return None

agent = Agent(
    knowledge_retriever=knowledge_retriever,
    search_knowledge=True,
)

agent.print_response("What ingredients do I need for Massaman Gai?")
```

#### 示例：查询重构

搜索前请展开或修改查询：

```python

from agno.knowledge.knowledge import Knowledge

knowledge = Knowledge(vector_db=vector_db)

def knowledge_retriever(query: str, num_documents: int = 5, **kwargs) -> list[dict]:
    # Expand common terms
    expanded_query = query.replace("vacation", "vacation PTO paid time off")
    expanded_query = expanded_query.replace("WFH", "work from home remote")

    # Search with expanded query
    results = knowledge.search(expanded_query, max_results=num_documents)

    return [doc.to_dict() for doc in results]
```

#### 示例：多源检索
整合来自多个知识库的结果：

```python
def knowledge_retriever(query: str, num_documents: int = 5, **kwargs) -> list[dict]:
    # Search multiple sources
    policy_results = policy_knowledge.search(query, max_results=3)
    faq_results = faq_knowledge.search(query, max_results=3)

    # Combine and deduplicate
    all_results = []
    seen_ids = set()

    for doc in policy_results + faq_results:
        if doc.id not in seen_ids:
            all_results.append(doc.to_dict())
            seen_ids.add(doc.id)

    return all_results[:num_documents]
```

#### 何时使用定制检索工具

| 用例 | 为什么选择定制检索器 |
| :--- | :--- |
| 直接数据库访问 | 为了提高性能，可以省略知识抽象。 |
| 查询扩展 | 搜索前请添加同义词或相关术语 |
| 多源搜索 | 整合多个知识库的结果 |
| 外部 API | 搜索第三方服务（Elasticsearch、Algolia 等） |
| 自定义排名 | 实施领域特定相关性评分 |
| 条件逻辑 | 根据查询类型应用不同的搜索策略 |
对于大多数使用场景，内置的知识库搜索功能已经足够。当您需要完全控制检索过程时，请使用自定义检索器。

## Readers
将文件、URL 和文本转换为可搜索的文档。

阅读器将原始内容转换为Document可分块、可嵌入并存储在知识库中的对象。每个阅读器处理特定格式（PDF、CSV、Markdown 等），并提取文本和元数据。

```python

from agno.knowledge.reader.pdf_reader import PDFReader

reader = PDFReader(chunk=True, chunk_size=5000)
documents = reader.read("company_handbook.pdf")
```

读者如何工作
1. 解析：使用特定格式逻辑读取原始内容
2. 提取：提取文本和元数据（页码、作者等）
3. 分块：将大内容拆分成小块（如果启用）
4. 返回Document：提供一个可供嵌入的对象列表

```python
# Output structure
Document(
    content="The extracted text...",
    id="unique_id",
    name="document_name",
    meta_data={"page": 1, "source": "handbook.pdf"},
)
```

### 受支持的读者

| 读者 | 描述 |
| :--- | :--- |
| PDFReader | 从PDF 文件中提取文本 |
| DoclingReader | 通过 Docling 处理多种格式 |
| TextReader | 纯文本文件 |
| MarkdownReader | Markdown 文件 |
| CSVReader | CSV 文件（行变成文档） |
| FieldLabeledCSVReader | CSV 行作为字段标签文本 |
| JSONReader | JSON 文件 |
| PPTXReader | PowerPoint演示文稿 |
| ArxivReader | 来自 arXiv 的学术论文 |
| WikipediaReader | 维基百科文章 |
| YouTubeReader | YouTube 文字稿 |
| WebsiteReader | 递归地爬取网站 |
| WebSearchReader | 网络搜索结果 |
| FirecrawlReader | 通过 Firecrawl API 进行网页抓取 |

### 利用知识型读者
传递一个读取器以knowledge.insert()覆盖自动格式检测：

```python
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader

knowledge = Knowledge(vector_db=vector_db)

# Use custom reader configuration
reader = PDFReader(chunk_size=3000, split_on_pages=True)
knowledge.insert(path="documents/", reader=reader)

```

### 自动选择
Agno 会根据文件扩展名或 URL 自动选择合适的阅读器：

```python
from agno.knowledge.reader.reader_factory import ReaderFactory

# By file extension
reader = ReaderFactory.get_reader_for_extension(".pdf")  # PDFReader
reader = ReaderFactory.get_reader_for_extension(".csv")  # CSVReader

# By URL
reader = ReaderFactory.get_reader_for_url("https://youtube.com/watch?v=...")  # YouTubeReader
```

### 配置
​
分块

```python
reader = PDFReader(
    chunk=True,           # Enable chunking (default: True)
    chunk_size=5000,      # Characters per chunk
)
```

### 格式特定选项
```python
# PDF with encryption and OCR
reader = PDFReader(
    password="secret",
    read_images=True,     # OCR for images
    split_on_pages=True,  # One document per page
)

# CSV with custom encoding
reader = CSVReader(
    encoding="latin-1",
)

# Text with encoding override
reader = TextReader(
    encoding="utf-8",
)

```

#### 运行时选项
调用时覆盖设置read()：

```python
documents = reader.read(
    "file.pdf",
    name="custom_document_name",  # Override default naming
    password="runtime_password",  # Password at read time
)
```

### 异步处理
所有读取器都支持异步操作，以提高 I/O 操作的性能：

```python
import asyncio

# Single file
documents = await reader.async_read("file.pdf")

# Batch processing
tasks = [reader.async_read(file) for file in files]
all_documents = await asyncio.gather(*tasks)
```


#### 自定义分块策略
覆盖默认分块行为：

```python
from agno.knowledge.chunking.semantic_chunking import SemanticChunking

reader = PDFReader(
    chunk=True,
    chunking_strategy=SemanticChunking(),
)
```

## 分块

复制页面

将文档拆分成更小的部分，以便进行有效的矢量搜索。

分块是指在将内容嵌入并存储到向量数据库之前，将其分割成更小的部分。您选择的策略会影响搜索质量和检索准确率。

```python
from agno.knowledge.chunking.semantic_chunking import SemanticChunking
from agno.knowledge.reader.pdf_reader import PDFReader

reader = PDFReader(
    chunking_strategy=SemanticChunking(),
)
```

### 为什么分块很重要
考虑采用不同的策略来处理食谱：

| 战略 | 结果 |
| :--- | :--- |
| 固定大小（5000个字符） | 可能会在烹饪过程中拆分食谱 |
| 语义 | 根据含义将完整的食谱放在一起 |
| 文档 | 每一页都变成一个区块。 |

正确的策略能返回完整、相关的结果，而错误的策略只会返回零散的结果。


### 可用策略
![](https://i-blog.csdnimg.cn/direct/aa2ca327d44d4d968aba0924d0293508.png)

### 与读者一起使用
将分块阅读策略传授给任何读者：

```python
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.chunking.fixed_size_chunking import FixedSizeChunking
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.pgvector import PgVector

reader = PDFReader(
    chunking_strategy=FixedSizeChunking(chunk_size=3000),
)

knowledge = Knowledge(
    vector_db=PgVector(table_name="docs", db_url=db_url),
)

knowledge.insert(path="documents/", reader=reader)
```

### 选择策略
| 内容类型 | 推荐策略 | 为什么 |
| :--- | :--- | :--- |
| 总则 | 语义 | 保持意义和语境 |
| 结构化文档 | 文档 | 保留各部分及其层级结构。 |
| Markdown 文件 | Markdown | 尊重标题结构 |
| CSV／表格数据 | CSV 行 | 每行都是一个逻辑单元 |
| 源代码 | 代码 | 在函数和类边界处进行拆分 |
| 混合内容 | 递归 | 可处理多种分隔符类型 |
| 需要保持一致性 | 固定尺寸 | 可预测的块维度 |

每个阅读器都有一个合理的默认值，但您可以根据内容和检索需求进行覆盖。

### 配置
大多数策略都接受配置选项：
```python
# Fixed size with overlap
FixedSizeChunking(
    chunk_size=5000,       # Characters per chunk
    overlap=200,           # Overlap between chunks
)

# Semantic with threshold
SemanticChunking(
    similarity_threshold=0.7,  # Lower = more splits
)

# Recursive with custom separators
RecursiveChunking(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=4000,
)
```

### 块大小指南
| 块大小 | 权衡 |
| :--- | :--- |
| 短篇（1000－3000 个字符） | 检索更精确，但可能会丢失上下文信息。 |
| 默认值（5000个字符） | 平衡的精确性和语境 |
| 大型（ $8000+$ 字符） | 更多背景信息，更少针对性结果 |

对于具体问题，较小的信息块效果更好。当上下文很重要时，较大的信息块效果更好。

## 嵌入器

将文本转换为向量表示以进行语义搜索。

嵌入器将文本转换为向量（数字列表），从而捕捉文本的含义。这些向量支持语义搜索，因此即使没有关键词匹配，“如何重置我的密码？”也能找到提及“更改密码”的文档。

```python
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="docs",
        db_url=db_url,
        embedder=OpenAIEmbedder(),  # Default
    ),
)
```

### 工作原理
- 插入：添加内容时，每个数据块都会转换为向量。
- 存储：矢量图保存在您的矢量图数据库中。
- 搜索：查询语句嵌入到存储的向量中，并根据相似度进行匹配。

AgnoOpenAIEmbedder默认使用，但您可以替换为任何受支持的嵌入器。

### 配置
```python
from agno.knowledge.embedder.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    id="text-embedding-3-small",
    dimensions=1536,
)
```

### 运用知识
```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="docs",
        db_url=db_url,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Content is embedded automatically on insert
knowledge.insert(path="documents/")
```
### 批量嵌入
在一次 API 调用中处理多个文本，以减少请求并提高性能：

```python
embedder = OpenAIEmbedder(
    id="text-embedding-3-small",
    dimensions=1536,
    enable_batch=True,
    batch_size=100,
)
```


### 批量嵌入
在一次 API 调用中处理多个文本，以减少请求并提高性能：
```python
embedder = OpenAIEmbedder(
    id="text-embedding-3-small",
    dimensions=1536,
    enable_batch=True,
    batch_size=100,
)
```
支持批量处理的嵌入器：OpenAI、Azure OpenAI、Gemini、Cohere、Voyage AI、Mistral、Fireworks、Together、Jina、Nebius。

### 最佳实践

更换模型时需要重新嵌入：不同嵌入器生成的向量不兼容。如果切换嵌入器，则必须重新嵌入所有内容。

测试检索质量：使用示例查询来验证是否找到了正确的数据块。如果结果不佳，请调整数据块划分策略或嵌入器。

匹配尺寸：确保嵌入器的输出尺寸与矢量数据库所期望的尺寸相匹配。


### 支持的嵌入器
| Embedder | Type | Cost | Notes |
| :--- | :--- | :--- | :--- |
| OpenAI | Hosted | $$ | Default, excellent quality |
| Gemini | Hosted | $$ | Multilingual, Google ecosystem |
| Cohere | Hosted | $$ | Strong retrieval performance |
| Voyage AI | Hosted | $$$ | Specialized for retrieval |
| Mistral | Hosted | $$ | European provider |
| Ollama | Local | Free | Privacy, offline |
| FastEmbed | Local | Free | Fast local embeddings |
| HuggingFace | Local/Hosted | Free/$ | Open source models |
| AWS Bedrock | Hosted | $$ | AWS ecosystem |
| Azure OpenAI | Hosted | $$ | Azure ecosystem |
| Fireworks | Hosted | $ | Fast inference |
| Together | Hosted | $ | Open source models |
| Jina | Hosted | $$ | Multilingual |
| Nebius | Hosted | $ | European provider |


### Choosing an Embedder

| Consideration | Recommendation |
| :--- | :--- |
| General use | OpenAI or Gemini |
| Privacy/offline | Ollama or FastEmbed |
| Multilingual | Gemini or Jina |
| Cost-sensitive | Local embedders (free) or Fireworks/Together ($) |
| Best retrieval quality | Voyage AI or Cohere |

关键因素：
- 托管 vs 本地部署：本地部署注重隐私且无需 API 费用；托管部署注重质量和便利性。
- 延迟和成本：较小的模型更便宜、速度更快；较大的模型通常能获得更好的结果。
- 语言支持：请确保您的嵌入器支持您内容的语言。
- 尺寸大小：与您的矢量数据库预期的嵌入尺寸相匹配


## 过滤

通过元数据筛选知识搜索结果，实现精确检索。

筛选器可将知识库搜索范围限制在符合特定条件的文档中。添加内容时可附加元数据，然后在搜索时按元数据进行筛选。

```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge

# Add content with metadata
knowledge.insert(
    path="resumes/",
    metadata={"user_id": "jordan_mitchell", "document_type": "cv", "year": 2025}
)

# Search with filters
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    knowledge_filters={"user_id": "jordan_mitchell"},
)
```

为什么要使用过滤器？
- 个性化：检索特定用户或群组的文档
- 访问控制：将搜索范围限制在授权内容范围内
- 精确度：通过将结果范围缩小到相关文档来降低噪声。

### 手动筛选
创建代理或搜索时显式传递筛选条件：
```python
# Filter at agent level
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    knowledge_filters={"user_id": "jordan_mitchell"},
)

# Filter at query time
agent.print_response(
    "What are Jordan's skills?",
    knowledge_filters={"document_type": "cv"}
)

# Direct search with filters
results = knowledge.search(
    query="programming experience",
    filters={"user_id": "jordan_mitchell", "year": 2025}
)
```
多个过滤器通过“与”逻辑组合使用。

### 智能体过滤
让代理程序自动从查询中提取筛选条件。代理程序会分析用户的问题，并确定要应用哪些筛选条件。
```python
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    enable_agentic_knowledge_filters=True,  # Agent infers filters from query
)

# Agent extracts "jordan_mitchell" as user filter from the query
agent.print_response("What skills does Jordan Mitchell have?")
```
### 手动过滤与自动过滤
| 方法 | 何时使用 |
| :--- | :--- |
| 手动的 | 自动化、可预测的过滤器、全面控制 |
| 代理 | 用户应用、自然语言查询 |


### 传统 RAG 与 Agentic RAG
过滤器适用于 RAG 的两种方法：

```python
# Agent decides when to search (default)
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
    knowledge_filters={"user_id": "jordan_mitchell"},
)


# Always inject context into prompt
agent = Agent(
    knowledge=knowledge,
    search_knowledge=False,
    add_knowledge_to_context=True,
    knowledge_filters={"user_id": "jordan_mitchell"},
)
```
一次只采用一种方法。search_knowledge=True对于大多数用例，建议使用Agentic RAG。

### 元数据设计
良好的元数据有助于实现有效的筛选：

```python
# Rich, filterable metadata
metadata = {
    "user_id": "jordan_mitchell",
    "document_type": "cv",
    "department": "engineering",
    "year": 2025,
    "access_level": "internal",
}

# Add with content
knowledge.insert(path="resume.pdf", metadata=metadata)
```

尖端：
- 使用一致的值（始终如此"engineering"，而不是有时如此"eng"）
- 包含时间数据以进行基于时间的滤波
- 添加基于权限的过滤访问级别


### 支持的向量数据库
支持筛选功能:
- ChromaDB
- LanceDB
- Milvus
- MongoDB
- PgVector
- Pinecone
- Qdrant
- Weaviate

## 隔离向量搜索
当多个实例共享同一个向量数据库时，范围搜索将限定在单个 Knowledge 实例内。

当多个Knowledge实例共享同一个向量数据库时，默认情况下搜索结果会来自所有实例。您可以设置此选项，isolate_vector_search=True将每个实例的搜索范围限定在其自身的数据范围内。

```python
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

vector_db = PgVector(
    table_name="shared_vectors",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

# Only returns results from documents this instance inserted
knowledge = Knowledge(
    name="support-docs",
    vector_db=vector_db,
    isolate_vector_search=True,
)
```
### 工作原理
什么时候isolate_vector_search=True：   
- 插入：每个文档都会获得linked_to设置为知识实例的元数据name。
- 搜索：linked_to系统会自动注入过滤器，因此只会返回匹配的文档。

isolate_vector_search=False（默认）   
- 插入：未linked_to添加任何元数据。
- 搜索：未linked_to应用任何筛选条件。搜索结果将涵盖矢量数据库中的所有文档。
​
### 何时使用
| 设想 | isolate＿vector＿search |
| :--- | :--- |
| 单一知识实例 | False（默认） |
| 多个实例，每个实例都有自己的矢量数据库 | False（默认） |
| 多个实例共享同一个向量数据库，需要隔离 | True |

### 例如：共享数据库，独立搜索
```python
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

vector_db = PgVector(
    table_name="shared_vectors",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

# Two knowledge instances sharing the same vector database
hr_knowledge = Knowledge(
    name="hr-docs",
    vector_db=vector_db,
    isolate_vector_search=True,
)

engineering_knowledge = Knowledge(
    name="engineering-docs",
    vector_db=vector_db,
    isolate_vector_search=True,
)

# Insert into each instance
hr_knowledge.insert(path="hr-policies/")
engineering_knowledge.insert(path="engineering-docs/")

# This agent only searches HR documents
hr_agent = Agent(knowledge=hr_knowledge, search_knowledge=True)

# This agent only searches engineering documents
eng_agent = Agent(knowledge=engineering_knowledge, search_knowledge=True)
```

### 向后兼容性
isolate_vector_search默认值为False。现有知识实例的行为与之前完全相同。
​
###  现有数据没有linked_to元数据
在该标志启用之前索引的文档，其矢量数据库元数据中不包含此字段linked_to。启用后isolate_vector_search=True，搜索将筛选此字段linked_to=<name>。缺少此元数据字段的文档将无法匹配，并且在隔离搜索中将不可见。

> 启用此功能isolate_vector_search=True后，如果矢量数据库缺少linked_to元数据，这些文档将从搜索结果中消失。您必须重新建立索引或手动更新元数据才能恢复它们。

### 结合手动过滤器
启用此功能后isolate_vector_search=True，linked_to无论过滤器格式如何，该过滤器都会自动与您传递的任何过滤器合并：

```python
# Dict-based filters: linked_to is merged automatically
results = hr_knowledge.search(
    query="vacation policy",
    filters={"department": "legal"},
)
# Searches for: linked_to="hr-docs" AND department="legal"


# List-based filters (FilterExpr): linked_to is also injected automatically
from agno.filters import EQ

results = hr_knowledge.search(
    query="vacation policy",
    filters=[EQ("department", "legal")],
)
# Searches for: linked_to="hr-docs" AND department="legal"

```
### 实例唯一性
每个 Knowledge 实例必须具有唯一的数据库和表组合name。如果两个实例名称相同，且指向相同的内容数据库和表，则ValueError启动时会引发异常。

```python
from agno.db.postgres import PostgresDb
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

contents_db = PostgresDb(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    knowledge_table="knowledge_contents",
)

vector_db = PgVector(
    table_name="shared_vectors",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

# These two instances will conflict because they share the same
# name, contents_db, and table
knowledge_a = Knowledge(
    name="my-docs",
    contents_db=contents_db,
    vector_db=vector_db,
)

knowledge_b = Knowledge(
    name="my-docs",          # same name
    contents_db=contents_db,  # same database and table
    vector_db=vector_db,
)
# ValueError: Duplicate knowledge instances detected
```

为了解决这个问题，可以给每个实例一个唯一的标识符name，或者将它们指向不同的内容数据库或表。

### 要求
知识实例必须有一个名称集。没有名称时，即使 isolate_vector_search=True，也不会添加linked_to元数据，也不会应用过滤。

矢量数据库必须支持元数据过滤。请参阅“过滤”部分，了解支持的数据库。



## 云存储来源
将 S3、GCS、SharePoint、GitHub 和 Azure Blob 中的内容加载到知识库中。

在知识实例中注册云存储提供商，并content_sources。每个提供者都有 .file（） 和 .folder（） 方法，创建内容引用，然后传递给 knowledge.insert（）。

```python
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.remote_content import S3Config

knowledge = Knowledge(
    vector_db=vector_db,
    contents_db=contents_db,
    content_sources=[
        S3Config(
            id="company-docs",
            name="Company Documents",
            bucket_name="my-docs-bucket",
            region="us-east-1",
        ),
    ],
)

# Insert a single file
knowledge.insert(
    name="Q4 Report",
    remote_content=knowledge.content_sources[0].file("reports/q4-2025.pdf"),
)

# Insert an entire folder
knowledge.insert(
    name="Engineering Specs",
    remote_content=knowledge.content_sources[0].folder("specs/"),
)
```

### 支持的提供商
| Provider | Config Class | Install |
| :--- | :--- | :--- |
| Amazon S3 | S3Config | pip install boto3 |
| Google Cloud Storage | GcsConfig | pip install google-cloud-storage |
| SharePoint | SharePointConfig | pip install msal requests |
| GitHub | GitHubConfig | pip install requests |
| Azure Blob Storage | AzureBlobConfig | pip install azure-identity azure-storage-blob |

### 提供商配置
​
#### S3配置

```python
from agno.knowledge.remote_content import S3Config

s3 = S3Config(
    id="s3-docs",
    name="S3 Documents",
    bucket_name="my-bucket",
    region="us-east-1",
    aws_access_key_id="...",       # optional, falls back to default credential chain
    aws_secret_access_key="...",   # optional, falls back to default credential chain
    prefix="documents/",           # optional, default prefix for browsing
)
```

#### GcsConfig

```python
from agno.knowledge.remote_content import GcsConfig

gcs = GcsConfig(
    id="gcs-docs",
    name="GCS Documents",
    bucket_name="my-gcs-bucket",
    project="my-gcp-project",
)
```

| 场地 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | 必需的 | 唯一标识符 |
| name | str | 必需的 | 显示名称 |
| bucket＿name | str | 必需的 | GCS 存储桶名称 |
| project | Optional［str］ | None | GCP 项目 ID |
| credentials＿path | Optional［str］ | None | GCP 凭据文件的路径 |
| prefix | Optional［str］ | None | 默认前缀 |


#### GitHubConfig

```python
from agno.knowledge.remote_content import GitHubConfig

github = GitHubConfig(
    id="my-repo",
    name="My Repository",
    repo="owner/repo",
    token="ghp_...",
    branch="main",
)

```

| 场地 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | 必需的 | 唯一标识符 |
| name | str | 必需的 | 显示名称 |
| repo | str | 必需的 | owner／repo 格式为 的存储库 |
| token | Optional［str］ | None | GitHub 个人访问令牌（需要 Contents：read 权限） |
| branch | Optional［str］ | None | 分支名称 |
| path | Optional［str］ | None | 默认路径过滤器 |


#### SharePointConfig

```python

from agno.knowledge.remote_content import SharePointConfig

sharepoint = SharePointConfig(
    id="sharepoint-docs",
    name="SharePoint Documents",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    hostname="contoso.sharepoint.com",
    site_path="/sites/Engineering",
)
```

| 场地 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | 必需的 | 唯一标识符 |
| name | str | 必需的 | 显示名称 |
| tenant＿id | str | 必需的 | Azure AD 租户 ID |
| client＿id | str | 必需的 | Azure AD 应用程序客户端 ID |
| client＿secret | str | 必需的 | Azure AD 应用程序客户端密钥 |
| hostname | str | 必需的 | SharePoint 主机名 |
| site＿path | Optional［str］ | None | 站点路径（例如，／sites／Engineering） |
| site＿id | Optional［str］ | None | 完整站点 ID |
| folder＿path | Optional［str］ | None | 默认文件夹路径 |


#### AzureBlobConfig

```python

from agno.knowledge.remote_content import AzureBlobConfig

azure = AzureBlobConfig(
    id="azure-docs",
    name="Azure Blob Documents",
    tenant_id="...",
    client_id="...",
    client_secret="...",
    storage_account="mystorageaccount",
    container="documents",
)
```
| 场地 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| id | str | 必需的 | 唯一标识符 |
| name | str | 必需的 | 显示名称 |
| tenant＿id | str | 必需的 | Azure AD 租户 ID |
| client＿id | str | 必需的 | Azure AD 应用程序客户端 ID |
| client＿secret | str | 必需的 | Azure AD 应用程序客户端密钥 |
| storage＿account | str | 必需的 | Azure 存储帐户名称 |
| container | str | 必需的 | Blob 容器名称 |
| prefix | Optional［str］ | None | 默认前缀 |

需要存储帐户上的存储 Blob 数据读取者（或贡献者）角色。


### 插入内容

每个配置都有.file()返回.folder()内容引用的方法knowledge.insert()。

```python
# Single file
knowledge.insert(
    name="Architecture Doc",
    remote_content=s3.file("docs/architecture.pdf"),
)

# Entire folder
knowledge.insert(
    name="All Specs",
    remote_content=gcs.folder("specs/"),
)

# GitHub file from a specific branch
knowledge.insert(
    name="README",
    remote_content=github.file("README.md", branch="develop"),
)

# SharePoint file from a specific site
knowledge.insert(
    name="Policy",
    remote_content=sharepoint.file("Shared Documents/policy.pdf", site_path="/sites/HR"),
)
```

### 浏览 S3 文件
S3Config支持分页文件列表list_files()。这对于构建文件选择器或在摄取之前浏览存储桶内容非常有用。

```python
result = s3.list_files(prefix="reports/", limit=50, page=1)

for folder in result.folders:
    print(f"Folder: {folder['name']}")

for file in result.files:
    print(f"File: {file['name']} ({file['size']} bytes)")

print(f"Page {result.page} of {result.total_pages}")
```

| 范围 | 类型 | 默认 | 描述 |
| :--- | :--- | :--- | :--- |
| prefix | Optional［str］ | None | 路径前缀过滤器。覆盖配置中的设置 prefix 。 |
| delimiter | str | ＂／＂ | 文件夹分隔符 |
| limit | int | 100 | 每页文件数（1－1000） |
| page | int | 1 | 页码（从1开始索引） |

也有异步变体 alist_files（） 使用相同签名。

### 多来源
在单个 Knowledge 实例上注册多个提供商。

```python
knowledge = Knowledge(
    vector_db=vector_db,
    contents_db=contents_db,
    content_sources=[s3, gcs, github, sharepoint, azure],
)

# Insert from different sources
knowledge.insert(name="S3 Doc", remote_content=s3.file("doc.pdf"))
knowledge.insert(name="GitHub Doc", remote_content=github.file("README.md"))
```

在运行 AgentOS 时，注册源通过 /knowledge/{id}/sources API 端点公开用于列表和浏览。


### 后续步骤
| 任务 | 指导 |
| :--- | :--- |
| 内容类型概述 | 内容类型 |
| 筛选搜索结果 | 过滤 |
| 建立矢量数据库 | 向量数据库 |

## 具备知识的代理人
Agno 代理默认使用 Agentic RAG，这意味着当我们向代理提供知识时，它会在运行时搜索该知识库，寻找完成任务所需的具体信息。

```python
import asyncio

from agno.agent import Agent
from agno.db.postgres.postgres import PostgresDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

db = PostgresDb(
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    knowledge_table="knowledge_contents",
)

# Create Knowledge Instance
knowledge = Knowledge(
    name="Basic SDK Knowledge Base",
    description="Agno 2.0 Knowledge Implementation",
    contents_db=db,
    vector_db=PgVector(
        table_name="vectors",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OpenAIEmbedder(),
    ),
)
# Add from URL to the knowledge base
asyncio.run(
    knowledge.ainsert(
        name="Recipes",
        url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
        metadata={"user_tag": "Recipes from website"},
    )
)

agent = Agent(
    name="My Agent",
    description="Agno 2.0 Agent Implementation",
    knowledge=knowledge,
    search_knowledge=True,
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup?",
    markdown=True,
)
```
我们可以通过以下方式让代理访问知识库：

- 我们可以设置 search_knowledge=True，为代理添加一个 search_knowledge_base（） 工具。如果你给代理添加知识，search_knowledge默认是真。
- 我们可以设置 add_knowledge_to_context=True，根据你的用户消息自动将知识库中的引用添加到代理的上下文中。这就是传统的RAG方法。

### 自定义知识检索
如果您需要完全控制知识库搜索，您可以传递knowledge_retriever具有以下签名的自定义函数：

```python

def knowledge_retriever(agent: Agent, query: str, num_documents: Optional[int], **kwargs) -> Optional[list[dict]]:
  ...
```

如何配置带有自定义检索器的代理示例：

```python
def knowledge_retriever(agent: Agent, query: str, num_documents: Optional[int], **kwargs) -> Optional[list[dict]]:
  ...

agent = Agent(
    knowledge_retriever=knowledge_retriever,
    search_knowledge=True,
)
```

该函数在代理运行时被调用search_knowledge_base()，并由代理用于从知识库中检索参考资料。

### 知识存储
知识内容在“内容数据库”中进行跟踪，并被向量化存储在“向量数据库”中。
​
### 内容数据库
内容数据库是一个数据库，用于存储您添加到知识库中的任何内容的名称、描述、元数据和其他信息。
以下是内容数据库的架构：

| 场地 | 类型 | 描述 |
| :--- | :--- | :--- |
| id | str | 知识内容的唯一标识符。 |
| name | str | 知识内容的名称。 |
| description | str | 知识内容的描述。 |
| metadata | dict | 知识内容的元数据。 |
| type | str | 知识内容的类型。 |
| size | int | 知识内容的大小。仅适用于文件。 |
| linked＿to | str | 此内容链接到的知识内容的 ID。 |
| access＿count | int | 该内容已被访问的次数。 |
| status | str | 知识内容的状态。 |
| status＿message | str | 与知识内容状态相关的消息。 |
| created＿at | int | 知识内容创建时的时间戳。 |
| updated＿at | int | 知识内容上次更新的时间戳。 |
| external＿id | str | 知识内容的外部 ID。用于使用外部矢量存储系统，例如 JightRAG。 |

该数据最好显示在AgentOS UI 的知识页面上

### 向量数据库
向量数据库为从密集信息中快速检索相关结果提供了最佳解决方案。
​
### 添加内容
知识库内容添加时的典型处理流程如下：  

1. 解析内容.   
阅读器用于根据插入的内容类型解析内容。  

2. 将信息分块处理.    
内容被拆分成更小的部分，以确保我们的搜索查询只返回相关结果。

3. 嵌入每个数据块.   
将数据块转换为嵌入向量并存储在向量数据库中。

例如，要将 PDF 文件添加到知识库：

```python
...
knowledge = Knowledge(
    name="Basic SDK Knowledge Base",
    description="Agno 2.0 Knowledge Implementation",
    vector_db=vector_db,
    contents_db=contents_db,
)

asyncio.run(
    knowledge.ainsert(
        name="CV",
        path="cookbook/08_knowledge/testing_resources/cv_1.pdf",
        metadata={"user_tag": "Engineering Candidates"},
    )
)
```

### 示例：代理 RAG 代理
让我们构建一个RAG 代理，它可以回答 PDF 中的问题。

1. 建立数据库

让我们将Postgres其用作内容数据库和矢量数据库。

安装Docker Desktop并使用以下命令在5532端口运行Postgres：

```python
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

`uv pip install -U pgvector pypdf psycopg sqlalchemy`

2. 执行代理 RAG

agentic_rag.py创建一个包含以下内容的文件

```python
import asyncio
from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

db = PostgresDb(
    db_url=db_url,
    knowledge_table="knowledge_contents",
)

knowledge = Knowledge(
    contents_db=db,
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    )
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.2"),
    db=db,
    knowledge=knowledge,
    markdown=True,
)
if __name__ == "__main__":
    asyncio.run(
        knowledge.ainsert(
            name="Recipes",
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            metadata={"user_tag": "Recipes from website"}
        )
    )
    # Create and use the agent
    asyncio.run(
        agent.aprint_response(
            "How do I make chicken and galangal in coconut milk soup?",
            markdown=True,
        )
    )
```

3. 运行代理

运行代理

python agentic_rag.py


### 拥有知识的团队

团队协作中使用知识库。

团队可以使用知识库来存储和检索信息，就像代理一样：isolate_vector_search在跨团队或租户共享向量数据库时，用于限定检索范围。

```python
from pathlib import Path

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.vectordb.lancedb import LanceDb

# Setup paths
cwd = Path(__file__).parent
tmp_dir = cwd.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Initialize knowledge base
agno_docs_knowledge = Knowledge(
    vector_db=LanceDb(
        uri=str(tmp_dir.joinpath("lancedb")),
        table_name="agno_docs",
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

agno_docs_knowledge.insert(url="https://docs.agno.com/llms-full.txt")

hackernews_agent = Agent(
    name="HackerNews Agent",
    role="Search HackerNews for tech news",
    model=OpenAIResponses(id="gpt-5.2"),
    tools=[HackerNewsTools()],
    instructions=["Always include sources"],
)

team_with_knowledge = Team(
    name="Team with Knowledge",
    members=[hackernews_agent],
    model=OpenAIResponses(id="gpt-5.2"),
    knowledge=agno_docs_knowledge,
    show_members_responses=True,
    markdown=True,
)

if __name__ == "__main__":
    team_with_knowledge.print_response("Tell me about the Agno framework", stream=True)

```

## 性能提示


优化知识库性能、搜索质量和内容加载速度。

Agno 的默认设置在大多数情况下都能很好地满足需求。但如果您遇到搜索速度慢、内存不足或搜索结果不佳等问题，进行一些策略性调整或许会有所帮助。

### 快速取胜
​
1. 选择合适的矢量数据库

数据库选择在大规模应用中影响最大：
| 数据库 | 用例 |
| :--- | :--- |
| LanceDB／ChromaDB | 开发、测试（零设置） |
| PgVector | 生产环境最多处理 100 万份文档，需要 SQL |
| 松果 | 托管服务，自动扩展 |

```python
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.pgvector import PgVector

# Development
dev_db = LanceDb(table_name="docs", uri="./local_db")

# Production
prod_db = PgVector(table_name="docs", db_url=db_url)
```

2. 跳过已处理的文件     
重新运行数据摄取时速度提升最大：
 
```python
knowledge.insert(
    path="documents/",
    skip_if_exists=True,  # Don't reprocess existing files
)

# Batch loading with filters
knowledge.insert_many(
    paths=["docs/", "policies/"],
    skip_if_exists=True,
    include=["*.pdf", "*.md"],
    exclude=["*temp*", "*draft*"]
)
```

3. 使用元数据过滤器.      
搜索前缩小搜索范围：

```python
# Slow: search everything
results = knowledge.search("deployment process")

# Fast: filter first, then search
results = knowledge.search(
    query="deployment process",
    filters={"department": "engineering", "type": "procedure"}
)

# Validate filters to catch typos
valid_filters, invalid_keys = knowledge.validate_filters({
    "department": "engineering",
    "invalid_key": "value"  # This gets flagged
})
```

4. 将分块与内容相匹配
| 战略 | 速度 | 质量 | 最适合 |
| :--- | :--- | :--- | :--- |
| 固定尺寸 | 快速地 | 好的 | 均匀含量 |
| 语义 | 慢点 | 最好的 | 复杂文件 |
| 递归 | 快速地 | 好的 | 结构化文档 |

```python
from agno.knowledge.chunking.fixed_size_chunking import FixedSizeChunking
from agno.knowledge.chunking.semantic_chunking import SemanticChunking

# Fast processing
FixedSizeChunking(chunk_size=5000, overlap=200)

# Better quality (slower)
SemanticChunking(similarity_threshold=0.5)
```


5. 使用异步进行批量操作
同时处理多个数据源：


```python
import asyncio

async def load_knowledge():
    await asyncio.gather(
        knowledge.ainsert(path="docs/hr/"),
        knowledge.ainsert(path="docs/engineering/"),
        knowledge.ainsert(url="https://company.com/api-docs"),
    )

asyncio.run(load_knowledge())
```

### 常见问题
​
**不相关的搜索结果**

原因：数据块过大/过小，数据块划分策略错误。

修复：
- 尝试语义分块以获得更好的上下文
- 增加max_results检查次数以查看相关结果是否排名较低
- 添加元数据筛选器以缩小范围

**内容加载缓慢**.  

原因：重新处理现有文件，对大型数据集进行语义分块。

修复：
- 使用skip_if_exists=True
- 切换到固定大小分块
- 分批处理

```python
# Only process new PDFs
knowledge.insert(
    path="documents/",
    include=["*.pdf"],
    exclude=["*draft*", "*backup*"],
    skip_if_exists=True,
)
```

**内存问题**       
原因：一次加载太多大文件，数据块大小过大。

修复：
- 小批量加工
- 减小块大小
- 使用包含/排除模式
- 清除过时的内容knowledge.remove_content_by_id(content_id)


### 高级优化
​
混合搜索
结合向量搜索和关键词搜索：

```python
from agno.vectordb.pgvector import PgVector, SearchType

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    search_type=SearchType.hybrid,
)

```

### 重新排名
优化结果排序：

```python
from agno.knowledge.reranker.cohere import CohereReranker

vector_db = PgVector(
    table_name="docs",
    db_url=db_url,
    reranker=CohereReranker(model="rerank-v3.5", top_n=10),
)
```

#### 更小的嵌入尺寸
牺牲一些质量来换取更快的搜索速度：
```python
from agno.knowledge.embedder.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    id="text-embedding-3-large",
    dimensions=1024,  # Instead of 3072
)
```

#### 监测

```python
import time

# Time searches
start = time.time()
results = knowledge.search("test query", max_results=5)
print(f"Search: {time.time() - start:.2f}s")

# Check failed content
content_list, total = knowledge.get_content()
for content in content_list:
    if content.status == "failed":
        status, message = knowledge.get_content_status(content.id)
        print(f"{content.name}: {message}")
```

