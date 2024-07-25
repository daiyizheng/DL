<!--
 * @Description: 
 * @Author: shenlei
 * @Modified: linhui
 * @Date: 2023-12-19 10:31:41
 * @LastEditTime: 2024-02-24 23:55:57
 * @LastEditors: shenlei
-->

<h1 align="center">BCEmbedding: Bilingual and Crosslingual Embedding for RAG</h1>

<div align="center">
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/license-Apache--2.0-yellow">
    </a>
        
    <a href="https://twitter.com/YDopensource">
      <img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}">
    </a>
        
</div>
<br>

<p align="center">
  <strong style="background-color: green;">English</strong>
  |
  <a href="./README_zh.md" target="_Self">简体中文</a>
</p>

<details open="open">
<summary>Click to Open Contents</summary>

- <a href="#-bilingual-and-crosslingual-superiority" target="_Self">🌐 Bilingual and Crosslingual Superiority</a>
- <a href="#-key-features" target="_Self">💡 Key Features</a>
- <a href="#-latest-updates" target="_Self">🚀 Latest Updates</a>
- <a href="#-model-list" target="_Self">🍎 Model List</a>
- <a href="#-manual" target="_Self">📖 Manual</a>
  - <a href="#installation" target="_Self">Installation</a>
  - <a href="#quick-start" target="_Self">Quick Start (`transformers`, `sentence-transformers`)</a>
  - <a href="#embedding-and-reranker-integrations-for-rag-frameworks" target="_Self">Embedding and Reranker Integrations for RAG Frameworks (`langchain`, `llama_index`)</a>
- <a href="#%EF%B8%8F-evaluation" target="_Self">⚙️ Evaluation</a>
  - <a href="#evaluate-semantic-representation-by-mteb" target="_Self">Evaluate Semantic Representation by MTEB</a>
  - <a href="#evaluate-rag-by-llamaindex" target="_Self">Evaluate RAG by LlamaIndex</a>
- <a href="#-leaderboard" target="_Self">📈 Leaderboard</a>
  - <a href="#semantic-representation-evaluations-in-mteb" target="_Self">Semantic Representation Evaluations in MTEB</a>
  - <a href="#rag-evaluations-in-llamaindex" target="_Self">RAG Evaluations in LlamaIndex</a>
- <a href="#-youdaos-bcembedding-api" target="_Self">🛠 Youdao's BCEmbedding API</a>
- <a href="#-wechat-group" target="_Self">🧲 WeChat Group</a>
- <a href="#%EF%B8%8F-citation" target="_Self">✏️ Citation</a>
- <a href="#-license" target="_Self">🔐 License</a>
- <a href="#-related-links" target="_Self">🔗 Related Links</a>

</details>
<br>

**B**ilingual and **C**rosslingual **Embedding** (`BCEmbedding`) in English and Chinese, developed by NetEase Youdao, encompasses `EmbeddingModel` and `RerankerModel`. The `EmbeddingModel` specializes in generating semantic vectors, playing a crucial role in semantic search and question-answering, and the `RerankerModel` excels at refining search results and ranking tasks.

`BCEmbedding` serves as the cornerstone of Youdao's Retrieval Augmented Generation (RAG) implementation, notably [QAnything](http://qanything.ai) [[github](https://github.com/netease-youdao/qanything)], an open-source implementation widely integrated in various Youdao products like [Youdao Speed Reading](https://read.youdao.com/#/home) and [Youdao Translation](https://fanyi.youdao.com/download-Mac?keyfrom=fanyiweb_navigation).

Distinguished for its bilingual and crosslingual proficiency, `BCEmbedding` excels in bridging Chinese and English linguistic gaps, which achieves

- **A high performance on <a href="#semantic-representation-evaluations-in-mteb" target="_Self">Semantic Representation Evaluations in MTEB</a>**;
- **A new benchmark in the realm of <a href="#rag-evaluations-in-llamaindex" target="_Self">RAG Evaluations in LlamaIndex</a>**.

<img src="./Docs/assets/rag_eval_multiple_domains_summary.jpg">

### Our Goals

Provide a bilingual and crosslingual two-stage retrieval model repository for the RAG community, which can be used directly without finetuning, including `EmbeddingModel` and `RerankerModel`:

- One Model: `EmbeddingModel` handle **bilingual and crosslingual** retrieval task in English and Chinese. `RerankerModel` supports **English, Chinese, Japanese and Korean**.
- One Model: **Cover common business application scenarios with RAG optimization**. e.g. Education, Medical Scenario, Law, Finance, Literature, FAQ, Textbook, Wikipedia, General Conversation.
- Easy to Integrate: We provide **API** in `BCEmbedding` for LlamaIndex and LangChain integrations.
- Others Points:
  - `RerankerModel` supports **long passages (more than 512 tokens, less than 32k tokens) reranking**;
  - `RerankerModel` provides **meaningful relevance score** that helps to remove passages with low quality.
  - `EmbeddingModel` **does not need specific instructions**.

### Third-party Examples

- RAG applications: [QAnything](https://github.com/netease-youdao/qanything), [HuixiangDou](https://github.com/InternLM/HuixiangDou), [ChatPDF](https://github.com/shibing624/ChatPDF).
- Efficient inference: [ChatLLM.cpp](https://github.com/foldl/chatllm.cpp), [Xinference](https://github.com/xorbitsai/inference).

## 🌐 Bilingual and Crosslingual Superiority

Existing embedding models often encounter performance challenges in bilingual and crosslingual scenarios, particularly in Chinese, English and their crosslingual tasks. `BCEmbedding`, leveraging the strength of Youdao's translation engine, excels in delivering superior performance across monolingual, bilingual, and crosslingual settings.

`EmbeddingModel` supports ***Chinese (ch) and English (en)*** (more languages support will come soon), while `RerankerModel` supports ***Chinese (ch), English (en), Japanese (ja) and Korean (ko)***.

## 💡 Key Features

- **Bilingual and Crosslingual Proficiency**: Powered by Youdao's translation engine, excelling in Chinese, English and their crosslingual retrieval task, with upcoming support for additional languages.
- **RAG-Optimized**: Tailored for diverse RAG tasks including **translation, summarization, and question answering**, ensuring accurate **query understanding**. See <a href="#rag-evaluations-in-llamaindex" target="_Self">RAG Evaluations in LlamaIndex</a>.
- **Efficient and Precise Retrieval**: Dual-encoder for efficient retrieval of `EmbeddingModel` in first stage, and cross-encoder of `RerankerModel` for enhanced precision and deeper semantic analysis in second stage.
- **Broad Domain Adaptability**: Trained on diverse datasets for superior performance across various fields.
- **User-Friendly Design**: Instruction-free, versatile use for multiple tasks without specifying query instruction for each task.
- **Meaningful Reranking Scores**: `RerankerModel` provides relevant scores to improve result quality and optimize large language model performance.
- **Proven in Production**: Successfully implemented and validated in Youdao's products.

## 🚀 Latest Updates

- ***2024-02-04***: **Technical Blog** - See <a href="https://zhuanlan.zhihu.com/p/681370855">为RAG而生-BCEmbedding技术报告</a>.
- ***2024-01-16***: **LangChain and LlamaIndex Integrations** - See <a href="#embedding-and-reranker-integrations-for-rag-frameworks" target="_Self">more</a>.
- ***2024-01-03***: **Model Releases** - [bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) and [bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1) are available.
- ***2024-01-03***: **Eval Datasets** [[CrosslingualMultiDomainsDataset](https://huggingface.co/datasets/maidalun1020/CrosslingualMultiDomainsDataset)] - Evaluate the performance of RAG, using [LlamaIndex](https://github.com/run-llama/llama_index).
- ***2024-01-03***: **Eval Datasets** [[Details](./BCEmbedding/evaluation/c_mteb/Retrieval.py)] - Evaluate the performance of crosslingual semantic representation, using [MTEB](https://github.com/embeddings-benchmark/mteb).

## 🍎 Model List

| Model Name            |     Model Type     |   Languages   | Parameters |                                                                          Weights                                                                          |
| :-------------------- | :----------------: | :------------: | :--------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| bce-embedding-base_v1 | `EmbeddingModel` |     ch, en     |    279M    | [Huggingface](https://huggingface.co/maidalun1020/bce-embedding-base_v1), [国内通道](https://hf-mirror.com/maidalun1020/bce-embedding-base_v1) |
| bce-reranker-base_v1  | `RerankerModel` | ch, en, ja, ko |    279M    |  [Huggingface](https://huggingface.co/maidalun1020/bce-reranker-base_v1), [国内通道](https://hf-mirror.com/maidalun1020/bce-reranker-base_v1)  |

## 📖 Manual

### Installation

First, create a conda environment and activate it.

```bash
conda create --name bce python=3.10 -y
conda activate bce
```

Then install `BCEmbedding` for minimal installation (To avoid cuda version conflicting, you should install [`torch`](https://pytorch.org/get-started/previous-versions/) that is compatible to your system cuda version manually first):

```bash
pip install BCEmbedding==0.1.3
```

Or install from source (**recommended**):

```bash
git clone git@github.com:netease-youdao/BCEmbedding.git
cd BCEmbedding
pip install -v -e .
```

### Quick Start

#### 1. Based on `BCEmbedding`

Use `EmbeddingModel`, and `cls` [pooler](./BCEmbedding/models/embedding.py#L24) is default.

```python
from BCEmbedding import EmbeddingModel

# list of sentences
sentences = ['sentence_0', 'sentence_1']

# init embedding model
model = EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")

# extract embeddings
embeddings = model.encode(sentences)
```

Use `RerankerModel` to calculate relevant scores and rerank:

```python
from BCEmbedding import RerankerModel

# your query and corresponding passages
query = 'input_query'
passages = ['passage_0', 'passage_1']

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# init reranker model
model = RerankerModel(model_name_or_path="maidalun1020/bce-reranker-base_v1")

# method 0: calculate scores of sentence pairs
scores = model.compute_score(sentence_pairs)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)
```

NOTE:

- In [`RerankerModel.rerank`](./BCEmbedding/models/reranker.py#L137) method, we provide an advanced preproccess that we use in production for making `sentence_pairs`, when "passages" are very long.

#### 2. Based on `transformers`

For `EmbeddingModel`:

```python
from transformers import AutoModel, AutoTokenizer

# list of sentences
sentences = ['sentence_0', 'sentence_1']

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-embedding-base_v1')
model = AutoModel.from_pretrained('maidalun1020/bce-embedding-base_v1')

device = 'cuda'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# get embeddings
outputs = model(**inputs_on_device, return_dict=True)
embeddings = outputs.last_hidden_state[:, 0]  # cls pooler
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # normalize
```

For `RerankerModel`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# init model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('maidalun1020/bce-reranker-base_v1')
model = AutoModelForSequenceClassification.from_pretrained('maidalun1020/bce-reranker-base_v1')

device = 'cuda'  # if no GPU, set "cpu"
model.to(device)

# get inputs
inputs = tokenizer(sentence_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

# calculate scores
scores = model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
scores = torch.sigmoid(scores)
```

#### 3. Based on `sentence_transformers`

For `EmbeddingModel`:

```python
from sentence_transformers import SentenceTransformer

# list of sentences
sentences = ['sentence_0', 'sentence_1', ...]

# init embedding model
## New update for sentence-trnasformers. So clean up your "`SENTENCE_TRANSFORMERS_HOME`/maidalun1020_bce-embedding-base_v1" or "～/.cache/torch/sentence_transformers/maidalun1020_bce-embedding-base_v1" first for downloading new version.
model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")

# extract embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)
```

For `RerankerModel`:

```python
from sentence_transformers import CrossEncoder

# init reranker model
model = CrossEncoder('maidalun1020/bce-reranker-base_v1', max_length=512)

# calculate scores of sentence pairs
scores = model.predict(sentence_pairs)
```

### Embedding and Reranker Integrations for RAG Frameworks

#### 1. Used in `langchain`

We provide `BCERerank` in `BCEmbedding.tools.langchain` that inherits the advanced preproc tokenization of `RerankerModel`.

- Install langchain first
```bash
pip install langchain==0.1.0
pip install langchain-community==0.0.9
pip install langchain-core==0.1.7
pip install langsmith==0.0.77
```

- Demo
```python
# We provide the advanced preproc tokenization for reranking.
from BCEmbedding.tools.langchain import BCERerank

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever


# init embedding model
embedding_model_name = 'maidalun1020/bce-embedding-base_v1'
embedding_model_kwargs = {'device': 'cuda:0'}
embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True, 'show_progress_bar': False}

embed_model = HuggingFaceEmbeddings(
  model_name=embedding_model_name,
  model_kwargs=embedding_model_kwargs,
  encode_kwargs=embedding_encode_kwargs
)

reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}
reranker = BCERerank(**reranker_args)

# init documents
documents = PyPDFLoader("BCEmbedding/tools/eval_rag/eval_pdfs/Comp_en_llama2.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# example 1. retrieval with embedding and reranker
retriever = FAISS.from_documents(texts, embed_model, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT).as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.3, "k": 10})

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

response = compression_retriever.get_relevant_documents("What is Llama 2?")
```

#### 2. Used in `llama_index`

We provide `BCERerank` in `BCEmbedding.tools.llama_index` that inherits the advanced preproc tokenization of `RerankerModel`.

- Install llama_index first

```bash
pip install llama-index==0.9.42.post2
```

- Demo
```python
# We provide the advanced preproc tokenization for reranking.
from BCEmbedding.tools.llama_index import BCERerank

import os
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever

# init embedding model and reranker model
embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cuda:0'}
embed_model = HuggingFaceEmbedding(**embed_args)

reranker_args = {'model': 'maidalun1020/bce-reranker-base_v1', 'top_n': 5, 'device': 'cuda:1'}
reranker_model = BCERerank(**reranker_args)

# example #1. extract embeddings
query = 'apples'
passages = [
        'I like apples', 
        'I like oranges', 
        'Apples and oranges are fruits'
    ]
query_embedding = embed_model.get_query_embedding(query)
passages_embeddings = embed_model.get_text_embedding_batch(passages)

# example #2. rag example
llm = OpenAI(model='gpt-3.5-turbo-0613', api_key=os.environ.get('OPENAI_API_KEY'), api_base=os.environ.get('OPENAI_BASE_URL'))
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

documents = SimpleDirectoryReader(input_files=["BCEmbedding/tools/eval_rag/eval_pdfs/Comp_en_llama2.pdf"]).load_data()
node_parser = SimpleNodeParser.from_defaults(chunk_size=400, chunk_overlap=80)
nodes = node_parser.get_nodes_from_documents(documents[0:36])
index = VectorStoreIndex(nodes, service_context=service_context)

query = "What is Llama 2?"

# example #2.1. retrieval with EmbeddingModel and RerankerModel
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10, service_context=service_context)
retrieval_by_embedding = vector_retriever.retrieve(query)
retrieval_by_reranker = reranker_model.postprocess_nodes(retrieval_by_embedding, query_str=query)

# example #2.2. query with EmbeddingModel and RerankerModel
query_engine = index.as_query_engine(node_postprocessors=[reranker_model])
query_response = query_engine.query(query)
```

## ⚙️ Evaluation

### Evaluate Semantic Representation by MTEB

We provide evaluation tools for `embedding` and `reranker` models, based on [MTEB](https://github.com/embeddings-benchmark/mteb) and [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB).

First, install `MTEB`:

```
pip install mteb==1.1.1
```

#### 1. Embedding Models

Just run following cmd to evaluate `your_embedding_model` (e.g. `maidalun1020/bce-embedding-base_v1`) in **bilingual and crosslingual settings** (e.g. `["en", "zh", "en-zh", "zh-en"]`).

```bash
python BCEmbedding/tools/eval_mteb/eval_embedding_mteb.py --model_name_or_path maidalun1020/bce-embedding-base_v1 --pooler cls
```

The total evaluation tasks contain ***114 datasets*** of **"Retrieval", "STS", "PairClassification", "Classification", "Reranking" and "Clustering"**.

***NOTE:***

- **All models are evaluated in their recommended pooling method (`pooler`)**.
  - `mean` pooler: "jina-embeddings-v2-base-en", "m3e-base", "m3e-large", "e5-large-v2", "multilingual-e5-base", "multilingual-e5-large" and "gte-large".
  - `cls` pooler: Other models.
- "jina-embeddings-v2-base-en" model should be loaded with `trust_remote_code`.

```bash
python BCEmbedding/tools/eval_mteb/eval_embedding_mteb.py --model_name_or_path {mean_pooler_models} --pooler mean

python BCEmbedding/tools/eval_mteb/eval_embedding_mteb.py --model_name_or_path jinaai/jina-embeddings-v2-base-en --pooler mean --trust_remote_code
```

#### 2. Reranker Models

Run following cmd to evaluate `your_reranker_model` (e.g. "maidalun1020/bce-reranker-base_v1") in **bilingual and crosslingual settings** (e.g. `["en", "zh", "en-zh", "zh-en"]`).

```bash
python BCEmbedding/tools/eval_mteb/eval_reranker_mteb.py --model_name_or_path maidalun1020/bce-reranker-base_v1
```

The evaluation tasks contain ***12 datasets*** of **"Reranking"**.

#### 3. Metrics Visualization Tool

We provide a one-click script to summarize evaluation results of `embedding` and `reranker` models as [Embedding Models Evaluation Summary](./Docs/EvaluationSummary/embedding_eval_summary.md) and [Reranker Models Evaluation Summary](./Docs/EvaluationSummary/reranker_eval_summary.md).

```bash
python BCEmbedding/evaluation/mteb/summarize_eval_results.py --results_dir {your_embedding_results_dir | your_reranker_results_dir}
```

### Evaluate RAG by LlamaIndex

[LlamaIndex](https://github.com/run-llama/llama_index) is a famous data framework for LLM-based applications, particularly in RAG. Recently, a [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) has evaluated the popular embedding and reranker models in RAG pipeline and attracts great attention. Now, we follow its pipeline to evaluate our `BCEmbedding`.

First, install LlamaIndex, and upgrade `transformers` to 4.36.0:

```bash
pip install transformers==4.36.0

pip install llama-index==0.9.22
```

Export your "openai" and "cohere" app keys, and openai base url (e.g. "https://api.openai.com/v1") to env:

```bash
export OPENAI_BASE_URL={openai_base_url}  # https://api.openai.com/v1
export OPENAI_API_KEY={your_openai_api_key}
export COHERE_APPKEY={your_cohere_api_key}
```

#### 1. Metrics Definition

- Hit Rate:

  Hit rate calculates the fraction of queries where the correct answer is found within the top-k retrieved documents. In simpler terms, it's about how often our system gets it right within the top few guesses. ***The larger, the better.***
- Mean Reciprocal Rank (MRR):

  For each query, MRR evaluates the system's accuracy by looking at the rank of the highest-placed relevant document. Specifically, it's the average of the reciprocals of these ranks across all the queries. So, if the first relevant document is the top result, the reciprocal rank is 1; if it's second, the reciprocal rank is 1/2, and so on. ***The larger, the better.***

#### 2. Reproduce [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)

In order to compare our `BCEmbedding` with other embedding and reranker models fairly, we provide a one-click script to reproduce results of the LlamaIndex Blog, including our `BCEmbedding`:

```bash
# There should be two GPUs available at least.
CUDA_VISIBLE_DEVICES=0,1 python BCEmbedding/tools/eval_rag/eval_llamaindex_reproduce.py
```

Then, summarize the evaluation results by:

```bash
python BCEmbedding/tools/eval_rag/summarize_eval_results.py --results_dir BCEmbedding/results/rag_reproduce_results
```

Results reproduced from the LlamaIndex Blog can be checked in ***[Reproduced Summary of RAG Evaluation](./Docs/EvaluationSummary/rag_eval_reproduced_summary.md)***, with some obvious ***conclusions***:

- In `WithoutReranker` setting, our `bce-embedding-base_v1` outperforms all the other embedding models.
- With fixing the embedding model, our `bce-reranker-base_v1` achieves the best performance.
- ***The combination of `bce-embedding-base_v1` and `bce-reranker-base_v1` is SOTA.***

#### 3. Broad Domain Adaptability

The evaluation of [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) is **monolingual, small amount of data, and specific domain** (just including "llama2" paper). In order to evaluate the **broad domain adaptability, bilingual and crosslingual capability**, we follow the blog to build a multiple domains evaluation dataset (includding "Computer Science", "Physics", "Biology", "Economics", "Math", and "Quantitative Finance". [Details](./BCEmbedding/tools/eval_rag/eval_pdfs/)), named [CrosslingualMultiDomainsDataset](https://huggingface.co/datasets/maidalun1020/CrosslingualMultiDomainsDataset):

- To prevent test data leakage, English eval data is selected from the latest English articles in various fields on ArXiv, up to date December 30, 2023. Chinese eval data is selected from high-quality, as recent as possible, Chinese articles in the corresponding fields on Semantic Scholar.
- Use OpenAI `gpt-4-1106-preview` to produce eval data for high quality.

First, run following cmd to evaluate the most popular and powerful embedding and reranker models:

```bash
# There should be two GPUs available at least.
CUDA_VISIBLE_DEVICES=0,1 python BCEmbedding/tools/eval_rag/eval_llamaindex_multiple_domains.py
```

Then, run the following script to summarize the evaluation results:

```bash
python BCEmbedding/tools/eval_rag/summarize_eval_results.py --results_dir BCEmbedding/results/rag_results
```

The summary of multiple domains evaluations can be seen in <a href="#1-multiple-domains-scenarios" target="_Self">Multiple Domains Scenarios</a>.

## 📈 Leaderboard

### Semantic Representation Evaluations in MTEB

#### 1. Embedding Models

| Model                               | Dimensions |  Pooler  | Instructions | Retrieval (47) |    STS (19)    | PairClassification (5) | Classification (21) | Reranking (12) | Clustering (15) | ***AVG*** (119) |
| :---------------------------------- | :--------: | :------: | :----------: | :-------------: | :-------------: | :--------------------: | :-----------------: | :-------------: | :-------------: | :---------------------: |
| bge-base-en-v1.5                    |    768    | `cls` |     Need     |      37.14      |      55.06      |         75.45         |        59.73        |      43.00      |      37.74      |          47.19          |
| bge-base-zh-v1.5                    |    768    | `cls` |     Need     |      47.63      |      63.72      |         77.40         |        63.38        |      54.95      |      32.56      |          53.62          |
| bge-large-en-v1.5                   |    1024    | `cls` |     Need     |      37.18      |      54.09      |         75.00         |        59.24        |      42.47      |      37.32      |          46.80          |
| bge-large-zh-v1.5                   |    1024    | `cls` |     Need     |      47.58      |      64.73      |    **79.14**    |        64.19        |      55.98      |      33.26      |          54.23          |
| gte-large                           |    1024    | `mean` |     Free     |      36.68      |      55.22      |         74.29         |        57.73        |      42.44      |      38.51      |          46.67          |
| gte-large-zh                        |    1024    | `cls` |     Free     |      41.15      |      64.62      |         77.58         |        62.04        |      55.62      |      33.03      |          51.51          |
| jina-embeddings-v2-base-en          |    768    | `mean` |     Free     |      31.58      |      54.28      |         74.84         |        58.42        |      41.16      |      34.67      |          44.29          |
| m3e-base                            |    768    | `mean` |     Free     |      46.29      |      63.93      |         71.84         |        64.08        |      52.38      |      37.84      |          53.54          |
| m3e-large                           |    1024    | `mean` |     Free     |      34.85      |      59.74      |         67.69         |        60.07        |      48.99      |      31.62      |          46.78          |
| e5-large-v2                         |    1024    | `mean` |     Need     |      35.98      |      55.23      |         75.28         |        59.53        |      42.12      |      36.51      |          46.52          |
| multilingual-e5-base                |    768    | `mean` |     Need     |      54.73      |      65.49      |         76.97         |        69.72        |      55.01      |      38.44      |          58.34          |
| multilingual-e5-large               |    1024    | `mean` |     Need     |      56.76      | **66.79** |         78.80         |   **71.61**   |      56.49      | **43.09** |     **60.50**     |
| ***bce-embedding-base_v1*** |    768    | `cls` |     Free     | **57.60** |      65.73      |         74.96         |        69.00        | **57.29** |      38.95      |          59.43          |

***NOTE:***

- Our ***bce-embedding-base_v1*** outperforms other open-source embedding models with comparable model sizes.
- ***114 datasets including 119 eval results*** (some dataset contains multiple languages) of "Retrieval", "STS", "PairClassification", "Classification", "Reranking" and "Clustering" in ***`["en", "zh", "en-zh", "zh-en"]` setting***, including **MTEB and CMTEB**.
- The [crosslingual evaluation datasets](./BCEmbedding/evaluation/c_mteb/Retrieval.py) we released belong to `Retrieval` task.
- More evaluation details should be checked in [Embedding Models Evaluations](./Docs/EvaluationSummary/embedding_eval_summary.md).

#### 2. Reranker Models

| Model                              | Reranking (12) | ***AVG*** (12) |
| :--------------------------------- | :-------------: | :--------------------: |
| bge-reranker-base                  |      59.04      |         59.04         |
| bge-reranker-large                 |      60.86      |         60.86         |
| ***bce-reranker-base_v1*** | **61.29** |  ***61.29***  |

***NOTE:***

- Our ***bce-reranker-base_v1*** outperforms other open-source reranker models.
- ***12 datasets*** of "Reranking" in ***`["en", "zh", "en-zh", "zh-en"]` setting***.
- More evaluation details should be checked in [Reranker Models Evaluations](./Docs/EvaluationSummary/reranker_eval_summary.md).

### RAG Evaluations in LlamaIndex

#### 1. Multiple Domains Scenarios

<img src="./Docs/assets/rag_eval_multiple_domains_summary.jpg">

***NOTE:***

- Data Quality: 
  - To prevent test data leakage, English eval data is selected from the latest English articles in various fields on ArXiv, up to date December 30, 2023. Chinese eval data is selected from high-quality, as recent as possible, Chinese articles in the corresponding fields on Semantic Scholar. 
  - Use OpenAI `gpt-4-1106-preview` to produce eval data for high quality.
- Evaluated in ***`["en", "zh", "en-zh", "zh-en"]` setting***. If you are interested in monolingual setting, please check in [Chinese RAG evaluations with ["zh"] setting](./Docs/EvaluationSummary/rag_eval_multiple_domains_summary_zh.md), and [English RAG evaluations with ["en"] setting](./Docs/EvaluationSummary/rag_eval_multiple_domains_summary_en.md).
- Consistent with our ***[Reproduced Results](./Docs/EvaluationSummary/rag_eval_reproduced_summary.md)*** of [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83).
- In `WithoutReranker` setting, our `bce-embedding-base_v1` outperforms all the other embedding models.
- With fixing the embedding model, our `bce-reranker-base_v1` achieves the best performance.
- **The combination of `bce-embedding-base_v1` and `bce-reranker-base_v1` is SOTA**.

## 🛠 Youdao's BCEmbedding API

For users who prefer a hassle-free experience without the need to download and configure the model on their own systems, `BCEmbedding` is readily accessible through Youdao's API. This option offers a streamlined and efficient way to integrate BCEmbedding into your projects, bypassing the complexities of manual setup and maintenance. Detailed instructions and comprehensive API documentation are available at [Youdao BCEmbedding API](https://ai.youdao.com/DOCSIRMA/html/aigc/api/embedding/index.html). Here, you'll find all the necessary guidance to easily implement `BCEmbedding` across a variety of use cases, ensuring a smooth and effective integration for optimal results.

## 🧲 WeChat Group

Welcome to scan the QR code below and join the WeChat group.

<img src="./Docs/assets/Wechat.jpg" width="20%" height="auto">

## ✏️ Citation

If you use `BCEmbedding` in your research or project, please feel free to cite and star it:

```
@misc{youdao_bcembedding_2023,
    title={BCEmbedding: Bilingual and Crosslingual Embedding for RAG},
    author={NetEase Youdao, Inc.},
    year={2023},
    howpublished={\url{https://github.com/netease-youdao/BCEmbedding}}
}
```

## 🔐 License

`BCEmbedding` is licensed under [Apache 2.0 License](./LICENSE)

## 🔗 Related Links

[Netease Youdao - QAnything](https://github.com/netease-youdao/qanything)

[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

[MTEB](https://github.com/embeddings-benchmark/mteb)

[C_MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)

[LLama Index](https://github.com/run-llama/llama_index) | [LlamaIndex Blog](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)
