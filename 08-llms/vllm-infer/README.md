# vLLM推理

## 离线批量推理

```
from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

```
 python inference.py --model_name /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf
```

## 基于vLLM进行本地化模型服务部署

```
## 1. 基于vLLM将model部署为本地服务
python -m vllm.entrypoints.api_server --host 0.0.0.0 --port 9909 --model /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf
```

```
curl http://localhost:9909/generate -d '{
        "prompt": "Who wrote the book Innovators dilemma?",
        "max_tokens": 300,
        "temperature": 0
    }'
```

```
# 2. 如果有多个GPU,则可以部署在多个GPU上
python api_server.py --host 0.0.0.0 --port 6006 --model  /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf --tensor-parallel-size 2
```

```
# 3. 还可以将vLLM托管的Llama 2部署为OpenAI兼容服务，以便使用OpenAI API轻松替换代码。
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9909 --model  /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf
```

```
# 访问服务方式-1

curl http://localhost:9909/v1/completions -H "Content-Type: application/json" -d '{
        "model": "/slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf",
        "prompt": "Who wrote the book Innovators dilemma?",
        "max_tokens": 300,
        "temperature": 0
    }'
```

```
# 访问服务方式-2 (在 notebook 中进行访问)

from langchain.llms import VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_key = "EMPTY",
    openai_api_base = "http://xxx",
    model_name = "/slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf",
    max_tokens = 300
)

print(llm("Who wrote the book godfather?"))

'''
Who wrote The Godfather?
Francis Ford Coppola wrote the screenplay for The Godfather, which was based on the novel of the same name by Mario Puzo.
Puzo's novel was published in 1969 and became a bestseller. Coppola's film adaptation of the book, also titled The Godfather, was released in 1972 and became one of the most successful and critically acclaimed films of all time.
So, to answer your question, Mario Puzo wrote the book The Godfather, and Francis Ford Coppola adapted it into a screenplay for the film.
'''
```
