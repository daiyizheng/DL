**第1步: 部署LLaMA-2-7B模型服务

```
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5005 --model /root/autodl-tmp/llama-7b-chat-hf
```

**第2步: 执行notebook中的代码

```
RAG_chatbot.ipynb
```
