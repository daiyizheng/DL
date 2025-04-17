# LLAMA

```shell
# 原来改过端口之后要加端口号才能正常运行命令：
OLLAMA_HOST=172.16.10.104:11434 ~/resources/soft/ollama/bin/ollama list
OLLAMA_HOST=127.0.0.1:10001 ollama list
OLLAMA_HOST=127.0.0.1:10001 ollama ps
OLLAMA_HOST=127.0.0.1:10001 ollama run qwen2:72b
```
我这里对应的`.service`文件是这样：
```shell
## 做成服务器

[Unit]
Description=Ollama Service
After=network-online.target
 
[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin"
Environment="OLLAMA_HOST=0.0.0.0:10001"
Environment="OLLAMA_KEEP_ALIVE=1h"
Environment="OLLAMA_NUM_PARALLEL=5"
 
[Install]
WantedBy=default.target
```
