# 1.下载开源项目

```shell
git clone https://github.com/facebookresearch/llama-recipes.git
```

# 2.下载开源模型

```shell
https://huggingface.co/meta-llama

# 本教程采用的模型是：
# meta-llama/Llama-2-7b-chat-hf
```

<img src="pngs/00.png" width=700 height=400/>

# 3.  创建虚拟环境

```shell
# 创建Conda新的虚拟环境（如已创建，请忽略！）
conda create -n llama python=3.10             # 构建一个虚拟环境，名为：llama
conda init bash && source /root/.bashrc   # 更新bashrc中的环境变量

# 将新的Conda虚拟环境加入jupyterlab中
conda activate llama                         # 切换到创建的虚拟环境：llama
conda install ipykernel
ipython kernel install --user --name=llama   # 设置kernel，--user表示当前用户，llama为虚拟环境名称
```

# 4. 安装项目依赖

```shell
# 进入llama-recipes项目目录下，执行下面的命令

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .[tests,auditnlg,vllm]
```

# 5. 推理案例实践

## 5.1 摘要生成

```shell
# 摘要生成
python examples/inference.py --model_name /root/autodl-tmp/llama-7b-chat-hf --prompt_file examples/samsum_prompt.txt
```

## 5.2 chat completion

```shell
# chat completion 
python examples/chat_completion/chat_completion.py --model_name /root/autodl-tmp/llama-7b-chat-hf --prompt_file examples/chat_completion/chats.json
```

## 5.3 code completion

```shell
# code completion 代码补全
python examples/code_llama/code_completion_example.py --model_name /root/autodl-tmp/llama-7b-chat-hf --prompt_file examples/code_llama/code_completion_prompt.txt --temperature 0.2 --top_p 0.9
```



## 5.4 code infilling

```shell
# code infilling 代码填充
python examples/code_llama/code_infilling_example.py --model_name /root/autodl-tmp/llama-7b-chat-hf --prompt_file examples/code_llama/code_infilling_prompt.txt --temperature 0.2 --top_p 0.9
```


