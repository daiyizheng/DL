# 1.下载开源项目

```shell
git clone https://github.com/facebookresearch/llama-recipes.git
```

# 2.下载开源模型

```shell
https://huggingface.co/meta-llama
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --local-dir meta-llama/Llama-2-7b-chat-hf

# 本教程采用的模型是：
# meta-llama/Llama-2-7b-chat-hf
```

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
python examples/inference.py --model_name /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf --prompt_file  examples/samsum_prompt.txt
```

## 5.2 chat completion

```shell
# chat completion 
python examples/chat_completion/chat_completion.py --model_name /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf --prompt_file examples/chat_completion/chats.json
```

## 5.3 code completion

```shell
# code completion 代码补全
python examples/code_llama/code_completion_example.py --model_name /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf --prompt_file examples/code_llama/code_completion_prompt.txt --temperature 0.2 --top_p 0.9
```

## 5.4 code infilling

```shell
# code infilling 代码填充
python examples/code_llama/code_infilling_example.py --model_name /slurm/home/yrd/shaolab/daiyizheng/resources/modelscope/shakechen/Llama-2-7b-chat-hf --prompt_file examples/code_llama/code_infilling_prompt.txt --temperature 0.2 --top_p 0.9
```


## 微调实践案例

### 基于llama-recipes微调

#### 第1步: 下载 alpaca_dataset 数据集

```
https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/alpaca
```


#### 第2步: 配置微调的参数

#####  **方法-1: 直接修改代码中的参数值**

```
# 根据本机GPU的情况, 在 src/llama_recipes/configs/training.py 中重新调整参数值

class train_config:
    '''用于配置神经网络模型的训练参数'''
    # 模型名称或模型文件的路径
    model_name: str="/root/autodl-tmp/llama-7b-chat-hf"
    # 是否启用全参数分片
    enable_fsdp: bool=False
    # 是否启用CPU卸载以优化FSDP的内存使用
    low_cpu_fsdp: bool=False
    # 是否在训练过程中运行验证
    run_validation: bool=True
    # 训练时的批量大小
    batch_size_training: int=8
    # 批处理策略，"packing"或"padding"
    batching_strategy: str="packing"
    # 输入的上下文长度
    context_length: int=1024
    # 梯度累积步数, 决定了在执行反向传播和更新权重之前要处理的批次数量。
    gradient_accumulation_steps: int=1
    # 是否应用梯度裁剪, 有助于防止梯度爆炸问题
    gradient_clipping: bool = False
    # 梯度裁剪阈值, 梯度裁剪的上限值
    gradient_clipping_threshold: float = 1.0
    # 训练的总轮数（epoch数）
    num_epochs: int=1
    # 数据加载时的工作线程数, 数据加载的并行程度。
    num_workers_dataloader: int=8
    # 学习率
    lr: float=1e-4
    # 权重衰减率, 正则化项，有助于防止过拟合。
    weight_decay: float=0.0
    # 学习率衰减系数, 影响学习率随时间的变化率。
    gamma: float= 0.85
    # 随机种子
    seed: int=42
    # 是否使用FP16精度
    use_fp16: bool=True
    # 是否使用混合精度训练
    mixed_precision: bool=True
    # 验证时的批量大小
    val_batch_size: int=2
    # 使用的数据集
    dataset = "alpaca_dataset"
    # 参数高效微调（PEFT）方法，如"lora"、"llama_adapter"或"prefix"。
    peft_method: str = "lora" # None , llama_adapter, prefix
    # 是否使用参数高效微调（PEFT）
    use_peft: bool=True
    # 保存PEFT模型的路径
    output_dir: str = "/root/autodl-tmp/PEFT/model"
    # 是否冻结部分层
    freeze_layers: bool = False
    # 冻结的层数
    num_freeze_layers: int = 1
    # 是否应用量化
    quantization: bool = True
    # 是否仅在一个GPU上训练
    one_gpu: bool = True
    # 是否保存模型
    save_model: bool = True
    # FSDP模型保存的根文件夹路径
    dist_checkpoint_root_folder: str="/root/autodl-tmp/FSDP/model" # will be used if using FSDP
    # FSDP模型微调后保存的文件夹名称
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    # 是否保存优化器状态
    save_optimizer: bool=False # will be used if using FSDP
    # 是否使用快速内核，如Flash Attention和Xformer。
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
```


##### **方法-2: 通过运行命令时传参**

```
# grammer_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```


#### 第3步: 运行微调命令

```
# 基于数据集alpaca_dataset进行微调
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --use_fp16
```
