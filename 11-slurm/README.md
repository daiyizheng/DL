# Slurm

## Slurm安装教程


## Slurm 命令教程

```bash


#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-5:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -p debug # 提交到哪一个分区
#SBATCH --mem=2000 # 所有核心可以使用的内存池大小，MB为单位
#SBATCH -o myjob.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e myjob.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=netid@nyu.edu # 把通知发送到哪一个邮箱
#SBATCH --constraint=2630v3  # the Features of the nodes, using command " showcluster " could find them.
#SBATCH --gres=gpu:n # 需要使用多少GPU，n是需要的数量


//下面是示范例子
#!/bin/bash
#SBATCH --job-name=alpaca_no_prompt_no_input_maxlen512_sft_qa_20231023_baichuan2_13b_chat
#SBATCH --partition=a100
#SBATCH --ntasks=16
#SBATCH --gres=gpu:8
#SBATCH --output=./logs/alpaca_no_prompt_no_input_maxlen512_sft_qa_20231023_baichuan2_13b_chat_%j.out
#SBATCH --error=./logs/alpaca_no_prompt_no_input_maxlen512_sft_qa_20231023_baichuan2_13b_chat_%j.err

module load miniconda3
source activate baichuan2

export WANDB_PROJECT="CMM"

model_type="Baichuan2-13B-Chat"
date_time=$(date +"%Y%m%d%H%M%S")
prefix="alpaca_no_prompt_no_input_maxlen512_sft_qa_20231023_baichuan2_13b_chat"

run_name="${model_type}-${date_time}"
model_name_or_path="/slurm/home/yrd/shaolab/daiyizheng/resources/hf_weights/baichuan/${model_type}/"
output_dir="output/${prefix}_${run_name}"
data_path="data/alpaca_no_prompt_no_input_maxlen512_sft_qa_20231023.json"

deepspeed --hostfile="" fine-tune.py  \
    --report_to "wandb" \
    --run_name ${run_name}  \
    --data_path ${data_path} \
    --model_name_or_path ${model_name_or_path} \
    --output_dir ${output_dir} \
    --model_max_length 512 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True
```
