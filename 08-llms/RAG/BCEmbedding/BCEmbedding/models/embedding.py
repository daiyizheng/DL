'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-01-19 12:10:58
@LastEditors: shenlei
'''
import logging
import torch

from tqdm import tqdm
# ndarray是numpy中的基础类，表示多维数组。
from numpy import ndarray
from typing import List, Dict, Tuple, Type, Union

from transformers import AutoModel, AutoTokenizer
from BCEmbedding.utils import logger_wrapper
# 创建并获取一个日志记录器实例，这个实例被用来记录关于EmbeddingModel模型的日志信息。
logger = logger_wrapper('BCEmbedding.models.EmbeddingModel')


class EmbeddingModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/bce-embedding-base_v1', # 预训练模型的名称或路径
            pooler: str='cls', # 池化策略，从模型的输出中提取固定大小的嵌入表示，'cls'表示使用CLS标记的输出，'mean'表示使用所有输出的平均值。
            use_fp16: bool=False, # 是否使用半精度浮点数（FP16）来加速模型的计算
            device: str=None, # 模型运行的设备
            **kwargs
        ):
        # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        # 记录加载信息
        logger.info(f"Loading from `{model_name_or_path}`.")
        # 断言pooler参数必须是'cls'或'mean'中的一个，否则抛出异常。
        assert pooler in ['cls', 'mean'], f"`pooler` should be in ['cls', 'mean']. 'cls' is recommended!"
        self.pooler = pooler

        # 确定模型运行的device
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        # 如果启用了FP16，将模型转换为半精度。
        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        # 如果有多个GPU可用，使用torch.nn.DataParallel来并行化模型的计算。
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16};\t embedding pooling type: {self.pooler};\t trust remote code: {kwargs.get('trust_remote_code', False)}")

    def encode(
            self,
            sentences: Union[str, List[str]], # 可以是单个字符串或字符串列表，表示要编码的文本。
            batch_size: int=256, # 每个批次的大小
            max_length: int=512, # 文本的最大长度
            normalize_to_unit: bool=True, # 是否将嵌入向量标准化到单位长度
            return_numpy: bool=True, # 是否以NumPy数组的形式返回嵌入向量
            enable_tqdm: bool=True, # 是否启用tqdm进度条来显示处理进度
            query_instruction: str="", # 附加的查询指令
            **kwargs
        ):
        # 如果有多个GPU可用，则根据GPU的数量调整批次大小，以充分利用多GPU并行处理的能力。
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        # 如果sentences是单个字符串（即单个句子），则将其转换为包含该字符串的列表。
        if isinstance(sentences, str):
            sentences = [sentences]
        
        with torch.no_grad():
            # 用于收集每个批次的嵌入向量
            embeddings_collection = []
            # desc='Extract embeddings'设置了进度条的描述。
            for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings', disable=not enable_tqdm):
                # 如果query_instruction是非空字符串，会将其添加到每个句子前面，然后对这个批次的句子进行分词和编码。
                if isinstance(query_instruction, str) and len(query_instruction) > 0:
                    sentence_batch = [query_instruction+sent for sent in sentences[sentence_id:sentence_id+batch_size]] 
                else:
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]
                # 对句子进行分词处理
                inputs = self.tokenizer(
                        sentence_batch, 
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    )
                # 将输入数据移动到模型所在的设备上
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                # 使用模型对处理好的输入数据进行编码，return_dict=True表示以字典形式返回输出。
                outputs = self.model(**inputs_on_device, return_dict=True)

                if self.pooler == "cls":
                    # 使用预训练模型输出的最后一层中的第一个向量（通常对应于CLS标记）作为整个输入文本的嵌入表示
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "mean":
                    # 计算所有输出的加权平均值作为嵌入向量，权重由注意力掩码决定。
                    # 权重是通过attention_mask来确定的，以确保只计算实际文本部分的平均值，忽略了填充部分。
                    # 这一池化策略适用于获取整个句子内容的平均表示。
                    attention_mask = inputs_on_device['attention_mask']
                    last_hidden = outputs.last_hidden_state
                    embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    # 将嵌入向量归一化到单位长度
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                # 将当前批次的嵌入向量移动到CPU并添加到embeddings_collection列表中
                embeddings_collection.append(embeddings.cpu())
            #  将所有批次的嵌入向量合并成一个张量
            embeddings = torch.cat(embeddings_collection, dim=0)
        
        if return_numpy and not isinstance(embeddings, ndarray):
            # 将embeddings转换为NumPy数组
            embeddings = embeddings.numpy()
        
        return embeddings
