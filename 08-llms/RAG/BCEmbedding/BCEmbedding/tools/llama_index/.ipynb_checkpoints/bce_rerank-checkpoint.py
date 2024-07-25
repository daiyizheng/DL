'''
@Description: 
@Author: shenlei
@Date: 2024-01-15 14:15:30
@LastEditTime: 2024-01-15 23:16:24
@LastEditors: shenlei
'''
from typing import Any, List, Optional
# Field用于定义数据模型的字段，PrivateAttr用于定义模型的私有属性。
from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
# CBEventType是一个枚举类型，用于定义回调事件的类型；
# EventPayload是一个数据类或者是一个类型标注，用于标注回调事件的数据载体。
from llama_index.legacy.callbacks import CBEventType, EventPayload
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
# MetadataMode是一个用于定义元数据模式的枚举或类；
# NodeWithScore是一个数据结构，用于表示带有评分的节点；
# QueryBundle是一个封装了查询相关信息的类。
from llama_index.legacy.schema import MetadataMode, NodeWithScore, QueryBundle
# 用于根据当前环境或配置自动推断应该使用的PyTorch设备（CPU或GPU）
from llama_index.legacy.utils import infer_torch_device


class BCERerank(BaseNodePostprocessor):
    model: str = Field(ddescription="Sentence transformer model name.")
    # 指定按分数排序后要返回的节点数量
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    # 用于标记这是一个私有属性，意味着它在类的外部不应直接访问。
    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 5,
        model: str = "maidalun1020/bce-reranker-base_v1",
        device: Optional[str] = None,
        **kwargs
    ):
        try:
            from BCEmbedding.models import RerankerModel
        except ImportError:
            raise ImportError(
                "Cannot import `BCEmbedding` package,",
                "please `pip install BCEmbedding>=0.1.2`",
            )
        self._model = RerankerModel(model_name_or_path=model, device=device, **kwargs)
        device = infer_torch_device() if device is None else device
        super().__init__(top_n=top_n, model=model, device=device)

    @classmethod
    def class_name(cls) -> str:
        '''类方法，名为class_name，它返回一个字符串，即类的名称。'''
        return "BCERerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        '''用于对节点进行后处理'''
        if query_bundle is None:
            # 说明调用时没有提供必要的查询包，因此抛出ValueError异常，提示缺少查询包。
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []
        # 从query_bundle中提取查询字符串
        query = query_bundle.query_str
        passages = [] # 存储节点内容
        valid_nodes = [] # 有效节点
        invalid_nodes = [] # 无效节点
        for node in nodes:
            # 获取节点内容，该内容以嵌入的元数据模式提取。
            passage = node.node.get_content(metadata_mode=MetadataMode.EMBED)
            # 检查提取的内容是否为非空字符串
            if isinstance(passage, str) and len(passage) > 0:
                # 如果是，将内容中的换行符替换为空格
                passages.append(passage.replace('\n', ' '))
                valid_nodes.append(node)
            else:
                # 如果内容不是非空字符串，将该节点添加到invalid_nodes列表。
                invalid_nodes.append(node)

        # 主要处理节点重排序的逻辑
        # 调用self.callback_manager的event方法来创建一个事件上下文
        # event方法接收两个参数：事件类型(CBEventType.RERANKING)和一个包含各种信息的payload字典。
        with self.callback_manager.event(
                CBEventType.RERANKING,
                payload={
                    EventPayload.NODES: nodes, # 节点列表
                    EventPayload.MODEL_NAME: self.model, # 模型名称
                    EventPayload.QUERY_STR: query_bundle.query_str, # 查询字符串
                    EventPayload.TOP_K: self.top_n, # 顶部节点数量
                },
            ) as event:
            # 返回重排序的结果，包括重排序后的分数和对应的ID。
            rerank_result = self._model.rerank(query, passages)
            # 存储重排序后的节点
            new_nodes = []
            for score, nid in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
                node = valid_nodes[nid]
                node.score = score
                new_nodes.append(node)

            for node in invalid_nodes:
                node.score = 0
                new_nodes.append(node)
            # 确保处理过程中没有丢失或错误地添加了节点
            assert len(new_nodes) == len(nodes)
            # 只保留分数最高的前top_n个节点
            new_nodes = new_nodes[:self.top_n]
            # 在上下文管理器的末尾调用event.on_end方法，传入一个包含重排序后的节点列表的payload。
            # 这可能用于记录事件结束的状态或触发与事件结束相关的回调。
            event.on_end(payload={EventPayload.NODES: new_nodes})
        # 返回重排序（并可能被裁剪）后的节点列表
        return new_nodes