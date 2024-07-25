'''
@Description: 
@Author: shenlei
@Date: 2023-11-29 17:21:19
@LastEditTime: 2024-01-07 00:19:52
@LastEditors: shenlei
'''
import argparse
import os
# 通常是为了解决某些情况下并行化tokenizers可能带来的问题，如避免不必要的警告或提高性能。
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import random
from tqdm import tqdm

from BCEmbedding.evaluation import c_mteb, YDDRESModel

from BCEmbedding.utils import query_instruction_for_retrieval_dict, passage_instruction_for_retrieval_dict
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.eval_embedding_mteb')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="maidalun1020/bce-embedding-base_v1", type=str, help="model name or path you want to eval")
    # 指定评估的任务类型，默认值为None，表示评估所有类型的任务。
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    # 指定评估的语言
    parser.add_argument('--task_langs', default=['en', 'zh', 'zh-CN'], type=str, nargs='+', help="eval langs")  # default: monolingual, bilingual and crosslingual
    parser.add_argument('--pooler', default='cls', type=str, help="pooling method of embedding model, default `cls`.")
    parser.add_argument('--batch_size', default=256, type=int, help="batch size to inference.")
    parser.add_argument('--use_fp16', default=False, action='store_true', help="eval in fp16 mode")
    parser.add_argument('--trust_remote_code', default=False, action='store_true', help="load huggingface model trust_remote_code")
    return parser.parse_args()


if __name__ == '__main__':
    # 获取命令行参数
    args = get_args()
    # 创建YDDRESModel模型实例
    model = YDDRESModel(
        model_name_or_path=args.model_name_or_path,
        normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
        pooler=args.pooler,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        trust_remote_code=args.trust_remote_code,
        )
    # 模型名称
    save_name = args.model_name_or_path.strip('/').split('/')[-1]
    # 创建任务列表，这里使用c_mteb.MTEB根据指定的任务类型和任务语言生成一个任务集合，然后从中提取任务列表。
    tasks = [t for t in c_mteb.MTEB(task_types=args.task_type, task_langs=args.task_langs).tasks]
    # 遍历任务列表，desc参数为进度条添加描述，unit参数定义进度条的单位。
    for task in tqdm(tasks, desc='EmbeddingModel Evaluating', unit='task'):
        # 判断任务名称是否为MSMARCOv2或BUCC
        if task.description["name"] in ['MSMARCOv2', 'BUCC']:
            # 这些任务由于数据量过大而被跳过
            logger.info('Skip task {}, which is too large to eval.'.format(task.description["name"]))
            continue

        # for retrieval and reranking tasks
        # 筛选出属于检索和重排序任务的任务类型
        if 'CQADupstack' in task.description["name"] or \
            'T2Reranking' in task.description["name"] or \
            'T2Retrieval' in task.description["name"] or \
            'MMarcoReranking' in task.description["name"] or \
            'MMarcoRetrieval' in task.description["name"] or \
            'CrosslingualRetrieval' in task.description["name"] or \
            task.description["name"] in [
                # en
                'Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', 

                # add
                'QuoraRetrieval', 'ArguAna', 'StackOverflowDupQuestions', 'SciDocsRR', 'MindSmallReranking', 'AskUbuntuDupQuestions',
                
                # zh
                'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval',
                'CovidRetrieval', 'CmedqaRetrieval',
                'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval',
                'T2Reranking', 'MMarcoReranking', 'CMedQAv1', 'CMedQAv2'
            ]:
            # 确保当前任务的类型必须是"Retrieval"（检索）或"Reranking"（重排序）
            assert task.description["type"] in ["Retrieval", "Reranking"], task.description
            # for query
            # 如果用户指定的模型名称或路径不在query_instruction_for_retrieval_dict字典中，说明没有为这个模型定义特定的查询指令。
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                # 将指令设置为None并记录一条信息日志
                instruction = None
                logger.info(f"{args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                # 如果找到了对应的指令，就从字典中获取这个指令并设置给模型的query_instruction_for_retrieval属性，
                # 同时记录一条包含指令内容的信息日志。
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
                logger.info(f"{args.model_name_or_path} in query_instruction_for_retrieval_dict, set instruction={instruction}")
            model.query_instruction_for_retrieval = instruction

            # for passage
            # 如果模型名称在字典中，则使用相应的指令
            if args.model_name_or_path not in passage_instruction_for_retrieval_dict:
                # 如果不在，则指令设置为None。
                instruction = None
                logger.info(f"{args.model_name_or_path} not in passage_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                instruction = passage_instruction_for_retrieval_dict[args.model_name_or_path]
                logger.info(f"{args.model_name_or_path} in passage_instruction_for_retrieval_dict, set instruction={instruction}")
            model.passage_instruction_for_retrieval = instruction
        # multilingual-e5 needs instruction for other task
        elif "e5-base" in save_name or "e5-large" in save_name:
            # 对于包含"e5-base"或"e5-large"在save_name中的模型，脚本会为其他类型的任务设置查询指令，但不会设置文档指令。
            assert task.description["type"] not in ["Retrieval", "Reranking"], task.description
            # for other tasks
            if args.model_name_or_path not in query_instruction_for_retrieval_dict:
                instruction = None
                logger.info(f"other tasks: {args.model_name_or_path} not in query_instruction_for_retrieval_dict, set instruction={instruction}")
            else:
                instruction = query_instruction_for_retrieval_dict[args.model_name_or_path]
                logger.info(f"other tasks: {args.model_name_or_path} in query_instruction_for_retrieval_dict, set instruction={instruction}")
            model.query_instruction_for_retrieval = instruction
            model.passage_instruction_for_retrieval = None
        else:
            # 如果任务既不是检索或重排序任务，也不包含"e5-base"或"e5-large"，则查询和文档指令都设置为None。
            assert task.description["type"] not in ["Retrieval", "Reranking"], task.description
            model.query_instruction_for_retrieval = None
            model.passage_instruction_for_retrieval = None
        # 专门用于评估当前的任务
        evaluation = c_mteb.MTEB(
            tasks=[task], 
            task_langs=args.task_langs, 
        )
        # 执行评估过程，将评估结果保存到指定的输出文件夹中。
        evaluation.run(
            model, 
            output_folder=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/embedding_results", save_name),
            overwrite_results=False, # 控制是否覆盖已存在的评估结果
        )


