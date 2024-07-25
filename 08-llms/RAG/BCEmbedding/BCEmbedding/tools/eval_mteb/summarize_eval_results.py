import argparse
import json
import os
import os.path as osp
from collections import defaultdict

from BCEmbedding.evaluation import c_mteb
from BCEmbedding.utils import logger_wrapper, query_instruction_for_retrieval_dict
logger = logger_wrapper('evaluation.summarize_eval_results')
# 这个列表包含了需要特定检索指令的模型名称
need_instruction_models = [model_name_or_path.strip('/').split('/')[-1] for model_name_or_path in query_instruction_for_retrieval_dict]
# 列出了那些在嵌入向量汇总时需要使用均值池化（mean pooling）方法的模型。
need_mean_pooler = ["jina-embeddings-v2-base-en", "jina-embeddings-v2-base-zh", "m3e-base", "m3e-large", "e5-large-v2", "multilingual-e5-base", "multilingual-e5-large", "gte-large"]

def read_results(task_types, # ：一个列表，包含要读取结果的任务类型。
                 except_tasks, # 一个列表，包含要排除的任务名称。
                 args # 一个包含其他配置（如结果目录、语言等）的对象
                 ):
    '''读取和汇总特定任务类型和语言的模型评估结果'''
    tasks_results = {} # 用于存储按语言和任务类型组织的评估结果
    model_dirs = {} #用于存储模型相关的目录路径
    for lang in args.lang:
        # 为每种语言创建一个空字典，用于存放该语言的评估结果。
        tasks_results[lang] = {}
        for t_type in task_types:
            if 'zh' == lang:
                # 如果语言是中文（'zh'），则同时考虑'zh'和'zh-CN'。
                mteb_tasks = c_mteb.MTEB(task_types=[t_type], task_langs=[lang, 'zh-CN']).tasks
            else:
                mteb_tasks = c_mteb.MTEB(task_types=[t_type], task_langs=[lang]).tasks
            # 如果获取到的任务列表为空，则继续下一次循环。
            if len(mteb_tasks) == 0: continue
            # 为每种任务类型创建一个空字典，用于存放该任务类型的评估结果。
            tasks_results[lang][t_type] = {}
            # 遍历获取到的任务列表
            for t in mteb_tasks:
                # 任务名称
                task_name = t.description["name"]
                # 如果任务名称在排除列表except_tasks中，则跳过当前任务。
                if task_name in except_tasks: continue
                # 列出所有模型名称，并对模型名称进行排序。
                models_names = os.listdir(args.results_dir)
                models_names.sort()
                # 如果模型名称列表为空，则继续下一次循环。
                if len(models_names) == 0: continue
                # 从任务描述中获取主要评价指标的名称
                metric = t.description["main_score"]
                # 遍历模型目录
                for model_name in models_names:
                    # 对于每一个模型名称，构建其完整路径，并检查这个路径确实指向一个目录。如果不是目录，则跳过处理。
                    model_dir = os.path.join(args.results_dir, model_name)
                    if not os.path.isdir(model_dir): continue
                    # 对于确认是目录的模型路径，将其记录下来
                    model_dirs[model_name] = model_dir
                    # 检查模型目录中是否存在以任务名称命名的JSON格式的评估结果文件
                    if os.path.exists(os.path.join(model_dir, task_name + '.json')):
                        # 如果当前任务名在tasks_results中还没有对应的记录，则为它创建一个新的defaultdict(None)，
                        # 这样即使后续访问不存在的键时也不会引发错误。
                        if task_name not in tasks_results[lang][t_type]:
                            tasks_results[lang][t_type][task_name] = defaultdict(None)
                        with open(os.path.join(model_dir, task_name + '.json')) as f:
                            # 使用json.load(f)从文件中加载评估结果数据
                            data = json.load(f)
                        # 尝试确定使用哪一个数据集分割来表示模型性能，这通常依据评估结果文件中包含的哪个键（如'test', 'dev', 'validation', 'dev2'等）。
                        # 一旦找到存在的分割，就中断循环。
                        for s in ['test', 'dev', 'validation', 'dev2']:
                            if s in data:
                                split = s
                                break
                        # 根据lang参数，代码选择适当的评估结果。
                        # 首先尝试找到与lang直接对应的评估结果，如果不存在，则退回到默认的结果。
                        if 'en' == lang:
                            if 'en-en' in data[split]:
                                temp_data = data[split]['en-en']
                            elif 'en' in data[split]:
                                temp_data = data[split]['en']
                            else:
                                temp_data = data[split]
                        elif 'zh' == lang:
                            if 'zh-zh' in data[split]:
                                temp_data = data[split]['zh-zh']
                            elif 'zh' in data[split]:
                                temp_data = data[split]['zh']
                            elif 'zh-CN' in data[split]:
                                temp_data = data[split]['zh-CN']
                            else:
                                temp_data = data[split]
                        elif 'en-zh' == lang:
                            if 'en-zh' in data[split]:
                                temp_data = data[split]['en-zh']
                            elif 'en-zh-CN' in data[split]:
                                temp_data = data[split]['en-zh-CN']
                            else:
                                temp_data = data[split]
                        elif 'zh-en' == lang:
                            if 'zh-en' in data[split]:
                                temp_data = data[split]['zh-en']
                            elif 'zh-CN-en' in data[split]:
                                temp_data = data[split]['zh-CN-en']
                            else:
                                temp_data = data[split]
                        # 读取性能指标
                        if metric == 'ap':
                            # 如果主要评估指标是'ap'（平均精度），则提取temp_data['cos_sim']['ap']的值，并乘以100转换为百分比形式。
                            tasks_results[lang][t_type][task_name][model_name] = temp_data['cos_sim']['ap'] * 100
                        elif metric == 'cosine_spearman':
                            # 如果主要评估指标是'cosine_spearman'，则提取temp_data['cos_sim']['spearman']的值，并同样乘以100转换。
                            tasks_results[lang][t_type][task_name][model_name] = temp_data['cos_sim']['spearman'] * 100
                        else:
                            # 对于其他指标，直接从temp_data中提取对应指标的值，并乘以100。
                            tasks_results[lang][t_type][task_name][model_name] = temp_data[metric] * 100

    return tasks_results, model_dirs


def output_markdown(tasks_results_wt_langs, # 按语言组织的任务结果字典
                    model_names, # 模型名称列表
                    model_type,  # 模型类型
                    save_file  # 输出Markdown文件的路径
                    ):
    with open(save_file, 'w') as f:
        # 写入一个Markdown标题
        f.write(f"# {model_type} Evaluation Results  \n")
        # 储不同任务类型在所有语言中的综合结果
        task_type_res_merge_lang = {}
        has_CQADupstack_overall = False
        # 对于tasks_results_wt_langs中的每一种语言及其对应的任务结果，写入一个二级标题来标明当前的语言。
        for lang, tasks_results in tasks_results_wt_langs.items():
            f.write(f'## Language: `{lang}`  \n')
            # 存储当前语言下每种任务类型的结果
            task_type_res = {}
            for t_type, type_results in tasks_results.items():
                has_CQADupstack = False
                task_cnt = 0
                # 检查当前任务类型是否存在结果，如果没有，则继续处理下一种任务类型。
                tasks_names = list(type_results.keys())
                if len(tasks_names) == 0:
                    continue

                task_type_res[t_type] = defaultdict()
                if t_type not in task_type_res_merge_lang:
                    task_type_res_merge_lang[t_type] = defaultdict(list)
                # 在Markdown文件中为每种任务类型添加一个三级标题。
                f.write(f'\n### Task Type: {t_type}  \n')
                # 准备表格的标题行（first_line）和分隔行（second_line）。
                first_line = "| Model |"
                second_line = "|:-------------------------------|"
                # 遍历任务名称
                for task_name in tasks_names:
                    # 检查特定任务
                    if "CQADupstack" in task_name:
                        has_CQADupstack = True
                        has_CQADupstack_overall = True
                        continue
                    # 更新表格标题行和分隔行
                    # 如果当前任务名称不是CQADupstack，则在first_line中添加该任务名称，
                    # 并在second_line中添加对应的列分隔符(":--------:|")。
                    first_line += f" {task_name} |"
                    second_line += ":--------:|"
                    task_cnt += 1
                # 如果检测到有CQADupstack任务，会在所有任务遍历完毕后，额外添加一个CQADupstack列到first_line中，
                # 并在second_line中也添加对应的列分隔符。这样做是为了确保CQADupstack任务（如果存在）在表格的最后一个任务列之后单独列出。
                if has_CQADupstack:
                    first_line += f" CQADupstack |"
                    second_line += ":--------:|"
                    task_cnt += 1 # 记录了除CQADupstack外的任务数量
                # 将更新后的first_line和second_line写入Markdown文件。
                # 此外，first_line末尾额外添加了一个" AVG |"列，用于展示模型在当前任务类型下所有任务的平均性能。
                f.write(first_line + ' ***AVG*** |  \n')
                f.write(second_line + ':--------:|  \n')
                # 遍历模型名称
                for model in model_names:
                    # 初始化write_line，这将是Markdown表格中的一行，首先写入模型名称。
                    write_line = f"| {model} |"
                    # 分别用于收集当前模型在当前任务类型下所有任务（除"CQADupstack"任务）的评分和仅"CQADupstack"任务的评分。
                    all_res = []
                    cqa_res = []
                    # 遍历任务名称
                    for task_name in tasks_names:
                        results = type_results[task_name]
                        # 如果当前任务是"CQADupstack"相关，则将得分加入cqa_res列表；
                        # 否则，将得分加入all_res列表。
                        if "CQADupstack" in task_name:
                            if model in results:
                                cqa_res.append(results[model])
                            continue

                        # 对于每个任务，如果当前模型有评分，则将评分格式化后写入write_line；
                        # 如果没有评分，则在对应位置留空。
                        if model in results:
                            write_line += " {:.2f} |".format(results[model])
                            all_res.append(results[model])
                        else:
                            write_line += f"  |"

                    # 如果cqa_res列表不为空，计算这些任务的平均得分并加入到all_res列表中，
                    # 同时在write_line中添加该平均得分。
                    if len(cqa_res) > 0:
                        write_line += " {:.2f} |".format(sum(cqa_res) / len(cqa_res))
                        all_res.append(sum(cqa_res) / len(cqa_res))
                    # 如果all_res列表的长度等于任务计数器task_cnt的值，计算all_res中所有得分的平均值，
                    # 并在write_line末尾添加该平均得分。
                    if len(all_res) == task_cnt:
                        write_line += " {:.2f} |".format(sum(all_res) / max(len(all_res), 1))
                        # 更新task_type_res和task_type_res_merge_lang字典，用于后续汇总和分析。
                        task_type_res[t_type][model] = all_res
                        task_type_res_merge_lang[t_type][model].extend(all_res)
                    else:
                        write_line += f"  |"
                    # 将构建好的write_line写入Markdown文件。
                    f.write(write_line + '  \n')
            # 通过写入'### *Summary on {lang}* '，在Markdown文件中为每种语言添加一个汇总部分的标题，
            # 用以概括该语言下所有任务类型的综合评估结果。
            f.write(f'\n### *Summary on `{lang}`*  \n')
            # 初始化标题行first_line和分隔线second_line
            first_line = "| Model |"
            second_line = "|:-------------------------------|"
            task_type_res_keys = list(task_type_res.keys())
            # 遍历task_type_res_keys（即当前语言下所有有结果的任务类型列表），
            # 为每种任务类型添加一个列标题到first_line中，并在second_line中添加对应的分隔符。
            for t_type in task_type_res_keys:
                first_line += f" {t_type} |"
                second_line += ":--------:|"
            # 在标题行末尾添加' ***AVG*** |'列，用于显示每个模型在当前语言下所有任务类型的平均得分。
            f.write(first_line + ' ***AVG*** |  \n')
            f.write(second_line + ':--------:|  \n')
            # 遍历模型并汇总评估结果
            for model in model_names:
                # 初始化write_line以构建表格中的一行，首先写入模型名称。
                write_line = f"| {model} |"
                # 收集当前模型在当前语言下所有任务类型的所有得分，以计算整体平均得分。
                all_res = []
                # 遍历所有任务类型，根据task_type_res字典获取每种任务类型下当前模型的评分，
                # 计算这些评分的平均值，并将其写入write_line中。
                for type_name in task_type_res_keys:
                    results = task_type_res[type_name]
                    if model in results:
                        write_line += " {:.2f} |".format(sum(results[model]) / max(len(results[model]), 1))
                        all_res.extend(results[model])
                    else:
                        # 如果当前模型在某任务类型下没有评分，则在对应位置留空。
                        write_line += f"  |"
                # 如果all_res列表不为空，计算该列表中所有得分的平均值，并将这个整体平均得分添加到write_line末尾。
                if len(all_res) > 0:
                    write_line += " {:.2f} |".format(sum(all_res) / len(all_res))
                # 将构建好的行write_line写入Markdown文件
                f.write(write_line + '  \n')
        
        # 通过f.write(f'## Summary on all langs: {list(tasks_results_wt_langs.keys())} \n')添加一个二级标题
        f.write(f'## Summary on all langs: `{list(tasks_results_wt_langs.keys())}`  \n')
        if model_type.lower() == 'embedding':
            # 如果model_type为'embedding'（小写），则表格的标题行包括模型名称、维度、池化器类型和是否使用指令，适用于评估嵌入模型。
            first_line = "| Model | Dimensions | Pooler | Instructions |"
            second_line = "|:--------|:--------:|:--------:|:--------:|"
        else:
            # 对于其他模型类型，表格标题行仅包含模型名称。
            first_line = "| Model |"
            second_line = "|:--------|"
        task_type_res_merge_lang_keys = list(task_type_res_merge_lang.keys())
        task_nums = 0
        # 遍历所有任务类型的列表，计算每种任务类型下评估的任务数量，并更新first_line和second_line来反映这些信息。
        for t_type in task_type_res_merge_lang_keys:
            task_num = max([len(tmp_metrics) for tmp_model_name, tmp_metrics in task_type_res_merge_lang[t_type].items()])
            # 如果任务类型为'Retrieval'并且整体评估包含'CQADupstack'任务，由于'CQADupstack'任务被合并处理，需要对任务数量进行相应的调整。
            if t_type == 'Retrieval' and has_CQADupstack_overall: # 'CQADupstack' has been merged 12 to 1. We should add 11.
                task_num +=  11
            task_nums += task_num
            first_line += f" {t_type} ({task_num}) |"
            second_line += ":--------:|"
        # 在表格标题行的末尾添加一个表示整体平均得分的列，包括整体的任务数量。
        f.write(first_line + f' ***AVG*** ({task_nums}) |  \n')
        # 将构建好的标题行和分隔线写入文件
        f.write(second_line + ':--------:|  \n')
        # 遍历模型列表
        for model in model_names:
            write_line = f"| {model} |"
            # 如果model_type为'embedding'，则在每行开始处写入模型名称后，继续添加模型的维度信息（基于模型名称中是否包含'base'来判断使用768还是1024维）、
            # 池化方法（如果模型在need_mean_pooler列表中，则为'mean'，否则为'cls'）、以及是否需要指令（如果模型在need_instruction_models列表中，
            # 则标记为'Need'，否则为'Free'）。
            if model_type.lower() == 'embedding':
                write_line += f" {768 if 'base' in model else 1024} |"
                write_line += f" {'`mean`' if model in need_mean_pooler else '`cls`'} |"
                write_line += f" {'Need' if model in need_instruction_models else 'Free'} |"
            # 收集模型在所有任务类型下的评分
            all_res = []
            for type_name in task_type_res_merge_lang_keys:
                # 遍历所有任务类型，根据task_type_res_merge_lang字典获取当前模型在各任务类型下的评分列表，
                # 计算这些评分的平均值，并将其格式化后添加到write_line中。
                results = task_type_res_merge_lang[type_name]
                if model in results:
                    write_line += " {:.2f} |".format(sum(results[model]) / max(len(results[model]), 1))
                    all_res.extend(results[model])
                else:
                    # 如果当前模型在某任务类型下没有评分，则在对应位置留空。
                    write_line += f"  |"

            if len(all_res) > 0:
                # 如果all_res列表不为空，计算该列表中所有得分的平均值，并将这个整体平均得分添加到write_line末尾。
                write_line += " {:.2f} |".format(sum(all_res) / len(all_res))
            # 将构建好的write_line写入Markdown文件
            f.write(write_line + '  \n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/embedding_results"), type=str, help="eval results path")
    parser.add_argument('--model_type', default="embedding", choices=['embedding', 'reranker'], type=str, help="model type, including `embedding` and `reranker` models")
    parser.add_argument('--lang', default="en-zh", choices=['en', 'zh', 'en-zh', 'zh-en'], type=str, help="choice which language eval results you want to collecte.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    save_name = osp.basename(args.results_dir.strip('/'))
    # 根据语言设置任务类型和排除的任务
    if 'embedding' in save_name.lower():
        args.model_type = 'embedding'
        save_name = 'embedding_eval_summary'
    elif 'reranker' in save_name.lower():
        args.model_type = 'reranker'
        save_name = 'reranker_eval_summary'
    
    if args.lang == 'zh':
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['zh', 'zh-CN']
    elif args.lang == 'en':
        task_types = ["Retrieval", "Clustering", "PairClassification", "Reranking", "STS", # "Summarization",
                      "Classification"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['en']
    elif args.lang in ['en-zh', 'zh-en']:
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['en', 'zh', 'en-zh', 'zh-en']
    else:
        raise NotImplementedError(f"args.lang must be zh or en, but {args.lang}")
    
    args.model_type = args.model_type.capitalize()

    # 日志记录
    logger.info(f'eval results path: {args.results_dir}')
    logger.info(f'model type: {args.model_type}')
    logger.info(f'collect languages: {args.lang}')

    # 读取评估结果
    task_results, model_dirs = read_results(task_types, except_tasks, args=args)
    # 调用output_markdown函数，将处理好的评估结果、模型目录列表、
    # 模型类型和保存文件路径作为参数，生成并保存Markdown格式的评估结果汇总文件。
    output_markdown(
        task_results, 
        model_dirs.keys(), # 获取所有评估过的模型名称列表
        args.model_type,
        save_file=os.path.join(
            args.results_dir, 
            save_name + ".md"
            )
        )
