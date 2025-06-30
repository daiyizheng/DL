#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :01-tutorial.py
@Description  :
@Time         :2025/06/28 22:38:40
@Author       :flow-laic
@Version      :1.0
'''
import sys
sys.path.insert(0, "/Users/a1-6/Documents/projects/DL/08-llms/RAG/metric/RAGAS")

from ragas import SingleTurnSample
from ragas.metrics import BleuScore
from dotenv import load_dotenv
import os
import asyncio
load_dotenv("/Users/a1-6/Documents/projects/DL/.env")

#######################################################################################
################################ 使用非法学硕士指标进行评估 ################################
#######################################################################################

# test_data = {
#     "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
#     "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
#     "reference": "The company reported an 8% growth in Q3 2024, primarily driven by strong sales in the Asian market, attributed to strategic marketing and localized products, with continued growth anticipated in the next quarter."
# }
# metric = BleuScore()
# test_data = SingleTurnSample(**test_data)
# metric.single_turn_score(test_data)

#########################################################################################
################################ 使用基于 LLM 的指标进行评估 ################################
#########################################################################################

# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=os.environ["api_key"], base_url=os.environ["base_url"]))
# evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.environ["api_key"], base_url=os.environ["base_url"]))

# from ragas import SingleTurnSample
# from ragas.metrics import AspectCritic

# """
# instruction:  Evaluate the Input based on the criterial defined. Use only 'Yes' (1) and 'No' (0) as verdict.\nCriteria Definition: Verify if the summary is accurate.

# generate_output_signature: Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:\n{"properties": {"reason": {"description": "Reason for the verdict", "title": "Reason", "type": "string"}, "verdict": {"description": "The verdict (0 or 1) for the submission", "title": "Verdict", "type": "integer"}}, "required": ["reason", "verdict"], "title": "AspectCriticOutput", "type": "object"}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

# Now perform the same with the following input

# Input :{\n    "user_input": "summarise given text\\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",\n    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter."\n}

# "Output: "
# """

# async def main():
#     test_data = {
#     "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
#     "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
# }
#     metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
#     test_data = SingleTurnSample(**test_data)
#     rea = await metric.single_turn_ascore(test_data)
#     print(f"Aspect Critic Score: {rea}")
#     return rea

# asyncio.run(main())



#########################################################################################
################################ 在数据集上进行评估 ########################################
#########################################################################################

test_data = [
    # Sample 1
    {
        "user_input": "summarise given text\nThe Q3 earnings report revealed a significant 15% increase in revenue, ...",
        "response": "The Q2 earnings report showed a 15% revenue increase, ...",
    },
    # Sample N
    {
        "user_input": "summarise given text\nIn 2023, North American sales experienced a 5% decline, ...",
        "response": "Companies are strategizing to adapt to market challenges and ...",
    }
]

from datasets import load_dataset, Dataset
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import AspectCritic
from ragas import evaluate

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", api_key=os.environ["api_key"], base_url=os.environ["base_url"]))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=os.environ["api_key"], base_url=os.environ["base_url"]))

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
# eval_dataset = load_dataset("explodinggradients/earning_report_summary",split="train")
eval_dataset = Dataset.from_list(test_data, split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
print("Features in dataset:", eval_dataset.features())
print("Total samples in dataset:", len(eval_dataset))

results = evaluate(eval_dataset, metrics=[metric])
results