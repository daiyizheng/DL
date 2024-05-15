# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import torch
import warnings
from typing import List
from string import Template # 字符串模板处理
from enum import Enum # 定义枚举类型


class AgentType(Enum):
    '''AgentType 的枚举类，包含两个成员：AGENT 和 USER'''
    AGENT = "Agent"
    USER = "User"

# Class for performing safety checks using AuditNLG library
class AuditNLGSensitiveTopics(object):
    def __init__(self):
        pass

    def __call__(self, output_text, **kwargs):
        '''__call__ 方法，使得类的实例可以像函数一样被调用。
           它接受一个参数 output_text（输出文本）和任意数量的关键字参数。'''
        try:
            # 从auditnlg库导入safety_scores函数
            from auditnlg.safety.exam import safety_scores
        except ImportError as e:
            print("Could not import optional dependency: auditnlg\nPlease install manually with:\n pip install auditnlg\nFollowed by:\npip install -r requirements.txt")
            raise e
            
        # 创建一个字典列表data，其中包含一个键为output的字典，其值为传入的output_text。
        data = [{"output": output_text}]
        # 这个函数似乎用于评估文本的安全性，特别是针对敏感话题。
        result = safety_scores(data=data, method="sensitive_topics")
        # 从 safety_scores 函数的返回结果中提取评分信息，并判断文本是否安全。
        # 如果预测的类别（pred_class）是 "none"，则认为文本是安全的。
        scores = result[1]["all_scores"][0]
        is_safe = scores["pred_class"] == "none"
        report = ""
        # 如果文本不安全，将生成一个包含敏感类别和相应评分的报告。
        if not is_safe:
            report += f"Predicted class: {scores['pred_class']}\n"
            report += "|" + "|".join(f"{n:^10}" for n in [list(k.keys())[0] for k in scores["class_scores"]]) + "|\n"
            report += "|" + "|".join(f"{n:^10.5}" for n in [list(k.values())[0] for k in scores["class_scores"]]) + "|\n"
        return "Sensitive Topics", is_safe, report
    
    
class SalesforceSafetyChecker(object):
    '''使用 Salesforce 提供的 safety-flan-t5-base 模型进行文本安全性检查。'''
    def __init__(self):
        pass

    def __call__(self, output_text, **kwargs):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
        # 加载了 Salesforce 提供的 safety-flan-t5-base 模型的配置、分词器和模型本身，并将模型设置为评估模式。
        config = AutoConfig.from_pretrained("Salesforce/safety-flan-t5-base")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/safety-flan-t5-base") 
        model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/safety-flan-t5-base", config=config)
        model.eval()
        # 构建了一个用于安全性检查的输入字符串，并使用分词器将其转换为模型可接受的输入ID。
        prefix = "Is the <Text> field safe or unsafe?"
        input_ids = tokenizer(prefix + " <Text> " + output_text + " <Context> ", return_tensors="pt").input_ids
        # 如果输入长度超过512个tokens，则发出警告，因为这可能会影响安全性检查的准确性。
        if len(input_ids[0]) > 512:
            warnings.warn(
                "Input length is > 512 token. Safety check result could be incorrect."
            )
        # 在不计算梯度的情况下生成模型的输出, 这是为了提高效率并防止内存泄漏。
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=20,
                )
        # 将模型的输出解码为文本，并检查第一个单词是否为 "safe"，以确定内容是否安全。
        is_safe = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).split(" ")[0] == "safe"    
        # 如果内容不安全，生成一个报告，显示各种不安全类别（如 "toxicity", "hate" 等）的得分。报告以表格的形式显示这些得分。
        report = ""
        if not is_safe:
            true_false_ids = tokenizer("true false").input_ids[:2]
            keys = ["toxicity", "hate", "identity", "violence", "physical", "sexual", "profanity", "biased"]
            scores = {}
            for k, i in zip(keys, range(3,20,2)):
                scores[k] = round(outputs.scores[i][0,true_false_ids].softmax(dim=0)[0].item(), 5)
            
            report += "|" + "|".join(f"{n:^10}" for n in scores.keys()) + "|\n"
            report += "|" + "|".join(f"{n:^10}" for n in scores.values()) + "|\n"
        return "Salesforce Content Safety Flan T5 Base", is_safe, report
        

    def get_total_length(self, data):
        '''使用类中已经定义的 tokenizer 对拼接后的字符串（prefix + input_sample）进行分词，
           并获取对应的输入ID（input_ids）。然后返回这些输入ID的长度，这代表了整个输入文本的令牌数量。
        '''
        prefix = "Is the <Text> field safe or unsafe "
        input_sample = "<Text> {output} <Context> ".format(**data[0])

        return len(self.tokenizer(prefix + input_sample)["input_ids"])


# Class for performing safety checks using Azure Content Safety service
class AzureSaftyChecker(object):
    def __init__(self):
        try:
            # ContentSafetyClient 用于与 Azure 的内容安全服务交互，
            # 而 AzureKeyCredential 用于提供访问密钥。
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential
            # 从环境变量中获取 Azure 内容安全服务的密钥和端点
            key = os.environ["CONTENT_SAFETY_KEY"]
            endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
        except ImportError:
            # 如果无法导入必需的包，代码会抛出一个异常
            raise Exception(
                "Could not import required package azure-ai-contentsafety. Install with: pip install azure-ai-contentsafety"
            )
        except KeyError:
            # 如果环境变量没有正确设置，代码会抛出另一个异常
            raise Exception(
                "Environment variables not set. Please set CONTENT_SAFETY_KEY and CONTENT_SAFETY_ENDPOINT."
            )
        # 使用提供的端点和密钥创建 ContentSafetyClient 的实例
        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    def __call__(self, output_text, **kwargs):
        # 导入了处理 Azure HTTP 响应错误的类 HttpResponseError，
        # 以及用于内容安全分析的 AnalyzeTextOptions 和 TextCategory 类。
        from azure.core.exceptions import HttpResponseError
        from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
        # 打印并检查输入文本的长度。
        # 如果长度超过1000个字符，抛出异常，因为该服务对输入长度有限制。
        print(len(output_text))
        if len(output_text) > 1000:
            raise Exception("Input length to safety check is too long (>1000).")
        # 定义了要检查的文本类别，包括暴力、自我伤害、性内容和仇恨言论。
        categories = [
            TextCategory.VIOLENCE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.HATE,
        ]
        # 创建了一个 AnalyzeTextOptions 实例，用于指定要分析的文本和类别。
        request = AnalyzeTextOptions(text=output_text, categories=categories)

        try:
            # 使用 Azure 内容安全客户端对文本进行分析。
            # 如果分析失败（比如因为网络问题或服务错误），打印错误信息并抛出异常。
            response = self.client.analyze_text(request)
        except HttpResponseError as e:
            print("Analyze text failed.")
            if e.error:
                print(f"Error code: {e.error.code}")
                print(f"Error message: {e.error.message}")
                raise
            print(e)
            raise e
        # 定义了一个字典 levels，用于将数值严重性级别映射为字符串描述。
        levels = {0: "Safe", 2: "Low", 4: "Medium", 6: "High"}
        # 从响应中提取每个类别的严重性级别
        severities = [
            getattr(response, c.name.lower() + "_result").severity for c in categories
        ]
        # 定义了默认的安全严重性级别，并判断文本是否安全。
        # 如果所有类别的严重性级别都低于或等于默认级别，则认为文本是安全的。
        DEFAULT_LEVELS = [0, 0, 0, 0]

        is_safe = all([s <= l for s, l in zip(severities, DEFAULT_LEVELS)])
        # 如果文本不安全，生成一个报告，显示各种类别的严重性级别。
        report = ""
        if not is_safe:
            report = "|" + "|".join(f"{c.name:^10}" for c in categories) + "|\n"
            report += "|" + "|".join(f"{levels[s]:^10}" for s in severities) + "|\n"

        return "Azure Content Saftey API", is_safe, report

class LlamaGuardSafetyChecker(object):
    '''用于使用 Meta 的 LlamaGuard-7b 模型进行文本的安全性检查'''
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "meta-llama/LlamaGuard-7b"
        # 初始化方法中加载了LlamaGuard-7b模型和对应的分词器。
        # 模型被配置为在自动选择的设备上以 8 位精度运行，这可能是为了优化内存使用和计算效率。
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
        pass

    def __call__(self, output_text, **kwargs):
        # 从关键字参数中获取 agent_type（代理类型）和 user_prompt（用户提示）。
        # 如果没有提供这些参数，则分别使用默认值 AgentType.USER 和空字符串。
        agent_type = kwargs.get('agent_type', AgentType.USER)
        user_prompt = kwargs.get('user_prompt', "")

        model_prompt = output_text.strip()
        # 根据 agent_type 来决定如何构建模型的输入。
        # 如果是代理类型，处理用户提示和代理提示来形成对话；否则，只包含用户的输入。
        if(agent_type == AgentType.AGENT):
            if user_prompt == "":
                print("empty user prompt for agent check, returning unsafe")
                return "Llama Guard", False, "Missing user_prompt from Agent response check"
            else:
                model_prompt = model_prompt.replace(user_prompt, "")
                user_prompt = f"User: {user_prompt}"
                agent_prompt = f"Agent: {model_prompt}"
                chat = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": agent_prompt},
                ]
        else:
            chat = [
                {"role": "user", "content": model_prompt},
            ]
        # 使用分词器将对话转换为模型的输入ID，并在 GPU 上生成模型的输出。接着，解码输出以获得文本结果。
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[-1]
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        # 将结果分割，并检查第一行是否是 "safe"，以确定内容是否安全。
        splitted_result = result.split("\n")[0];
        is_safe = splitted_result == "safe"    

        report = result
        # 将完整的结果作为报告，并返回一个包含服务名称、文本是否安全的布尔值和生成的报告的元组。
        return "Llama Guard", is_safe, report
        

# Function to load the PeftModel for performance optimization
# Function to determine which safety checker to use based on the options selected
def get_safety_checker(enable_azure_content_safety,
                       enable_sensitive_topics,
                       enable_salesforce_content_safety,
                       enable_llamaguard_content_safety):
    '''它的作用是根据传入的选项决定使用哪些安全检查器'''
    safety_checker = []
    # 是否启用Azure内容安全检查器、AuditNLG敏感话题检查器、Salesforce内容安全检查器和LlamaGuard内容安全检查器。
    if enable_azure_content_safety:
        safety_checker.append(AzureSaftyChecker())
    if enable_sensitive_topics:
        safety_checker.append(AuditNLGSensitiveTopics())
    if enable_salesforce_content_safety:
        safety_checker.append(SalesforceSafetyChecker())
    if enable_llamaguard_content_safety:
        safety_checker.append(LlamaGuardSafetyChecker())
    return safety_checker

