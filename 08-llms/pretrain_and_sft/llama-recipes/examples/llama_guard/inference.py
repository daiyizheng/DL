import fire
from transformers import AutoTokenizer, AutoModelForCausalLM


from llama_recipes.inference.prompt_format_utils import build_prompt, create_conversation, LLAMA_GUARD_CATEGORY
from typing import List, Tuple
from enum import Enum

class AgentType(Enum):
    '''一个枚举类型，用于区分“代理”和“用户”的角色。'''
    AGENT = "Agent"
    USER = "User"

def main():
    """
    Entry point of the program for generating text using a pretrained model.
    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """
    # 一个包含元组的列表，每个元组包含一组字符串和一个 AgentType。
    # 这些字符串代表对话中的用户或代理的发言
    prompts: List[Tuple[List[str], AgentType]] = [
        (["<Sample user prompt>"], AgentType.USER),

        (["<Sample user prompt>",
        "<Sample agent response>"], AgentType.AGENT),
        
        (["<Sample user prompt>",
        "<Sample agent response>",
        "<Sample user reply>",
        "<Sample agent response>",], AgentType.AGENT),

    ]

    model_id = "meta-llama/LlamaGuard-7b"
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 设置 load_in_8bit 为 True 和 device_map 为 "auto"，以优化内存使用和计算。
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

    
    for prompt in prompts:
        # 使用 build_prompt 函数来构建格式化的提示。
        formatted_prompt = build_prompt(
                prompt[1], 
                LLAMA_GUARD_CATEGORY, 
                create_conversation(prompt[0]))
        # 使用分词器将格式化的提示文本转换为张量，然后将这个张量发送到GPU（如果可用）
        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        # 计算输入张量中的tokens数量
        prompt_len = input["input_ids"].shape[-1]
        # 调用模型的generate方法，传入输入张量和其他参数（如生成的最大新tokens数和填充tokens ID）来生成文本。
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0)
        # 使用分词器将生成的输出tokens解码成文本，跳过特殊tokens，并从提示文本之后的部分开始解码。
        results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
       
        
        print(prompt[0])
        print(f"> {results}")
        # 打印一条分隔线来清晰地区分不同的对话实例
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)