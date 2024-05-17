# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 已经解读
import fire
import sys
sys.path.append("./src")
from llama_recipes.finetuning import main

if __name__ == "__main__":
    fire.Fire(main)