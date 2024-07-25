'''
@Description: A text embedding model and reranking model produced by YouDao Inc., which can be use for dense embedding retrieval and reranking in RAG workflow.
@Author: shenlei
@Date: 2023-11-28 17:53:45
@LastEditTime: 2024-01-23 18:30:59
@LastEditors: shenlei
'''
# setup函数用于安装包，find_packages函数用于自动查找当前目录下的所有包（一个包对应一个目录，并且包含__init__.py文件）。
from setuptools import setup, find_packages

# 读取README.md文件的全部内容，并将其存储在变量readme中。
with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name='BCEmbedding', # 包的名称
    version='0.1.3', # 版本号
    license='apache-2.0', # 许可证
    description='A text embedding model and reranking model produced by Netease Youdao Inc., which can be use for dense embedding retrieval and reranking in RAG workflow.',
    long_description=readme, # 使用之前读取的README.md文件内容作为包的长描述。
    long_description_content_type="text/markdown", # 指定长描述的内容类型为Markdown
    author='Netease Youdao, Inc.', # 指定包的作者为网易有道公司
    author_email='shenlei02@corp.netease.com', # 包作者的电子邮件地址
    url='https://gitlab.corp.youdao.com/ai/BCEmbedding', # 给出了包的源代码所在的URL
    packages=find_packages(), # 调用find_packages()自动查找并列出所有应该包含在包内的子包。
    install_requires=[ # 定义了安装这个包需要预先安装的依赖包列表
        'torch>=1.6.0',
        'transformers>=4.35.0,<4.37.0',
        'datasets',
        'sentence-transformers'
    ]
)
