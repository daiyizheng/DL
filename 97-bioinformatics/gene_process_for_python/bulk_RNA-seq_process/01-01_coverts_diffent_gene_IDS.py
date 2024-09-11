# -*- encoding: utf-8 -*-
'''
Filename         :coverts_diffent_gene_IDS.py
Description      :基因标签转换
Time             :2024/07/11 20:36:24
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

# !pip install mygene

### Get gene IDs for one gene symbol
import mygene
mg = mygene.MyGeneInfo()
res = mg.query("ZNF263", 
               scopes="symbol", 
               fields="uniprot", 
               species="human")
print(res)

res = mg.query("ZNF263", 
               scopes="symbol", 
               fields="uniprot,ensembl,entrezgene,refseq", 
               species="human")
print(res)


## Convert diffenent IDs to each other
res = mg.query("O14978", 
               scopes="uniprot", 
               fields="entrezgene,ensembl", 
               species="human")
print(res)

## Convert gene IDs in batch
import pandas as pd
df_gene = pd.read_csv("97-bioinformatics/datasets/ZF_KRAB_list.csv")
gene_list = list(df_gene["Gene symbol"])
gene_info = mg.querymany(gene_list, 
                         scopes="symbol", 
                         fields="uniprot,ensembl,entrezgene,refseq", species="human")
print(gene_info)

uni_ids = []
for info_dict in gene_info:
    query = info_dict["query"]
    uni_id = info_dict["uniprot"]["Swiss-Prot"]
    if isinstance(info_dict["ensembl"], dict):
        ensembl_id = info_dict["ensembl"]["gene"]
    if isinstance(info_dict["ensembl"], list):
        ensembl_id = [dit["gene"] for dit in info_dict["ensembl"]]
    entrezgene_id = info_dict["entrezgene"]
    refseq_id = info_dict["refseq"]["genomic"]
    print(query, uni_id, ensembl_id, refseq_id)
    uni_ids.append(uni_id)
    
df_gene["UniprotID"] = uni_id