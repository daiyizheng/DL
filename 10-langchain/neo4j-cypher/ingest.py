# -*- encoding: utf-8 -*-
'''
Filename         :ingest.py
Description      :
Time             :2024/04/17 20:25:17
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url=os.environ["ENO4J_URL"], 
                   username=os.environ["ENO4J_USER"], 
                   password=os.environ["EN04J_PASSWORD"])

res = graph.query(
"""
MERGE (m:Movie {name:"Top Gun"})
WITH m
UNWIND ["Tom Cruise", "Val Kilmer", "Anthony Edwards", "Meg Ryan"] AS actor
MERGE (a:Actor {name:actor})
MERGE (a)-[:ACTED_IN]->(m)
"""
)
print(res)
