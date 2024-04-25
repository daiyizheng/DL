# -*- encoding: utf-8 -*-
'''
Filename         :ingest.py
Description      :
Time             :2024/04/17 20:25:17
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(url="bolt://192.168.26.200:7687", 
                   username="neo4j", 
                   password="zju20230808")

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
