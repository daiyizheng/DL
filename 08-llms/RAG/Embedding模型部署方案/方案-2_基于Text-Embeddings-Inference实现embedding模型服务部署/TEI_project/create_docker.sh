# --name text_embeddings_container 为容器指定一个名称
# --gpus all 使用本机的所有GPU,也可以指定GPU,例如: --gpus '"device=0"' 
# -d 容器运行在后台
# -p 8080:80 主机端口是8080, 容器内端口是80
# -v /root/TEI/models:/data 将主机的/root/TEI/models目录挂载到容器的/data目录
# ghcr.io/huggingface/text-embeddings-inference:turing-1.1 指定使用的Docker镜像以及其版本标签
# --model-id /data/bge-large-zh-v1.5 指定使用的模型ID
docker run -d --name text_embeddings_container --gpus all -p 8080:80 -v /root/TEI/models:/models ghcr.io/huggingface/text-embeddings-inference:turing-1.1 --dtype float16 --model-id /models/bge-large-zh-v1.5