# 使用官方Python 3.10镜像作为基础镜像
FROM python:3.10-slim
# 为镜像添加元数据标签
LABEL authors="唐国梁Tommy"

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录下的所有文件复制到容器的/app目录中
COPY . /app

# 使用pip命令安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露容器的8000端口
EXPOSE 8000

# 定义容器启动时执行的命令，使用uvicorn运行FastAPI应用
CMD ["uvicorn", "embedding_app:app", "--host", "0.0.0.0", "--port", "8000"]
