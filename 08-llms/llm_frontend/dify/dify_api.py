import requests
import os

api_key = "app-tG37hdnKNZFzh7I6nVVKReG7"
base_url = "http://127.0.0.1:5001/v1"
user = "abc-123"
### 1. 对话
def chat_message(query, api_key):
    param = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": user,
            "files": [
            {
                "type": "image",
                "transfer_method": "remote_url",
                "url": "https://cloud.dify.ai/logo/logo-site.png"
            }
            ]
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(f"{base_url}/chat-messages", headers=headers, json=param)
    print(response.json())

## 上传文件
def upload_file(local_file_path, 
                api_key):
    """
    上传文件
    """
    # 替换为你的实际文件类型
    file_type =  'image/png'# 可以根据实际情况修改为 jpeg、jpg、webp、gif 等

    # API 端点
    url = f'{base_url}/files/upload'

    # 设置请求头
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    # 打开本地文件
    with open(local_file_path, 'rb') as file:
        # 构建表单数据
        files = {
            'file': (os.path.basename(local_file_path), file, file_type) # 使用绝对路径
        }
        data = {
            'user': user
        }

        # 发送 POST 请求
        response = requests.post(url, 
                                 headers=headers, 
                                 files=files, 
                                 data=data)

    # 检查响应状态码
    if response.status_code == 201:
        print("文件上传成功")
        print(response.json())
        id = response.json()['id']
        return id
    else:
        print(f"文件上传失败，状态码: {response.status_code}")
        print(response.text)

def stop_chat_message(task_id):
    url = f"{base_url}/chat-messages/{task_id}/stop"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "user": user
    }
    response = requests.post(url, 
                             headers=headers,
                             json=param)
    if response.status_code == 200:
        print("停止对话成功")
        print(response.json())
    else:
        print(f"停止对话失败，状态码: {response.status_code}")
        print(response.json())

## 消息反馈（点赞）
def message_feedback(message_id):
    url = f"{base_url}/messages/{message_id}/feedbacks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "rating": "like",
        "user": user,
        "content": "message feedback information"
    }
    response = requests.post(url, 
                             headers=headers,
                             json=param)
    if response.status_code == 200:
        print("消息反馈成功")
        print(response.json())
    else:
        print(f"消息反馈失败，状态码: {response.status_code}")


def message_suggested(message_id):
    url = f"{base_url}/messages/{message_id}/suggested?user={user}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "user": user
    }
    response = requests.get(url, 
                             headers=headers, 
                             json=param)
    if response.status_code == 200:
        print("消息推荐成功")
        print(response.json())
    else:
        print(f"消息推荐失败，状态码: {response.status_code}")
        print(response.json())

def get_messages(conversation_id, first_id="", limit=20): # error
    url = f"{base_url}/messages?user={user}&conversation_id={conversation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("获取消息成功")
        print(response.json())
    else:
        print(f"获取消息失败，状态码: {response.status_code}")
        print(response.json())

def get_conversations(user, last_id, limit):
    url = f"{base_url}/conversations?user={user}&last_id={last_id}&limit={limit}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("获取会话成功")
        print(response.json())
    else:
        print(f"获取会话失败，状态码: {response.status_code}")
        print(response.json())


def delete_conversation(conversation_id):
    url = f"{base_url}/conversations/{conversation_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "user": user
    }
    response = requests.delete(url, headers=headers, json=param)
    if response.status_code == 200:
        print("删除会话成功")
        print(response.json())
    else:
        print(f"删除会话失败，状态码: {response.status_code}")
        print(response.json())


def update_conversation_name(conversation_id, name, auto_generate=False):
    url = f"{base_url}/conversations/{conversation_id}/name"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "name": name, 
        "auto_generate": auto_generate, 
        "user": user
    }
    response = requests.post(url, headers=headers, json=param)
    if response.status_code == 200:
        print("更新会话名称成功")
        print(response.json())
    else:
        print(f"更新会话名称失败，状态码: {response.status_code}")
        print(response.json())

def audio_to_text(local_file_path): # error
    url = f"{base_url}/audio-to-text"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    with open(local_file_path, 'rb') as file:
        files = {
            'file': (os.path.basename(local_file_path), file, 'audio/mp3')
        }
        response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        print("音频转文本成功")
        print(response.json())
    else:
        print(f"音频转文本失败，状态码: {response.status_code}")
        print(response.json())

def text_to_audio(text, message_id): ## error
    url = f"{base_url}/text-to-audio"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    param = {
        "text": text,
        "user": user,
        "message_id": message_id
    }
    response = requests.post(url, headers=headers, json=param)
    if response.status_code == 200:
        print("文本转音频成功")
        print(response.json())
    else:
        print(f"文本转音频失败，状态码: {response.status_code}")
        print(response.json())

def get_info():
    url = f"{base_url}/info"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    print(response.json())
    if response.status_code == 200:
        print("获取信息成功")
        print(response.json())
    else:
        print(f"获取信息失败，状态码: {response.status_code}")
        print(response.json())

def get_parameters():
    url = f"{base_url}/parameters"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    print(response.json())
    if response.status_code == 200:
        print("获取参数成功")
        print(response.json())
    else:
        print(f"获取参数失败，状态码: {response.status_code}")
        print(response.json())

def get_meta():
    url = f"{base_url}/meta"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    print(response.json())
    if response.status_code == 200:
        print("获取元数据成功")
        print(response.json())
    else:
        print(f"获取元数据失败，状态码: {response.status_code}")
        print(response.json())

if __name__ == '__main__':
    ## 对话
    query = "介绍一下嘉兴？"
    chat_message(query, api_key)
    ## 上传文件
    # local_file_path = "public/logo/logo.png"
    # upload_file(local_file_path, api_key)
    ## 停止对话
    # task_id = "2a1e7daf-053f-4f49-b6a5-aa3f060df69d"
    # stop_chat_message(task_id)
    # message_id = "cce3efd8-aac8-400c-90d2-7466ea5af31d"
    # message_feedback(message_id)

    ## 获取下一轮建议问题列表
    # message_id = "d91aa7bd-9134-4df8-9813-1325895adfe5"
    # message_suggested(message_id)
    ## 获取历史消息
    # conversation_id = "de8a8732-d09a-48c4-a12e-c7640d62c025"
    # get_messages(conversation_id)
    ## 获取会话列表
    # last_id = ""
    # limit = 20
    # get_conversations(user, last_id, limit)
    ## 删除会话
    # conversation_id = "6116f16c-4f7b-4439-b996-d1e68b07b6ee"
    # delete_conversation(conversation_id)
    ## 更新会话名称
    # conversation_id = "a8b4f71a-f069-4318-bcce-faca65f352b4"
    # name = "iphone15"
    # update_conversation_name(conversation_id, name)
    ## 音频转文本
    # local_file_path = "tests/16k16bit.mp3"
    # audio_to_text(local_file_path)
    ## 文本转音频
    # text = "你好Dify"
    # message_id = "5ad4cb98-f0c7-4085-b384-88c403be6290"
    # text_to_audio(text, message_id)
    ## 获取应用基本信息
    # get_info()
    ## 获取应用参数
    # get_parameters()
    ## 获取应用Meta信息
    # get_meta()
