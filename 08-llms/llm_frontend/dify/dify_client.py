#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :dify_client_test.py
@Description  :
@Time         :2025/04/15 10:54:37
@Author       :daiyizheng
@Version      :1.0
'''
import os,sys
sys.path.append("./")
import time
import unittest

api_key = "dataset-2br8pWJgKwkqsKfwHEhCz7sS" # "app-tG37hdnKNZFzh7I6nVVKReG7"
base_url = "http://127.0.0.1:5001/v1"
app_id = "710e070a-d8d4-4142-9fa4-50aece1b0078"
user = "abc-123"

API_KEY = os.environ.get("API_KEY", api_key)
APP_ID = os.environ.get("APP_ID", app_id)
API_BASE_URL = os.environ.get("API_BASE_URL", base_url)
FILE_PATH_BASE = os.path.dirname(os.path.dirname(__file__))



from dify_client import (
    ChatClient,
    CompletionClient,
    DifyClient,
    KnowledgeBaseClient,
)


class TestKnowledgeBaseClient(unittest.TestCase):
    def setUp(self):
        self.knowledge_base_client = KnowledgeBaseClient(API_KEY, base_url=API_BASE_URL)
        self.README_FILE_PATH = os.path.abspath(
            os.path.join(FILE_PATH_BASE, "README.md")
        )
        self.dataset_id = None
        self.document_id = None
        self.segment_id = None
        self.batch_id = None

    def _get_dataset_kb_client(self):
        """
        获取知识库客户端
        """
        self.assertIsNotNone(self.dataset_id)
        return KnowledgeBaseClient(
            API_KEY, base_url=API_BASE_URL, dataset_id=self.dataset_id
        )

    def test_001_create_dataset(self):
        ## 创建空的知识库

        """
        参数， 知识数据库
        知识数据库 name
        "permission": "only_me"
        
        """
        param = {"params":
                    {
                    "permission": "only_me",
                    "provider": "vendor",
                    }
                }
        response = self.knowledge_base_client.create_dataset(name="test_dataset", **param)
        data = response.json()
        self.assertIn("id", data)
        self.dataset_id = data["id"]
        self.assertEqual("test_dataset", data["name"])

        # the following tests require to be executed in order because they use
        # the dataset/document/segment ids from the previous test
        self._test_002_list_datasets()
        self._test_003_create_document_by_text()
        time.sleep(1)
        self._test_004_update_document_by_text()
        # self._test_005_batch_indexing_status()
        time.sleep(1)
        self._test_006_update_document_by_file()
        time.sleep(1)
        self._test_007_list_documents()
        self._test_008_delete_document()
        self._test_009_create_document_by_file()
        time.sleep(1)
        self._test_010_add_segments()
        self._test_011_query_segments()
        self._test_012_update_document_segment()
        self._test_013_delete_document_segment()
        self._test_014_delete_dataset()

    def _test_002_list_datasets(self):
        ## 获取知识库列表
        response = self.knowledge_base_client.list_datasets()
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("total", data)

    def _test_003_create_document_by_text(self):
        ## 通过文本创建文档
        client = self._get_dataset_kb_client()

        """
        参数， 文档
        知识库名称 name
        文本内容 text
        额外参数 extra_params
        {
            'indexing_technique': 'high_quality',
            'doc_form': 'text_model',
            'process_rule': {
                'rules': {
                    'pre_processing_rules': [
                        {'id': 'remove_extra_spaces', 'enabled': True},
                        {'id': 'remove_urls_emails', 'enabled': True}
                    ],
                    'segmentation': {
                        'separator': '\n',
                        'max_tokens': 500
                    }
                },
                'mode': 'custom'
            }
        }
        """
        extra_params = {
            'indexing_technique': 'high_quality',
            'doc_form': 'text_model',
            'process_rule': {
                'rules': {
                    'pre_processing_rules': [
                        {'id': 'remove_extra_spaces', 'enabled': True},
                        {'id': 'remove_urls_emails', 'enabled': True}
                    ],
                    'segmentation': {
                        'separator': '\n',
                        'max_tokens': 500
                    }
                },
                'mode': 'custom'
            }
        }
        response = client.create_document_by_text(name="test_document", text="test_text", extra_params=extra_params)
        data = response.json()
        self.assertIn("document", data)
        self.document_id = data["document"]["id"]
        self.batch_id = data["batch"]

    def _test_004_update_document_by_text(self):
        ## 通过文本更新文档
        client = self._get_dataset_kb_client()
        self.assertIsNotNone(self.document_id)
        response = client.update_document_by_text(
            document_id=self.document_id, name="test_document_updated", text="test_text_updated"
        )
        data = response.json()
        self.assertIn("document", data)
        self.assertIn("batch", data)
        self.batch_id = data["batch"]

    def _test_005_batch_indexing_status(self):
        client = self._get_dataset_kb_client()
        response = client.batch_indexing_status(self.batch_id)
        response.json()
        self.assertEqual(response.status_code, 200)

    def _test_006_update_document_by_file(self):
        ## 通过文件更新文档
        client = self._get_dataset_kb_client()
        self.assertIsNotNone(self.document_id)
        response = client.update_document_by_file(
            document_id=self.document_id, file_path=self.README_FILE_PATH
        )
        data = response.json()
        self.assertIn("document", data)
        self.assertIn("batch", data)
        self.batch_id = data["batch"]

    def _test_007_list_documents(self):
        ## 知识库文档列表
        client = self._get_dataset_kb_client()
        response = client.list_documents()
        data = response.json()
        self.assertIn("data", data)

    def _test_008_delete_document(self):
        ## 删除文档
        client = self._get_dataset_kb_client()
        self.assertIsNotNone(self.document_id)
        response = client.delete_document(document_id=self.document_id)
        data = response.json()
        self.assertIn("result", data)
        self.assertEqual("success", data["result"])

    def _test_009_create_document_by_file(self):
        ## 通过文件创建文档
        client = self._get_dataset_kb_client()
        response = client.create_document_by_file(file_path=self.README_FILE_PATH)
        data = response.json()
        self.assertIn("document", data)
        self.document_id = data["document"]["id"]
        self.batch_id = data["batch"]

    def _test_010_add_segments(self):
        ## 添加文档子分段
        client = self._get_dataset_kb_client()
        response = client.add_segments(
            self.document_id, [{"content": "test text segment 1"}]
        )
        data = response.json()
        self.assertIn("data", data)
        self.assertGreater(len(data["data"]), 0)
        segment = data["data"][0]
        self.segment_id = segment["id"]

    def _test_011_query_segments(self):
        ## 查询文档子分段
        client = self._get_dataset_kb_client()
        response = client.query_segments(self.document_id)
        data = response.json()
        self.assertIn("data", data)
        self.assertGreater(len(data["data"]), 0)

    def _test_012_update_document_segment(self):
        ## 更新文档子分段
        client = self._get_dataset_kb_client()
        self.assertIsNotNone(self.segment_id)
        response = client.update_document_segment(
            self.document_id,
            self.segment_id,
            {"content": "test text segment 1 updated"},
        )
        data = response.json()
        self.assertIn("data", data)
        self.assertGreater(len(data["data"]), 0)
        segment = data["data"]
        self.assertEqual("test text segment 1 updated", segment["content"])

    def _test_013_delete_document_segment(self):
        ## 删除文档子分段
        client = self._get_dataset_kb_client()
        self.assertIsNotNone(self.segment_id)
        response = client.delete_document_segment(self.document_id, self.segment_id)
        data = response.json()
        self.assertIn("result", data)
        self.assertEqual("success", data["result"])

    def _test_014_delete_dataset(self):
        ## 删除知识库
        client = self._get_dataset_kb_client()
        response = client.delete_dataset()
        self.assertEqual(204, response.status_code)


class TestChatClient(unittest.TestCase):
    def setUp(self):
        self.chat_client = ChatClient(API_KEY, base_url=API_BASE_URL)

    def create_chat_message(self):
        ## 发送对话消息
        response = self.chat_client.create_chat_message(
            {}, "介绍一下嘉兴", "test_user"
        )
        self.assertIn("answer", response.text)

    def create_chat_message_with_vision_model_by_remote_url(self): 
        # 使用远程图片发送对话消息
        files = [
            {"type": "image", "transfer_method": "remote_url", "url": "http://shaolab.scienceai.top/images/journals/scniche_j.jpg"}
        ]
        response = self.chat_client.create_chat_message(
            {}, "Describe the picture.", "test_user", files=files
        )
        self.assertIn("answer", response.text)

    def create_chat_message_with_vision_model_by_local_file(self): 
        # 使用本地图片发送对话消息
        files = [
            {
                "type": "image",
                "transfer_method": "local_file",
                "upload_file_id": "018b7602-7fe5-4823-8d43-441612b04f6a",
            }
        ]
        response = self.chat_client.create_chat_message(
            {}, "Describe the picture.", "test_user", files=files
        )
        self.assertIn("answer", response.text)

    def get_conversation_messages(self):
        ## 获取历史对话消息
        response = self.chat_client.get_conversation_messages(
            "test_user", "de8a8732-d09a-48c4-a12e-c7640d62c025"
        )
        self.assertIn("answer", response.text)

    def get_conversations(self):
        ## 获取会话信息
        response = self.chat_client.get_conversations("test_user")
        self.assertIn("data", response.text)


class TestCompletionClient(unittest.TestCase):
    def setUp(self):
        self.completion_client = CompletionClient(API_KEY, base_url=API_BASE_URL)

    def create_completion_message(self):
        ## 生成对话消息
        response = self.completion_client.create_completion_message(
            {"query": "介绍一下嘉兴"}, "blocking", "test_user"
        )
        self.assertIn("answer", response.text)

    def create_completion_message_with_vision_model_by_remote_url(self):
        ## 使用远程图片生成对话消息
        files = [
            {"type": "image", "transfer_method": "remote_url", "url": "http://shaolab.scienceai.top/images/journals/scniche_j.jpg"}
        ]
        response = self.completion_client.create_completion_message(
            {"query": "Describe the picture."}, "blocking", "test_user", files
        )
        self.assertIn("answer", response.text)

    def create_completion_message_with_vision_model_by_local_file(self):
        ## 使用本地图片生成对话消息
        files = [
            {
                "type": "image",
                "transfer_method": "local_file",
                "upload_file_id": "018b7602-7fe5-4823-8d43-441612b04f6a",
            }
        ]
        response = self.completion_client.create_completion_message(
            {"query": "Describe the picture."}, "blocking", "test_user", files
        )
        self.assertIn("answer", response.text)


class TestDifyClient(unittest.TestCase):
    def setUp(self):
        self.dify_client = DifyClient(API_KEY, base_url=API_BASE_URL)

    def message_feedback(self):
        ## 反馈消息
        response = self.dify_client.message_feedback(
            "658fc601-4e8b-4fef-b374-1f3da7bbff2d", "like", "test_user"
        )
        self.assertIn("success", response.text)

    def get_application_parameters(self):
        ## 获取应用参数
        response = self.dify_client.get_application_parameters("test_user")
        self.assertIn("user_input_form", response.text)

    def file_upload(self):
        ## 上传文件
        file_path = "/Users/a1-6/Documents/projects/web_projects/dify-1.2.0/web/public/logo/logo-site.png"
        file_name = "logo-site.png"
        mime_type = "image/png"

        with open(file_path, "rb") as file:
            files = {"file": (file_name, file, mime_type)}
            response = self.dify_client.file_upload("test_user", files)
            self.assertIn("name", response.text)


if __name__ == "__main__":
    unittest.main()