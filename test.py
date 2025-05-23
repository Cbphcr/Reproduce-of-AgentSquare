import os
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import uuid


class MemoryBase:
    def __init__(self, memory_type: str):
        self.embedding = OpenAIEmbeddings()
        self.db_path = os.path.join("./db", memory_type)
        # self.db_path = os.path.join(self.db_path, str(uuid.uuid4()))
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding, persist_directory=self.db_path
        )
        # 输出底层
        print(f"[backbone] {dir(self.scenario_memory._collection._client)}")

    def add_memory(self, text: str):
        # 简单添加一些文本（真实用法会使用文档对象）
        self.scenario_memory.add_documents([Document(page_content=text)])
        print(f"[Added] {text}")

    def __del__(self):
        # 关闭数据库连接

        self.scenario_memory.reset_collection()

        print(f"[Closed] {self.db_path}")


def use_memory():
    m = MemoryBase("test_memory")
    m.add_memory("This is a memory.")
    # 没有调用 m.scenario_memory.close() 或 m = None


if __name__ == "__main__":
    print("=== First call ===")
    use_memory()  # 第一次成功

    print("\n=== Second call ===")
    use_memory()  # 第二次可能报错（如 PermissionError / Database is locked）
