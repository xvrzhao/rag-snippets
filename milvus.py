"""
Milvus 向量数据库的基本概念和操作
"""

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import torch


class MilvusOperator:
    def __init__(self, collection_name: str):
        self.model = self.init_model()
        self.collection_name = collection_name
        self.client = MilvusClient(uri="http://localhost:19530")

    def init_model(self):
        """ 初始化 embedding 模型"""

        print(">>> 初始化模型: 开始")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer("BAAI/bge-m3", device=device)
        print(">>> 初始化模型: 结束")

        return model

    def create_collection(self):
        """ 创建 collection，相当于关系型数据库中的表 """

        if not self.client.has_collection(self.collection_name):
            print("创建 collection: 开始")

            # 定义 collection 的结构：schema
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("doc_id", DataType.INT64)
            schema.add_field("page", DataType.INT32)
            schema.add_field("section_path", DataType.VARCHAR, max_length=512)

            # 定义 collection 的索引：index
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding", # 在哪个字段上建立索引
                index_type="HNSW",      # 索引类型 HNSW
                metric_type="COSINE",   # 构建索引时，相似度算法采用余弦相似度

                # 以下两个参数涉及到 HNSW 索引的机制
                # 具体见：https://milvus.io/docs/zh/hnsw.md
                # M 表示构建索引时，每个节点的最大连接数，即每个向量点与周围多少个点建立邻居关系
                # efConstruction 代表构建索引时的候选搜索范围，即选择周围邻居的范围，在范围内选择 M 个点建立邻居关系
                # (ef 是候选列表大小的意思，construction 代表构建索引时的候选列表。此外在查询阶段，传入的 ef 代表搜索阶段的候选列表大小)
                params={"M": 16, "efConstruction": 200}, # 常见的默认/推荐值
            )

            # 创建 collection
            self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            print("创建 collection: 结束")

        print("collection 信息：", self.client.describe_collection(self.collection_name))
        print("collection 索引信息：", self.client.describe_index(self.collection_name, index_name="embedding"))

    def insert_chunks(self, chunks: list[dict], encode_batch_size: int = 32, insert_batch_size: int = 500) -> int:
        """ 批量 encode 并分批写入 Milvus，返回插入总条数 """

        if not chunks:
            return 0

        print(">>> 向量化并入库：开始")

        total = 0
        for start in range(0, len(chunks), insert_batch_size):
            batch = chunks[start : start + insert_batch_size]

            embeddings = self.model.encode(
                [c["content"] for c in batch],
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            rows = [
                {**c, "embedding": emb.tolist()}
                for c, emb in zip(batch, embeddings)
            ]

            res = self.client.insert(self.collection_name, rows)
            total += res["insert_count"]

        print(">>> 向量化并入库：结束")
        return total

    def query(self, query: str, top_k: int = 3) -> list[dict]:
        """ 根据 query 查询最相似的 top_k 个文本 """

        print(f">>> 检索用户问题：{query}")

        query_embedding = self.model.encode(query, normalize_embeddings=True)

        self.client.load_collection(self.collection_name) # search 前需要先 load collection，此操作幂等
        hits = self.client.search(
            collection_name=self.collection_name, 
            data=[query_embedding], 
            anns_field="embedding", 
            limit=top_k, 
            output_fields=["content", "page"],
            search_params={
                "metric_type": "COSINE",    # 需和构建索引时的 metric_type 一致
                "params": {"ef": 64},       # 检索时的候选列表大小，从中选择 top-k 个向量作为返回值
            }
        )[0]

        return hits


if __name__ == "__main__":
    operator = MilvusOperator(collection_name="chunks")
    operator.create_collection()
    chunks = [
        # {
        #     "content": "Attu 是独立的图形化工具,面向开发者,负责数据库管理和操作任务。",
        #     "doc_id": 1,
        #     "page": 4,
        #     "section_path": "第1章 > 第4节",
        # }
    ]
    n = operator.insert_chunks(chunks)
    print(f">>> 入库 {n} 条")

    hits = operator.query("milvus如何查看和操作数据")
    for i, hit in enumerate(hits):
        print(f">>> top-{i}:", hit, "\n")