"""
Milvus 向量数据库的基本概念和操作
"""

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
import torch


def init_model():
    """ 初始化 embedding 模型"""

    print("初始化模型: 开始")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    print("初始化模型: 结束")

    return model


def create_collection(client: MilvusClient, collection_name: str = "chunks"):
    """ 创建 collection，相当于关系型数据库中的表 """

    if not client.has_collection(collection_name):
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
            # 具体见：https://claude.ai/share/9914bec2-f242-407e-b7fb-3966630133ce
            # M 表示构建索引时，每个节点的最大连接数，即每个向量点与周围多少个点建立邻居关系
            # efConstruction 代表构建索引时的候选搜索范围，即选择周围邻居的范围，在范围内选择 M 个点建立邻居关系
            # (ef 是候选列表大小的意思，construction 代表构建索引时的候选列表。此外在查询阶段，传入的 ef 代表搜索阶段的候选列表大小)
            params={"M": 16, "efConstruction": 200}, # 常见的默认/推荐值
        )

        # 创建 collection
        client.create_collection(collection_name, schema=schema, index_params=index_params)
        print("创建 collection: 结束")

    print("collection 信息：", client.describe_collection(collection_name))
    print("collection 索引信息：", client.describe_index(collection_name, index_name="embedding"))


def insert_chunks(client: MilvusClient, chunks: list[dict], model: SentenceTransformer, collection_name = "chunks") -> dict:
    for chunk in chunks:
        embedding = model.encode(chunk["content"], normalize_embeddings=True)
        chunk["embedding"] = embedding
    print(chunks)
    return client.insert(collection_name, chunks)


def query(client: MilvusClient, query: str, collection_name = "chunks") -> list[dict]:
    model = init_model()
    query_embedding = model.encode(query)
    hits = client.search(
        collection_name=collection_name, 
        data=[query_embedding], 
        anns_field="embedding", 
        limit=3, 
        output_fields=["content", "page"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}}
    )[0]
    return hits


if __name__ == "__main__":
    client = MilvusClient()
    create_collection(client)
    # chunks = [
    #     {
    #         "content": "向量作为神经网络模型的输出数据格式，能够有效编码信息，并在知识库、语义搜索、检索增强生成（RAG）等人工智能应用中发挥关键作用。",
    #         "doc_id": 1,
    #         "page": 1,
    #         "section_path": "第1章 > 第1节",
    #     },
    #     {
    #         "content": "Milvus 是一款开源向量数据库，适用于各种规模的 AI 应用，从在 Jupyter Notebook 中运行演示版聊天机器人，到构建服务数十亿用户的 Web 级搜索系统。在本指南中，我们将带您了解如何在几分钟内本地部署 Milvus，并使用 Python 客户端库来生成、存储和搜索向量。",
    #         "doc_id": 1,
    #         "page": 2,
    #         "section_path": "第1章 > 第2节",
    #     },
    #     {
    #         "content": "在 Milvus 中，我们需要一个 Collection 来存储向量及其关联的元数据。您可以将其视为传统 SQL 数据库中的表。创建 Collection 时，您可以定义 Schema 和索引参数来配置向量规格，例如维度、索引类型和距离度量。 此外，还有一些复杂的概念用于优化索引以提升向量搜索性能。目前，我们仅关注基础内容，并尽可能使用默认设置。至少，您只需设置Collection名称和Collection中向量字段的维度。",
    #         "doc_id": 1,
    #         "page": 3,
    #         "section_path": "第1章 > 第3节",
    #     }
    # ]
    # # insert_chunks(client, chunks, init_model())

    # hits = query(client, "向量在ai中的作用")
    # for hit in hits:
    #     print(hit, "\n")