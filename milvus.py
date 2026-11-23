"""
Milvus 向量数据库的基本概念和操作

本文件演示：
1. collection 的创建：稠密向量字段 + 稀疏向量字段，以及各自的索引
2. 用 BGE-M3 同时生成稠密向量和稀疏词权重（lexical weights）
3. 从 chunks.json 加载 chunk 数据并入库
4. 稠密检索、稀疏检索、RRF 混合检索（手动实现 / Milvus 原生实现）
"""

import json
from pathlib import Path

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
from FlagEmbedding import BGEM3FlagModel
import torch


def load_chunks(json_path: str | Path) -> list[dict]:
    """ 从 JSON 文件加载 chunk 数据 """

    with open(json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # 简单校验每条 chunk 的字段，避免入库时才报错
    required_fields = {"content", "doc_id", "page", "section_path"}
    for i, chunk in enumerate(chunks):
        missing = required_fields - chunk.keys()
        if missing:
            raise ValueError(f"第 {i} 条 chunk 缺少字段：{sorted(missing)}")

    return chunks


class MilvusOperator:
    DENSE_FIELD = "embedding"
    SPARSE_FIELD = "sparse"
    OUTPUT_FIELDS = ["content", "page", "section_path"]

    def __init__(self, collection_name: str):
        self.model = self.init_model()
        self.collection_name = collection_name
        self.client = MilvusClient(uri="http://localhost:19530")

    def init_model(self):
        """ 初始化 BGE-M3 模型：一次编码同时得到稠密向量和稀疏词权重 """

        print(">>> 初始化模型: 开始")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # FlagEmbedding 的 BGEM3FlagModel 既能输出 dense_vecs（稠密向量），
        # 也能输出 lexical_weights（稀疏词权重，格式为 {token_id: 权重}）
        # use_fp16=False：演示环境优先保证稳定，避免 MPS 上 fp16 的兼容问题
        model = BGEM3FlagModel(
            "BAAI/bge-m3",
            normalize_embeddings=True,  # 稠密向量做 L2 归一化，配合 COSINE 使用
            devices=device,
            use_fp16=False,
        )
        print(">>> 初始化模型: 结束")

        return model

    def encode(self, texts: list[str], batch_size: int = 32) -> tuple[list[list[float]], list[dict[int, float]]]:
        """ 同时生成稠密向量和稀疏向量

        稀疏向量（lexical weights）是 {token_id: 权重} 的字典，
        Milvus 的 SPARSE_FLOAT_VECTOR 字段直接接收这种字典。
        """

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # 不需要 ColBERT 多向量，省算力
        )

        dense_vectors = output["dense_vecs"].tolist()

        # FlagEmbedding 返回的 token_id 是字符串、权重是 numpy.float32，
        # 这里统一转成 Milvus 需要的 Python 类型
        sparse_vectors = [
            {int(token_id): float(weight) for token_id, weight in weights.items()}
            for weights in output["lexical_weights"]
        ]

        return dense_vectors, sparse_vectors

    def create_collection(self):
        """ 创建 collection；如果已存在，则确保它有稀疏字段和索引 """

        if not self.client.has_collection(self.collection_name):
            print("创建 collection: 开始")

            # 定义 collection 的结构：schema
            schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field(self.DENSE_FIELD, DataType.FLOAT_VECTOR, dim=1024)
            # 稀疏向量长度不固定，只有非零维度会被存储，因此不需要指定 dim
            schema.add_field(self.SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("doc_id", DataType.INT64)
            schema.add_field("page", DataType.INT32)
            schema.add_field("section_path", DataType.VARCHAR, max_length=512)

            # 定义 collection 的索引：index
            index_params = MilvusClient.prepare_index_params()

            # 稠密向量索引
            index_params.add_index(
                field_name=self.DENSE_FIELD, # 在哪个字段上建立索引
                index_type="HNSW",           # 索引类型 HNSW
                metric_type="COSINE",        # 构建索引时，相似度算法采用余弦相似度

                # 以下两个参数涉及到 HNSW 索引的机制
                # 具体见：https://milvus.io/docs/zh/hnsw.md
                # M 表示构建索引时，每个节点的最大连接数，即每个向量点与周围多少个点建立邻居关系
                # efConstruction 代表构建索引时的候选搜索范围，即选择周围邻居的范围，在范围内选择 M 个点建立邻居关系
                # (ef 是候选列表大小的意思，construction 代表构建索引时的候选列表。此外在查询阶段，传入的 ef 代表搜索阶段的候选列表大小)
                params={"M": 16, "efConstruction": 200}, # 常见的默认/推荐值
            )

            # 稀疏向量索引: https://milvus.io/docs/zh/sparse-inverted-index.md
            # SPARSE_INVERTED_INDEX 是稀疏倒排索引；稀疏向量用 IP（内积）计算相似度
            # drop_ratio_build：构建索引时丢弃权重最小的 20% 维度，用来减小索引体积
            index_params.add_index(
                field_name=self.SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            # 创建 collection
            self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            print("创建 collection: 结束")

        self.client.load_collection(self.collection_name)  # search 前需要先 load collection，此操作幂等

        print("collection 信息：", self.client.describe_collection(self.collection_name))
        for field in (self.DENSE_FIELD, self.SPARSE_FIELD):
            print(f"collection 索引信息（{field}）：", self.client.describe_index(self.collection_name, index_name=field))

    def insert_chunks(self, chunks: list[dict], encode_batch_size: int = 32, insert_batch_size: int = 500) -> int:
        """ 批量 encode 并分批写入 Milvus，返回插入总条数 """

        if not chunks:
            return 0

        print(">>> 向量化并入库：开始")

        total = 0
        for start in range(0, len(chunks), insert_batch_size):
            batch = chunks[start : start + insert_batch_size]

            dense_vectors, sparse_vectors = self.encode(
                [c["content"] for c in batch],
                batch_size=encode_batch_size,
            )

            rows = [
                {**c, self.DENSE_FIELD: dense, self.SPARSE_FIELD: sparse}
                for c, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
            ]

            res = self.client.insert(self.collection_name, rows)
            total += res["insert_count"]

        self.client.flush(self.collection_name)  # 让新写入的数据立即可见

        print(">>> 向量化并入库：结束")
        return total

    def _encode_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        """ 把查询文本编码成 (稠密向量, 稀疏向量) """

        dense_vectors, sparse_vectors = self.encode([query], batch_size=1)
        return dense_vectors[0], sparse_vectors[0]

    def _format_hits(self, hits: list[dict]) -> list[dict]:
        """ 把 Milvus 的原始搜索结果整理成易读的 dict """

        results = []
        for hit in hits:
            entity = hit.get("entity") or {}
            results.append({
                "id": hit["id"],
                "score": hit["distance"],
                "content": entity.get("content"),
                "page": entity.get("page"),
                "section_path": entity.get("section_path"),
            })
        return results

    def dense_search(self, query: str, top_k: int = 3) -> list[dict]:
        """ 稠密向量检索：擅长语义相似（近义词、换一种说法） """

        print(f">>> 稠密检索用户问题：{query}")

        dense_vector, _ = self._encode_query(query)

        self.client.load_collection(self.collection_name)  # search 前需要先 load collection，此操作幂等
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vector],
            anns_field=self.DENSE_FIELD,
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS,
            search_params={
                "metric_type": "COSINE",    # 需和构建索引时的 metric_type 一致
                "params": {"ef": 64},       # 检索时的候选列表大小，从中选择 top-k 个向量作为返回值
            },
        )[0]

        return self._format_hits(hits)

    def sparse_search(self, query: str, top_k: int = 3) -> list[dict]:
        """ 稀疏向量检索：擅长关键词/术语精确匹配（类似 BM25） """

        print(f">>> 稀疏检索用户问题：{query}")

        _, sparse_vector = self._encode_query(query)

        self.client.load_collection(self.collection_name)
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[sparse_vector],
            anns_field=self.SPARSE_FIELD,
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS,
            search_params={
                "metric_type": "IP",                    # 稀疏索引使用内积
                "params": {"drop_ratio_search": 0.2},   # 检索时忽略权重最小的 20% 维度
            },
        )[0]

        return self._format_hits(hits)

    def hybrid_search(self, query: str, top_k: int = 3, candidate_k: int | None = None, rrf_k: int = 60) -> list[dict]:
        """ 混合检索（手动 RRF）

        分别取稠密和稀疏的前 candidate_k 条结果，用 RRF 公式融合：

            RRF(d) = Σ 1 / (rrf_k + rank_i(d))

        其中 rank_i(d) 是文档 d 在第 i 路检索结果中的排名（从 1 开始）；
        rrf_k 是平滑参数（常取 60），用来缩小排名靠前结果之间的权重差距。
        RRF 只看排名、不看原始分数，因此不需要把余弦相似度和内积归一化到同一量纲。
        """

        if candidate_k is None:
            candidate_k = top_k * 4

        print(f">>> 混合检索（手动 RRF）用户问题：{query}")

        dense_hits = self.dense_search(query, top_k=candidate_k)
        sparse_hits = self.sparse_search(query, top_k=candidate_k)

        fused: dict[int, dict] = {}

        def merge(source: str, hits: list[dict]):
            for rank, hit in enumerate(hits, start=1):
                item = fused.setdefault(hit["id"], {
                    "id": hit["id"],
                    "content": hit["content"],
                    "page": hit["page"],
                    "section_path": hit["section_path"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_score": None,
                    "sparse_score": None,
                })
                item[f"{source}_rank"] = rank
                item[f"{source}_score"] = hit["score"]
                item["rrf_score"] += 1.0 / (rrf_k + rank)

        merge("dense", dense_hits)
        merge("sparse", sparse_hits)

        results = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        return results[:top_k]

    def hybrid_search_native(self, query: str, top_k: int = 3, candidate_k: int | None = None, rrf_k: int = 60) -> list[dict]:
        """ 混合检索（Milvus 原生 hybrid_search + RRFRanker）

        把两路检索封装成 AnnSearchRequest，由 Milvus 在数据库侧完成 RRF 融合，
        不需要把两路原始结果全部传回客户端。
        """

        if candidate_k is None:
            candidate_k = top_k * 4

        print(f">>> 混合检索（Milvus 原生 RRF）用户问题：{query}")

        dense_vector, sparse_vector = self._encode_query(query)

        self.client.load_collection(self.collection_name)
        requests = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field=self.DENSE_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=candidate_k,
            ),
            AnnSearchRequest(
                data=[sparse_vector],
                anns_field=self.SPARSE_FIELD,
                param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                limit=candidate_k,
            ),
        ]

        hits = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=requests,
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=self.OUTPUT_FIELDS,
        )[0]

        return self._format_hits(hits)


if __name__ == "__main__":
    operator = MilvusOperator(collection_name="chunks")
    operator.create_collection()

    # 1. 从单独的 JSON 文件加载 chunk 数据
    chunks_path = Path(__file__).with_name("chunks.json")
    chunks = load_chunks(chunks_path)
    print(f">>> 从 {chunks_path.name} 加载 {len(chunks)} 条 chunks")

    # 2. 向量化并入库
    n = operator.insert_chunks(chunks)
    print(f">>> 入库 {n} 条")

    # 3. 四种检索方式对比
    query = "milvus如何查看和操作数据"
    for name, search in [
        ("稠密检索", operator.dense_search),
        ("稀疏检索", operator.sparse_search),
        ("混合检索（手动 RRF）", operator.hybrid_search),
        ("混合检索（Milvus 原生 RRF）", operator.hybrid_search_native),
    ]:
        print(f"\n===== {name} =====")
        for i, hit in enumerate(search(query)):
            print(f">>> top-{i}:", hit)
