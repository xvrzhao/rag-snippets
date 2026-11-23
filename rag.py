from FlagEmbedding import BGEM3FlagModel
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BGEM3Embeddings(Embeddings):
    """把 FlagEmbedding 的 bge-m3 包装成 LangChain 的 Embeddings"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "mps"):
        self.model = BGEM3FlagModel(model_name, devices=device, use_fp16=False)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=16)["dense_vecs"].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


embeddings = BGEM3Embeddings()
vector_store = InMemoryVectorStore(embedding=embeddings)

# 1. 加载并切分
docs = TextLoader("./doc.md", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
for chunk in chunks:
    print(chunk)
    print("\n\n")

print(f"切分出 {len(chunks)} 个 chunk")

# 2. 入库
vector_store.add_documents(chunks)

# 3. 搜索
query = "长笛应该如何拾音"
for doc, score in vector_store.similarity_search_with_score(query, k=3):
    print(f"\n[{score:.4f}] {doc.page_content[:120]!r}")
