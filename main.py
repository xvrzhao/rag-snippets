# from FlagEmbedding import BGEM3FlagModel

# model = BGEM3FlagModel('BAAI/bge-m3')

# sentences_1 = ["BGE M3 是什么", "BM25 的定义"]
# sentences_2 = ["BGE M3 是一个嵌入模型", "BM25 是..."]

# embeddings_1 = model.encode(sentences_1)["dense_vecs"]
# print(embeddings_1)

from sentence_transformers import SentenceTransformer
import torch

# Apple Silicon 上用 mps 加速，否则退回 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer("BAAI/bge-m3", device=device)

texts = ["如何重置密码", "忘记密码怎么办", "今天天气不错"]
embeddings = model.encode("texts", normalize_embeddings=True)
print(embeddings)
# print(embeddings.shape)   # (3, 1024)

# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
# from langchain_text_splitters import