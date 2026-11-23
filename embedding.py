from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', normalize_embeddings=True, devices="mps", return_sparse=True)

sentences = ["密集向量和稀疏向量分别是什么"]

embeddings = model.encode(sentences, return_sparse=True)

print("dense_vecs:", embeddings["dense_vecs"][0])
print("lexical_weights:", embeddings["lexical_weights"][0])
