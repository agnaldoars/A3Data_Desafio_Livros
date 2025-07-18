# ================================
# 1. Importações
# ================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch

# Para sumarização com LLM (exemplo usando OpenAI)
# from openai import OpenAI

# ================================
# 2. Carregar dados processados
# ================================
df = pd.read_csv("avaliacoes_processadas2.csv")
df4 = df.copy()
df = df.dropna(
    subset=["review_clean"]
)  # removendo todas as linhas do DataFrame df onde a coluna "review_clean" tem valor NaN (nulo ou ausente).
df = df[:100]
df.shape
df.to_csv("Slice.csv", index=False)
# df = df4.copy()
# ================================
# 3. Gerar Embeddings com SentenceTransformer
# ================================

# Modelo multilíngue leve (ajustável)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Geração dos embeddings
texts = df["review_clean"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# ================================
# 4. Criar índice FAISS
# ================================
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

print(f"Base indexada com {index.ntotal} avaliações.")


# ================================
# 5. Função de Busca Semântica
# ================================
# df.columns
def buscar_reviews(pergunta, top_k=3):
    pergunta_emb = model.encode([pergunta])
    D, I = index.search(np.array(pergunta_emb).astype("float32"), top_k)
    resultados = df.iloc[I[0]][["Title_clean", "score", "sentiment", "text"]]
    return resultados


# Exemplo de uso:
print("🔍 Exemplo de busca:")
# What are common themes in the Biography & Autobiography reviews?
# How do readers feel about Philip Nel’s Dr. Seuss: American Icon?
# What is the general feedback on books in the Religion category?
# How do Fiction reviews differ from Religion reviews?
# What makes a review positive or negative?
print(buscar_reviews("What are common themes in the Biography?"))

# ================================
# 6. Sumarização com LLM
# ================================

from transformers import pipeline

# Carregando pipeline de summarization local ou via Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def resumir_reviews(reviews_df):
    textos = " ".join(reviews_df["text"].tolist())
    if len(textos) > 2000:
        textos = textos[:2000]  # truncar texto se for muito grande
    resumo = summarizer(textos, max_length=130, min_length=30, do_sample=False)
    return resumo[0]["summary_text"]


# Exemplo de uso:
# How do readers feel about Philip Nel’s Dr. Seuss: American Icon?
# Which reviews mention well-illustrated books?
# Are there any well-illustrated books in this dataset?
# resultados = buscar_reviews("livros com boas ilustrações")
# resultados = buscar_reviews("Which reviews mention well-illustrated books?")
resultados = buscar_reviews(
    "How do readers feel about Philip Nel’s Dr. Seuss: American Icon?"
)

resumo = resumir_reviews(resultados)
print("\n🧠 Resumo com LLM:\n", resumo)
# 🧠 Resumo com LLM:
# "Dr. Seuss: American Icon" by Philip Nel is a thoughtful deconstruction
# of the life and work of Theodore Geisel. In this thoughtful book, Mr.
# Nel deepens our appreciation for Seuss as a distinctively American poet,
# artist and educator.
