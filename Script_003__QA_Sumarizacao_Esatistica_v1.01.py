# Sumarização estatística da base
# 🎯 Objetivo:
# Gerar estatísticas úteis para negócio e análise exploratória, incluindo:
# 📚 Total de livros, autores e categorias
# 👤 Total de usuários únicos que avaliaram
# 📝 Média de score e sentimento por autor e por categoria
# 🔢 Quantidade de reviews por autor
# 🔍 Distribuição de tamanho das reviews
# 📈 Correlação entre score e sentimento

import pandas as pd
import numpy as np

df = pd.read_csv("avaliacoes_processadas.csv")
# df4 = df.copy()

ratings = pd.read_csv("Books_rating.csv")
books = pd.read_csv("books_data.csv")

# Títulos normalizados
ratings["Title_clean"] = ratings["Title"].str.strip().str.lower()
books["Title_clean"] = books["Title"].str.strip().str.lower()


# Total de reviews
total_reviews = ratings.shape[0]


# Total de títulos
# total_books = books.shape[0]


# Total de títulos únicos
total_books = books["Title_clean"].nunique()
# Title unicos 212403 -- unique 212403 review
# Title unicos 212403 -- unique 212403 review
# Listar títulos únicos em cada base
titulos_review = set(ratings["Title_clean"].unique())
titulos_books = set(books["Title_clean"].unique())

# Total de autores únicos
total_authors = books["authors"].nunique()

# Total de categorias únicas
total_categories = books["categories"].nunique()

# Total de usuários únicos
total_users = ratings["profileName"].nunique()

print(f"Total de avaliações: {total_reviews}")
print(f"Total de livros únicos: {total_books}")
print(f"Total de autores: {total_authors}")
print(f"Total de categorias: {total_categories}")
print(f"Total de usuários únicos: {total_users}")
print(f"Total de livros por Usuario: {total_books/total_users}")
print(f"Total de Usuario por livros: {total_users/total_books}")
books.info()
books["ratingsCount"].describe()

books.info()
books["ratingsCount"].describe()

df.info()
for collumn in ["Price", "score", "ratingsCount", "sentiment"]:
    print(df.loc[:, [collumn]].describe())

# df.loc[:,['Price','score','ratingsCount','sentiment']].describe()

# 📊 Estatísticas por Autor
# Média, contagem e sentimento por autor
autores_stats = (
    df.groupby("authors")
    .agg(
        quantidade_reviews=("text", "count"),
        media_score=("score", "mean"),
        media_sentimento=("sentiment", "mean"),
    )
    .sort_values(by="quantidade_reviews", ascending=False)
)

autores_stats.head(10)

# 📚 Estatísticas por Categoria
categorias_stats = (
    df.groupby("categories")
    .agg(
        qtd_reviews=("text", "count"),
        media_score=("score", "mean"),
        media_sentimento=("sentiment", "mean"),
    )
    .sort_values(by="qtd_reviews", ascending=False)
)

categorias_stats.head(10)

# 📝 Tamanho médio das avaliações
# Número de palavras por review
df["review_length"] = df["review_clean"].apply(lambda x: len(str(x).split()))

# Média e máximo
print("Média de palavras por review:", df["review_length"].mean())
print("Review mais longa:", df["review_length"].max(), "palavras")
print(df["review_length"].describe())

# 📈 Correlação entre score e sentiment
corr = df[["score", "sentiment"]].corr()
print("Correlação entre nota e sentimento:\n", corr)

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x="score", y="sentiment", alpha=0.2)
plt.title("Correlação entre Score e Sentimento")
plt.xlabel("Nota atribuída")
plt.ylabel("Polaridade do Sentimento")
plt.show()

# Listar títulos únicos em cada base
titulos_review_v2 = set(ratings["Title_clean"].unique())
titulos_books_v2 = set(books["Title_clean"].unique())

# Encontrar diferenças entre os conjuntos
# Títulos nos metadados que não têm review
sem_review = list(titulos_books - titulos_review)

# Títulos nas reviews que não estão nos metadados
sem_book = list(titulos_review - titulos_books)

# (casos não encontrado)Mostrar títulos sem review (com metadados disponíveis)
# print("\n🔍 Títulos em books_data2.csv que não aparecem nas reviews:")
# for titulo in sem_review[:10]:
#    original = books[books["Title_clean"] == titulo]["Title"].values[0]
#    print(f"📘 {original} (não encontrado nas avaliações)")
#
# Mostrar títulos sem metadado
# print("\n🔍 Títulos nas reviews que não existem em books_data2.csv:")
# for titulo in sem_book[:10]:
#    original = ratings[ratings["Title_clean"] == titulo]["Title"].values[0]
#    print(f"📕 {original} (sem metadado correspondente)")


# 🧠 Conclusão executiva:
# Com a automação do processo de análise via embeddings e LLMs, é possível economizar até R$166 mil/ano em custos operacionais, liberando os analistas para tarefas mais estratégicas como curadoria, relacionamento com autores, ou novas iniciativas de dados.


import matplotlib.pyplot as plt

# Dados
cenarios = ["Semanal", "Mensal", "Anual"]
custo_atual = [177216, 40896, 3408]
custo_modelo = [10400, 2400, 200]
economia = [a - b for a, b in zip(custo_atual, custo_modelo)]

# Cores
cores = ["#FF9999", "#FFCC99", "#99CCFF"]

# Gráfico de barras comparando custo atual x modelo
fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(cenarios, custo_atual, label="Custo Atual", color="#FF9999")
bar2 = ax.bar(cenarios, custo_modelo, label="Custo com Modelo", color="#99CCFF")

# Anotações
for i in range(len(cenarios)):
    ax.text(i, custo_atual[i] + 2000, f"R${custo_atual[i]:,}", ha="center", fontsize=9)
    ax.text(
        i, custo_modelo[i] + 1000, f"R${custo_modelo[i]:,}", ha="center", fontsize=9
    )

# Configurações
ax.set_title("Comparativo de Custos: Análise Manual vs Modelo", fontsize=14)
ax.set_ylabel("Custo Anual Estimado (R$)")
ax.legend()
plt.tight_layout()
plt.show()

# Gráfico de barras da economia
fig, ax2 = plt.subplots(figsize=(10, 6))
bar3 = ax2.bar(cenarios, economia, color="#90EE90")

# Anotações
for i in range(len(cenarios)):
    ax2.text(i, economia[i] + 2000, f"R${economia[i]:,}", ha="center", fontsize=10)

# Configurações
ax2.set_title("Economia Anual Estimada com Automação", fontsize=14)
ax2.set_ylabel("Valor Economizado (R$)")
plt.tight_layout()
plt.show()
