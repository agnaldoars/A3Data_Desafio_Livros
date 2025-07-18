# EDA_NLP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import ast

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")

# ==============================#


# =============================
# 1. Leitura e junção dos dados
# =============================
ratings = pd.read_csv("Books_rating.csv")
books = pd.read_csv("books_data.csv")
# df = pd.read_csv("avaliacoes_processadas.csv")

# Espiar os dados as colunas do DataFrame
# Campos ou features (características) ou atributos.
print("Tamanho das bases:", ratings.shape, books.shape)
# Tamanho das bases: (3000000, 9) (212404, 10)

print("Cabeçalho dos Books:", books["Title"].head(10))
print("Cabeçalho dos Ratings:", ratings["Title"].head(10))

# A base de livros (books) está bem normalizada,
# sem repetições de título.
# Isso afeta os joins com a base de
# avaliações (ratings):
# ⚠️ Se títulos repetidos existissem,
# teriamos que desambiguar por autor, edição, etc.
# Ajuda na hora de fazer buscas, matching e FAISS,
# pois garante consistência dos títulos.
books["Title"].describe()
# count     212403
# unique    212403
# top       Its Only Art If Its Well Hung!
# freq      1

# A base ratings contém avaliações de usuários,
# com repetição de títulos, um mesmo livro pode
# receber muitas avaliações.
# Há exatamente 212.403 títulos distintos, o que coincide
# com a base books. Mas:
# Isso não garante que todos os títulos estejam
# em ambas as bases.
# É necessário cruzamento (merge) para identificar títulos
# que só aparecem em uma base.
# freq = 22023 para The Hobbit	Esse livro é extremamente
# popular, útil para análise de sentimento
# Há uma pequena quantidade de linhas (208) sem título,

# 🧠 A base de ratings é muito rica e densa, com milhões de
# avaliações de usuários.
# Embora tenhamos 212403 mil livros únicos avaliados, há uma
# alta concentração de reviews em alguns títulos
# populares (como "The Hobbit").
# Há uma pequena quantidade de linhas (208) sem título,
# que podem ser filtradas e analisadas no futuro me de 1%

ratings["Title"].describe()
# count        2999792
# unique        212403
# top       The Hobbit
# freq           22023
# linhas (208) sem título

# ratings.shape
# ratings["Title"].isna() | (ratings["Title"].str.strip() == '')
ratings["Title"].isna().sum()
books["Title"].isna().sum()
ratings_Nulos = ratings[ratings["Title"].isna()].copy()
ratings_Nulos.to_csv("ratings_Nulos.csv", index=False)
books.loc[:10, "Title"]
ratings.loc[:10, "Title"]

# Um caso para teste de merge
ratings.loc[ratings["Title"] == "The Hobbit"]

# Normalização dos títulos para facilitar o merge
ratings["Title_clean"] = ratings["Title"].str.strip().str.lower()
books["Title_clean"] = books["Title"].str.strip().str.lower()

# Normalização dos authors e categories, lista para texto
books["authors"] = books["authors"].apply(
    lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) else x
)
books["categories"] = books["categories"].apply(
    lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) else x
)
books["categories"].head(3)


# Merge via título normalizado
df = pd.merge(
    ratings, books, on="Title_clean", how="left", suffixes=("_rating", "_book")
)
df.shape
# Via Left Join (3320453, 20) Avaliar diferença se significativa

df = pd.merge(
    ratings, books, on="Title_clean", how="inner", suffixes=("_rating", "_book")
)
df.shape
# Via Inner Join (3320453, 20)

# Espiar os dados do join, colunas limpas
df.loc[:3, ["authors", "categories", "Title_clean"]]
print("Tamanho das bases:", df.shape, ratings.shape, books.shape, end="\n")
# Tamanho das bases: df (3320453, 20) ratings (3000000, 10) books (212404, 11)

# word_tokenize("text to test")
df3 = df.copy()
# df = df3.copy()
# df = df.loc[:100000,].copy()

# =============================
# 2. Limpeza de texto (coluna 'text')
# =============================
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


# 2. Limpeza de texto - word_tokenize -- stop_words
df["review_clean"] = df["text"].fillna("").apply(clean_text)


# =============================
# 3. Análise de sentimento
# =============================
def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0


# Analise de Sentimentos da Coluna Text
df["sentiment"] = df["text"].fillna("").apply(get_sentiment)

# Sentimento Geral das Avaliações
# Foram analisadas 3,3 milhões de reviews com NLP.
# A média de sentimento é +0,21, indicando tendência positiva.
# 25% das avaliações têm sentimento quase neutro (< 0,10).
# Há presença de críticas fortemente negativas (mínimo = -1), mas são minoria.
# A dispersão dos dados mostra uma base rica para extração de insights.
# 👉 Isso permite priorizar avaliações negativas para correções e
# identificar fãs para ações de marketing.

df["sentiment"].describe()
# count    3.320453e+06
# mean     2.098236e-01
# std      1.947297e-01
# min     -1.000000e+00
# 25%      9.598966e-02
# 50%      1.927083e-01
# 75%      3.092105e-01
# max      1.000000e+00

# =============================
# 4. Visualizações
# =============================

# Distribuição de notas
# Este gráfico mostra a distribuição geral
# das notas que os leitores deram aos livros
# analisados.
# Cada barra representa a quantidade de avaliações
# dentro de uma faixa de nota (score),
# variando de 1 a 5.
# 📌 Por que isso importa:
# Ele nos ajuda a entender se os leitores, no geral,
# estão satisfeitos ou insatisfeitos com os livros.
# Por exemplo, um pico em notas 4 e 5 indica boa
# aceitação, enquanto muitas notas abaixo de 3
# alertam para possíveis problemas de conteúdo,
# posicionamento ou expectativas.
plt.figure(figsize=(6, 4))
sns.histplot(df["score"], bins=5, kde=False)
plt.title("Rating Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Top dez autores
# Este gráfico mostra como variam as avaliações (notas) dos
# livros dos 3 autores mais avaliados.
# Cada barra horizontal representa o intervalo de notas recebidas pelos
# livros de um autor.
# O traço central representa a mediana (nota mais comum).
# As bordas da caixa indicam os 25% e 75% dos valores (distribuição).
# Os pontos fora da caixa são notas extremas (muito altas ou muito baixas).
# 📌 Por que isso importa:
# Esse gráfico permite identificar autores com avaliações mais consistentes
# (caixas menores) ou mais polêmicos (caixas grandes ou com muitos outliers).
# Isso ajuda na curadoria de autores para campanhas, destaques ou reavaliação editorial.
df["authors"] = df["authors"].astype(str)
top_authors = df["authors"].value_counts().nlargest(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["authors"].isin(top_authors)], x="authors", y="score")
plt.title("Score Distribution by Top Authors")
plt.xticks(rotation=45)
plt.show()

# books["authors"].isna().sum()
# sem_autor = books["authors"].isna() | (books["authors"].str.strip() == "")
# print("Total de livros sem autor:", sem_autor.sum())

# Quantos livros estão sem autor identificado:
books["authors"] = books["authors"].fillna("")
sem_autor = books["authors"].str.strip() == ""
print("Total de livros sem autor:", sem_autor.sum())
# Total de livros sem autor: 31414

# Top 10 Sentimento por categoria
# Este gráfico mostra como os leitores se sentem, em média,
# sobre os livros de cada uma das 10 categorias mais populares.
# Aqui não estamos olhando para a nota, mas sim para o tom emocional
# da avaliação textual — se a linguagem usada é positiva, neutra ou negativa.
# Os valores vão de -1.0 (muito negativo) até +1.0 (muito positivo):
# A mediana mostra o sentimento mais comum.
# A largura da caixa mostra a diversidade de sentimentos.
# 📌 Por que isso importa:
# Isso nos permite detectar gêneros com maior carga emocional
# positiva ou negativa.
# Por exemplo, categorias com sentimento consistentemente positivo
# indicam alta satisfação — mesmo que a nota não seja a mais alta.
# Já categorias com grande variação de sentimento (ou tendendo ao negativo)
# podem sinalizar que os livros geram frustração ou são mal compreendidos.

# df["categories"] = df["categories"].astype(str)
# top_cats = df["categories"].value_counts().nlargest(10).index
# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df[df["categories"].isin(top_cats)], x="categories", y="sentiment")
# plt.title("Sentiment by Book Category")
# plt.xticks(rotation=45)
# plt.show()

import matplotlib.pyplot as plt

top_categorias = df["categories"].value_counts().nlargest(10).index
df_top_categorias = df[df["categories"].isin(top_categorias)]
contagem = df_top_categorias["categories"].value_counts()

contagem.plot(kind="barh", figsize=(8, 5))
plt.title("Top 10 Categorias com Mais Livros")
plt.xlabel("Quantidade de livros")
plt.gca().invert_yaxis()  # categorias mais populares no topo
plt.tight_layout()
plt.show()

books["categories"] = books["categories"].fillna("")
sem_categories = books["categories"].str.strip() == ""
print("Total de livros sem categories:", sem_categories.sum())
# Total de livros sem categories: 41199

# Top 10 Sentimento por editora
df["publisher"] = df["publisher"].astype(str)
top_publisher = df["publisher"].value_counts().nlargest(10).index
df_top_publisher = df[df["publisher"].isin(top_publisher)]
contagem = df_top_publisher["publisher"].value_counts()

books["publisher"] = books["publisher"].fillna("")
sem_publisher = books["publisher"].str.strip() == ""
print("Total de livros sem publisher:", sem_publisher.sum())
# Total de livros sem publisher: 75886

contagem.plot(kind="barh", figsize=(8, 5))
plt.title("Top 10 publisher com Mais Livros")
plt.xlabel("Quantidade de livros")
plt.gca().invert_yaxis()  # publisher mais populares no topo
plt.tight_layout()
plt.show()

# =============================
# 5. Wordclouds
# =============================

# Extrair da base os 3 livros mais contraditórios em dois cenários:
# 1. Nota alta (score ≥ 4.5) + Sentimento negativo (sentiment ≤ 0)
# 2. Nota baixa (score ≤ 2.5) + Sentimento positivo (sentiment ≥ 0.3)
# Esses casos são valiosos para:
# Identificar livros mal avaliados por engano (ex: review positiva, mas nota baixa)
# Detectar leitores que gostaram da experiência mas deram nota baixa por um
# fator específico
# Entender ambiguidade entre o texto e a nota
# Top 3 casos com nota alta, mas sentimento negativo
caso1 = (
    df[(df["score"] >= 4.5) & (df["sentiment"] <= 0)]
    .sort_values(by="sentiment")
    .head(3)
)
# 🟦 Notas Altas com Sentimentos Negativos
# 📘 Título: in dark places
# 👤 Usuário: Creepee
# ⭐ Score: 5.0 | 🧠 Sentimento: -1.0
# 📝 Resumo: Great book!
# 💬 Texto:
# Another well written book by Michael Prescott! Makes you keep guessing
# and just when you think you know who the bad guy is...nope you don't!!...


# Top 3 casos com nota baixa, mas sentimento positivo
caso2 = (
    df[(df["score"] <= 2.5) & (df["sentiment"] >= 0.3)]
    .sort_values(by="sentiment", ascending=False)
    .head(3)
)


# 🧠 Interpretação de Negócio
# 🔍 Alerta de inconsistência: pode indicar erro de nota, ou avaliação emocionalmente complexa
# 📈 Útil para entrevistas: leitores que expressam sentimentos positivos apesar de nota baixa trazem insights ricos
# 🤖 Relevante para fine-tuning de LLM: melhora detecção de ironia, ambiguidade e contexto
def mostrar_casos(df_casos, titulo):
    print(f"\n🟦 {titulo}")
    for i, row in df_casos.iterrows():
        print(f"\n📘 Título: {row['Title_clean']}")
        print(f"👤 Usuário: {row['profileName']}")
        print(f"⭐ Score: {row['score']} | 🧠 Sentimento: {round(row['sentiment'], 3)}")
        print(f"📝 Resumo: {row['summary']}")
        print(f"💬 Texto:\n{row['text'][:500]}...\n{'-'*80}")


mostrar_casos(caso1, "Notas Altas com Sentimentos Negativos")
mostrar_casos(caso2, "Notas Baixas com Sentimentos Positivos")


def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Positivas
positive_text = " ".join(df[df["sentiment"] > 0.3]["review_clean"].tolist())
plot_wordcloud(positive_text, "Frequent Words in Positive Reviews")

# Negativas
negative_text = " ".join(df[df["sentiment"] < -0.1]["review_clean"].tolist())
plot_wordcloud(negative_text, "Frequent Words in Negative Reviews")

# =============================
# 6. Exportar base processada
# =============================

df.to_csv("avaliacoes_processadas2.csv", index=False)
