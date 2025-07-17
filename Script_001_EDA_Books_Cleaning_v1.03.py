# EDA_NLP_Livros.py

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
# ==============================#


# =============================
# 1. Leitura e junção dos dados
# =============================

ratings = pd.read_csv("Books_rating.csv")
books = pd.read_csv("books_data.csv")

# Espiar os dados
print("Tamanho das bases:", ratings.shape, books.shape)

print("Describe Ratings:", ratings.describe)
print("Describe Books:", books.describe)

print("Cabeçalho dos Books:", books.head(10))
print("Cabeçalho dos Ratings:", ratings.head(10))

books.loc[:10, "Title"]
ratings.loc[:10, "Title"]

# Um caso para teste de merge
ratings.loc[ratings["Title"] == "Its Only Art If Its Well Hung!"]

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

# Espiar os dados do join
df.loc[:10]
print("Tamanho das bases:", df.shape, ratings.shape, books.shape)


# =============================
# 2. Limpeza de texto (coluna 'text')
# =============================

stop_words = set(stopwords.words("english"))

nltk.download("punkt_tab")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


# word_tokenize("text to test")
df3 = df.copy()
# df = df3.copy()

# df3 = df.loc[:1000,].copy()
# ver colunas atuais e na base reduzida
df.columns
df["review_clean"] = df["text"].fillna("").apply(clean_text)

# =============================
# 3. Análise de sentimento
# =============================


def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0


df["sentiment"] = df["text"].fillna("").apply(get_sentiment)

df.loc[:10, ["sentiment"]]
# =============================
# 4. Visualizações
# =============================

# Distribuição de notas
# Este gráfico mostra a distribuição geral das notas que os leitores deram aos livros analisados.
# Cada barra representa a quantidade de avaliações dentro de uma faixa de nota (score), variando de 1 a 5.
# 📌 Por que isso importa:
# Ele nos ajuda a entender se os leitores, no geral, estão satisfeitos ou insatisfeitos com os livros.
# Por exemplo, um pico em notas 4 e 5 indica boa aceitação, enquanto muitas notas abaixo de 3 alertam para possíveis problemas de conteúdo, posicionamento ou expectativas.
plt.figure(figsize=(6, 4))
sns.histplot(df["score"], bins=5, kde=False)
plt.title("Rating Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Top dez autores
# Este gráfico mostra como variam as avaliações (notas) dos livros dos 10 autores mais avaliados.
# Cada barra horizontal representa o intervalo de notas recebidas pelos livros de um autor.
# O traço central representa a mediana (nota mais comum).
# As bordas da caixa indicam os 25% e 75% dos valores (distribuição).
# Os pontos fora da caixa são notas extremas (muito altas ou muito baixas).
# 📌 Por que isso importa:
# Esse gráfico permite identificar autores com avaliações mais consistentes (caixas menores) ou mais polêmicos (caixas grandes ou com muitos outliers).
# Isso ajuda na curadoria de autores para campanhas, destaques ou reavaliação editorial.
df["authors"] = df["authors"].astype(str)
top_authors = df["authors"].value_counts().nlargest(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["authors"].isin(top_authors)], x="authors", y="score")
plt.title("Score Distribution by Top Authors")
plt.xticks(rotation=45)
plt.show()


# plt.figure(figsize=(6, 4))
# ax = sns.histplot(df["score"], bins=5, kde=False)
#
## Adiciona os valores no topo de cada barra
# for p in ax.patches:
#    height = p.get_height()
#    ax.text(
#        p.get_x() + p.get_width() / 2,     # posição x (centro da barra)
#        height + 0.5,                      # posição y (acima da barra)
#        f"{int(height)}",                  # texto com valor inteiro
#        ha="center"                        # alinhamento horizontal
#    )
#
# plt.title("Rating Distribution")
# plt.xlabel("Score")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

# Top 10 Sentimento por categoria
# Este gráfico mostra como os leitores se sentem, em média, sobre os livros de cada uma das 10 categorias mais populares.
# Aqui não estamos olhando para a nota, mas sim para o tom emocional da avaliação textual — se a linguagem usada é positiva, neutra ou negativa.
# Os valores vão de -1.0 (muito negativo) até +1.0 (muito positivo):
# A mediana mostra o sentimento mais comum.
# A largura da caixa mostra a diversidade de sentimentos.
# 📌 Por que isso importa:
# Isso nos permite detectar gêneros com maior carga emocional positiva ou negativa.
# Por exemplo, categorias com sentimento consistentemente positivo indicam alta satisfação — mesmo que a nota não seja a mais alta.
# Já categorias com grande variação de sentimento (ou tendendo ao negativo) podem sinalizar que os livros geram frustração ou são mal compreendidos.

df["categories"] = df["categories"].astype(str)
top_cats = df["categories"].value_counts().nlargest(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df["categories"].isin(top_cats)], x="categories", y="sentiment")
plt.title("Sentiment by Book Category")
plt.xticks(rotation=45)
plt.show()

# =============================
# 5. Wordclouds
# =============================

# Extrair da base os 3 livros mais contraditórios em dois cenários:
# 1. Nota alta (score ≥ 4.5) + Sentimento negativo (sentiment ≤ 0)
# 2. Nota baixa (score ≤ 2.5) + Sentimento positivo (sentiment ≥ 0.3)
# Esses casos são valiosos para:
# Identificar livros mal avaliados por engano (ex: review positiva, mas nota baixa)
# Detectar leitores que gostaram da experiência mas deram nota baixa por um fator específico
# Entender ambiguidade entre o texto e a nota
# Top 3 casos com nota alta, mas sentimento negativo
caso1 = (
    df[(df["score"] >= 4.5) & (df["sentiment"] <= 0)]
    .sort_values(by="sentiment")
    .head(3)
)

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

df.to_csv("avaliacoes_processadas.csv", index=False)
