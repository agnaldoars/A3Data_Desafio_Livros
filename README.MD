# 📚 A3Data — Desafio Técnico: Análise de Avaliações de Livros com NLP e LLM

Este repositório contém a solução desenvolvida para o desafio técnico da A3Data, 
com foco em **análise de texto**, **sumarização**, e **Q&A semântico** aplicado à base de dados de avaliações de livros.

---

## 🎯 Objetivo

A proposta é automatizar a análise de avaliações de livros feita atualmente de forma manual por analistas. 
Utilizando técnicas de NLP e LLMs, construímos uma solução capaz de:

- Responder perguntas em linguagem natural sobre autores, gêneros e temas
- Explorar sentimentos e opiniões dos usuários
- Resumir comentários longos
- Gerar insights acionáveis com economia de tempo e custo

---

## ⚙️ Etapas da Solução

1. **EDA e Análise de Sentimento** (Script_001_EDA_Books_Cleaning_v1.03.py)
   - Limpeza textual (tokenização, stopwords, stemming)
   - Visualizações de score, autores, categorias
   - Correlação entre sentimento e nota

2. **Embeddings + FAISS** (Script_002__QA_Sumarizacao_v1.00.py)
   - Geração de embeddings com `SentenceTransformer`
   - Criação de índice FAISS para busca semântica rápida

3. **Sumarização com LLM** (Script_003__QA_Sumarizacao_Esatistica_v1.01.py)
   - Respostas com base em texto filtrado via FAISS
   - Sumarização com modelos da `transformers` (ex: BART)

4. **Estimativa de Impacto**
   - Comparação entre esforço manual e solução automatizada
   - Potencial de economia superior a R$160.000/ano

---

## 📁 Estrutura do Projeto

A3Data_Desafio_Livros/
├── data/ # Dados de entrada (livros e reviews) --(Veja Links dos arquivos no PDF, e renomeios conforme os scripts)
├── notebooks/
│ ├── Script_001_EDA_Books_Cleaning_v1.py # Limpeza, EDA, sentimento
│ ├── Script_002__QA_Sumarizacao_v1.01.py # FAISS + Q&A
│ └── Script_003__QA_Sumarizacao_Esatistica_v1.01.py # Estatísticas por autor, etc.
├── slides/
│ └── Apresentacao_A3Data_Livros.pptx # Apresentação com insights
├── requirements.txt
└── README.md




## 🚀 Como Rodar

1. **Clone o repositório**
   bash
   git clone https://github.com/seuusuario/A3Data_Desafio_Livros.git
   cd A3Data_Desafio_Livros
   
## 🚀 Instale os requisitos

bash
Copy
Edit
pip install -r requirements.txt  


## 🚀 Execute os scripts
Script_001_EDA_Books_Cleaning_v1.py: análise textual e sentimentos
Script_002__QA_Sumarizacao_v1.01.py: interface de perguntas e sumarização
Script_003__QA_Sumarizacao_Esatistica_v1.01.py: estatísticas avançadas

## 💡 Tecnologias Utilizadas
pandas, numpy
matplotlib, seaborn, wordcloud
nltk, TextBlob
sentence-transformers, faiss
transformers (HuggingFace)
torch

##💰 Estimativa de Impacto
Frequência de uso	Custo manual anual	Custo com modelo	Economia estimada
Semanal	R$177.216,00	R$10.400,00	R$166.816,00 ✅
Mensal	R$40.896,00	R$2.400,00	R$38.496,00
Anual	R$3.408,00	R$200,00	R$3.208,00

##🔮 Próximos Passos (Roadmap)
Interface web para analistas consultarem o modelo
Integração com banco de dados para feedback contínuo
Fine-tuning com dados internos da editora
Dashboard com autorranking e análise temporal
