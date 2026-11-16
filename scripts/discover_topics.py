"""Descobrir os ASSUNTOS das reclamações com clustering (Fase 1, categorias).

Ideia: TF-IDF transforma cada relato em números; KMeans junta os parecidos
em K grupos SEM saber nomes. Depois NÓS olhamos os grupos e damos nomes.

Só usamos o split de TREINO: validação e teste não podem influenciar
nenhuma decisão (nem a escolha das categorias).

Uso:  .venv/Scripts/python.exe scripts/discover_topics.py [K]
"""

import sys
import textwrap

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.config import get_settings

sys.stdout.reconfigure(encoding="utf-8")

K = int(sys.argv[1]) if len(sys.argv) > 1 else 12

# Palavras que não dizem nada sobre o ASSUNTO. Os nomes das operadoras saem
# de propósito: senão o KMeans agrupa por empresa ("grupo da Tim"), não por tema.
STOPWORDS = """
a o as os um uma uns umas de do da dos das no na nos nas em por para pra com sem
e ou mas que se nao sim ja ate ao aos a e foi ser sao era estou esta estao tem
ter tenho tinha fui fiz faz feito meu minha meus minhas me mim eu ele ela eles
voce voces seu sua seus suas isso isto esse essa este esta aquele mais menos
muito pouco como quando onde porque pois entao tambem ainda so apenas dia dias
mes ano vez vezes hoje ontem agora la aqui ha the vou vai quero gostaria
vivo tim claro oi net embratel telefonica empresa operadora
contato numero sendo porem consigo mesmo fazer atendente linha servico problema
""".split()
# Linha acima (rodada 2): palavras que apareceram no topo de QUASE TODOS os
# grupos na 1ª rodada. Se estão em todo lugar, não ajudam a separar assuntos.

settings = get_settings()
df = pd.read_parquet(settings.paths.processed_dir / "telecom_ml.parquet")
train = df[df["split"] == "train"].reset_index(drop=True)
print(f"Relatos de treino: {len(train):,}  |  K = {K} grupos")

vectorizer = TfidfVectorizer(
    strip_accents="unicode",  # "cobrança" e "cobranca" viram a mesma palavra
    lowercase=True,
    stop_words=STOPWORDS,
    ngram_range=(1, 2),  # pega também pares: "cobranca indevida", "sem sinal"
    min_df=5,  # ignora palavras em menos de 5 relatos (erros de digitação, nomes)
    max_df=0.4,  # ignora palavras em mais de 40% dos relatos (genéricas demais)
    # Só palavras com 3+ letras, sem números. O "(?!x+\b)" descarta tokens
    # feitos só de "x" (xxx, xxxxx): é a marca da anonimização (protocolo
    # XXX.XXX, telefone XXXXX-XXXX), não diz nada sobre o assunto.
    token_pattern=r"(?u)\b(?!x+\b)[a-zA-Z]{3,}\b",
)
X = vectorizer.fit_transform(train["relato"])
print(f"Vocabulário: {X.shape[1]:,} termos")

km = KMeans(n_clusters=K, n_init=10, random_state=settings.seed)
train["grupo"] = km.fit_predict(X)

terms = np.array(vectorizer.get_feature_names_out())
rng = np.random.default_rng(settings.seed)
for g in train["grupo"].value_counts().index:  # do maior para o menor
    rows = train[train["grupo"] == g]
    top = terms[km.cluster_centers_[g].argsort()[::-1][:12]]
    print(f"\n{'=' * 90}")
    print(f"GRUPO {g:>2}  |  {len(rows):>5} relatos ({len(rows) / len(train):.0%})", end="")
    print(f"  |  risco {rows['risco'].mean():.0%}")
    print("Termos:", ", ".join(top))
    for idx in rng.choice(rows.index, size=min(2, len(rows)), replace=False):
        print("  •", textwrap.shorten(train.at[idx, "relato"], 180, placeholder=" [...]"))
