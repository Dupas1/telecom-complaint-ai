"""Primeira olhada no dataset bruto (Fase 1, passo 1).

Objetivo: entender o formato ANTES de decidir qualquer coisa
(tamanho, colunas, tipos, quais empresas existem).

Uso:  .venv/Scripts/python.exe scripts/explore_raw.py
"""

import sys
import time

import pandas as pd

from src.core.config import get_settings

# No Windows o print sai em cp1252 e os acentos quebram; forçamos UTF-8.
sys.stdout.reconfigure(encoding="utf-8")

settings = get_settings()
path = settings.paths.raw_data

t0 = time.perf_counter()
# O arquivo é UM array JSON gigante ([ {...}, {...} ]), então read_json
# lê tudo de uma vez para a memória. dtype=False evita que o pandas
# "adivinhe" tipos (ex.: transformar "nota" em número sozinho).
df = pd.read_json(path, dtype=False)
print(f"Carregado em {time.perf_counter() - t0:.1f}s")

print(f"\nLinhas x colunas: {df.shape}")
print(f"Memória: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

print("\nTipos de cada coluna:")
print(df.dtypes)

print("\nTop 40 empresas por número de reclamações:")
print(df["empresa"].value_counts().head(40).to_string())

# Passo 2: procurar candidatas a telecom em TODOS os nomes, não só no top 40.
# \b = fronteira de palavra: "\boi\b" pega "Oi Fixo" mas não "Coisas".
# A lista é larga de propósito (pega falsos positivos); depois revisamos à mão.
TELECOM_PATTERN = (
    r"\b(?:vivo|tim|claro|oi|net|sky|nextel|algar|embratel|telef[oô]nica"
    r"|brisanet|desktop|unifique|vero|ligga|sercomtel|telecom|internet"
    r"|fibra|banda larga|celular|m[oó]vel)\b"
)
mask = df["empresa"].str.contains(TELECOM_PATTERN, case=False, regex=True, na=False)
print("\nCandidatas a telecom (todas as que casam com o padrão):")
print(df.loc[mask, "empresa"].value_counts().to_string())
