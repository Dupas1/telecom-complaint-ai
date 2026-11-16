"""Sorteia relatos da VALIDAÇÃO para o gold set de categorias (Fase 1).

Por que validação: as regras foram ajustadas olhando o treino; medir no
treino daria uma nota otimista. 9 por categoria prevista (estratificado) para
que categorias pequenas também sejam avaliadas.

O CSV NÃO traz a categoria da regra: quem rotula não deve ser influenciado.

Uso:  .venv/Scripts/python.exe scripts/sample_gold.py
"""

import sys

import pandas as pd

from src.core.config import get_settings
from src.ml.categories import add_category

PER_CATEGORY = 9

settings = get_settings()
df = pd.read_parquet(settings.paths.processed_dir / "telecom_ml.parquet")
val = add_category(df[df["split"] == "val"])

sample = (
    val.groupby("categoria", group_keys=False)
    .sample(n=PER_CATEGORY, random_state=settings.seed)
    .sample(frac=1, random_state=settings.seed)  # embaralha: sem pistas pela ordem
)
out = sample[["id", "relato"]].assign(categoria_claude="", correto="", categoria_certa="")

# .xlsx e não .csv: o Excel em pt-BR espera ";" e abriria o CSV numa coluna só.
path = settings.paths.processed_dir / "gold_categorias.xlsx"
if path.exists():  # o arquivo guarda revisão humana: nunca sobrescrever
    sys.exit(f"{path} já existe (tem rótulos revisados). Apague à mão se quiser re-sortear.")
out.to_excel(path, index=False)
print(f"{len(out)} relatos salvos em {path}")
