"""Sorteia relatos da VALIDAÇÃO para um gold set de categorias (Fase 1).

Por que validação: as regras foram ajustadas olhando o treino; medir no
treino daria uma nota otimista. Estratificado pela categoria que a regra
prevê, para que categorias pequenas também sejam avaliadas.

Gold N > 1 exclui os relatos dos golds anteriores: as regras foram ajustadas
olhando os erros deles, então reusá-los daria de novo uma nota otimista.

O arquivo NÃO traz a categoria da regra: quem rotula não deve ser influenciado.

Uso:  .venv/Scripts/python.exe scripts/sample_gold.py [N] [POR_CATEGORIA]
      (padrão: N=1, 9 por categoria -> gold_categorias.xlsx;
       N=2 -> gold_categorias_2.xlsx, e assim por diante)
"""

import sys

import pandas as pd

from src.core.config import get_settings
from src.ml.categories import add_category

number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
per_category = int(sys.argv[2]) if len(sys.argv) > 2 else 9


def gold_path(n: int):
    suffix = "" if n == 1 else f"_{n}"
    return settings.paths.processed_dir / f"gold_categorias{suffix}.xlsx"


settings = get_settings()
path = gold_path(number)
if path.exists():  # o arquivo guarda revisão humana: nunca sobrescrever
    sys.exit(f"{path} já existe (tem rótulos revisados). Apague à mão se quiser re-sortear.")

df = pd.read_parquet(settings.paths.processed_dir / "telecom_ml.parquet")
val = add_category(df[df["split"] == "val"])
used = {i for n in range(1, number) for i in pd.read_excel(gold_path(n), dtype=str)["id"]}
val = val[~val["id"].astype(str).isin(used)]

sample = (
    val.groupby("categoria", group_keys=False)
    .sample(n=per_category, random_state=settings.seed + number)
    .sample(frac=1, random_state=settings.seed)  # embaralha: sem pistas pela ordem
)
out = sample[["id", "relato"]].assign(categoria_claude="", correto="", categoria_certa="")

# .xlsx e não .csv: o Excel em pt-BR espera ";" e abriria o CSV numa coluna só.
out.to_excel(path, index=False)
print(f"{len(out)} relatos salvos em {path} ({len(used)} de golds anteriores excluídos)")
