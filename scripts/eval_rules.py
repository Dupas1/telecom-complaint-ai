"""Mede as regras de categoria contra o gold set revisado (Fase 1).

O gold foi sorteado ESTRATIFICADO pela categoria que a regra previu (9 por
categoria). Consequências:
- precisão por categoria é estimada corretamente (é o que sorteamos);
- acurácia simples dos 99 é distorcida (categoria rara pesa igual à comum);
  por isso reportamos a acurácia PONDERADA = soma(precisão_c x fatia_c na val).

Uso:  .venv/Scripts/python.exe scripts/eval_rules.py
"""

import sys
import textwrap

import pandas as pd

from src.core.config import get_settings
from src.ml.categories import add_category

sys.stdout.reconfigure(encoding="utf-8")

settings = get_settings()
gold = pd.read_excel(settings.paths.processed_dir / "gold_categorias.xlsx", dtype=str)
# Rótulo final: a correção humana se houver, senão o rótulo aprovado.
wrong = gold["correto"].str.strip().str.lower() == "n"
gold["gold"] = gold["categoria_claude"].where(~wrong, gold["categoria_certa"])
gold = add_category(gold).rename(columns={"categoria": "regra"})
gold["acertou"] = gold["regra"] == gold["gold"]

df = pd.read_parquet(settings.paths.processed_dir / "telecom_ml.parquet")
share = add_category(df[df["split"] == "val"])["categoria"].value_counts(normalize=True)

per_cat = gold.groupby("regra").agg(n=("acertou", "size"), precisao=("acertou", "mean"))
per_cat["fatia_val"] = share
per_cat = per_cat.sort_values("precisao")
weighted = (per_cat["precisao"] * per_cat["fatia_val"]).sum()

print(f"Acurácia simples (99, distorcida): {gold['acertou'].mean():.0%}")
print(f"Acurácia PONDERADA (estimativa real): {weighted:.0%}\n")
print(per_cat.round(2).to_string())

print("\nConfusões mais comuns (regra -> gold):")
errors = gold[~gold["acertou"]]
print(errors.groupby(["regra", "gold"]).size().sort_values(ascending=False).to_string())

print("\nErros (para entender o porquê):")
for _, r in errors.iterrows():
    print(f"  regra={r['regra']:<17} gold={r['gold']:<17} ", end="")
    print(textwrap.shorten(r["relato"], 110, placeholder=" [...]"))
