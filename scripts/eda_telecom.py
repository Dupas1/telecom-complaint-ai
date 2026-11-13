"""EDA do recorte de telecom (Fase 1, passo 4).

Uso:  .venv/Scripts/python.exe scripts/eda_telecom.py
"""

import sys

import pandas as pd

from src.core.config import get_settings
from src.ml.data import filter_telecom, load_raw

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 120)


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


df = filter_telecom(load_raw(get_settings().paths.raw_data))
print(f"Reclamações de telecom: {len(df):,}")

section("Por operadora")
print(df["operadora"].value_counts().to_string())

section("Status (dropna=False mostra também os vazios)")
print(df["status"].value_counts(dropna=False).to_string())

section("Ausentes por coluna: None/NaN  vs  texto vazio ''")
text_cols = df.columns.drop(["id", "operadora"])
missing = pd.DataFrame(
    {
        "nulos": df[text_cols].isna().sum(),
        "vazios": (df[text_cols].fillna("x").apply(lambda s: s.str.strip()) == "").sum(),
    }
)
missing["% total"] = ((missing["nulos"] + missing["vazios"]) / len(df) * 100).round(1)
print(missing.to_string())

section("Nota (convertida para número; texto inválido vira NaN)")
df["nota_num"] = pd.to_numeric(df["nota"], errors="coerce")
print(df["nota_num"].value_counts(dropna=False).sort_index().to_string())

section("Status x Nota (% da linha) — nota ajuda a separar os status?")
print(
    pd.crosstab(df["status"], df["nota_num"].fillna(-1), normalize="index")
    .mul(100)
    .round(1)
    .rename(columns={-1: "sem nota"})
    .to_string()
)

section("Status por operadora (% da linha)")
print(pd.crosstab(df["operadora"], df["status"], normalize="index").mul(100).round(1).to_string())

section("Tamanho do relato")
chars = df["relato"].fillna("").str.len()
words = df["relato"].fillna("").str.split().str.len()
print(
    pd.DataFrame({"caracteres": chars, "palavras": words})
    .describe(percentiles=[0.5, 0.9, 0.99])
    .round(0)
    .to_string()
)
print(f"Relatos com mais de 400 palavras (~512 tokens do BERT): {(words > 400).mean():.1%}")

section("Datas")
df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
print(f"Datas inválidas: {df['data_dt'].isna().sum()}")
print(f"De {df['data_dt'].min().date()} até {df['data_dt'].max().date()}")
print(df["data_dt"].dt.to_period("M").value_counts().sort_index().to_string())
