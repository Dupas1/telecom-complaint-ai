"""Gera data/processed/telecom_ml.parquet (Fase 1, passo 8).

Pipeline: bruto -> só telecom -> alvo -> sem duplicados -> split temporal.
Uso:  .venv/Scripts/python.exe scripts/build_dataset.py
"""

import sys

import pandas as pd

from src.core.config import get_settings
from src.ml.data import (
    add_target,
    drop_duplicate_relatos,
    filter_telecom,
    load_raw,
    temporal_split,
)

sys.stdout.reconfigure(encoding="utf-8")

# Só o que existe NO MOMENTO da reclamação + o alvo + o split.
# status/nota/resposta/comentario ficam de fora de propósito (vazamento).
COLUMNS = ["id", "data", "operadora", "empresa", "local", "relato", "risco", "split"]

settings = get_settings()
df = load_raw(settings.paths.raw_data)
print(f"Bruto:              {len(df):>6}")
df = filter_telecom(df)
print(f"Só telecom:         {len(df):>6}")
df = drop_duplicate_relatos(add_target(df))
print(f"Sem duplicados:     {len(df):>6}")
df = temporal_split(df, settings.split.val_start, settings.split.test_start)

out = df[COLUMNS].assign(data=pd.to_datetime(df["data"]))
out_path = settings.paths.processed_dir / "telecom_ml.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(out_path, index=False)

summary = out.groupby("split").agg(
    linhas=("risco", "size"),
    positivos=("risco", "sum"),
    pct_risco=("risco", "mean"),
    de=("data", "min"),
    ate=("data", "max"),
)
summary["pct_risco"] = (summary["pct_risco"] * 100).round(1)
summary = summary.reindex(["train", "val", "test"])
print(f"\n{summary.to_string()}")
print(f"\nSalvo em {out_path.relative_to(settings.paths.processed_dir.parents[1])}")
