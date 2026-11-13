"""Auditoria do rótulo (Fase 1, passo 5): a etiqueta faz sentido?

Uso:  .venv/Scripts/python.exe scripts/audit_labels.py
"""

import sys
import textwrap

import pandas as pd

from src.core.config import get_settings
from src.ml.data import filter_telecom, load_raw

sys.stdout.reconfigure(encoding="utf-8")

settings = get_settings()
df = filter_telecom(load_raw(settings.paths.raw_data))
# Notas fora de 1..5 (ex.: 10) viram NaN: são erro de coleta.
df["nota_num"] = pd.to_numeric(df["nota"], errors="coerce").where(lambda s: s.between(1, 5))

nao_resolvido = df["status"] == "Não Resolvido"
nota_baixa = df["nota_num"] <= 2

print("========== Tamanho de cada alvo candidato ==========")
for nome, y in {
    "Não Resolvido": nao_resolvido,
    "nota <= 2": nota_baixa,
    "Não Resolvido OU nota <= 2": nao_resolvido | nota_baixa,
}.items():
    print(f"{nome:<28} {y.sum():>5} positivos  ({y.mean():.1%})")
print(f"Resolvido mas nota <= 2:     {(~nao_resolvido & nota_baixa).sum():>5}")
print(f"Não Resolvido mas nota >= 4: {(nao_resolvido & (df['nota_num'] >= 4)).sum():>5}")


def show(rows: pd.DataFrame, title: str) -> None:
    print(f"\n{'#' * 15} {title} {'#' * 15}")
    for i, r in enumerate(rows.itertuples(), 1):
        print(f"\n--- [{i}] {r.operadora} | {r.data} | status={r.status} | nota={r.nota}")
        print("RELATO:    ", textwrap.shorten(r.relato, 400, placeholder=" [...]"))
        print("RESPOSTA:  ", textwrap.shorten(r.resposta, 250, placeholder=" [...]"))
        print("COMENTÁRIO:", textwrap.shorten(r.comentario, 300, placeholder=" [...]"))


# random_state = seed do config: todo mundo que rodar vê os MESMOS exemplos.
show(df[nao_resolvido].sample(8, random_state=settings.seed), "NÃO RESOLVIDO (amostra)")
show(
    df[~nao_resolvido & nota_baixa].sample(5, random_state=settings.seed),
    "RESOLVIDO, MAS NOTA 1-2 (amostra)",
)

print("\n========== Duplicados (relato normalizado) ==========")
norm = df["relato"].str.lower().str.split().str.join(" ")
dup = norm.duplicated(keep=False)
print(f"Linhas cujo relato aparece mais de uma vez: {dup.sum()} ({dup.mean():.1%})")
grupos = df[dup].groupby(norm[dup])["status"].nunique()
print(f"Grupos de relatos repetidos: {len(grupos)}")
print(f"  ... com status DIFERENTES entre si: {(grupos > 1).sum()}")
