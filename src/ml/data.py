"""Carregamento e recorte do dataset bruto de reclamações.

Decisão de escopo (Fase 1): só as 4 grandes operadoras (Tim, Claro, Vivo, Oi).
SKY, provedores regionais e falsos positivos da busca por regex
("Sky Airline", "Oi Paggo Administradora de Crédito") ficam de fora.
"""

from datetime import date
from pathlib import Path

import pandas as pd

# Nome exato no dataset -> grupo padronizado ("operadora").
# Lista fechada e revisada à mão: nome novo/desconhecido NÃO entra.
TELECOM_OPERATORS: dict[str, str] = {
    "Tim": "Tim",
    "Vivo - Telefônica": "Vivo",
    "Claro Celular": "Claro",
    "Claro Residencial": "Claro",
    "Claro Fixo - Embratel": "Claro",
    "NET (DESATIVADA) (ATUAL CLARO)": "Claro",
    "Oi Fixo": "Oi",
    "Oi Celular": "Oi",
}


def load_raw(path: Path) -> pd.DataFrame:
    """Lê o JSON bruto sem deixar o pandas adivinhar tipos (tudo vira texto)."""
    return pd.read_json(path, dtype=False)


def filter_telecom(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém só as operadoras do escopo e adiciona a coluna 'operadora'."""
    out = df[df["empresa"].isin(TELECOM_OPERATORS)].copy()
    out["operadora"] = out["empresa"].map(TELECOM_OPERATORS)
    return out.reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria 'nota_num' (1..5, inválida -> NaN) e o alvo binário 'risco'.

    risco = 1 se status == "Não Resolvido" OU nota <= 2 (decisão da Fase 1:
    pega também o "resolveu, mas o cliente saiu furioso" -> churn/ação judicial).

    ATENÇÃO (vazamento): status, nota, resposta e comentario só existem DEPOIS
    da reclamação. Servem para criar o alvo, NUNCA como feature do modelo.
    """
    out = df.copy()
    nota = pd.to_numeric(out["nota"], errors="coerce")
    out["nota_num"] = nota.where(nota.between(1, 5))
    # NaN <= 2 é False: nota inválida não marca risco sozinha.
    out["risco"] = ((out["status"] == "Não Resolvido") | (out["nota_num"] <= 2)).astype(int)
    return out


def drop_duplicate_relatos(df: pd.DataFrame) -> pd.DataFrame:
    """Remove relatos repetidos (mesmo texto ignorando maiúsculas/espaços).

    Mantém a ocorrência mais RECENTE, que reflete o desfecho final. Sem isso, o
    mesmo texto pode cair no treino e no teste e inflar a métrica (vazamento).
    """
    norm = df["relato"].str.lower().str.split().str.join(" ")
    order = pd.to_datetime(df["data"]).argsort(kind="stable")
    keep = ~norm.iloc[order].duplicated(keep="last")
    return df.iloc[order][keep.to_numpy()].sort_index().reset_index(drop=True)


def temporal_split(df: pd.DataFrame, val_start: date, test_start: date) -> pd.DataFrame:
    """Adiciona a coluna 'split' (train/val/test) cortando pela DATA, não por sorteio.

    Simula produção: o modelo aprende com o passado e é avaliado no "futuro".
    """
    out = df.copy()
    dt = pd.to_datetime(out["data"])
    out["split"] = "train"
    out.loc[dt >= pd.Timestamp(val_start), "split"] = "val"
    out.loc[dt >= pd.Timestamp(test_start), "split"] = "test"
    return out
