import pandas as pd

from src.ml.data import add_target, drop_duplicate_relatos, filter_telecom


def _df(empresas: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"empresa": empresas, "relato": ["x"] * len(empresas)})


def test_keeps_big_four_and_maps_operadora():
    out = filter_telecom(_df(["Tim", "Claro Fixo - Embratel", "Oi Celular", "Vivo - Telefônica"]))
    assert out["operadora"].tolist() == ["Tim", "Claro", "Oi", "Vivo"]


def test_drops_out_of_scope_and_false_positives():
    out = filter_telecom(
        _df(["SKY", "Sky Airline", "Oi Paggo Administradora de Crédito", "Brisanet Telecom"])
    )
    assert out.empty


def test_does_not_mutate_input():
    df = _df(["Tim", "Nubank"])
    filter_telecom(df)
    assert "operadora" not in df.columns and len(df) == 2


def test_add_target_union_rule_and_invalid_nota():
    df = pd.DataFrame(
        {
            "status": ["Não Resolvido", "Resolvido", "Resolvido", "Resolvido", "Resolvido"],
            "nota": ["5", "1", "3", "10", ""],
        }
    )
    out = add_target(df)
    # NR com nota 5 -> risco; resolvido nota 1 -> risco; nota 3 -> ok;
    # nota 10 e vazia são inválidas (NaN) e não marcam risco sozinhas.
    assert out["risco"].tolist() == [1, 1, 0, 0, 0]
    assert out["nota_num"].isna().tolist() == [False, False, False, True, True]


def test_drop_duplicates_keeps_most_recent():
    df = pd.DataFrame(
        {
            "relato": ["Sem  SINAL", "sem sinal", "outro"],
            "data": ["2024-05-01", "2023-01-01", "2023-06-01"],
            "status": ["Resolvido", "Não Resolvido", "Resolvido"],
        }
    )
    out = drop_duplicate_relatos(df)
    assert len(out) == 2
    assert out.loc[out["relato"] == "Sem  SINAL", "data"].item() == "2024-05-01"


def test_temporal_split_uses_cutoffs_inclusively():
    from datetime import date

    from src.ml.data import temporal_split

    df = pd.DataFrame({"data": ["2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01"]})
    out = temporal_split(df, val_start=date(2024, 1, 1), test_start=date(2024, 7, 1))
    assert out["split"].tolist() == ["train", "val", "val", "test"]
