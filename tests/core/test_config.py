import pytest
from pydantic import ValidationError

from src.core.config import PROJECT_ROOT, Settings


@pytest.fixture
def settings_isolado(monkeypatch):
    """Cria Settings ignorando o .env da máquina: mesmo resultado em qualquer PC."""
    for var in ("SEED", "PATHS__RAW_DATA", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    def _criar():
        return Settings(_env_file=None)

    return _criar


def test_carrega_valores_do_yaml(settings_isolado):
    s = settings_isolado()
    assert s.seed == 42
    assert s.hf_token is None


def test_caminho_relativo_vira_absoluto_na_raiz(settings_isolado):
    s = settings_isolado()
    assert s.paths.raw_data == PROJECT_ROOT / "data" / "raw" / "dados2025.json"
    assert s.paths.raw_data.is_absolute()


def test_variavel_de_ambiente_sobrescreve_yaml(settings_isolado, monkeypatch):
    monkeypatch.setenv("SEED", "7")
    monkeypatch.setenv("PATHS__RAW_DATA", "outro/arquivo.json")
    s = settings_isolado()
    assert s.seed == 7
    assert s.paths.raw_data == PROJECT_ROOT / "outro" / "arquivo.json"


def test_tipo_invalido_falha_cedo(settings_isolado, monkeypatch):
    monkeypatch.setenv("SEED", "quarenta-e-dois")
    with pytest.raises(ValidationError):
        settings_isolado()


def test_token_nao_vaza_em_print(settings_isolado, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_segredo123")
    s = settings_isolado()
    assert "hf_segredo123" not in repr(s)
    assert s.hf_token.get_secret_value() == "hf_segredo123"
