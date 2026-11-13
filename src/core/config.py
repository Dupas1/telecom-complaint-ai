"""Configuração central do projeto.

Uso em qualquer módulo:
    from src.core.config import get_settings
    settings = get_settings()
    settings.paths.raw_data

Ordem de precedência (a primeira que tiver o valor vence):
    1. argumentos passados direto para Settings(...)   (útil em testes)
    2. variáveis de ambiente                            (CI, Colab)
    3. arquivo .env                                     (segredos locais)
    4. configs/base.yaml                                (valores do projeto)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# src/core/config.py -> parents[0]=core, [1]=src, [2]=raiz do projeto.
# Assim os caminhos funcionam não importa de qual pasta o script é executado.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    raw_data: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    @field_validator("*", mode="after")
    @classmethod
    def _resolver_relativo_a_raiz(cls, valor: Path) -> Path:
        return valor if valor.is_absolute() else PROJECT_ROOT / valor


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=PROJECT_ROOT / "configs" / "base.yaml",
        yaml_file_encoding="utf-8",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # PATHS__RAW_DATA -> paths.raw_data
        extra="ignore",  # variáveis no .env que não são campos daqui não dão erro
    )

    seed: int
    paths: PathsConfig

    # SecretStr esconde o valor em print/log: aparece como '**********'.
    # Use settings.hf_token.get_secret_value() só onde o token é realmente necessário.
    hf_token: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Por padrão o pydantic-settings não lê YAML; aqui o adicionamos
        # como ÚLTIMA fonte (menor prioridade).
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )


@lru_cache
def get_settings() -> Settings:
    """Carrega a configuração uma única vez e reaproveita nas chamadas seguintes."""
    return Settings()
