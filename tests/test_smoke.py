"""Teste de fumaça: confirma que o ambiente e o pacote estão de pé."""

import sys


def test_python_version():
    assert sys.version_info[:2] == (3, 11)


def test_pacote_src_importa():
    import src  # noqa: F401
