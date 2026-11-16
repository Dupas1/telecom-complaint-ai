import pandas as pd

from src.ml.categories import CATEGORIES, add_category, categorize, normalize


def test_normalize_removes_accents_and_case():
    assert normalize("Cobrança INDEVIDA, Não") == "cobranca indevida, nao"


def test_each_category_has_a_clear_example():
    examples = {
        "telemarketing": "Recebo ligações de telemarketing todo dia, mesmo no Não Me Perturbe",
        "negativacao": "Meu nome foi negativado no Serasa por dívida que desconheço",
        "portabilidade": "Solicitei a portabilidade do meu número e não concluíram",
        "recarga": "Fiz uma recarga de 20 reais e o saldo não entrou",
        "mudanca_plano": "Pedi a migração do pré-pago para o plano controle",
        "cancelamento": "Solicitei o cancelamento e não conseguem cancelar",
        "qualidade_tecnica": "Estou sem internet, sem sinal, e o técnico não veio",
        "reajuste_oferta": "Tive um aumento no valor e perdi o desconto da fidelidade",
        "cobranca": "A fatura veio com cobrança indevida, já paguei o boleto",
        "atendimento": "Péssimo atendimento, fiquei 40 minutos de espera e desligaram",
    }
    for expected, text in examples.items():
        assert categorize(text) == expected, text


def test_no_keyword_is_outros():
    assert categorize("Comprei um aparelho e veio com defeito na tela") == "outros"


def test_rare_signal_beats_repeated_generic_word():
    text = "A fatura veio errada. Na fatura seguinte, a cobrança continuou. Pedi portabilidade."
    assert categorize(text) == "portabilidade"


def test_more_mentions_win_between_similar_weights():
    text = "Quero cancelar. Pedi o cancelamento duas vezes, cancelaram só um. A fatura veio."
    assert categorize(text) == "cancelamento"


def test_generic_category_still_wins_alone():
    assert categorize("A fatura veio com valor errado") == "cobranca"


def test_tie_goes_to_the_more_specific_category():
    assert categorize("Pedi portabilidade e a fatura veio errada") == "portabilidade"


def test_negativa_of_a_request_is_not_negativacao():
    text = "Recebo a negativa da migração para o pós-pago"
    assert categorize(text) == "mudanca_plano"


def test_atendimento_is_only_a_fallback():
    # Reclamar do atendimento é estilo: o assunto concreto vence.
    assert categorize("Péssimo atendimento, descaso total. A fatura veio errada") == "cobranca"


def test_receiving_calls_alone_is_not_telemarketing():
    text = "Não consigo receber ligação, cai tudo na caixa postal"
    assert categorize(text) == "qualidade_tecnica"


def test_mentioning_current_plan_is_not_plan_change():
    assert categorize("Tenho um plano pós-pago e a fatura veio errada") == "cobranca"


def test_add_category_keeps_input_intact():
    df = pd.DataFrame({"relato": ["sem sinal em casa", "oi"]})
    out = add_category(df)
    assert out["categoria"].tolist() == ["qualidade_tecnica", "outros"]
    assert "categoria" not in df.columns
    assert set(out["categoria"]) <= set(CATEGORIES)
