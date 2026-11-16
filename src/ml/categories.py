"""Categoria (assunto) de cada reclamação por regras de palavras-chave.

As categorias vieram do clustering exploratório (scripts/discover_topics.py)
e foram revisadas à mão. Regras são uma 1ª versão MEDIDA contra um gold set
revisado por humano; depois o BERT aprende com esses rótulos.

Como decide: pontuação = (nº de menções da categoria) x (peso da categoria);
vence a maior. Empate -> a que vem primeiro em RULES (específica antes de
genérica). Sem nenhuma menção -> "outros".
"""

import re
import unicodedata

import pandas as pd

# Ordem = prioridade no desempate. Padrões rodam sobre texto normalizado
# (minúsculo, sem acento), por isso não há acentos aqui.
RULES: dict[str, str] = {
    # "receber ligação" sozinho NÃO: "não consigo receber ligação" é problema
    # técnico. Exige sinal de insistência perto de "ligação" ([^.]{0,40} = até
    # 40 caracteres na mesma frase).
    "telemarketing": (
        r"telemarketing|nao me perturbe|"
        r"ligac\w*[^.]{0,40}(insistent|automati|indesejad|abusiv|gravad|diariamente|"
        r"todos os dias|todo dia|por dia|ofertando|oferecendo)|"
        r"(insistent|muitas|varias|diversas|inumeras|absurd)\w*[^.]{0,30}ligac|"
        r"(ligam|liga|ligando) (todos os dias|diariamente|toda hora|varias vezes)"
    ),
    # "negativa" sozinho NÃO: "recebo a negativa da migração" é recusa, não Serasa.
    "negativacao": (
        r"serasa|\bspc\b|\bscpc\b|boa vista|negativad[oa]s?|negativacao|negativaram|"
        r"nome (sujo|limpo)|divida\w* (desconhecida|inexistente)|desconheco (a |essa |esta )?divida"
    ),
    "portabilidade": r"portabilidade|\bportar\b|\bportad[oa]\b",
    "recarga": r"\brecarga|\brecarreg|\bsaldo\b(?! devedor)|\bcreditos\b",
    # Citar o plano ATUAL ("tenho um pós-pago") não é pedir mudança: exige verbo.
    "mudanca_plano": (
        r"\bmigra(cao|r|ram|ei)\b|"
        r"(troca|mudanca|alteracao) (de|do) plano|"
        r"(trocar|mudar|alterar|reduzir|diminuir) (o |meu )?plano|"
        r"para (o |um )?(plano )?(pre|pos|controle)\b|"
        r"(volt\w+|virar) (a ser )?(o |para o )?pre\b|plano mais barato"
    ),
    "cancelamento": r"cancel\w*|multa rescisoria|rescisao",
    "qualidade_tecnica": (
        r"\bsinal\b|sem (a |minha )?(internet|rede)|caixa postal|"
        r"internet (lenta|caindo|cai|caiu|parou|instavel|oscilando)|"
        r"velocidade|instabilidade|oscila\w*|\btecnic[oa]s?\b|visita|reparo|"
        r"sem servico|nao funciona|sem conexao|conexao|canais|queda"
    ),
    "reajuste_oferta": (
        r"reajuste|aumento|aumentou|aumentaram|desconto|fidelidade|"
        r"promoc\w*|oferta|acordad[oa]|combinad[oa]|\bacordo\b(?! com)|"
        r"negociad[oa]|renegocia\w*|prometid[oa]|promete\w*|por 12 meses"
    ),
    "cobranca": (
        r"cobran\w*|cobrad[oa]s?|cobrando|fatura\w*|\bconta\b|boleto|pagamento|"
        r"paguei|\bpago\b|debito|estorno|reembolso|comprovante|\bpix\b|descontad[oa]|"
        r"nao contratei|sem (meu )?consentimento|sem (minha )?autorizacao"
    ),
    # FALLBACK: reclamar do atendimento é um ESTILO, não um assunto (quase todo
    # relato de fatura/cancelamento diz "péssimo atendimento"). Só vence quando
    # nenhuma categoria de assunto casou. Precisão foi 0/9 quando competia.
    "atendimento": (
        r"(pessimo|mau|ruim|falta de) atendimento|minutos (de espera|esperando|aguardando)|"
        r"tempo de espera|derrub\w+ a ligacao|desligaram|nao consigo (atendimento|falar|contato)|"
        r"aguardando atendimento|hora\w* (de espera|esperando|aguardando)|\bura\b|"
        r"descaso|falta de respeito|ninguem resolve|nao resolvem"
    ),
}
FALLBACK = "atendimento"
CATEGORIES: list[str] = [*RULES, "outros"]

# Mesma ideia do IDF: sinal raro vale mais que sinal comum. Sem isso, "fatura"
# (cita-se em 68% dos relatos) roubava telemarketing ("ligações de cobrança")
# e negativação ("cobrando dívida que desconheço"). Peso = log(N / df), com df =
# nº de relatos de TREINO em que a regra casa (calculado em 2026-11-16, N=12.431).
# Recalcule sempre que mudar um padrão. FALLBACK não tem peso (não compete).
WEIGHTS: dict[str, float] = {
    "telemarketing": 3.1,
    "negativacao": 2.8,
    "portabilidade": 2.2,
    "recarga": 2.7,
    "mudanca_plano": 2.4,
    "cancelamento": 1.1,
    "qualidade_tecnica": 1.8,
    "reajuste_oferta": 1.1,
    "cobranca": 0.4,
}

_COMPILED = {cat: re.compile(pattern) for cat, pattern in RULES.items()}


def normalize(text: str) -> str:
    """Minúsculas e sem acento: 'Cobrança' e 'cobranca' viram o mesmo texto."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def categorize(text: str) -> str:
    """Categoria de maior pontuação; 'outros' se nenhuma regra casar."""
    norm = normalize(text)
    scores = {cat: len(_COMPILED[cat].findall(norm)) * w for cat, w in WEIGHTS.items()}
    best = max(scores, key=scores.get)  # max() devolve o 1º em empate = prioridade
    if scores[best] > 0:
        return best
    return FALLBACK if _COMPILED[FALLBACK].search(norm) else "outros"


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia com a coluna 'categoria'."""
    out = df.copy()
    out["categoria"] = out["relato"].map(categorize)
    return out
