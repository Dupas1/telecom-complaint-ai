# Telecom Complaint AI — Tecnologias e Modelos

> Documento de estudo: o que é cada tecnologia, qual modelo usamos e **por quê**.
> Restrições do projeto: **100% gratuito**, **sem API de IA externa** (nada de OpenAI/Azure), foco em **aprendizado**.

---

## Visão geral: o caminho de uma reclamação

> *"Fui cobrado 3 meses por um serviço que cancelei e a Vivo não resolve."*

```
Reclamação
   │
   ▼
[1] TRIAGEM (ML)        → categoria: "Cobrança indevida" | risco de não-resolução: 78%
   │
   ▼
[2] RAG (busca)         → Art. X e Art. Y do Regulamento da Anatel (RGC)
   │
   ▼
[3] LLM (seu modelo)    → resposta empática, fundamentada nos artigos
   │
   ▼
[4] PRODUÇÃO            → API entrega + registra latência, uso de GPU e drift
```

---

## Camada 1 — Triagem (ML Clássico + Deep Learning)

### TF-IDF — transformar texto em números
Modelos não entendem palavras, só números. O TF-IDF dá um peso a cada palavra:
- **TF (Term Frequency):** quantas vezes a palavra aparece *nesta* reclamação.
- **IDF (Inverse Document Frequency):** quão **rara** a palavra é *no conjunto todo*.

"Cancelamento" aparece em muitos textos e pesa pouco. "Portabilidade" é rara e pesa muito.
**Limitação:** não entende contexto. Para ele, "não resolveram" e "resolveram" são quase iguais.

### LightGBM — o modelo de ML clássico
Algoritmo de **árvores de decisão em sequência** (*gradient boosting*). Cada árvore faz perguntas de sim ou não (*"tem a palavra 'cobrança'? a empresa é X?"*). Uma árvore sozinha é fraca, então o boosting treina **centenas de árvores, cada uma corrigindo os erros da anterior**.

**Por que o LightGBM?**
- Em **dados tabulares** (empresa, UF, tamanho do texto), modelos de árvore ainda costumam ganhar das redes neurais.
- **Muito rápido**: roda em CPU em milissegundos, sem GPU.
- Lida bem com **desbalanceamento** (poucos "não resolvidos" contra muitos "resolvidos").
- Alternativas: XGBoost e CatBoost, que são parecidos. O LightGBM costuma ser o mais rápido para treinar.

### SHAP — explicar cada previsão
Responde *"por que o modelo disse 78% de risco?"*. Por exemplo: "a palavra 'Procon' somou +15%, a empresa somou +8%, o texto curto tirou 3%".
Num atendimento real, ninguém confia num modelo que não explica a decisão.

### BERTimbau — o modelo de Deep Learning
**BERT** é um modelo **Transformer** que lê a frase inteira e entende o **contexto** ("banco de dados" ≠ "banco da praça").
**BERTimbau** (`neuralmind/bert-base-portuguese-cased`) é um BERT treinado com textos em **português do Brasil**.

**Por que usar os dois modelos?** Para aprender a decisão que existe no mundo real:

| | LightGBM + TF-IDF | BERTimbau |
|---|---|---|
| Entende contexto | Não | Sim |
| Velocidade | ~1 ms (CPU) | ~20–50 ms (GPU) |
| Precisa de GPU | Não | Sim, para treinar |
| Explicabilidade | Fácil (SHAP) | Mais difícil |

Se o BERT ganhar só 1% de F1, provavelmente não compensa. **Medir isso é a lição.**
Tamanho: ~110 milhões de parâmetros. Treina na GTX 1660 (6 GB) com `fp16` e lotes pequenos.

### PyTorch
A **biblioteca de Deep Learning** em que o BERT e o Qwen rodam. O Hugging Face `transformers` usa o PyTorch por baixo.

---

## Camada 2 — RAG (busca no regulamento)

### O que é RAG
**Retrieval-Augmented Generation** significa "geração com busca". Um LLM sozinho **inventa** coisas (*alucinação*). Com RAG, ele **consulta** os artigos verdadeiros antes de responder. É como uma prova com consulta, em vez de uma prova de memória.

### Docling — ler o PDF direito
PDFs são ruins de ler por programa: tabelas quebram e cabeçalhos se misturam no texto. O **Docling** (IBM) entende a **estrutura** do documento (capítulo, artigo, parágrafo). Isso permite dividir o texto **por artigo** (*chunking*), em vez de cortar no meio de uma frase.

### Embeddings — `BAAI/bge-m3`
Um **embedding** transforma texto num vetor que representa o **significado**. Textos parecidos ficam "perto". Assim, "quero cancelar" encontra o artigo sobre "rescisão contratual" sem nenhuma palavra em comum.
**Por que o bge-m3?** É multilíngue (bom em português), aceita textos longos e gera **vetor denso e esparso de uma vez**. Tem ~568M parâmetros e ocupa ~2 GB.

### Busca híbrida e o RRF
- **Busca densa:** acha o significado, mas pode errar termos exatos ("Art. 52").
- **Busca esparsa (BM25):** acha palavras exatas, mas não entende sinônimos.
- **RRF (Reciprocal Rank Fusion):** junta as duas listas olhando só a posição de cada item em cada lista. As duas buscas juntas acham mais coisas certas do que qualquer uma sozinha.

### Reranker — `BAAI/bge-reranker-v2-m3`
A busca devolve, por exemplo, 20 candidatos, de forma rápida mas aproximada. O **reranker** (*cross-encoder*) lê **a pergunta e cada trecho juntos** e dá uma nota precisa. Ficamos com os 3–5 melhores.
**Por que não usar só o reranker?** É lento demais para comparar com todos os trechos. Padrão da indústria: **busca rápida e ampla primeiro, filtro preciso depois**.

### Milvus — banco de vetores
Banco feito para responder rápido à pergunta "quais vetores estão mais perto deste?". Suporta busca híbrida nativamente, é open-source e roda no Docker.
Alternativas: FAISS (só uma biblioteca, sem banco), Chroma (mais simples) e Qdrant.

### LangChain / LangGraph — orquestração
**Conectam as peças:** pergunta → Milvus → reranker → prompt → LLM.
O **LangGraph** desenha o fluxo como um grafo, o que permite regras como *"se não achou artigo relevante, não invente; encaminhe o caso"*.

### RAGAS — nota para o RAG
- **Faithfulness:** a resposta se baseia nos artigos, ou inventou algo?
- **Context Precision:** os trechos buscados eram relevantes?
- **Context Recall:** a busca achou *tudo* o que precisava?

Usa um **LLM como juiz**. Sem API paga, o juiz será um **modelo local de outra família** (ex.: Llama ou Gemma pequeno), para evitar o **viés de autoavaliação**, que é o Qwen dar nota alta para as próprias respostas.

---

## Camada 3 — Fine-Tuning (o seu modelo)

### Qwen2.5-7B-Instruct
LLM open-source da Alibaba com **7 bilhões de parâmetros**, já treinado para seguir instruções.
**Por que ele?** Português muito bom para o tamanho, licença Apache 2.0 e **cabe numa T4 gratuita** quando quantizado.
Alternativas: Llama-3.1-8B (português pior) e Gemma-2-9B (maior). Se a GPU local apertar na hora de servir: **Qwen2.5-3B**.

### Quantização (4-bit)
Guardar cada parâmetro com **4 bits** em vez de 16.
7 bilhões × 2 bytes = **~14 GB** → em 4-bit fica **~4–5 GB**, com pouca perda de qualidade.

### LoRA e QLoRA
Treinar os 7 bilhões de parâmetros exigiria várias GPUs caras. O **LoRA** **congela o modelo original** e treina só umas **matrizes pequenas** acopladas a ele (~0,5% dos parâmetros), como um "adesivo" de conhecimento.
**QLoRA** é o LoRA aplicado a um modelo **quantizado em 4-bit**, o que faz o treino caber numa GPU gratuita.

> **Fine-tuning ensina *como* falar (estilo, empatia, estrutura). O RAG fornece *o que* falar (os fatos).**

### PEFT, TRL e bitsandbytes
- **PEFT:** implementa o LoRA.
- **TRL:** fornece o `SFTTrainer` (treino supervisionado de "pergunta → resposta ideal").
- **bitsandbytes:** faz a quantização em 4-bit.

### Training loss vs. validation loss
O **loss** mede o tamanho do erro. Se o erro no **treino** cai e o da **validação** sobe, o modelo está **decorando** os exemplos (*overfitting*). É o gráfico principal do treino.

---

## Camada 4 — Produção e MLOps

| Tecnologia | O que é | Por que usar |
|---|---|---|
| **FastAPI** | Framework para APIs em Python | Rápido, assíncrono, valida dados e gera a documentação sozinho |
| **Pydantic** | Validação de dados | Garante entrada no formato certo (texto não vazio, tamanho máximo) |
| **Evidently** | Monitoramento de **data drift** | Detecta quando as reclamações novas ficam diferentes das do treino, e aí o modelo pode piorar sem ninguém perceber |
| **Docker** | Empacota a aplicação com tudo o que ela precisa | Acaba com o "na minha máquina funciona" |
| **Docker Compose** | Sobe vários containers juntos | A API e o Milvus sobem com um comando |
| **GitHub Actions** | CI/CD automático | A cada `push`: verifica o código, roda os testes e monta a imagem |
| **Ruff** | Linter (verificador de estilo e erros simples) | Aponta problemas de código, extremamente rápido |
| **Pytest** | Framework de testes | Testa cada função automaticamente |

---

## Resumo dos modelos

| Modelo | Tamanho | Onde treina | Onde roda | Função |
|---|---|---|---|---|
| LightGBM | Pequeno | CPU (local ou Colab) | CPU | Prever risco e categoria (baseline) |
| BERTimbau-base | 110M | Colab ou GTX 1660 | GPU/CPU | Categoria com entendimento de contexto |
| bge-m3 | 568M | Não treina | CPU/GPU | Embeddings para a busca |
| bge-reranker-v2-m3 | 568M | Não treina | CPU/GPU | Reordenar os trechos buscados |
| Qwen2.5-7B + QLoRA | 7B | **Colab (T4)** | GPU local (4-bit / GGUF) | Gerar a resposta |
| Juiz do RAGAS | ~3–8B | Não treina | Colab | Avaliar o RAG |

---

## Custo zero: onde cada coisa roda

| Recurso | Gratuito | Observação |
|---|---|---|
| Bibliotecas | Sim | Todas open-source |
| Milvus | Sim | Docker local |
| GPU para treino | Sim | Google Colab gratuito (T4); Kaggle como reserva |
| Hugging Face Hub | Sim | Guarda os adapters LoRA |
| GitHub Actions | Sim | Ilimitado em repositório público |
| Docker Desktop | Sim | Uso pessoal |
| Demo pública | Sim | Hugging Face Spaces (CPU grátis, modelo leve GGUF) |
