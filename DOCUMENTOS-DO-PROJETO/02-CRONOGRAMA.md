# Telecom Complaint AI — Cronograma

> **Premissa:** ~8–10 horas por semana de estudo e prática. Se tiver mais tempo, as fases encurtam.
> **Regra:** uma fase só termina quando o critério de "Pronto quando" for atingido. Errar e refazer faz parte do aprendizado.
> **Ambientes:** 💻 = PC local (GTX 1660) · ☁️ = Google Colab gratuito (GPU T4)

---

## Visão geral

| Fase | Semana(s) | Tema | Ambiente |
|---|---|---|---|
| 0 | 1 | Ambiente e fundações | 💻 |
| 1 | 2–3 | Análise exploratória (EDA) e preparação dos dados | ☁️ + 💻 |
| 2 | 4 | Baseline: TF-IDF + Regressão Logística | 💻 |
| 3 | 5 | LightGBM + validação cruzada + SHAP | 💻 / ☁️ |
| 4 | 6–7 | Deep Learning: BERTimbau | ☁️ (ou 💻) |
| 5 | 8 | RAG I: ingestão (Docling + chunking + embeddings) | ☁️ + 💻 |
| 6 | 9 | RAG II: Milvus, busca híbrida, reranker | 💻 |
| 7 | 10 | RAG III: LangGraph + avaliação com RAGAS | ☁️ |
| 8 | 11–12 | Fine-tuning QLoRA do Qwen2.5-7B | ☁️ |
| 9 | 13 | Servir o LLM localmente (merge + GGUF) | ☁️ + 💻 |
| 10 | 14 | API FastAPI integrando tudo | 💻 |
| 11 | 15 | Observabilidade: latência, custo de GPU, drift | 💻 |
| 12 | 16 | Docker, CI/CD e demo pública | 💻 + GitHub + HF Spaces |

**Total estimado: ~16 semanas (4 meses).**

---

## Fase 0 — Ambiente e fundações (semana 1) 💻

**Você vai aprender:** ambientes virtuais, gerenciamento de dependências, Git, CUDA.

- [x] Instalar Python 3.11 e criar o venv do projeto
- [x] Criar o `pyproject.toml` com grupos de dependências (`ml`, `rag`, `api`, `dev`)
- [x] Instalar o PyTorch com CUDA e confirmar que a GTX 1660 é reconhecida
- [ ] Instalar WSL2 + Docker Desktop
- [ ] Criar contas gratuitas: GitHub, Hugging Face (e usar a conta Google para o Colab)
- [x] `git init`, primeiro commit, repositório público no GitHub
- [x] CI mínimo no GitHub Actions: Ruff + Pytest (com um teste "hello world")
- [x] `src/core/config.py` lendo configurações (YAML + `.env`)

**Pronto quando:** `torch.cuda.is_available()` retorna `True` e o CI fica verde no GitHub.

---

## Fase 1 — EDA e preparação dos dados (semanas 2–3) ☁️ + 💻

**Você vai aprender:** pandas, análise exploratória, definição de alvo (*target*), vazamento de dados (*data leakage*), divisão treino/teste.

- [ ] Carregar o `dados2025.json` (200k+ registros) e filtrar as operadoras de telecom
- [ ] EDA: distribuição de `status`, `nota`, empresas, tamanho dos textos, valores ausentes
- [ ] **Definir o alvo de risco** (ex.: `status == "Não Resolvida"` ou nota baixa) e medir o desbalanceamento
- [ ] **Decidir como obter as categorias.** O dataset não tem coluna de categoria. Opções: regras por palavras-chave, clustering ou rotulagem manual de uma amostra. Vamos discutir isso
- [ ] Separar treino/validação/teste (avaliar divisão **temporal** para simular produção)
- [ ] Gerar `data/processed/telecom_ml.parquet`
- [ ] Gerar `data/processed/telecom_finetuning.jsonl` (1.200 pares, status "Resolvida" e nota 5)

**Pronto quando:** os dois arquivos existem, são gerados por um script e o desbalanceamento está documentado.

---

## Fase 2 — Baseline: TF-IDF + Regressão Logística (semana 4) 💻

**Você vai aprender:** vetorização de texto, por que começar simples, métricas (acurácia × F1 × ROC-AUC).

- [ ] Pipeline do scikit-learn: TF-IDF → Regressão Logística
- [ ] Entender por que a **acurácia engana** em dados desbalanceados
- [ ] Medir F1, ROC-AUC e PR-AUC e montar a matriz de confusão

**Pronto quando:** existe um número de referência (baseline) que os próximos modelos precisam superar.

---

## Fase 3 — LightGBM + validação cruzada + SHAP (semana 5) 💻 / ☁️

**Você vai aprender:** gradient boosting, validação cruzada estratificada, ajuste de hiperparâmetros, ajuste do limiar de decisão (*threshold*), explicabilidade.

- [ ] TF-IDF + metadados (empresa, UF, tamanho do texto) → LightGBM
- [ ] `StratifiedKFold` (5 folds)
- [ ] Tratar o desbalanceamento (`class_weight` / `scale_pos_weight`)
- [ ] Busca de hiperparâmetros com Optuna (☁️ Colab, para não ocupar o PC)
- [ ] SHAP: gráfico global + explicação de previsões individuais
- [ ] Salvar o modelo em `models/lgbm/`

**Pronto quando:** o LightGBM supera o baseline e o gráfico SHAP faz sentido para você.

---

## Fase 4 — Deep Learning: BERTimbau (semanas 6–7) ☁️ (ou 💻)

**Você vai aprender:** Transformers, tokenização, fine-tuning de classificador, `fp16`, *gradient accumulation*, early stopping.

- [ ] Entender a tokenização do BERT (WordPiece, limite de 512 tokens)
- [ ] Fine-tuning com `transformers` no Colab (T4, mais rápido) ou na GTX 1660 (batch pequeno + `fp16`)
- [ ] Acompanhar a curva de loss de treino × validação
- [ ] **Comparação final:** LightGBM × BERTimbau em F1, latência (ms por previsão) e memória
- [ ] Salvar o checkpoint em `models/bert/` ou no HF Hub

**Pronto quando:** existe uma tabela comparativa e você consegue dizer qual modelo usaria em produção e por quê.

---

## Fase 5 — RAG I: ingestão (semana 8) ☁️ + 💻

**Você vai aprender:** parsing de documentos, estratégias de chunking, embeddings.

- [ ] Baixar o PDF da Resolução Anatel nº 632/2014 para `data/raw/`
- [ ] Docling: PDF → estrutura (capítulos, artigos)
- [ ] Chunking **por artigo**, com metadados (número do artigo, capítulo)
- [ ] Gerar os embeddings com o `bge-m3` (denso + esparso)

**Pronto quando:** cada chunk é um artigo completo e legível, com metadados.

---

## Fase 6 — RAG II: Milvus, busca híbrida e reranker (semana 9) 💻

**Você vai aprender:** bancos vetoriais, busca densa × esparsa, RRF, cross-encoders.

- [ ] Subir o Milvus com Docker Compose
- [ ] Criar a coleção com campos denso + esparso e indexar os chunks
- [ ] Busca híbrida com `RRFRanker`
- [ ] Reranker `bge-reranker-v2-m3` (top-20 → top-5)
- [ ] Testar manualmente com 10 perguntas reais

**Pronto quando:** para as perguntas de teste, o artigo certo aparece no top-5.

---

## Fase 7 — RAG III: LangGraph + RAGAS (semana 10) ☁️

**Você vai aprender:** orquestração de LLMs, avaliação quantitativa, LLM como juiz e seus vieses.

- [ ] Fluxo no LangGraph: busca → rerank → decisão ("tem contexto suficiente?") → geração
- [ ] Montar um **dataset de avaliação** (30–50 perguntas com o artigo correto esperado)
- [ ] RAGAS com **juiz local** (modelo de outra família) rodando no Colab
- [ ] Comparar: só busca densa × busca híbrida × híbrida + reranker

**Pronto quando:** existe uma tabela mostrando que cada componente melhora (ou não) as métricas.

---

## Fase 8 — Fine-tuning QLoRA (semanas 11–12) ☁️

**Você vai aprender:** quantização, LoRA, SFT, overfitting em LLMs, avaliação de geração.

- [ ] Formatar o `.jsonl` no template de chat do Qwen
- [ ] Notebook `notebooks/finetuning_qlora.ipynb` no Colab (T4)
- [ ] Escolher os hiperparâmetros (rank `r`, `alpha`, learning rate, épocas) e entender cada um
- [ ] Monitorar a loss de treino × validação e salvar checkpoints
- [ ] Comparar respostas: modelo base × modelo ajustado (avaliação lado a lado)
- [ ] Publicar o adapter no Hugging Face Hub

**Pronto quando:** o adapter está no HF Hub e o modelo ajustado é visivelmente melhor no estilo de resposta.

---

## Fase 9 — Servir o LLM localmente (semana 13) ☁️ + 💻

**Você vai aprender:** merge de adapter, formato GGUF, quantização para inferência, limites de VRAM.

- [ ] Fazer o merge do adapter com o modelo base (☁️, precisa de bastante RAM)
- [ ] Converter para GGUF Q4 com `llama.cpp` (☁️)
- [ ] Rodar localmente com `llama-cpp-python` na GTX 1660
- [ ] Medir VRAM, tokens/s e latência. Se não couber, avaliar o Qwen2.5-3B

**Pronto quando:** o modelo responde localmente e os números de desempenho estão medidos.

---

## Fase 10 — API FastAPI (semana 14) 💻

**Você vai aprender:** APIs assíncronas, validação, versionamento, testes com mock.

- [ ] Endpoints `/v1/triage`, `/v1/retrieve`, `/v1/resolve`, `/health`
- [ ] Schemas Pydantic, rate limiting e tratamento de erros
- [ ] Carregar os modelos uma vez na inicialização (*lifespan*), e não a cada requisição
- [ ] Testes com Pytest usando um **LLM falso** (o CI não tem GPU)

**Pronto quando:** `/v1/resolve` recebe uma reclamação e devolve categoria + risco + artigos + resposta.

---

## Fase 11 — Observabilidade (semana 15) 💻

**Você vai aprender:** métricas de produção, custo de inferência, data drift.

- [ ] Registrar por requisição: latência de cada camada, tokens, VRAM e predição
- [ ] Custo estimado: tempo de GPU × custo de referência por hora
- [ ] Evidently: comparar os dados de produção com os de treino e gerar o relatório de drift
- [ ] Simular um drift (ex.: injetar reclamações de um tipo novo) e ver o alerta disparar

**Pronto quando:** o relatório de drift detecta a mudança que você simulou.

---

## Fase 12 — Docker, CI/CD e demo (semana 16) 💻 + GitHub + HF Spaces

**Você vai aprender:** containers multi-stage, Docker Compose, pipeline de CI/CD, deploy.

- [ ] Dockerfile multi-stage (imagem enxuta, sem dependências de treino)
- [ ] `docker-compose.yml`: API + Milvus
- [ ] GitHub Actions completo: Ruff → Pytest → build da imagem
- [ ] Demo no Hugging Face Spaces (CPU grátis, modelo leve)
- [ ] README final com arquitetura, métricas e decisões

**Pronto quando:** qualquer pessoa consegue clonar o repositório e subir o projeto com `docker compose up`.

---

## Estratégia para reduzir a carga no PC local

| Tarefa pesada | Onde rodar | Por quê |
|---|---|---|
| Carregar e limpar os 200k registros | ☁️ Colab (~12 GB de RAM) | Poupa a RAM do PC |
| Busca de hiperparâmetros (Optuna) | ☁️ Colab (CPU) | Horas de CPU a 100% |
| Fine-tuning do BERTimbau | ☁️ Colab T4 | A T4 tem 16 GB; a GTX 1660 tem 6 GB |
| Gerar embeddings em lote | ☁️ Colab | Processamento único, depois só importa os vetores |
| Avaliação RAGAS com o juiz | ☁️ Colab | O juiz é um LLM grande |
| Fine-tuning QLoRA | ☁️ Colab | Impossível em 6 GB |
| Merge + conversão para GGUF | ☁️ Colab | Precisa de muita RAM |
| Desenvolvimento, testes, Milvus, API | 💻 Local | Precisa ficar rodando continuamente |

**Boas práticas na nuvem gratuita:**
- **Colab gratuito:** GPU T4 sem garantia; sessão cai após ~90 min de inatividade e dura no máximo ~12 h; existe um limite de uso diário variável
- Guardar os dados no **Google Drive** e montá-lo no notebook (envia uma vez, reutiliza sempre)
- **Kaggle** (~30 h/semana de T4) fica como reserva se o Colab bloquear a GPU
- Salvar **checkpoints** com frequência e enviar os resultados para o **HF Hub** (a sessão pode cair)
- Usar notebooks só para tarefas pesadas e manter a lógica em `src/` (o notebook importa do repositório via `git clone`)
