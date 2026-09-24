"""
POC: Avaliacao de Agentes com LangFuse + OpenAI Evals Pattern
==============================================================
Baseado no padrao do OpenAI Evals (github.com/openai/evals) e
integrado com LangFuse para tracing e observabilidade.

Compativel com LangFuse SDK v3 (pip install "langfuse>=3").

Rode com: python3 eval_langfuse.py

Requisitos:
  pip install langfuse openai

Configuracao:
  export LANGFUSE_PUBLIC_KEY="pk-..."
  export LANGFUSE_SECRET_KEY="sk-..."
  export LANGFUSE_HOST="https://cloud.langfuse.com"  # ou self-hosted
  export OPENAI_API_KEY="sk-..."      # ou LLM_API_KEY
  export LLM_MODEL="gpt-4o-mini"      # opcional
  export LLM_BASE_URL="http://..."    # opcional (vazio = OpenAI oficial)

O que este POC demonstra:
  1. Como criar um dataset de avaliacao no LangFuse
  2. Como instrumentar um agente com tracing (@observe)
  3. Como avaliar saidas contra um gabarito (padrao OpenAI Evals)
  4. Como visualizar metricas no dashboard do LangFuse

Baseado em experiencia real com agentes em producao.
Nao contem dados, nomes ou metricas de nenhuma empresa.
"""

import json
import os

from openai import OpenAI
from langfuse import Langfuse, observe

# ============================================================
# CONFIGURACAO
# ============================================================
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DATASET_NAME = "avaliacao-sentimento-parceiros"

# Inicializa clientes
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)
_openai_kwargs = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    _openai_kwargs["base_url"] = LLM_BASE_URL
openai_client = OpenAI(**_openai_kwargs)


# ============================================================
# 1. DATASET DE AVALIACAO (padrao OpenAI Evals)
# ============================================================
# No OpenAI Evals, cada entrada tem: input, ideal (gabarito), metadata.
# Aqui seguimos o mesmo padrao, armazenando no LangFuse.

EXEMPLOS = [
    {
        "input": "O prazo de entrega atrasou de novo. Terceira vez esse mes.",
        "ideal": "negativo",
        "metadata": {"categoria": "reclamacao", "dificuldade": "facil"},
    },
    {
        "input": "Obrigado pelo suporte rapido! Resolveram em 2 horas.",
        "ideal": "positivo",
        "metadata": {"categoria": "elogio", "dificuldade": "facil"},
    },
    {
        "input": "Qual o horario de funcionamento da central?",
        "ideal": "neutro",
        "metadata": {"categoria": "duvida", "dificuldade": "facil"},
    },
    {
        "input": "A parceria tem sido otima. Queremos renovar contrato.",
        "ideal": "positivo",
        "metadata": {"categoria": "renovacao", "dificuldade": "medio"},
    },
    {
        "input": "A taxa esta muito acima do mercado. Precisamos renegociar.",
        "ideal": "negativo",
        "metadata": {"categoria": "reclamacao", "dificuldade": "medio"},
    },
    {
        "input": "Vamos lancar uma acao promocional no fim de semana. Consegue apoiar?",
        "ideal": "neutro",
        "metadata": {"categoria": "solicitacao", "dificuldade": "dificil"},
    },
    {
        "input": "A integracao com o sistema de voces esta instavel. Caiu 3x hoje.",
        "ideal": "negativo",
        "metadata": {"categoria": "problema_tecnico", "dificuldade": "medio"},
    },
    {
        "input": "Seu gerente de conta e muito atencioso. Melhor que o anterior.",
        "ideal": "positivo",
        "metadata": {"categoria": "elogio", "dificuldade": "facil"},
    },
    {
        "input": "Estou pensando em migrar pra concorrencia. Me convenca a ficar.",
        "ideal": "negativo",
        "metadata": {"categoria": "risco_churn", "dificuldade": "dificil"},
    },
    {
        "input": "O dashboard novo ficou excelente. Muito mais intuitivo.",
        "ideal": "positivo",
        "metadata": {"categoria": "elogio", "dificuldade": "facil"},
    },
]


def criar_dataset_se_nao_existe() -> str:
    """Cria o dataset no LangFuse se ainda nao existir."""
    try:
        existing = langfuse.get_dataset(DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' ja existe ({len(existing.items)} itens).")
        return existing.id
    except Exception:
        pass

    dataset = langfuse.create_dataset(
        name=DATASET_NAME,
        description="Classificacao de sentimento de mensagens — dataset anotado manualmente",
        metadata={"criado_por": "poc-evals", "versao": "1.0"},
    )

    for ex in EXEMPLOS:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"mensagem": ex["input"]},
            expected_output={"sentimento": ex["ideal"]},
            metadata=ex["metadata"],
        )

    print(f"Dataset '{DATASET_NAME}' criado com {len(EXEMPLOS)} exemplos.")
    return dataset.id


# ============================================================
# 2. AGENTE SOB TESTE (instrumentado com LangFuse)
# ============================================================
@observe(name="classificador-sentimento", as_type="generation")
def classificar_sentimento(mensagem: str) -> dict:
    """
    Agente que classifica o sentimento de uma mensagem.
    Em producao, isso seria um agente com tool calling, RAG, etc.
    O @observe envia automaticamente tracing pro LangFuse.
    """
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Classifique o sentimento da mensagem. Responda APENAS com um JSON: {\"sentimento\": \"positivo\"|\"negativo\"|\"neutro\", \"confianca\": 0.0-1.0, \"explicacao\": \"uma frase\"}",
            },
            {"role": "user", "content": mensagem},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    resultado = json.loads(response.choices[0].message.content)
    return resultado


# ============================================================
# 3. FUNCAO DE SCORING (o "eval" do OpenAI Evals)
# ============================================================
def avaliar_resposta(esperado: str, obtido: str) -> tuple[bool, float]:
    """
    Compara o sentimento obtido com o esperado.
    Retorna (acertou, score).

    Score usa match parcial: acerto exato = 1.0, erro = 0.0.
    Poderia ser estendido com fuzzy match, similaridade semantica, etc.
    """
    acertou = obtido.strip().lower() == esperado.strip().lower()
    return acertou, 1.0 if acertou else 0.0


# ============================================================
# 4. LOOP DE AVALIACAO (registrando scores no LangFuse)
# ============================================================
@observe(name="avaliacao-completa")
def executar_avaliacao():
    """Percorre o dataset inteiro, avalia cada exemplo, registra no LangFuse."""

    print("=" * 60)
    print("AVALIACAO DE AGENTES — LangFuse + OpenAI Evals Pattern")
    print("=" * 60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Modelo: {MODEL}")
    print(f"LangFuse: {LANGFUSE_HOST}")
    print()

    dataset = langfuse.get_dataset(DATASET_NAME)

    acertos = 0
    total = 0
    resultados_por_categoria = {}

    for item in dataset.items:
        mensagem = item.input["mensagem"]
        esperado = item.expected_output["sentimento"]
        categoria = item.metadata.get("categoria", "sem_categoria")

        # Cria um span individual por item (padrao OpenAI Evals)
        with langfuse.start_as_current_span(
            name=f"eval-item-{item.id}",
            input=item.input,
        ) as span:

            resultado = classificar_sentimento(mensagem)
            obtido = resultado.get("sentimento", "")
            confianca = resultado.get("confianca", 0)
            acertou, score = avaliar_resposta(esperado, obtido)

            # Registra score no span (visivel no dashboard)
            span.score(
                name="exact_match",
                value=score,
                comment=f"Esperado: {esperado}, Obtido: {obtido}" if not acertou else None,
            )

            span.update(output={"obtido": obtido, "esperado": esperado, "acertou": acertou})

        total += 1
        if acertou:
            acertos += 1

        if categoria not in resultados_por_categoria:
            resultados_por_categoria[categoria] = {"acertos": 0, "total": 0}
        resultados_por_categoria[categoria]["total"] += 1
        if acertou:
            resultados_por_categoria[categoria]["acertos"] += 1

        status = "✓" if acertou else "✗"
        print(
            f"[{status}] acerto_exato={score:.0f} | confianca_modelo={confianca:.2f} | "
            f"{esperado} | {obtido} | {mensagem[:60]}..."
        )

    # Metricas finais (tambem registradas no LangFuse)
    acuracia = acertos / total * 100 if total > 0 else 0

    print()
    print("--- Metricas Finais ---")
    print(f"Total: {total} | Acertos: {acertos} | Acuracia: {acuracia:.1f}%")
    print()

    print("--- Por Categoria ---")
    for cat, r in sorted(resultados_por_categoria.items()):
        pct = r["acertos"] / r["total"] * 100
        barra = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {cat:<20} {barra} {r['acertos']}/{r['total']} ({pct:.0f}%)")

    print()
    print("--- Diagnostico ---")
    if acuracia >= 90:
        print("✓ Performance solida. Agente pronto para producao com monitoramento.")
    elif acuracia >= 70:
        print("⚠ Performance aceitavel. Investigue as categorias com < 80% de acerto.")
        categorias_ruins = [
            cat
            for cat, r in resultados_por_categoria.items()
            if r["acertos"] / r["total"] < 0.8
        ]
        for cat in categorias_ruins:
            print(f"  → Categoria '{cat}' precisa de atencao (refine prompts ou dataset)")
    else:
        print("✗ Performance insuficiente. Reveja prompts, modelo, ou qualidade do dataset.")

    print()
    print(f"Dashboard LangFuse: {LANGFUSE_HOST}/project/{dataset.project_id}")
    print("Procure pela trace 'avaliacao-completa' para ver o detalhe de cada item.")

    # Limpa recursos
    langfuse.flush()


# ============================================================
# 5. ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    if not LANGFUSE_PUBLIC_KEY:
        print("ERRO: Configure as variaveis de ambiente do LangFuse.")
        print("  export LANGFUSE_PUBLIC_KEY='pk-...'")
        print("  export LANGFUSE_SECRET_KEY='sk-...'")
        print("  export LANGFUSE_HOST='https://cloud.langfuse.com'")
        print()
        print("Crie uma conta gratuita em: https://cloud.langfuse.com")
        exit(1)

    criar_dataset_se_nao_existe()
    executar_avaliacao()
