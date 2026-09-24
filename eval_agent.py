"""
Mini-POC: Avaliador de Agentes com Dataset de Referencia
========================================================
Versao minima: dataset anotado + agente + funcao de scoring.
Sem LangFuse — so pra entender o fluxo antes de instrumentar.

Baseado em experiencia real com agentes em producao.
Nao contem dados, nomes ou metricas de nenhuma empresa.

Rode com: python3 eval_agent.py

Requisito: pip install openai
"""

import json
import os
from dataclasses import dataclass
from typing import Any

# ============================================================
# CONFIGURE AQUI
# ============================================================
# Agnostico de provider: OpenAI oficial (deixe LLM_BASE_URL vazio)
# ou qualquer endpoint OpenAI-compativel (vLLM, Ollama, LM Studio, etc.).
#   export OPENAI_API_KEY="sk-..."    # ou LLM_API_KEY
#   export LLM_MODEL="gpt-4o-mini"    # opcional
#   export LLM_BASE_URL="http://..."  # opcional (vazio = OpenAI oficial)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Troque pelo seu modelo

# ============================================================
# 1. SIMULE UM AGENTE (troque pelo seu agente real)
# ============================================================
def meu_agente(prompt: str) -> str:
    """Aqui voce conecta seu agente real. Exemplo: chamada LLM simples."""
    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL or None, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Voce e um assistente comercial que classifica o sentimento de mensagens de parceiros. Responda apenas: positivo, negativo, ou neutro."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip().lower()


# ============================================================
# 2. DATASET DE REFERENCIA (a parte mais importante)
# ============================================================
# Dataset anotado manualmente. Cada entrada: prompt + resposta esperada.
# Isso e o que separa "parece que funciona" de "a gente SABE que funciona".
DATASET = [
    {
        "id": 1,
        "prompt": "O prazo de entrega atrasou de novo. Terceira vez esse mes.",
        "esperado": "negativo",
        "categoria": "reclamacao"
    },
    {
        "id": 2,
        "prompt": "Obrigado pelo suporte rapido! Resolveram em 2 horas.",
        "esperado": "positivo",
        "categoria": "elogio"
    },
    {
        "id": 3,
        "prompt": "Qual o horario de funcionamento da central?",
        "esperado": "neutro",
        "categoria": "duvida"
    },
    {
        "id": 4,
        "prompt": "A parceria tem sido otima. Queremos renovar contrato.",
        "esperado": "positivo",
        "categoria": "renovacao"
    },
    {
        "id": 5,
        "prompt": "A taxa esta muito acima do mercado. Precisamos renegociar.",
        "esperado": "negativo",
        "categoria": "reclamacao"
    },
    {
        "id": 6,
        "prompt": "Vamos lancar uma acao promocional no fim de semana. Consegue apoiar?",
        "esperado": "neutro",
        "categoria": "solicitacao"
    },
    {
        "id": 7,
        "prompt": "A integracao com o sistema de voces esta instavel. Caiu 3x hoje.",
        "esperado": "negativo",
        "categoria": "problema_tecnico"
    },
    {
        "id": 8,
        "prompt": "Seu gerente de conta e muito atencioso. Melhor que o anterior.",
        "esperado": "positivo",
        "categoria": "elogio"
    },
]


# ============================================================
# 3. EXECUCAO E METRICAS
# ============================================================
@dataclass
class Resultado:
    id: int
    prompt: str
    esperado: str
    obtido: str
    acertou: bool
    categoria: str

def avaliar() -> list[Resultado]:
    resultados = []
    for item in DATASET:
        obtido = meu_agente(item["prompt"])
        acertou = obtido == item["esperado"]
        resultados.append(Resultado(
            id=item["id"],
            prompt=item["prompt"],
            esperado=item["esperado"],
            obtido=obtido,
            acertou=acertou,
            categoria=item["categoria"]
        ))
        status = "✓" if acertou else "✗"
        print(f"[{status}] #{item['id']} | Esperado: {item['esperado']} | Obtido: {obtido} | {item['prompt'][:60]}...")

    return resultados


def metricas(resultados: list[Resultado]) -> dict[str, Any]:
    total = len(resultados)
    acertos = sum(1 for r in resultados if r.acertou)
    acuracia = acertos / total * 100

    # Acuracia por categoria
    por_categoria = {}
    for r in resultados:
        if r.categoria not in por_categoria:
            por_categoria[r.categoria] = {"total": 0, "acertos": 0}
        por_categoria[r.categoria]["total"] += 1
        if r.acertou:
            por_categoria[r.categoria]["acertos"] += 1

    # Matriz de confusao simples
    erros = [r for r in resultados if not r.acertou]

    return {
        "total": total,
        "acertos": acertos,
        "acuracia": f"{acuracia:.1f}%",
        "por_categoria": {
            cat: f"{d['acertos']}/{d['total']} ({d['acertos']/d['total']*100:.0f}%)"
            for cat, d in por_categoria.items()
        },
        "erros": [
            {"id": r.id, "esperado": r.esperado, "obtido": r.obtido, "prompt": r.prompt}
            for r in erros
        ]
    }


# ============================================================
# 4. RODA TUDO
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AVALIADOR DE AGENTES — Mini-POC")
    print("=" * 60)
    print(f"Dataset: {len(DATASET)} exemplos anotados")
    print(f"Modelo: {MODEL}")
    print()

    print("--- Executando avaliacoes ---")
    resultados = avaliar()

    print()
    print("--- Metricas ---")
    m = metricas(resultados)
    print(json.dumps(m, indent=2, ensure_ascii=False))

    print()
    print("--- Diagnostico ---")
    acuracia = float(m["acuracia"].replace("%", ""))
    if acuracia >= 90:
        print("✓ Agente com performance solida. Pode ir pra producao com monitoramento.")
    elif acuracia >= 70:
        print("⚠ Agente precisa de ajuste. Analise os erros e refine o prompt ou o dataset.")
    else:
        print("✗ Agente nao esta pronto. Reveja o prompt, modelo, ou a qualidade do dataset.")

    print()
    print("Proximos passos:")
    print("1. Aumente o dataset para 50+ exemplos balanceados por categoria")
    print("2. Adicione avaliacao qualitativa (humana) nos casos ambguos")
    print("3. Integre com LangFuse ou similar para tracing")
    print("4. Automatize no CI/CD pra rodar a cada mudanca de prompt")
