# Avaliando agentes de IA: v0 com código

Projeto mínimo e autossuficiente pra quem está começando a explorar
avaliação de agentes de IA. Acompanha o artigo "Avaliação de agentes de IA:
um projeto v0 pra começar" (série "IA em produção: o que ninguém te conta").

Baseado em experiência real. Não contém dados, nomes ou métricas de nenhuma
empresa.

## Dois níveis

- `eval_agent.py` — versão mínima, só OpenAI. Dataset anotado + função de
  scoring. Bom pra entender o fluxo em 5 minutos, sem depender de LangFuse.
- `eval_langfuse.py` — dataset versionado + tracing (`@observe`) + scoring no
  LangFuse. O fluxo completo que você leva pra produção.

## Requisitos

- Python 3.9+
- `pip install openai` (só `eval_agent.py`)
- `pip install openai langfuse` (pra `eval_langfuse.py` — SDK LangFuse v3)

## Configuração

Copie o `.env.example` pra `.env` e preencha. O `.env` NÃO vai pro git.

    cp .env.example .env

Ou exporte direto no shell. O código é agnóstico de provider:

    LLM_API_KEY    chave (ou OPENAI_API_KEY)
    LLM_MODEL      modelo (default gpt-4o-mini)
    LLM_BASE_URL   endpoint OpenAI-compatível (vazio = OpenAI oficial)

    # eval_langfuse.py também precisa de:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST

## Rodando

    python3 eval_agent.py
    python3 eval_langfuse.py

O `eval_langfuse.py` cria o dataset no LangFuse automaticamente e registra
cada score no trace — dá pra ver o breakdown por categoria no dashboard.

## Disclaimer

Baseado em experiência real com agentes em produção. Não contém dados, nomes
ou métricas de nenhuma empresa.
