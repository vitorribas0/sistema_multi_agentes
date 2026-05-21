import os
from langchain_groq import ChatGroq


def create_model(model_name: str | None = None) -> ChatGroq:
    """
    Fábrica do LLM — único lugar do projeto que instancia o modelo.
    O nome do modelo pode ser sobrescrito via argumento ou variável de ambiente MODEL_NAME.
    max_tokens limita o tamanho das respostas geradas para evitar estouro do limite de TPM.
    """
    name = model_name or os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "1024"))
    return ChatGroq(model=name, max_tokens=max_tokens)
