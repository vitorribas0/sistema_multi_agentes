import os
from langchain_groq import ChatGroq


def create_model(model_name: str | None = None) -> ChatGroq:
    """
    Fábrica do LLM — único lugar do projeto que instancia o modelo.
    O nome do modelo pode ser sobrescrito via argumento ou variável de ambiente MODEL_NAME.
    """
    name = model_name or os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    return ChatGroq(model=name)
