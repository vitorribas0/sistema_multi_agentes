from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from prompts.loader import load_prompt


def create_orchestrator(tools):
    """
    Cria o agente orquestrador principal.
    Recebe todas as tools disponíveis e decide qual usar a cada interação.
    Para expandir para múltiplos agentes, crie funções análogas neste pacote
    (ex: create_analyst_agent, create_nlp_agent) e adicione um supervisor aqui.
    """
    model = ChatGroq(model="openai/gpt-oss-120b")
    memory = MemorySaver()
    prompt = load_prompt("1_orquestrador.txt")

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=memory,
    )
    return agent
