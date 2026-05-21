import uuid

from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def create_agent(model, tools: list, prompt: str, use_memory: bool = True):
    """
    Cria um agente ReAct.
    - use_memory=True  → orquestrador: mantém histórico entre turnos do usuário.
    - use_memory=False → sub-agentes: stateless, sem acumulação de contexto entre chamadas.
    """
    checkpointer = MemorySaver() if use_memory else None
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        version="v2",
    )


def agent_as_tool(agent, name: str, description: str):
    """
    Encapsula um sub-agente como Tool async.
    Cada invocação usa um thread_id único para evitar acumulação de histórico
    e estouro de tokens (rate limit TPM).
    """
    async def _ainvoke(message: str) -> str:
        # thread_id único por chamada — sub-agentes são stateless por design
        config = {"configurable": {"thread_id": f"{name}_{uuid.uuid4().hex}"}}
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
        return result["messages"][-1].content

    return StructuredTool.from_function(
        coroutine=_ainvoke,
        name=name,
        description=description,
    )
