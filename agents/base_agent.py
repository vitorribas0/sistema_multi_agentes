from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def create_agent(model, tools: list, prompt: str):
    """
    Cria um agente ReAct com memória.
    Reutilizável por qualquer agente do sistema.
    """
    memory = MemorySaver()
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        checkpointer=memory,
    )


def agent_as_tool(agent, name: str, description: str):
    """
    Encapsula um agente como uma Tool async para que o orquestrador possa delegá-lo.
    Usa ainvoke para compatibilidade com tools MCP que são async-only.
    """
    async def _ainvoke(message: str) -> str:
        config = {"configurable": {"thread_id": f"{name}_thread"}}
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
