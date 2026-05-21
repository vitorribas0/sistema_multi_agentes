from agents.base_agent import create_agent, agent_as_tool
from prompts.loader import load_prompt

# Tools do domínio de dados que este agente pode usar
ANALYST_TOOL_NAMES = {
    "load_file",
    "get_session_info",
    "filter_session",
    "search_session",
    "describe_session",
    "detect_anomalies",
    "get_text_from_session",
    "export_session",
    "clear_session",
    "sequential_thinking",
}


def create_analyst_agent(model, all_tools: list):
    """
    Agente especialista em análise de dados e CSV.
    Recebe todas as tools do MCP e filtra apenas as de seu domínio.
    """
    tools = [t for t in all_tools if t.name in ANALYST_TOOL_NAMES]
    prompt = load_prompt("2_agente_analista.txt")
    agent = create_agent(model, tools, prompt, use_memory=False)

    return agent_as_tool(
        agent,
        name="analyst_agent",
        description=(
            "Agente especialista em análise de dados. "
            "Use quando a tarefa envolver leitura de CSV ou Excel, filtragem de dados, "
            "estatísticas descritivas ou detecção de anomalias."
        ),
    )
