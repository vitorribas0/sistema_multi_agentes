from agents.base_agent import create_agent, agent_as_tool
from prompts.loader import load_prompt

# Tools do domínio de NLP que este agente pode usar
NLP_TOOL_NAMES = {
    "summarize_text",
    "classify_sentiment",
    "extract_keywords",
    "preprocess_text_column",
    "quantify_keywords",
    "sequential_thinking",
}


def create_nlp_agent(model, all_tools: list):
    """
    Agente especialista em processamento de linguagem natural.
    Recebe todas as tools do MCP e filtra apenas as de seu domínio.
    """
    tools = [t for t in all_tools if t.name in NLP_TOOL_NAMES]
    prompt = load_prompt("3_agente_nlp.txt")
    agent = create_agent(model, tools, prompt, use_memory=False)

    return agent_as_tool(
        agent,
        name="nlp_agent",
        description=(
            "Agente especialista em NLP e texto. "
            "Use quando a tarefa envolver resumo de texto, análise de sentimento "
            "ou extração de palavras-chave."
        ),
    )
