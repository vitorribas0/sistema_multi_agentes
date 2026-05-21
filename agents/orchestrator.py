from agents.base_agent import create_agent
from agents.analyst_agent import create_analyst_agent
from agents.nlp_agent import create_nlp_agent
from core.model_factory import create_model
from prompts.loader import load_prompt


def create_orchestrator(all_tools: list):
    """
    Orquestrador (supervisor dos sub-agentes).
    Não executa tarefas diretamente — delega para o agente especialista correto.

    Para adicionar um novo agente:
      1. Crie agents/novo_agente.py com create_novo_agent(model, all_tools)
      2. Importe e instancie abaixo
      3. Adicione à lista sub_agent_tools
    """
    model = create_model()

    # Instancia sub-agentes como tools
    analyst_tool = create_analyst_agent(model, all_tools)
    nlp_tool = create_nlp_agent(model, all_tools)

    sub_agent_tools = [analyst_tool, nlp_tool]

    prompt = load_prompt("1_orquestrador.txt")
    return create_agent(model, sub_agent_tools, prompt)
