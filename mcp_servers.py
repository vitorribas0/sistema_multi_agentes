import sys
from typing import Optional
from fastmcp import FastMCP
from pydantic import Field

MCP_SERVERS_CONFIG = {
    "sequential_thinking": {
        "transport": "stdio",  # Define o transporte como stdio
        "command": "python",  # Comando para iniciar o servidor
        "args": ["mcp_servers.py"]  # Argumentos para o comando
    }
}

# Inicializa o servidor MCP com o nome do serviço
mcp = FastMCP("Sequential Thinking")

@mcp.tool(
        description=("""Ferramenta para realizar pensamento sequencial e autoreflexivo, 
        ideal para decompor problemas complexos, validar lógica, debugar 
        códigos difíceis e corrigir rumos antes de entregar a resposta final ao usuário.""")
)
def sequential_thinking(
    thought: str = Field(
        description="O raciocínio detalhado ou hipótese para este passo específico."
    ),
    thought_number: int = Field(
        description="O número do passo atual (ex: 1, 2, 3)."
    ),
    total_thoughts: int = Field(
        description="A estimativa atual de quantos passos serão necessários no total para resolver o problema."
    ),
    next_thought_needed: bool = Field(
        description="Defina como True se você precisa de outro passo para continuar analisando, ou False se concluiu o raciocínio."
    ),
    is_revision: Optional[bool] = Field(
        default=False,
        description="Defina como True se este passo corrige, refuta ou revisa um pensamento anterior."
    )
) -> str:
    """
    Uma ferramenta dedicada para realizar um processo de pensamento profundo, sequencial e autoreflexivo.
    Use esta ferramenta para quebrar problemas complexos, validar lógica, debugar códigos difíceis 
    e corrigir rumos antes de entregar a resposta final ao usuário.
    """
    
    # Marcador visual de revisão
    status_tag = "🔄 [REVISÃO]" if is_revision else "🧠 [PENSAMENTO]"
    
    # Mensagem de log enviada para o painel do Cliente MCP (via stderr para não quebrar o protocolo stdio)
    log_message = (
        f"\n{status_tag} Passo {thought_number}/{total_thoughts}\n"
        f"Raciocínio: {thought}\n"
        f"Próximo passo necessário? {'Sim' if next_thought_needed else 'Não, raciocínio concluído.'}\n"
        f"{'-'*40}\n"
    )
    print(log_message, file=sys.stderr, flush=True)
    
    # Resposta de confirmação interna que o Modelo (LLM) recebe de volta da ferramenta
    if next_thought_needed:
        return f"Passo {thought_number} registrado. Continue para o próximo passo se necessário."
    else:
        return f"Raciocínio finalizado no passo {thought_number}. Você pode responder ao usuário agora."


@mcp.tool(
    description="Ferramenta para ler um arquivo CSV do caminho especificado e " \
    "retornar as primeiras linhas como string.")
def read_csv(file_path: str) -> str:
    """Lê um arquivo CSV do caminho especificado e retorna as primeiras linhas como string.
    Args:
        file_path (str): Caminho completo para o arquivo CSV.

    Returns:
        str: As primeiras linhas do arquivo CSV como string.
    """
    try:
        df = pd.read_csv(file_path)
        return df.head().to_string()
    except Exception as e:
        return f"Erro ao ler o arquivo CSV: {e}"

@mcp.tool(
    description=("Soma dois números inteiros e retorna o resultado.")
)
def sum_numbers(a: int, b: int) -> int:
    """Soma dois números inteiros e retorna o resultado."""
    return a + b


#----------------- FIM DAS TOOLS -----------------#

if __name__ == "__main__":
    # Roda o servidor usando o protocolo padrão STDIO (Entrada/Saída padrão)
    mcp.run(transport="stdio")