from fastmcp import FastMCP

from tools.general_tools import register_tools as register_general_tools
from tools.data_tools import register_tools as register_data_tools
from tools.nlp_tools import register_tools as register_nlp_tools

MCP_SERVERS_CONFIG = {
    "main_server": {
        "transport": "stdio",
        "command": "python",
        "args": ["mcp_servers.py"],
    }
}

# Inicializa o servidor MCP principal
mcp = FastMCP("Main Server")

# Registra os grupos de tools por domínio
# Para adicionar um novo domínio: crie tools/novo_dominio.py e registre aqui
register_general_tools(mcp)
register_data_tools(mcp)
register_nlp_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")