from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient

from mcp_servers import MCP_SERVERS_CONFIG
from agents.orchestrator import create_orchestrator

import asyncio

load_dotenv()

async def main():
    mcp_client = MultiServerMCPClient(MCP_SERVERS_CONFIG)
    tools = await mcp_client.get_tools()

    agent_executor = create_orchestrator(tools)

    config = {
        'configurable': {'thread_id': '1'},
        'recursion_limit': 100,
    }

    while True:
        input_message = {
            'role': 'user',
            'content': input('Digite: '),
        }
        async for step in agent_executor.astream(
            {'messages': [input_message]}, config, stream_mode='values'
        ):
            step['messages'][-1].pretty_print()

asyncio.run(main())