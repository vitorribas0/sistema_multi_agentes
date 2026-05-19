from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from mcp_servers import MCP_SERVERS_CONFIG
from prompts import AGENT_SYSTEM_PROMPT

import asyncio

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')
memory = MemorySaver()

system_prompt = """
Você é um Especialista de multi-agentes na auditoria interna, 
você eh responsável por
auxiliar 
"""

tools = [list_cars, add_car, read_csv]


agent_executor = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

config = {'configurable': {'thread_id': '1'}}

while True:
    input_message = {
        'role': 'user',
        'content': input('Digite: '),
    }
    for step in agent_executor.stream(
        {'messages': [input_message]},config, stream_mode='values'
    ):
        step['messages'][-1].pretty_print()