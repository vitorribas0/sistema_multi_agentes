import sys
from typing import Optional
from pydantic import Field


def register_tools(mcp):

    @mcp.tool(
        description=(
            "Ferramenta para realizar pensamento sequencial e autoreflexivo, "
            "ideal para decompor problemas complexos, validar lógica, debugar "
            "códigos difíceis e corrigir rumos antes de entregar a resposta final ao usuário."
        )
    )
    def sequential_thinking(
        thought: str = Field(description="O raciocínio detalhado ou hipótese para este passo específico."),
        thought_number: int = Field(description="O número do passo atual (ex: 1, 2, 3)."),
        total_thoughts: int = Field(description="A estimativa atual de quantos passos serão necessários no total."),
        next_thought_needed: bool = Field(description="True se precisa de outro passo, False se concluiu o raciocínio."),
        is_revision: Optional[bool] = Field(default=False, description="True se este passo corrige ou revisa um pensamento anterior."),
    ) -> str:
        """Pensamento sequencial e autoreflexivo para raciocínio passo a passo."""
        status_tag = "🔄 [REVISÃO]" if is_revision else "🧠 [PENSAMENTO]"
        log_message = (
            f"\n{status_tag} Passo {thought_number}/{total_thoughts}\n"
            f"Raciocínio: {thought}\n"
            f"Próximo passo necessário? {'Sim' if next_thought_needed else 'Não, raciocínio concluído.'}\n"
            f"{'-' * 40}\n"
        )
        print(log_message, file=sys.stderr, flush=True)

        if next_thought_needed:
            return f"Passo {thought_number} registrado. Continue para o próximo passo se necessário."
        return f"Raciocínio finalizado no passo {thought_number}. Você pode responder ao usuário agora."

    @mcp.tool(description="Soma dois números inteiros e retorna o resultado.")
    def sum_numbers(a: int, b: int) -> int:
        """Soma dois números inteiros e retorna o resultado."""
        return a + b
