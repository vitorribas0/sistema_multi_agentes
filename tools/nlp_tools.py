import re
from collections import Counter


def register_tools(mcp):

    @mcp.tool(
        description=(
            "Realiza uma sumarização extrativa de um texto, retornando as primeiras "
            "frases mais relevantes. Útil para resumir documentos longos rapidamente."
        )
    )
    def summarize_text(text: str, num_sentences: int = 3) -> str:
        """Retorna as primeiras N frases do texto como resumo extrativo."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary = sentences[:num_sentences]
        return " ".join(summary) if summary else "Texto vazio ou muito curto para resumir."

    @mcp.tool(
        description=(
            "Classifica o sentimento de um texto como Positivo, Negativo ou Neutro "
            "com base em palavras-chave. Útil para análise rápida de feedbacks e avaliações."
        )
    )
    def classify_sentiment(text: str) -> str:
        """Classifica o sentimento de um texto em Positivo, Negativo ou Neutro."""
        positive_words = {
            "bom", "ótimo", "excelente", "incrível", "maravilhoso", "perfeito",
            "adorei", "gostei", "parabéns", "sucesso", "feliz", "satisfeito",
            "eficiente", "rápido", "recomendo", "positivo", "great", "good",
            "excellent", "amazing", "love", "perfect", "happy",
        }
        negative_words = {
            "ruim", "péssimo", "horrível", "terrível", "odeio", "odiei",
            "problema", "erro", "falha", "lento", "insatisfeito", "decepcionante",
            "negativo", "bad", "terrible", "horrible", "hate", "slow", "broken",
        }
        words = re.findall(r'\b\w+\b', text.lower())
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)

        if pos > neg:
            return f"Sentimento: Positivo (score positivo={pos}, negativo={neg})"
        elif neg > pos:
            return f"Sentimento: Negativo (score positivo={pos}, negativo={neg})"
        return f"Sentimento: Neutro (score positivo={pos}, negativo={neg})"

    @mcp.tool(
        description=(
            "Extrai as palavras-chave mais frequentes de um texto, ignorando "
            "stopwords comuns. Retorna as N palavras mais relevantes."
        )
    )
    def extract_keywords(text: str, top_n: int = 10) -> str:
        """Extrai as palavras mais frequentes de um texto, ignorando stopwords."""
        stopwords = {
            "a", "o", "e", "de", "do", "da", "em", "um", "uma", "para",
            "com", "se", "que", "não", "por", "as", "os", "ao", "dos",
            "das", "the", "is", "in", "of", "to", "and", "a", "an",
        }
        words = re.findall(r'\b[a-záéíóúâêîôûãõçàü]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        most_common = Counter(filtered).most_common(top_n)

        if not most_common:
            return "Nenhuma palavra-chave encontrada."

        result = "\n".join(f"{word}: {count}x" for word, count in most_common)
        return f"Palavras-chave extraídas:\n{result}"
