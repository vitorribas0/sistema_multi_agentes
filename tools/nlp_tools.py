import re
import json
from collections import Counter
from functools import lru_cache

import spacy

from tools.data_tools import get_session_df, _SESSION_DATA


# ─────────────────────────────────────────────────────────────
# Carregamento lazy do modelo spaCy — carregado uma única vez
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_nlp_model():
    """Carrega o modelo pt_core_news_sm uma única vez (lazy + cache)."""
    return spacy.load("pt_core_news_sm", disable=["parser", "ner"])


# ─────────────────────────────────────────────────────────────
# Stopwords em português (complementadas com as do spaCy)
# ─────────────────────────────────────────────────────────────
_PT_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em",
    "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas",
    "este", "estes", "eu", "foi", "fomos", "for", "foram", "há", "isso",
    "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus",
    "minha", "minhas", "muito", "na", "nas", "nem", "no", "nos", "nós",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "não", "nós", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
    "quando", "que", "quem", "se", "seja", "sem", "ser", "seu", "seus",
    "si", "sobre", "sua", "suas", "também", "te", "tem", "temos", "tenho",
    "ter", "teu", "teus", "tua", "tuas", "tudo", "um", "uma", "umas", "uns",
    "você", "vocês", "vos", "à", "às", "é", "são",
}


def _normalize_text(text: str) -> str:
    """Normaliza um texto: lowercase, remove pontuação e caracteres especiais."""
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúâêîôûãõçàü]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_stopwords(text: str) -> str:
    """Remove stopwords portuguesas de um texto já normalizado."""
    words = text.split()
    return " ".join(w for w in words if w not in _PT_STOPWORDS)


def _lemmatize_text(text: str) -> str:
    """Lematiza um texto usando spaCy pt_core_news_sm."""
    nlp = _get_nlp_model()
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_space)


def register_tools(mcp):

    # ─────────────────────────────────────────────────────────
    # TOOLS EXISTENTES
    # ─────────────────────────────────────────────────────────

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
        words = re.findall(r'\b[a-záéíóúâêîôûãõçàü]{3,}\b', text.lower())
        filtered = [w for w in words if w not in _PT_STOPWORDS]
        most_common = Counter(filtered).most_common(top_n)

        if not most_common:
            return "Nenhuma palavra-chave encontrada."

        result = "\n".join(f"  {word}: {count}x" for word, count in most_common)
        return f"Palavras-chave extraídas:\n{result}"

    # ─────────────────────────────────────────────────────────
    # NOVAS TOOLS: PIPELINE DE PRÉ-PROCESSAMENTO + QUANTIFICAÇÃO
    # ─────────────────────────────────────────────────────────

    @mcp.tool(
        description=(
            "Pré-processa uma coluna de texto do DataFrame da sessão: normaliza (lowercase, "
            "remove pontuação), remove stopwords em português e lematiza usando spaCy. "
            "O resultado é salvo em uma nova coluna '<coluna>_processado' na mesma sessão. "
            "Retorna um preview das primeiras 3 linhas originais vs. processadas."
        )
    )
    def preprocess_text_column(session_id: str, column: str) -> str:
        """Normaliza, remove stopwords e lematiza uma coluna de texto. Salva resultado na sessão."""
        df = get_session_df(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}"

        target_col = f"{column}_processado"

        def _process(text):
            if not isinstance(text, str) or not text.strip():
                return ""
            normalized = _normalize_text(text)
            no_stop = _remove_stopwords(normalized)
            lemmatized = _lemmatize_text(no_stop)
            return lemmatized

        df[target_col] = df[column].apply(_process)
        _SESSION_DATA[session_id] = df

        preview_rows = df[[column, target_col]].head(3)
        preview = preview_rows.to_string(index=False)

        return (
            f"Pré-processamento concluído.\n"
            f"Session ID: {session_id}\n"
            f"Coluna original: '{column}'\n"
            f"Coluna processada salva como: '{target_col}'\n"
            f"Total de registros processados: {len(df)}\n\n"
            f"Preview (original vs. processado):\n{preview}"
        )

    @mcp.tool(
        description=(
            "Cria colunas de quantificação de menções a grupos de palavras-chave no DataFrame da sessão. "
            "Para cada grupo informado, conta quantas vezes as palavras do grupo aparecem em cada registro "
            "da coluna especificada e salva o resultado como uma nova coluna numérica. "
            "Use a coluna '_processado' gerada por preprocess_text_column para maior precisão. "
            "O parâmetro 'keyword_groups' deve ser um JSON com formato: "
            "{\"nome_coluna\": [\"palavra1\", \"palavra2\", ...], ...}. "
            "Exemplo: {\"renegociacao\": [\"renegoci\", \"renegocia\", \"acordo\"], \"cobranca\": [\"cobr\", \"divida\"]}. "
            "Retorna session_id, nomes das colunas criadas e preview com as primeiras 3 linhas."
        )
    )
    def quantify_keywords(session_id: str, source_column: str, keyword_groups: str) -> str:
        """Cria colunas de contagem de menções por grupo de palavras. keyword_groups = JSON dict."""
        df = get_session_df(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        if source_column not in df.columns:
            return f"Coluna '{source_column}' não encontrada. Colunas disponíveis: {list(df.columns)}"

        try:
            groups: dict[str, list[str]] = json.loads(keyword_groups)
        except json.JSONDecodeError as e:
            return (
                f"Erro ao interpretar keyword_groups: {e}\n"
                f"Formato esperado: {{\"nome_grupo\": [\"palavra1\", \"palavra2\"], ...}}"
            )

        if not isinstance(groups, dict) or not groups:
            return "keyword_groups deve ser um dicionário não vazio."

        created_columns = []

        for group_name, keywords in groups.items():
            if not keywords:
                continue
            col_name = f"qtd_{group_name}"
            pattern = "|".join(re.escape(kw.lower()) for kw in keywords)

            def _count_mentions(text, pat=pattern):
                if not isinstance(text, str):
                    return 0
                return len(re.findall(pat, text.lower()))

            df[col_name] = df[source_column].apply(_count_mentions)
            created_columns.append(col_name)

        _SESSION_DATA[session_id] = df

        if not created_columns:
            return "Nenhuma coluna criada. Verifique se os grupos de palavras-chave estão corretos."

        preview_cols = [source_column] + created_columns
        preview = df[preview_cols].head(3).to_string(index=False)

        totals = "\n".join(
            f"  {col}: {df[col].sum()} menções totais, {(df[col] > 0).sum()} registros com ao menos 1 menção"
            for col in created_columns
        )

        return (
            f"Quantificação concluída.\n"
            f"Session ID: {session_id}\n"
            f"Coluna fonte: '{source_column}'\n"
            f"Colunas criadas: {created_columns}\n\n"
            f"Totais por grupo:\n{totals}\n\n"
            f"Preview (primeiras 3 linhas):\n{preview}"
        )

