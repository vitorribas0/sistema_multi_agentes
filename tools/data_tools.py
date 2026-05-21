import os
import pandas as pd

# Cache de DataFrames por session_id.
# Permite encadear tools (load → filter → nlp) sem reler o arquivo.
_SESSION_DATA: dict[str, pd.DataFrame] = {}


def _validate_path(file_path: str) -> str:
    """Valida e normaliza o caminho do arquivo para evitar path traversal."""
    normalized = os.path.normpath(os.path.abspath(file_path))
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"Arquivo não encontrado: {normalized}")
    return normalized


def get_session_df(session_id: str) -> pd.DataFrame | None:
    """Retorna o DataFrame da sessão. Usado por outras tools (ex: nlp_tools)."""
    return _SESSION_DATA.get(session_id)


def register_tools(mcp):

    @mcp.tool(description="Carrega um arquivo CSV na sessão para uso encadeado com outras tools. Retorna as colunas e total de linhas.")
    def load_csv(file_path: str, session_id: str) -> str:
        """Carrega um CSV na sessão. Use o mesmo session_id nas tools seguintes."""
        try:
            path = _validate_path(file_path)
            df = pd.read_csv(path)
            _SESSION_DATA[session_id] = df
            return (
                f"CSV carregado na sessão '{session_id}'.\n"
                f"Linhas: {len(df)} | Colunas: {list(df.columns)}"
            )
        except Exception as e:
            return f"Erro ao carregar CSV: {e}"

    @mcp.tool(description="Lê um arquivo CSV e retorna as primeiras linhas como string, sem armazenar em sessão.")
    def read_csv(file_path: str) -> str:
        """Lê um arquivo CSV e retorna as primeiras linhas. Não armazena em sessão."""
        try:
            path = _validate_path(file_path)
            df = pd.read_csv(path)
            return df.head().to_string()
        except Exception as e:
            return f"Erro ao ler o arquivo CSV: {e}"

    @mcp.tool(description="Filtra o DataFrame da sessão por coluna e valor. O resultado substitui o DataFrame da sessão para uso nas próximas tools.")
    def filter_session(session_id: str, column: str, value: str) -> str:
        """Filtra o DataFrame da sessão. O filtro é aplicado sobre o estado atual."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_csv primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não existe. Colunas disponíveis: {list(df.columns)}"

        filtered = df[df[column].astype(str) == value]
        _SESSION_DATA[session_id] = filtered
        return (
            f"Filtro aplicado: {column} == '{value}'.\n"
            f"Linhas restantes: {len(filtered)}\n"
            f"{filtered.head().to_string()}"
        )

    @mcp.tool(description="Retorna estatísticas descritivas (count, mean, std, min, max) do DataFrame da sessão atual.")
    def describe_session(session_id: str) -> str:
        """Retorna o resumo estatístico do DataFrame atual da sessão."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_csv primeiro."
        return df.describe().to_string()

    @mcp.tool(description="Detecta anomalias (outliers) via IQR em uma coluna numérica do DataFrame da sessão atual.")
    def detect_anomalies(session_id: str, column: str) -> str:
        """Detecta outliers no DataFrame atual da sessão usando o método IQR."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_csv primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}"

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[column] < lower) | (df[column] > upper)]
        if outliers.empty:
            return f"Nenhuma anomalia detectada na coluna '{column}'."

        return (
            f"Anomalias na coluna '{column}' "
            f"(limite inferior: {lower:.2f}, limite superior: {upper:.2f}):\n"
            f"{outliers.to_string()}"
        )

    @mcp.tool(description="Extrai o conteúdo de uma coluna de texto do DataFrame da sessão e retorna como string única para uso em tools de NLP.")
    def get_text_from_session(session_id: str, column: str) -> str:
        """Extrai o conteúdo de uma coluna texto da sessão para uso em NLP."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_csv primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}"

        texts = df[column].dropna().astype(str).tolist()
        return "\n".join(texts)

    @mcp.tool(description="Remove a sessão da memória quando não for mais necessária.")
    def clear_session(session_id: str) -> str:
        """Libera o DataFrame da memória ao fim do fluxo."""
        if session_id in _SESSION_DATA:
            del _SESSION_DATA[session_id]
            return f"Sessão '{session_id}' removida da memória."
        return f"Sessão '{session_id}' não encontrada."
