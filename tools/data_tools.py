import os
import re
import json
from pathlib import Path
import pandas as pd

# Cache de DataFrames por session_id.
# Permite encadear tools (load_file → filter_session → nlp) sem reler o arquivo.
_SESSION_DATA: dict[str, pd.DataFrame] = {}

_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Limite de caracteres para outputs de tools — evita estouro de TPM no LLM
_MAX_OUTPUT_CHARS = 2000


def _truncate(text: str) -> str:
    """Trunca o output de uma tool se ultrapassar _MAX_OUTPUT_CHARS."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n... [output truncado — {len(text)} chars totais]"


def _validate_path(file_path: str) -> str:
    """Valida e normaliza o caminho do arquivo para evitar path traversal."""
    normalized = os.path.normpath(os.path.abspath(file_path))
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"Arquivo não encontrado: {normalized}")
    ext = os.path.splitext(normalized)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato '{ext}' não suportado. Use: {', '.join(_SUPPORTED_EXTENSIONS)}"
        )
    return normalized


def _read_file_to_df(path: str) -> pd.DataFrame:
    """Lê CSV ou Excel e retorna um DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _build_file_summary(df: pd.DataFrame, session_id: str, file_path: str) -> str:
    """Gera o resumo padronizado que o LLM recebe após carregar qualquer arquivo."""
    schema_lines = "\n".join(
        f"  {col}: {dtype}" for col, dtype in df.dtypes.items()
    )
    preview = df.head(3).to_string(index=False)
    return (
        f"Arquivo carregado na sessão '{session_id}': {os.path.basename(file_path)}\n"
        f"Total de linhas: {len(df)} | Total de colunas: {len(df.columns)}\n\n"
        f"Schema (coluna: tipo):\n{schema_lines}\n\n"
        f"Primeiras 3 linhas:\n{preview}"
    )


def get_session_df(session_id: str) -> pd.DataFrame | None:
    """Retorna o DataFrame da sessão. Usado por outras tools (ex: nlp_tools)."""
    return _SESSION_DATA.get(session_id)


def register_tools(mcp):

    @mcp.tool(
        description=(
            "Carrega um arquivo CSV ou Excel (.csv, .xlsx, .xls) na sessão identificada por session_id. "
            "Retorna o schema completo (colunas e tipos), as 3 primeiras linhas e o total de registros. "
            "Use o mesmo session_id em todas as tools seguintes (filter_session, describe_session, etc.)."
        )
    )
    def load_file(file_path: str, session_id: str) -> str:
        """Carrega CSV ou Excel na sessão. Retorna schema + 3 linhas de preview."""
        try:
            path = _validate_path(file_path)
            df = _read_file_to_df(path)
            _SESSION_DATA[session_id] = df
            return _build_file_summary(df, session_id, path)
        except Exception as e:
            return f"Erro ao carregar arquivo: {e}"

    @mcp.tool(
        description=(
            "Verifica se uma sessão já possui dados carregados. "
            "Retorna o schema (colunas e tipos) e o total de registros se a sessão existir, "
            "ou uma mensagem informando que a sessão não existe. "
            "Use esta tool ANTES de load_file para evitar recarregamentos desnecessários."
        )
    )
    def get_session_info(session_id: str) -> str:
        """Retorna informações de uma sessão existente ou avisa que não existe."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' NÃO existe. É necessário chamar load_file antes."
        schema_lines = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
        return (
            f"Sessão '{session_id}' JÁ EXISTE com dados carregados.\n"
            f"Total de linhas: {len(df)} | Total de colunas: {len(df.columns)}\n\n"
            f"Schema (coluna: tipo):\n{schema_lines}\n\n"
            f"Prossiga diretamente com a operação solicitada — NÃO chame load_file novamente."
        )

    @mcp.tool(description="Filtra o DataFrame da sessão por coluna e valor. O resultado substitui o DataFrame da sessão para uso nas próximas tools.")
    def filter_session(session_id: str, column: str, value: str) -> str:
        """Filtra o DataFrame da sessão. O filtro é aplicado sobre o estado atual."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não existe. Colunas disponíveis: {list(df.columns)}"

        filtered = df[df[column].astype(str) == value]
        _SESSION_DATA[session_id] = filtered
        return _truncate(
            f"Filtro aplicado: {column} == '{value}'.\n"
            f"Linhas restantes: {len(filtered)}\n"
            f"{filtered.head(3).to_string()}"
        )

    @mcp.tool(
        description=(
            "Busca registros no DataFrame da sessão onde uma coluna de texto contenha "
            "qualquer um dos termos informados (busca por substring, case-insensitive). "
            "NÃO substitui a sessão — retorna apenas o resultado da busca como preview. "
            "O parâmetro 'terms' deve ser uma lista JSON de strings. "
            "Exemplo: '[\"ajuda\", \"preciso\", \"necessito\"]'. "
            "Retorna o total de ocorrências e as primeiras 10 linhas encontradas."
        )
    )
    def search_session(session_id: str, column: str, terms: str) -> str:
        """Busca registros que contêm qualquer dos termos na coluna especificada. Não modifica a sessão."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não existe. Colunas disponíveis: {list(df.columns)}"

        try:
            term_list: list[str] = json.loads(terms)
        except json.JSONDecodeError:
            return f"Parâmetro 'terms' inválido. Forneça uma lista JSON. Exemplo: '[\"ajuda\", \"preciso\"]'"

        if not term_list:
            return "A lista de termos está vazia."

        pattern = "|".join(re.escape(t.lower()) for t in term_list)
        mask = df[column].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
        result = df[mask]

        if result.empty:
            return (
                f"Nenhum registro encontrado em '{column}' com os termos: {term_list}.\n"
                f"Sugestão: tente termos mais curtos ou radicais (ex: 'ajud' em vez de 'ajuda')."
            )

        preview = result.head(5).to_string(index=False)
        return _truncate(
            f"Busca em '{column}' | Termos: {term_list}\n"
            f"Total encontrado: {len(result)} de {len(df)} registros ({len(result)/len(df)*100:.1f}%)\n\n"
            f"Primeiras 5 ocorrências:\n{preview}"
        )

    @mcp.tool(description="Retorna estatísticas descritivas (count, mean, std, min, max) do DataFrame da sessão atual.")
    def describe_session(session_id: str) -> str:
        """Retorna o resumo estatístico do DataFrame atual da sessão."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        return _truncate(df.describe().to_string())

    @mcp.tool(description="Detecta anomalias (outliers) via IQR em uma coluna numérica do DataFrame da sessão atual.")
    def detect_anomalies(session_id: str, column: str) -> str:
        """Detecta outliers no DataFrame atual da sessão usando o método IQR."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
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

        return _truncate(
            f"Anomalias na coluna '{column}' "
            f"(limite inferior: {lower:.2f}, limite superior: {upper:.2f}):\n"
            f"{outliers.head(10).to_string()}"
        )

    @mcp.tool(description="Extrai o conteúdo de uma coluna de texto do DataFrame da sessão e retorna como string única para uso em tools de NLP.")
    def get_text_from_session(session_id: str, column: str) -> str:
        """Extrai o conteúdo de uma coluna texto da sessão para uso em NLP."""
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."
        if column not in df.columns:
            return f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}"

        texts = df[column].dropna().astype(str).tolist()
        return _truncate("\n".join(texts))

    @mcp.tool(description="Remove a sessão da memória quando não for mais necessária.")
    def clear_session(session_id: str) -> str:
        """Libera o DataFrame da memória ao fim do fluxo."""
        if session_id in _SESSION_DATA:
            del _SESSION_DATA[session_id]
            return f"Sessão '{session_id}' removida da memória."
        return f"Sessão '{session_id}' não encontrada."

    @mcp.tool(
        description=(
            "Exporta o DataFrame atual da sessão para CSV ou Excel. "
            "Use quando o usuário pedir para baixar, exportar ou salvar o resultado. "
            "O arquivo é salvo na pasta Downloads do usuário. "
            "format deve ser 'csv' ou 'excel'."
        )
    )
    def export_session(session_id: str, format: str = "csv", filename: str = "") -> str:
        """Exporta o DataFrame da sessão para CSV ou Excel na pasta Downloads."""
        import re as _re
        df = _SESSION_DATA.get(session_id)
        if df is None:
            return f"Sessão '{session_id}' não encontrada. Use load_file primeiro."

        fmt = format.strip().lower()
        if fmt not in ("csv", "excel", "xlsx"):
            return "Formato inválido. Use 'csv' ou 'excel'."

        # Nome do arquivo: usa o fornecido ou gera um padrão seguro
        safe_name = _re.sub(r"[^\w\-]", "_", filename) if filename else session_id
        downloads = Path.home() / "Downloads"
        downloads.mkdir(exist_ok=True)

        if fmt == "csv":
            out_path = downloads / f"{safe_name}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
        else:
            out_path = downloads / f"{safe_name}.xlsx"
            df.to_excel(out_path, index=False)

        return (
            f"Arquivo exportado com sucesso!\n"
            f"Caminho: {out_path}\n"
            f"Linhas: {len(df)} | Colunas: {len(df.columns)}"
        )
