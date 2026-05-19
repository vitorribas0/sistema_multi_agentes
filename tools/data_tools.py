import os
import pandas as pd


def _validate_path(file_path: str) -> str:
    """Valida e normaliza o caminho do arquivo para evitar path traversal."""
    normalized = os.path.normpath(os.path.abspath(file_path))
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"Arquivo não encontrado: {normalized}")
    return normalized


def register_tools(mcp):

    @mcp.tool(description="Lê um arquivo CSV e retorna as primeiras linhas como string.")
    def read_csv(file_path: str) -> str:
        """Lê um arquivo CSV do caminho especificado e retorna as primeiras linhas."""
        try:
            path = _validate_path(file_path)
            df = pd.read_csv(path)
            return df.head().to_string()
        except Exception as e:
            return f"Erro ao ler o arquivo CSV: {e}"

    @mcp.tool(description="Retorna estatísticas descritivas (count, mean, std, min, max) de um arquivo CSV.")
    def describe_csv(file_path: str) -> str:
        """Retorna o resumo estatístico de um arquivo CSV."""
        try:
            path = _validate_path(file_path)
            df = pd.read_csv(path)
            return df.describe().to_string()
        except Exception as e:
            return f"Erro ao descrever o arquivo CSV: {e}"

    @mcp.tool(
        description=(
            "Detecta anomalias (outliers) em uma coluna numérica de um arquivo CSV "
            "usando o método IQR (Interquartile Range). "
            "Retorna as linhas que são consideradas outliers."
        )
    )
    def detect_anomalies(file_path: str, column: str) -> str:
        """Detecta outliers em uma coluna numérica de um CSV usando IQR."""
        try:
            path = _validate_path(file_path)
            df = pd.read_csv(path)

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
                f"Anomalias detectadas na coluna '{column}' "
                f"(limite inferior: {lower:.2f}, limite superior: {upper:.2f}):\n"
                f"{outliers.to_string()}"
            )
        except Exception as e:
            return f"Erro ao detectar anomalias: {e}"
