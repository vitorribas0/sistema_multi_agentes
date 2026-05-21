from langchain_core.tools import tool
import pandas as pd
import os

CARS = ['Marea', 'Corsa', 'Uno', 'Gol']

@tool
def list_cars() -> list:
    """Lista os carros disponíveis."""
    return CARS


@tool
def add_car(car_name: str) -> list:
    """Adiciona um carro à lista de carros disponíveis."""
    CARS.append(car_name)
    return CARS


@tool
def load_file(file_path: str) -> str:
    """Lê um arquivo CSV ou Excel (.csv, .xlsx, .xls) e retorna o schema, as 3 primeiras linhas e o total de registros.
    Args:
        file_path (str): Caminho completo para o arquivo.

    Returns:
        str: Schema (colunas e tipos), 3 primeiras linhas e total de registros.
    """
    try:
        normalized = os.path.normpath(os.path.abspath(file_path))
        if not os.path.exists(normalized):
            return f"Arquivo não encontrado: {normalized}"
        ext = os.path.splitext(normalized)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(normalized)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(normalized)
        else:
            return f"Formato '{ext}' não suportado. Use .csv, .xlsx ou .xls."
        schema_lines = "\n".join(f"  {col}: {dtype}" for col, dtype in df.dtypes.items())
        preview = df.head(3).to_string(index=False)
        return (
            f"Arquivo: {os.path.basename(normalized)}\n"
            f"Total de linhas: {len(df)} | Total de colunas: {len(df.columns)}\n\n"
            f"Schema (coluna: tipo):\n{schema_lines}\n\n"
            f"Primeiras 3 linhas:\n{preview}"
        )
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"
