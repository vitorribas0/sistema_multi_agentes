from langchain_core.tools import tool
import pandas as pd

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
def read_csv(file_path: str) -> str:
    """Lê um arquivo CSV do caminho especificado e retorna as primeiras linhas como string.
    Args:
        file_path (str): Caminho completo para o arquivo CSV.

    Returns:
        str: As primeiras linhas do arquivo CSV como string.
    """
    try:
        df = pd.read_csv(file_path)
        return df.head().to_string()
    except Exception as e:
        return f"Erro ao ler o arquivo CSV: {e}"