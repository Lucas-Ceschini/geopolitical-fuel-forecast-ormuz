"""Data loading and temporal split helpers."""

from __future__ import annotations 

import pandas as pd

from scripts.modeling.config import DATA_PATH

# Função para carregar o dataset de combustível, lendo um arquivo CSV, convertendo a coluna "Date" para formato de 
# data, ordenando por data, removendo linhas com valores nulos na coluna "brent_log_return", e criando uma nova 
# coluna "target_pct" que representa o retorno logarítmico do Brent multiplicado por 100 para facilitar a interpretação 
# como porcentagem
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True) 
    df = df.dropna(subset=["brent_log_return"]).copy() # Remover linhas com valores nulos na coluna "brent_log_return" para garantir que o modelo de regressão tenha dados completos para a variável alvo
    df["target_pct"] = df["brent_log_return"] * 100 # Criar uma nova coluna "target_pct" que representa o retorno logarítmico do Brent multiplicado por 100 para facilitar a interpretação como porcentagem, tornando os coeficientes do modelo mais intuitivos em termos de impacto percentual das variáveis independentes sobre o preço do combustível
    return df

# Função para realizar uma divisão temporal dos dados em conjuntos de treinamento e teste, com base em uma proporção 
# de treinamento especificada, retornando dois DataFrames separados para treinamento e teste
def train_test_split_temporal(
    df: pd.DataFrame, train_ratio: float = 0.7 # Proporção de dados a serem usados para treinamento, onde o restante será usado para teste
) -> tuple[pd.DataFrame, pd.DataFrame]: # Realizar uma divisão temporal dos dados em conjuntos de treinamento e teste, com base em uma proporção de treinamento especificada, retornando dois DataFrames separados para treinamento e teste
    split_idx = int(len(df) * train_ratio) # Calcular o índice de divisão com base na proporção de treinamento, garantindo que os dados sejam divididos de forma temporal, onde os dados mais antigos são usados para treinamento e os mais recentes para teste, evitando vazamento de dados futuros para o modelo
    train_df = df.iloc[:split_idx].copy() # Criar o DataFrame de treinamento com os dados mais antigos, garantindo que o modelo seja treinado com informações históricas e avaliado em dados futuros para simular um cenário real de previsão
    test_df = df.iloc[split_idx:].copy() # Criar o DataFrame de teste com os dados mais recentes, garantindo que a avaliação do modelo seja feita em um conjunto de dados que simula um cenário real de previsão, onde o modelo é testado em informações futuras que não estavam disponíveis durante o treinamento
    return train_df, test_df
