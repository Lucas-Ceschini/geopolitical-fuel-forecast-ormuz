"""
O script de preparação de dados tem como objetivo transformar o dataset bruto gerado pelo data_download.py 
em uma base pronta para análise exploratória e modelagem econométrica, de forma organizada, robusta e reprodutível. 
Primeiramente, ele garante que as pastas de destino para salvar os arquivos processados existam, criando-as caso 
necessário, o que evita erros de caminho inexistente e mantém a estrutura do projeto consistente. Em seguida, o 
dataset final consolidado, que inclui preços do Brent, índice do dólar, taxa de juros do Fed, produção e estoques 
de petróleo, índice de risco geopolítico e dummy do Estreito de Ormuz, é carregado e a coluna de datas é convertida 
para o formato datetime, permitindo manipulações temporais precisas, como defasagens e resampling.

Uma das etapas centrais é a criação de variáveis defasadas (lags) para capturar efeitos temporais das variáveis 
explicativas sobre o preço do petróleo. Cada variável econômica ou geopolítica relevante recebe colunas com 
defasagens de 1, 2 e 3 meses, permitindo que a modelagem considere atrasos naturais na reação do mercado. Por exemplo, 
uma alta no dólar pode não impactar imediatamente o preço do Brent, mas refletir-se nos meses seguintes; essas 
defasagens ajudam a capturar esses efeitos de forma estruturada. Paralelamente, médias móveis são calculadas, 
principalmente para o índice de risco geopolítico, suavizando variações bruscas causadas por eventos extremos, como 
conflitos militares ou crises financeiras, evitando que picos isolados distorçam a regressão.

O script também realiza um tratamento de dados faltantes, verificando e preenchendo valores ausentes de forma 
inteligente: variáveis contínuas são interpoladas linearmente para preservar a tendência, enquanto dummies, que 
representam eventos binários, recebem zero quando ausentes. Esse cuidado garante que a base esteja completa, sem 
valores nulos, permitindo que modelos de regressão e séries temporais operem corretamente. Para tornar os 
coeficientes das regressões mais interpretáveis e comparáveis, as variáveis contínuas são padronizadas, 
transformando-as em escala com média zero e desvio padrão um. Essa padronização é crucial para modelos econométricos, 
especialmente quando variáveis de escalas diferentes, como dólar, juros e índice de risco, entram simultaneamente.

Ao final, todas essas transformações — criação de lags, médias móveis, padronização e preenchimento de 
faltantes — são consolidadas em um dataset final pronto para análise, que é salvo como fuel_dataset_prepared.csv. 
Este arquivo contém todas as informações necessárias para estudar o impacto de variáveis macroeconômicas e 
geopolíticas sobre o preço do petróleo, mantendo uma estrutura coerente para análise exploratória, modelagem 
econométrica e eventualmente previsão do efeito de eventos extremos, como bloqueios no Estreito de Ormuz, crises 
financeiras ou conflitos internacionais. Dessa forma, o script cria uma base sólida, permitindo que todas as etapas 
seguintes do estudo sejam reproduzíveis e confiáveis, mantendo boas práticas de ciência de dados e análise econômica 
aplicada.
"""

import os
import pandas as pd
import numpy as np

# -----------------------------
# 1. Configurações de pastas / Folder settings
# -----------------------------
"""
Define o caminho para a pasta data/processed onde o dataset final será salvo. Cria a pasta caso não exista 
(os.makedirs(..., exist_ok=True)). Garante que qualquer script que rode o projeto encontre as pastas corretas, 
evitando erros de FileNotFound. Permite organização profissional do projeto.
Exemplo: Se não existir ../data/processed, o script cria automaticamente para salvar o CSV final.
"""
script_dir = os.path.dirname(__file__)
processed_data_path = os.path.join(script_dir, "../data/processed")
os.makedirs(processed_data_path, exist_ok=True)

# -----------------------------
# 2. Carregar dataset final gerado pelo data_download.py
# Load dataset from data_download.py
# -----------------------------
"""
Lê o CSV fuel_dataset.csv, que já contém Brent, USD, juros, produção, estoque, índice geopolítico e dummy Ormuz.
Converte a coluna Date para datetime. Por que precisamos do dataset unificado como ponto de partida para criar 
variáveis derivadas. Converter datas é essencial para fazer resampling, defasagens e médias móveis corretamente.
"""
input_file = os.path.join(processed_data_path, "fuel_dataset.csv")
df = pd.read_csv(input_file, parse_dates=['Date'])
print(f"Dataset loaded successfully / Dataset carregado: {input_file}")

# -----------------------------
# 3. Criar variáveis defasadas / Lagged variables
# -----------------------------
"""
Objetivo / Purpose:
- Variáveis macroeconômicas e de risco muitas vezes afetam preços com defasagem.
- Criar lags permite capturar efeito temporal em regressões e modelos de séries temporais.
O código lê o CSV fuel_dataset.csv, que já contém Brent, USD, juros, produção, estoque, índice geopolítico e
dummy Ormuz. Converte a coluna Date para datetime. Precisamos do dataset unificado como ponto de partida para 
criar variáveis derivadas. Converter datas é essencial para fazer resampling, defasagens e médias móveis 
corretamente.
"""
lags = [1, 2, 3]  # lags de 1, 2 e 3 meses
columns_to_lag = ['usd_index', 'interest_rate', 'oil_production', 'oil_stock', 'geopolitical_risk', 'ormuz_dummy']
for col in columns_to_lag:
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# -----------------------------
# 4. Criar médias móveis / Moving averages
# -----------------------------
"""
Objetivo / Purpose:
- Suavizar variações mensais extremas e capturar tendência de curto prazo.
- Ex: média móvel de 3 meses do índice geopolítico.
O script cria colunas com defasagem de 1, 2 e 3 meses para variáveis explicativas. Por exemolo.: usd_index_lag1 
é o valor do mês anterior do dólar. Isso ocorre pois muitas variáveis econômicas não afetam o preço do Brent 
imediatamente. Ex.: um aumento do USD Index pode influenciar o preço do petróleo apenas no próximo mês. Incluir 
lags permite capturar esses efeitos temporais na regressão.
"""
df['geo_risk_ma3'] = df['geopolitical_risk'].rolling(window=3, min_periods=1).mean()

# -----------------------------
# 5. Checagem de dados faltantes / Missing values check
# -----------------------------
"""
Verifica se há valores faltantes (NaN) no dataset. Para variáveis contínuas, usa interpolação linear. Para 
dummies (0/1), preenche com 0. Modelos de regressão não funcionam com valores ausentes. Interpolação mantém 
série contínua e realista sem distorcer tendências. Dummies preenchidas com zero fazem sentido: ausência de 
evento → 0.
"""
missing = df.isnull().sum()
if missing.any():
    print("Missing values detected / Valores faltantes detectados:")
    print(missing[missing > 0])
    # Preenchimento simples (interpolação linear para séries contínuas)
    continuous_cols = ['brent_price', 'brent_log_return', 'usd_index', 'interest_rate',
                       'oil_production', 'oil_stock', 'geopolitical_risk', 'geo_risk_ma3']
    df[continuous_cols] = df[continuous_cols].interpolate(method='linear')
    # Dummies (Ormuz) preencher com zero
    df['ormuz_dummy'] = df['ormuz_dummy'].fillna(0)
    print("Missing values handled / Valores faltantes tratados.")
else:
    print("No missing values detected / Nenhum valor faltante.")

# -----------------------------
# 6. Normalização / Standardization
# -----------------------------
"""
Objetivo / Purpose:
- Colocar todas as variáveis contínuas em mesma escala para facilitar regressão
- Coeficientes se tornam comparáveis
- Dummy variables permanecem 0/1
O que é feito: Verifica se há valores faltantes (NaN) no dataset. Para variáveis contínuas, usa interpolação 
linear. Para dummies (0/1), preenche com 0. Isso ocorre pois os modelos de regressão não funcionam com valores 
ausentes. Interpolação mantém série contínua e realista sem distorcer tendências. Dummies preenchidas com zero f
azem sentido: ausência de evento → 0.
"""
continuous_cols = ['usd_index', 'interest_rate', 'oil_production', 'oil_stock', 'geopolitical_risk', 'geo_risk_ma3']
for col in continuous_cols:
    mean = df[col].mean()
    std = df[col].std()
    df[f"{col}_std"] = (df[col] - mean) / std

# -----------------------------
# 7. Salvar dataset preparado / Save prepared dataset
# -----------------------------
output_file = os.path.join(processed_data_path, "fuel_dataset_prepared.csv")
df.to_csv(output_file, index=False)
print(f"Prepared dataset saved / Dataset preparado salvo: {output_file}")