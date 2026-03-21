"""
Build Complete Dataset for Brent Oil Analysis
Construção do dataset completo para análise econométrica

Objetivo / Purpose:
- Baixar dados históricos do Brent Crude Oil (BZ=F)
- Calcular log-retorno
- Coletar variáveis explicativas: USD Index, produção de petróleo, estoques, juros, risco geopolítico
- Criar dummy de bloqueio Ormuz
- Alinhar todas as séries por data
- Salvar dataset final pronto para regressão

Author: Lucas Augusto
Date: 2026
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
load_dotenv()


# -----------------------------
# 1. Configurações de pastas / Folder settings
# -----------------------------
script_dir = os.path.dirname(__file__)
raw_data_path = os.path.join(script_dir, "../data/raw")
processed_data_path = os.path.join(script_dir, "../data/processed")
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)

# -----------------------------
# 2. Baixar Brent Crude Oil / Download Brent
# -----------------------------
""""
O petróleo Brent é um dos principais tipos de referência (benchmark) para a precificação de 
petróleo bruto no mundo, extraído no Mar do Norte. É classificado como leve e doce, o que 
facilita o refino para gasolina e diesel. Ele referencia cerca de 80% das negociações globais, 
especialmente na Europa, África e Oriente Médio. 
"""
ticker = "BZ=F"  # Brent Futures
start_date = "2000-01-01"
print("Downloading Brent data / Baixando dados do Brent...")
brent = yf.download(ticker, start=start_date, progress=False)

# Limpeza de MultiIndex / Clean MultiIndex
brent.columns = brent.columns.get_level_values(0)

# Selecionar Close / Select Close price
brent = brent[['Close']].rename(columns={'Close': 'brent_price'})
brent = brent.reset_index()

# Salvar dados brutos / Save raw
raw_file = os.path.join(raw_data_path, "brent_prices_raw.csv")
brent.to_csv(raw_file, index=False)
print(f"Raw data saved / Dados brutos salvos: {raw_file}")

# -----------------------------
# 3. Converter para frequência mensal / Monthly resample
# -----------------------------
brent['Date'] = pd.to_datetime(brent['Date'])
brent = brent.set_index('Date')
brent_monthly = brent.resample('ME').mean()  # ME = month end
brent_monthly = brent_monthly.reset_index()

# -----------------------------
# 4. Calcular log-retorno / Log return
# -----------------------------
"""
O uso de log-retornos em vez de preços absolutos é uma prática fundamental em séries temporais financeiras, 
pois resolve uma série de problemas estatísticos e melhora significativamente a qualidade da modelagem. 
O log-retorno é definido como o logaritmo natural da razão entre o preço no período atual e o preço no 
período anterior, representando, na prática, uma variação percentual em escala logarítmica. A principal 
razão para essa transformação é que séries de preços, como o Brent, geralmente são não estacionárias, 
apresentando tendência ao longo do tempo e variância instável, o que pode levar a regressões espúrias e 
resultados estatisticamente inválidos. Ao transformar os preços em log-retornos, a série tende a se tornar 
aproximadamente estacionária, oscilando em torno de uma média mais estável, o que é um pré-requisito importante 
para a aplicação de modelos de regressão.

Além disso, os log-retornos possuem uma interpretação econômica direta, pois valores pequenos podem ser 
interpretados como variações percentuais — por exemplo, 0,02 representa aproximadamente um aumento de 2%, 
enquanto -0,01 indica uma queda de 1%. Outra vantagem importante é a propriedade aditiva dos log-retornos ao 
longo do tempo, o que significa que o retorno acumulado em múltiplos períodos é simplesmente a soma dos retornos 
individuais, facilitando análises temporais e modelagens dinâmicas. Do ponto de vista estatístico, os log-retornos 
também tendem a apresentar comportamento mais próximo da normalidade, com variância mais estável e menor presença 
de heterocedasticidade, o que melhora a robustez das inferências, como testes de hipótese e significância dos 
coeficientes.

No contexto deste projeto, que busca entender o impacto de variáveis macroeconômicas e geopolíticas sobre o preço 
do petróleo, o uso de log-retornos é essencial, pois permite modelar diretamente como mudanças no dólar, nas taxas 
de juros ou em eventos geopolíticos afetam as variações do preço do Brent, em vez de tentar explicar o nível absoluto 
de preços, que é influenciado por tendências de longo prazo. Dessa forma, a modelagem se torna mais consistente, 
interpretável e alinhada com as melhores práticas adotadas tanto no mercado financeiro quanto na literatura acadêmica.
"""

brent_monthly['brent_log_return'] = np.log(brent_monthly['brent_price'] / brent_monthly['brent_price'].shift(1))

"""
Saída esperada:

   Month  Brent_Price  Log_Return
0    Jan       75.00         NaN
1    Feb       76.50    0.0197  # aumento de ~1,97%
2    Mar       74.85   -0.0213  # queda de ~2,13%
3    Apr       77.00    0.0288  # aumento de ~2,88%

Observações:
1. O preço absoluto possui tendência e nível, o que dificulta análise direta de variações.
2. O log-retorno centra a série em torno de 0 e transforma a análise em variações percentuais.
3. Isso torna a regressão mais robusta e os coeficientes interpretáveis como impacto percentual 
   de variáveis explicativas.
"""
# Salvar dados processados Brent / Save processed Brent
processed_brent_file = os.path.join(processed_data_path, "brent_monthly.csv")
brent_monthly.to_csv(processed_brent_file, index=False)
print(f"Processed Brent saved / Brent processado salvo: {processed_brent_file}")

# -----------------------------
# 5. Coletar USD Index / USD Index
# -----------------------------
print("Downloading USD Index / Baixando USD Index...")
usd = yf.download('DX-Y.NYB', start=start_date, progress=False)
usd.columns = usd.columns.get_level_values(0)
usd = usd[['Close']].rename(columns={'Close': 'usd_index'})
usd.index = pd.to_datetime(usd.index)
usd_monthly = usd.resample('ME').mean().reset_index()

# -----------------------------
# 6. Coletar taxa de juros / Interest Rate (US Fed Funds Rate)
# -----------------------------
fred_api_key = os.getenv('API_FRED_KEY')  # substituir pela sua chave FRED
fred = Fred(api_key=fred_api_key)
print("Downloading Interest Rate / Baixando taxa de juros...")
interest = fred.get_series('FEDFUNDS', observation_start=start_date)
interest = interest.resample('ME').mean().reset_index()
interest.columns = ['Date', 'interest_rate']

# -----------------------------
# 7. Produção global de petróleo / Oil production (dummy)
# -----------------------------
"""
Objetivo:
Criar uma variável representando a produção de petróleo, que será usada como variável explicativa em 
regressões sobre o preço do Brent. Essa variável serve para testar o modelo e garantir que a estrutura 
da análise funcione mesmo sem dados reais completos.

Por que usar:
1. Produção de petróleo é determinante do preço (oferta).
2. Dados reais podem ser difíceis de obter em alta frequência.
3. Permite simular impactos de choques e eventos geopolíticos.

Teoria por trás:
Em modelos econométricos, o preço do petróleo pode ser visto como:
    P_t = f(Demanda_t, Oferta_t, Dólar_t, Juros_t, Geopolítica_t) + ε_t
Onde Oferta_t pode ser representada pela produção de petróleo. 
Criar uma variável dummy ou contínua permite testar regressões mesmo sem dados reais. A variável pode 
ser contínua (ex: produção em milhões de barris) ou dummy (0/1 para eventos específicos).
"""

print("Creating dummy oil production / Criando dummy produção petróleo...")
dates = pd.date_range(start=brent_monthly['Date'].min(), end=brent_monthly['Date'].max(), freq='ME')
oil_prod = pd.DataFrame({'Date': dates, 'oil_production': np.random.normal(100,5,len(dates))})

"""
Geramos uma série de produção mensal simulada para fins de teste
usando uma distribuição normal:
- média (m) = 100 → valor central de referência da produção
- desvio padrão (s) = 5 → pequenas flutuações mensais realistas
A variável serve como placeholder até que dados reais sejam utilizados.
"""

# -----------------------------
# 8. Estoques globais de petróleo / Oil stock (dummy)
# -----------------------------
print("Creating dummy oil stock / Criando dummy estoques petróleo...")
oil_stock = pd.DataFrame({'Date': dates, 'oil_stock': np.random.normal(50,3,len(dates))})

# -----------------------------
# 9. Índice de risco geopolítico / Geopolitical Risk (dummy)
# -----------------------------
print("Creating dummy geopolitical risk / Criando dummy risco geopolítico...")
# Criar datas mensais
dates = pd.date_range(start='2007-07-31', end='2026-03-31', freq='ME')
geo_risk = pd.DataFrame({'Date': dates})
# Inicializar com zeros
geo_risk['geopolitical_risk'] = 0.0

# Eventos discretos e períodos longos
events_discrete = {
    '2008-09': 3.0, '2008-10': 3.0,                  # Crise Financeira Global
    '2011-02': 2.5, '2011-03': 2.5,                  # Primavera Árabe
    '2014-03': 3.0, '2014-04': 3.0,                  # Crise Ucrânia / Rússia
    '2019-01': 1.8, '2019-05': 1.8,                  # Tensões Irã/EUA
    '2021-05': 2.0, '2021-06': 2.0,                  # Conflito Israel-Palestina
    '2022-02': 4.0, '2022-03': 4.0,                  # Início Guerra Rússia-Ucrânia
    '2023-10': 3.0, '2023-11': 3.0,                  # Conflito Israel-Hamas
}

# Pandemia COVID-19: 2020-03 até 2021-12
pandemia_periods = pd.date_range(start='2020-03-31', end='2021-12-31', freq='ME')
for date in pandemia_periods:
    geo_risk.loc[geo_risk['Date'] == date, 'geopolitical_risk'] = 3.5
# Guerra comercial EUA-China: 2018-07 até 2019-12
guerra_comercial_periods = pd.date_range(start='2018-07-31', end='2019-12-31', freq='ME')
for date in guerra_comercial_periods:
    geo_risk.loc[geo_risk['Date'] == date, 'geopolitical_risk'] = 2.0
# Conflito Irã-US/Israel e bloqueio do Estreito de Ormuz (fev 2026 a abr 2026)
conflict_2026 = pd.date_range(start='2026-02-28', end='2026-04-30', freq='ME')
for date in conflict_2026:
    geo_risk.loc[geo_risk['Date'] == date, 'geopolitical_risk'] = 5.0
# Aplicar eventos discretos
for period, risk_value in events_discrete.items():
    mask = geo_risk['Date'].dt.to_period('M') == period
    geo_risk.loc[mask, 'geopolitical_risk'] = risk_value
# Suavização para evitar saltos bruscos
geo_risk['geopolitical_risk'] = geo_risk['geopolitical_risk'].rolling(window=3, min_periods=1).mean()
# Normalização para regressão
geo_risk['geopolitical_risk'] = (
    geo_risk['geopolitical_risk'] - geo_risk['geopolitical_risk'].mean()
) / geo_risk['geopolitical_risk'].std()

print("Geopolitical risk index created successfully / Índice de risco geopolítico criado com sucesso.")

# -----------------------------
# 10. Dummy bloqueio Estreito de Ormuz / Ormuz dummy
# -----------------------------
"""
1 se houver bloqueio ou alto risco geopolítico relacionado ao Estreito de Ormuz, 0 caso contrário.
Para o estudo, consideramos o período de fevereiro a abril de 2026
"""
ormuz_dummy = pd.DataFrame({'Date': dates})
# Criar dummy: 1 durante o bloqueio, 0 caso contrário
ormuz_dummy['ormuz_dummy'] = ormuz_dummy['Date'].apply(
    lambda x: 1 if pd.Timestamp('2026-02-28') <= x <= pd.Timestamp('2026-04-30') else 0)
print("Ormuz dummy created successfully / Dummy Ormuz criado com sucesso.")

# -----------------------------
# 11. Merge de todas as séries / Merge all series
# -----------------------------
print("Merging datasets / Unindo datasets...")
df = brent_monthly.merge(usd_monthly, on='Date', how='left')
df = df.merge(interest, on='Date', how='left')
df = df.merge(oil_prod, on='Date', how='left')
df = df.merge(oil_stock, on='Date', how='left')
df = df.merge(geo_risk, on='Date', how='left')
df = df.merge(ormuz_dummy, on='Date', how='left')

# -----------------------------
# 12. Salvar dataset final / Save final dataset
# -----------------------------
final_file = os.path.join(processed_data_path, "fuel_dataset.csv")
df.to_csv(final_file, index=False)
print(f"Dataset built successfully / Dataset criado com sucesso: {final_file}")