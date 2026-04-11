"""Project configuration for econometric modeling."""

from pathlib import Path

# Configurações de caminho para dados e diretórios de saída, além de listas de recursos base e de defasagem 
# para uso em modelos econométricos
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "fuel_dataset_prepared.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Lista de recursos base, incluindo volatilidades de índices econômicos, produção e estoque de petróleo, 
# risco geopolítico e um dummy para o Estreito de Ormuz
BASE_FEATURES = [
    "usd_index_std",
    "interest_rate_std",
    "oil_production_std",
    "oil_stock_std",
    "geopolitical_risk_std",
    "geo_risk_ma3_std",
    "ormuz_dummy",
]

# Lista de recursos de defasagem, incluindo defasagens de índices econômicos, risco geopolítico e o dummy do 
# Estreito de Ormuz, para capturar efeitos temporais nos modelos econométricos
LAG_FEATURES = [
    "usd_index_lag1",
    "usd_index_lag2",
    "interest_rate_lag1",
    "interest_rate_lag2",
    "geopolitical_risk_lag1",
    "geopolitical_risk_lag2",
    "ormuz_dummy_lag1",
]
