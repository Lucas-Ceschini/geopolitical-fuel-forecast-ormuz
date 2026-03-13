# Geopolitical Fuel Forecast: Previsão de Preço do Combustível sob Risco Geopolítico – Bloqueio do Estreito de Ormuz

## Descrição do Projeto
Este projeto analisa e prevê o preço do combustível (gasolina, diesel ou petróleo Brent) considerando **eventos geopolíticos críticos**, como o bloqueio do Estreito de Ormuz pelo Irã.  

O objetivo é **quantificar e modelar o impacto de fatores macroeconômicos e geopolíticos** sobre o preço do combustível, permitindo simular cenários futuros de risco.

O projeto combina:  
- Séries temporais de preços e indicadores econômicos  
- Regressão multivariada interpretável  
- Simulação de cenários geopolíticos para previsão de preços  

---

## Objetivos
- Modelar o preço do combustível com base em indicadores econômicos e sinais geopolíticos.  
- Prever preços futuros considerando cenários de risco (bloqueios ou tensões internacionais).  
- Demonstrar interpretação econômica e estatística, conectando dados financeiros, macroeconômicos e geopolíticos.

---

## Fontes dos Dados

| Tipo de dado | Fonte | Frequência | Observações |
|--------------|-------|------------|-------------|
| Preço do petróleo Brent | Investing.com, FRED, EIA | Diário ou mensal | Ajustado em USD; referência global do preço do combustível |
| Preço do combustível local | ANP (Brasil) / EIA (EUA) | Mensal | Preço médio, usado como variável dependente |
| Produção global de petróleo | OPEC, EIA | Mensal | Indicador da oferta global, influencia preços |
| Estoques estratégicos | EIA | Mensal | Reflete disponibilidade imediata de petróleo e derivados |
| Taxa de câmbio (USD/BRL) | Banco Central / Yahoo Finance | Mensal | Afeta preço importado do petróleo |
| Taxa SELIC | Banco Central | Mensal | Proxy para custo de capital e financiamento de combustíveis |
| Inflação (IPCA ou CPI) | IBGE / FRED | Mensal | Ajuste para efeito geral de preços na economia |
| Sinais geopolíticos | Fontes de notícias / índices de risco | Diário ou mensal | Dummy binária ou índice para bloqueio ou tensão no Estreito de Ormuz |

> Observação: Todos os dados são transformados para a **mesma frequência temporal** (mensal) para consistência na modelagem.

---

## Metodologia

A metodologia foi estruturada para maximizar **interpretabilidade, robustez e aplicabilidade prática**.

### 1. Pré-processamento
- Uniformização da frequência temporal: agregação de dados diários em médias mensais.  
- Transformação de preços e indicadores em **variação percentual**, permitindo interpretação econômica direta.  
- Criação de **dummy geopolítica**: 0 = período normal, 1 = bloqueio ou tensão alta.  
- Tratamento de valores ausentes via interpolação linear ou remoção de meses incompletos.  
- Análise exploratória: estatísticas descritivas, correlação e identificação de outliers.

### 2. Construção do Modelo
- **Regressão Linear Múltipla:**

ΔP_comb,t = β0 + β1 * ΔP_Brent,t + β2 * Dólar_t + β3 * SELIC_t + β4 * Inflação_t + β5 * Geopolítica_t + ε_t

- Regularização (Ridge ou Lasso) opcional para reduzir multicolinearidade e aumentar robustez.  
- Coeficientes interpretáveis: permitem entender **impacto percentual de cada variável**.

### 3. Validação do Modelo
- Divisão temporal: treino (70%) e teste (30%).  
- Métricas de avaliação: R², RMSE, MAE.  
- Análise de coeficientes e p-values para interpretação econômica.  

### 4. Previsão de Cenários
- Cenário normal: Geopolítica = 0  
- Cenário de risco moderado: Geopolítica = 0.5 (tensão alta)  
- Cenário de bloqueio total: Geopolítica = 1, com coeficiente ajustado historicamente  

### 5. Visualização e Interpretação
- Séries temporais: preço real x previsto  
- Heatmaps de correlação  
- Coeficientes do modelo com intervalos de confiança  
- Tabelas de previsão por cenário geopolítico  

---

## Tecnologias e Bibliotecas
- Python 3.x  
- Pandas, NumPy, Scikit-learn, Statsmodels  
- Matplotlib, Seaborn  
- yfinance, requests (para download de dados)  

---

## Resultados Esperados
- Modelo que explica a variação do preço do combustível com base em variáveis econômicas e geopolíticas.  
- Previsão de preços para cenários de risco, permitindo simular impactos de eventos raros.  
- Visualizações intuitivas e tabelas de interpretação para apresentação em portfólio.  

---

## Autor
**Lucas Augusto**  
- [LinkedIn](#)  
- [Portfólio GitHub](#)
