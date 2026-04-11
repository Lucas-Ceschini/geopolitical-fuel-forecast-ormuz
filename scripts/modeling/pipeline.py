"""End-to-end modeling pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.modeling.config import BASE_FEATURES, LAG_FEATURES, OUTPUT_DIR
from scripts.modeling.data import load_dataset, train_test_split_temporal
from scripts.modeling.econometrics import build_diagnostics, ols_inference, regression_metrics


def build_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    x_train_base = train_df[BASE_FEATURES]
    x_test_base = test_df[BASE_FEATURES]
    y_train_base = train_df["brent_log_return"]
    y_test_base = test_df["brent_log_return"]

    full_features = BASE_FEATURES + LAG_FEATURES
    x_train_lag = train_df[full_features].dropna()
    x_test_lag = test_df[full_features].dropna()
    y_train_lag = train_df.loc[x_train_lag.index, "brent_log_return"]
    y_test_lag = test_df.loc[x_test_lag.index, "brent_log_return"]

    ols_base = LinearRegression().fit(x_train_base, y_train_base)
    ols_lag = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ).fit(x_train_lag, y_train_lag)
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ).fit(x_train_lag, y_train_lag)
    lasso = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000)),
        ]
    ).fit(x_train_lag, y_train_lag)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "x_train_base": x_train_base,
        "x_test_base": x_test_base,
        "y_train_base": y_train_base,
        "y_test_base": y_test_base,
        "x_train_lag": x_train_lag,
        "x_test_lag": x_test_lag,
        "y_train_lag": y_train_lag,
        "y_test_lag": y_test_lag,
        "ols_base": ols_base,
        "ols_lag": ols_lag,
        "ridge": ridge,
        "lasso": lasso,
    }


def evaluate_models(model_bundle: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = {
        "OLS contemporâneo": {
            "treino": model_bundle["ols_base"].predict(model_bundle["x_train_base"]),
            "teste": model_bundle["ols_base"].predict(model_bundle["x_test_base"]),
        },
        "OLS com lags": {
            "treino": model_bundle["ols_lag"].predict(model_bundle["x_train_lag"]),
            "teste": model_bundle["ols_lag"].predict(model_bundle["x_test_lag"]),
        },
        "Ridge": {
            "teste": model_bundle["ridge"].predict(model_bundle["x_test_lag"]),
        },
        "Lasso": {
            "teste": model_bundle["lasso"].predict(model_bundle["x_test_lag"]),
        },
    }

    metrics_rows = [
        {
            "modelo": "OLS contemporâneo",
            "amostra": "treino",
            **regression_metrics(model_bundle["y_train_base"], predictions["OLS contemporâneo"]["treino"]),
        },
        {
            "modelo": "OLS contemporâneo",
            "amostra": "teste",
            **regression_metrics(model_bundle["y_test_base"], predictions["OLS contemporâneo"]["teste"]),
        },
        {
            "modelo": "OLS com lags",
            "amostra": "treino",
            **regression_metrics(model_bundle["y_train_lag"], predictions["OLS com lags"]["treino"]),
        },
        {
            "modelo": "OLS com lags",
            "amostra": "teste",
            **regression_metrics(model_bundle["y_test_lag"], predictions["OLS com lags"]["teste"]),
        },
        {
            "modelo": "Ridge",
            "amostra": "teste",
            **regression_metrics(model_bundle["y_test_lag"], predictions["Ridge"]["teste"]),
        },
        {
            "modelo": "Lasso",
            "amostra": "teste",
            **regression_metrics(model_bundle["y_test_lag"], predictions["Lasso"]["teste"]),
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["amostra", "RMSE"]).reset_index(drop=True)

    predictions_df = pd.DataFrame(
        {
            "Date": model_bundle["test_df"].loc[model_bundle["x_test_base"].index, "Date"],
            "real_log_return": model_bundle["y_test_base"],
            "pred_ols_base": predictions["OLS contemporâneo"]["teste"],
        }
    )

    lag_predictions_df = pd.DataFrame(
        {
            "Date": model_bundle["test_df"].loc[model_bundle["x_test_lag"].index, "Date"],
            "pred_ols_lag": predictions["OLS com lags"]["teste"],
            "pred_ridge": predictions["Ridge"]["teste"],
            "pred_lasso": predictions["Lasso"]["teste"],
        }
    )

    predictions_df = predictions_df.merge(lag_predictions_df, on="Date", how="left")
    return metrics_df, predictions_df


def simulate_scenarios(df: pd.DataFrame, model_bundle: dict[str, object]) -> pd.DataFrame:
    latest = df.iloc[-1].copy()
    scenarios = {
        "Normalização geopolítica": {
            "geopolitical_risk_std": df["geopolitical_risk_std"].quantile(0.25),
            "geo_risk_ma3_std": df["geo_risk_ma3_std"].quantile(0.25),
            "ormuz_dummy": 0,
        },
        "Tensão regional elevada": {
            "geopolitical_risk_std": df["geopolitical_risk_std"].quantile(0.75),
            "geo_risk_ma3_std": df["geo_risk_ma3_std"].quantile(0.75),
            "ormuz_dummy": 0,
        },
        "Bloqueio severo de Ormuz": {
            "geopolitical_risk_std": df["geopolitical_risk_std"].max(),
            "geo_risk_ma3_std": df["geo_risk_ma3_std"].max(),
            "ormuz_dummy": 1,
        },
    }

    rows = []
    for scenario_name, updates in scenarios.items():
        row = latest[BASE_FEATURES].copy()
        for feature, value in updates.items():
            row[feature] = value
        prediction = model_bundle["ols_base"].predict(pd.DataFrame([row]))[0]
        rows.append(
            {
                "cenario": scenario_name,
                "retorno_previsto_log": prediction,
                "retorno_previsto_pct_aprox": prediction * 100,
            }
        )

    return pd.DataFrame(rows).sort_values("retorno_previsto_pct_aprox").reset_index(drop=True)


def save_outputs(
    metrics_df: pd.DataFrame,
    coefficients_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    coefficients_df.to_csv(OUTPUT_DIR / "ols_coefficients.csv", index=False)
    diagnostics_df.to_csv(OUTPUT_DIR / "model_diagnostics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    scenarios_df.to_csv(OUTPUT_DIR / "scenario_forecasts.csv", index=False)


def run_forecast_pipeline() -> dict[str, pd.DataFrame]:
    df = load_dataset()
    train_df, test_df = train_test_split_temporal(df)
    model_bundle = build_models(train_df, test_df)

    metrics_df, predictions_df = evaluate_models(model_bundle)
    coefficients_df, residuals, fitted = ols_inference(
        model_bundle["x_train_base"],
        model_bundle["y_train_base"],
        BASE_FEATURES,
    )
    diagnostics_df = build_diagnostics(residuals, fitted)
    scenarios_df = simulate_scenarios(df, model_bundle)
    save_outputs(metrics_df, coefficients_df, diagnostics_df, predictions_df, scenarios_df)

    return {
        "dataset": df,
        "train_df": train_df,
        "test_df": test_df,
        "metrics": metrics_df,
        "coefficients": coefficients_df,
        "diagnostics": diagnostics_df,
        "predictions": predictions_df,
        "scenarios": scenarios_df,
    }
