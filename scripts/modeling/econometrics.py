"""Econometric helpers used by the forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def durbin_watson(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    return float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))


def ols_inference(
    x_values: pd.DataFrame, y_values: pd.Series, feature_names: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_design = np.column_stack([np.ones(len(x_values)), x_values])

    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    beta = xtx_inv @ (x_design.T @ y_values)
    fitted = x_design @ beta
    residuals = y_values - fitted

    n_obs, n_params = x_design.shape
    sigma2 = (residuals @ residuals) / max(n_obs - n_params, 1)
    cov = sigma2 * xtx_inv
    std_err = np.sqrt(np.diag(cov))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err != 0)

    degrees_freedom = max(n_obs - n_params, 1)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=degrees_freedom))
    critical_value = stats.t.ppf(0.975, df=degrees_freedom)
    ci_low = beta - critical_value * std_err
    ci_high = beta + critical_value * std_err

    summary = pd.DataFrame(
        {
            "variable": ["const"] + feature_names,
            "coef": beta,
            "std_err": std_err,
            "t_stat": t_stats,
            "p_value": p_values,
            "ci_2.5%": ci_low,
            "ci_97.5%": ci_high,
        }
    )
    return summary, residuals, fitted


def build_diagnostics(residuals: np.ndarray, fitted: np.ndarray) -> pd.DataFrame:
    jb_result = stats.jarque_bera(residuals)
    return pd.DataFrame(
        {
            "metric": [
                "Durbin-Watson",
                "Jarque-Bera stat",
                "Jarque-Bera p-value",
                "Corr(abs(residual), fitted)",
            ],
            "value": [
                durbin_watson(residuals),
                float(jb_result.statistic),
                float(jb_result.pvalue),
                float(np.corrcoef(np.abs(residuals), fitted)[0, 1]),
            ],
        }
    )
