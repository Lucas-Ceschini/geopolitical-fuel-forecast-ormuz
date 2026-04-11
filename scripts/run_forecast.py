"""CLI entrypoint for the forecasting pipeline."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.modeling.pipeline import run_forecast_pipeline


def main() -> None:
    results = run_forecast_pipeline()
    train_df = results["train_df"]
    test_df = results["test_df"]

    print("Modelo estimado com sucesso.")
    print(
        f"Janela de treino: {train_df['Date'].min().date()} até {train_df['Date'].max().date()} | "
        f"{len(train_df)} observações"
    )
    print(
        f"Janela de teste:  {test_df['Date'].min().date()} até {test_df['Date'].max().date()} | "
        f"{len(test_df)} observações"
    )
    print("\nMétricas:")
    print(results["metrics"].to_string(index=False))
    print("\nCenários:")
    print(results["scenarios"].to_string(index=False))


if __name__ == "__main__":
    main()
