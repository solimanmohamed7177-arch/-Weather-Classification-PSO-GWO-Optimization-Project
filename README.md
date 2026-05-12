# Weather Classification — PSO & GWO Optimization Project

## Quick Start

### Option A — Jupyter Notebook (Recommended)
Open `weather_pso_gwo_optimization.ipynb`.
Set N_PARTICLES=12, N_ITER=15 for a fast demo (~10 min).
Set N_PARTICLES=25, N_ITER=35 for full results (~45-90 min).

### Option B — Command Line
```bash
cd src/
pip install scikit-learn xgboost pandas numpy matplotlib seaborn
python main.py --quick    # fast demo
python main.py            # full run
```

## File Structure
| File | Description |
|------|-------------|
| weather_pso_gwo_optimization.ipynb | Master Jupyter Notebook |
| src/data_loading.py | Dataset generation + noise injection |
| src/preprocessing.py | IQR capping, encoding, scaling, split |
| src/feature_engineering.py | Domain features, polynomial, transforms |
| src/models.py | Baselines (LR, RF, XGB) + stacking |
| src/optimization.py | PSO + GWO from scratch |
| src/evaluation.py | All metrics, plots, comparison table |
| src/main.py | End-to-end CLI orchestrator |

