# ESN_Tuned — Lorenz CTF Model

Echo State Network with evolved hyperparameters for the [CTF for Science](https://ctf-for-science.github.io/ctf4science/) Lorenz benchmark.

---

**Developed with [inZOR-ND](https://dumitrunovic-svg.github.io/inZOR-ND/) · Built using [Cursor](https://cursor.com)**

---

## Method

Leaky Echo State Network (ESN) with ridge regression readout.
Hyperparameters were discovered through evolutionary search on the ODE_Lorenz development dataset.

## Hyperparameters

| Parameter | Value |
|---|---|
| spectral_radius | 1.002 |
| ridge_alpha | 3.35e-05 |
| input_scaling | 0.1 |
| leaking_rate | 1.0 |
| reservoir_size | 500 |
| washout | 100 |

## Results (ODE_Lorenz dev set, 12 metrics)

- Mean score: **35.3**
- vs default ESN (ρ=0.9, ridge=1e-6): **+35% improvement**

## Usage

```bash
pip install -e /path/to/ctf4science
python run.py config_Lorenz_Official.yaml
```

## Requirements

```
numpy>=1.24
scipy>=1.10
pyyaml>=6.0
```
