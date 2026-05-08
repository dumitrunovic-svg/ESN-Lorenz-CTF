# ESN_Tuned — Lorenz CTF Model

Echo State Network with evolved hyperparameters for the CTF for Science Lorenz benchmark.

## Usage

```bash
pip install -e /path/to/ctf4science
python run.py config_Lorenz_Official.yaml
```

## Hyperparameters
- spectral_radius: 1.002
- ridge_alpha: 3.35e-05
- input_scaling: 0.1
- leaking_rate: 1.0
- reservoir_size: 500
- washout: 100
