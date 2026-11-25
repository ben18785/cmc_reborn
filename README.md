# cmc_reborn

Iterated, block-wise Contour Monte Carlo (CMC) for estimating cell-to-cell parameter
heterogeneity from snapshot data when the forward map is stochastic (measurement or
intrinsic noise) and there are many output dimensions.

- `cmc/models.py`: forward models (scalar Gaussian toy, linear-Gaussian with 24 snapshots,
  growth-factor ODE with log-normal measurement noise, three-gene induced transcription with
  Poisson intrinsic noise and binomial capture).
- `cmc/ratio.py`: density-ratio estimators (KDE, logistic regression on quadratic features,
  neural-network classifier, and the vine-copula KDE used by the original CMC).
- `cmc/infer.py`: `iterate_blocks` (block-wise Richardson-Lucy / iterative proportional fitting
  on weighted particles) and `one_step_joint` (original CMC; with `noise=True`, the augmented
  update of Butler et al. 2020).
- `cmc/gaussian.py`: closed-form iterates for the linear-Gaussian examples (used in tests).

Reproduce: `pip install -r requirements-lock.txt && pip install -e .`, then `pytest` and
`snakemake --cores 4` (or `--use-conda`). Fits are run on GitHub Actions
(`gh workflow run experiments.yml`); every setting lives in `config/config.yaml`.
The write-up is in the Overleaf project (cloned at `../cmc_stochastic_overleaf`).
