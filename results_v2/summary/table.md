| example | fitted dims (blocks) | method | θ energy dist. | fit ED / floor | held-out ED / floor | sd ratio | cov. rel. error | minutes |
|---|---|---|---|---|---|---|---|---|
| scalar | 1 (1) | Iterated, logistic | 0.00121 ± 0.00061 | 0.83 ± 0.72 | – | 1.01 ± 0.012 | 0.0327 ± 0.019 | 0.1 |
| scalar | 1 (1) | Iterated, MLP | 0.00403 ± 0.0028 | 3.05 ± 0.63 | – | 0.79 ± 0.049 | 0.374 ± 0.076 | 1.0 |
| linear | 40 (20) | Iterated, logistic | 0.00124 ± 0.00048 | 1.24 ± 0.072 | 1.24 ± 0.14 | 0.992 ± 0.0081 | 0.0708 ± 0.011 | 1.9 |
| linear | 40 (20) | Iterated, MLP | 0.0194 ± 0.0015 | 7.14 ± 1 | 8.05 ± 2.4 | 0.688 ± 0.026 | 0.505 ± 0.014 | 10.7 |
| growth | 20 (10) | Iterated, logistic | 0.25 ± 0.019 | 1.24 ± 0.53 | 2.38 ± 0.3 | 2.38 ± 0.047 | – | 1.3 |
| growth | 20 (10) | Iterated, MLP | 0.365 ± 0.1 | 27.7 ± 14 | 44.7 ± 31 | 2.39 ± 0.33 | – | 5.9 |
| gene | 18 (6) | Iterated, logistic | 0.0342 ± 0.0044 | 1.77 ± 0.39 | 1.88 ± 1.3 | 1.4 ± 0.038 | – | 0.9 |
| gene | 18 (6) | Iterated, MLP | 0.0166 ± 0.011 | 14.3 ± 6.4 | 24.3 ± 18 | 0.966 ± 0.034 | – | 3.9 |

θ energy distance: posterior vs true population (parameters scaled by prior sd). ED / floor: mean energy distance between observed and predicted snapshots divided by the distance between two independent data sets (1 = as good as the data allow). sd ratio: posterior sd / true sd averaged over parameters that vary across cells. Mean ± sd over seeds.
