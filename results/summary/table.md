| example | fitted dims (blocks) | method | θ energy dist. | fit ED / floor | held-out ED / floor | sd ratio | cov. rel. error | minutes |
|---|---|---|---|---|---|---|---|---|
| scalar | 1 (1) | CMC (noise ignored) | 0.017 ± 0.0024 | 5.69 ± 0.49 | – | 1.45 ± 0.012 | 1.09 ± 0.035 | 0.1 |
| scalar | 1 (1) | CMC (augmented) | 0.0296 ± 0.00045 | 12.2 ± 1.3 | – | 1.62 ± 0.0058 | 1.62 ± 0.019 | 0.1 |
| scalar | 1 (1) | Iterated, KDE | 0.00556 ± 0.0058 | 5.21 ± 4.1 | – | 1.26 ± 0.21 | 0.625 ± 0.55 | 6.6 |
| scalar | 1 (1) | Iterated, logistic | 0.000611 ± 0.00042 | 0.526 ± 0.24 | – | 1.02 ± 0.0091 | 0.0307 ± 0.018 | 0.0 |
| scalar | 1 (1) | Iterated, MLP | 0.00147 ± 0.0014 | 0.39 ± 0.1 | – | 1.01 ± 0.0069 | 0.0175 ± 0.012 | 0.2 |
| linear | 40 (20) | CMC (noise ignored) | 0.313 ± 0.14 | 108 ± 55 | 166 ± 69 | 0.792 ± 0.25 | 1.19 ± 0.57 | 1.1 |
| linear | 40 (20) | CMC (augmented) | 2.02 ± 0.45 | 788 ± 2.2e+02 | 714 ± 8.7 | 7.41e-06 ± 1e-05 | 1 ± 3.7e-10 | 1.1 |
| linear | 40 (20) | Iterated, KDE | 0.00755 ± 0.0062 | 4.86 ± 4.1 | 3.82 ± 2.4 | 1.11 ± 0.029 | 0.38 ± 0.086 | 80.9 |
| linear | 40 (20) | Iterated, logistic | 0.000836 ± 0.00014 | 0.987 ± 0.071 | 1.12 ± 0.16 | 0.99 ± 0.0082 | 0.0641 ± 0.0037 | 0.3 |
| linear | 40 (20) | Iterated, MLP | 0.00297 ± 0.00053 | 1.92 ± 0.33 | 2.05 ± 0.71 | 0.891 ± 0.0066 | 0.22 ± 0.019 | 2.2 |
| growth | 20 (10) | CMC (noise ignored) | 1.66 ± 0.42 | 62.5 ± 14 | 80.1 ± 4.3 | 4.96 ± 0.54 | – | 0.5 |
| growth | 20 (10) | CMC (augmented) | 0.997 ± 0.22 | 51.1 ± 5.1 | 70.2 ± 6.5 | 4.66 ± 0.69 | – | 0.4 |
| growth | 20 (10) | Iterated, KDE | 0.332 ± 0.031 | 2.68 ± 1.3 | 3.51 ± 1.8 | 2.8 ± 0.15 | – | 35.3 |
| growth | 20 (10) | Iterated, logistic | 0.624 ± 0.06 | 3.9 ± 3.6 | 4.31 ± 3 | 3.73 ± 0.12 | – | 0.1 |
| growth | 20 (10) | Iterated, MLP | 0.626 ± 0.074 | 2.26 ± 0.81 | 4.29 ± 0.66 | 3.8 ± 0.076 | – | 1.2 |
| gene | 18 (6) | CMC (noise ignored) | 0.511 ± 0.058 | 34.9 ± 8.7 | 47.5 ± 18 | 2.1 ± 0.24 | – | 0.2 |
| gene | 18 (6) | CMC (augmented) | 0.949 ± 0.45 | 87.5 ± 50 | 112 ± 55 | 1.99 ± 0.74 | – | 0.3 |
| gene | 18 (6) | Iterated, KDE | 0.133 ± 0.019 | 9.39 ± 4.1 | 12.7 ± 4.5 | 1.96 ± 0.13 | – | 22.2 |
| gene | 18 (6) | Iterated, logistic | 0.099 ± 0.0069 | 1.74 ± 0.083 | 2.06 ± 0.4 | 1.88 ± 0.032 | – | 0.1 |
| gene | 18 (6) | Iterated, MLP | 0.0703 ± 0.0039 | 3.8 ± 1.2 | 6.01 ± 1.3 | 1.66 ± 0.027 | – | 0.7 |

θ energy distance: posterior vs true population (parameters scaled by prior sd). ED / floor: mean energy distance between observed and predicted snapshots divided by the distance between two independent data sets (1 = as good as the data allow). sd ratio: posterior sd / true sd averaged over parameters that vary across cells. Mean ± sd over seeds.
