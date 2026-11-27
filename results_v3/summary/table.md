| example | fitted dims (blocks) | method | θ energy dist. | fit ED / floor | held-out ED / floor | sd ratio | cov. rel. error | minutes |
|---|---|---|---|---|---|---|---|---|
| scalar | 1 (1) | Iterated, logistic | 0.000455 ± 0.00026 | 0.732 ± 0.34 | – | 1.01 ± 0.016 | 0.037 ± 0.011 | 0.1 |
| scalar | 1 (1) | Iterated, MLP | 0.00371 ± 0.0018 | 2.69 ± 1.5 | – | 0.804 ± 0.03 | 0.352 ± 0.047 | 0.8 |
| linear | 40 (20) | Iterated, logistic | 0.000865 ± 0.00015 | 1.04 ± 0.15 | 0.948 ± 0.21 | 0.993 ± 0.007 | 0.0525 ± 0.0054 | 1.9 |
| linear | 40 (20) | Iterated, MLP | 0.0246 ± 0.0026 | 8.9 ± 1.2 | 8.56 ± 1.4 | 0.639 ± 0.0095 | 0.563 ± 0.03 | 14.8 |
| growth | 20 (10) | Iterated, logistic | 0.244 ± 0.017 | 1.22 ± 0.28 | 2.12 ± 0.76 | 2.35 ± 0.015 | – | 1.0 |
| growth | 20 (10) | Iterated, MLP | 0.257 ± 0.0062 | 8.5 ± 4.7 | 13.8 ± 4.4 | 2.33 ± 0.034 | – | 8.9 |
| gene | 18 (6) | Iterated, logistic | 0.0534 ± 0.0041 | 1.01 ± 0.32 | 1.89 ± 0.3 | 1.55 ± 0.034 | – | 0.7 |
| gene | 18 (6) | Iterated, MLP | 0.0079 ± 0.0017 | 7.1 ± 2.4 | 13.2 ± 6.1 | 1.03 ± 0.061 | – | 5.2 |

θ energy distance: posterior vs true population (parameters scaled by prior sd). ED / floor: mean energy distance between observed and predicted snapshots divided by the distance between two independent data sets (1 = as good as the data allow). sd ratio: posterior sd / true sd averaged over parameters that vary across cells. Mean ± sd over seeds.
