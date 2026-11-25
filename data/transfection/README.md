# Single-cell mRNA transfection data

GFP fluorescence time courses of single cells after mRNA transfection, from Fröhlich et al. (2018),
"Multi-experiment nonlinear mixed effect modeling of single-cell translation kinetics after
transfection", npj Systems Biology and Applications 4:42. Source: the paper's Zenodo archive
(doi 10.5281/zenodo.1228899, `code.zip`, files `code/project/data/{20160427,20160513,20160603}_mean_{eGFP,d2eGFP}.xlsx`),
licensed CC BY-SA 4.0; these converted files are distributed under the same licence.

`eGFP.csv` (2565 cells) and `d2eGFP.csv` (2923 cells): first row cell ids, second row experiment
date, then one row per time point (hours, up to 30 h; 10-minute spacing). Regenerate with
`python workflow/scripts/prep_transfection.py --zip code.zip --out data/transfection`.
This is the data set used by Arruda et al. (2024, ICML) to benchmark amortised mixed-effects inference.
