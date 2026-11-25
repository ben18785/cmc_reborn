# Stickleback functional-response trials

Subset of `data/df_covariates_better_date.csv` from
https://github.com/rccreswell/functional-response (commit ff3fd1a), the data behind
Newport et al. (2026), "Heatwaves and turbidity influxes have interacting and lasting effects on foraging
behaviour of three-spined sticklebacks *Gasterosteus aculeatus*".

One row per trial: a fish given `n_prey_initial` prey for one trial period, and the number it ate
(`n_prey_eaten`). Rows with `exclude == "Yes"` are dropped. Columns kept: fish id, exclusion flag,
condition (acclimation x exposure treatment), week, initial prey, prey eaten, weight.
