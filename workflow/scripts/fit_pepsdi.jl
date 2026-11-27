#=
PEPSDI baseline: the method of Persson et al. (2022, PLoS Comput Biol 18(5):e1010082), run with their
code (github.com/cvijoviclab/PEPSDI, cloned at a fixed commit by the Snakefile rule `pepsdi_repo`;
not redistributed) on the stochastic transfection model (cmc/models.py, TransfectionSSA).

    julia --project=workflow/envs/pepsdi fit_pepsdi.jl --repo external/PEPSDI --work <dir>

<dir> holds data.csv (PEPSDI's tidy format) and settings.csv, written by
`workflow/scripts/pepsdi_io.py prep`; this script writes draws.csv (thinned posterior draws of the
population parameters) and info.csv there, and `pepsdi_io.py collect` turns them into the npz that
fit.py scores. Environment: Julia 1.5.2, as the authors used, with packages resolved against the
General registry as of 2021-02-28 (see the CI step and workflow/envs/pepsdi/Project.toml). Their
repository has a Project.toml without versions and no Manifest; their notebook output loads
DynamicPPL from a directory whose name identifies version 0.10.7 (current from 2021-01-26 to
2021-03-29, so the snapshot reproduces their environment).

Method (their defaults where they have one): PEPSDI running option 2, their recommended default (the
cell constants kappa and the noise parameter xi vary weakly between cells, epsilon = 0.01); the
log-normal population with full correlation and NUTS updates of the population parameters (PopNormalLKJ,
50 warm-up steps, as their clock-model notebook); RAM proposals for the cell parameters (their
recommended sampler; initial covariance 0.1 I and 250 steps before adaptation, as the clock notebook);
a pilot run followed by their particle tuning (per-cell number of particles such that the variance of
the log-likelihood estimate is below 2 twice in a row, 50 filter runs per trial), then the main run
started from the pilot's end point with the tuned particles. Stochastic model and filter: their
Extrande simulator and bootstrap filter (Extrande is exact for propensities that change in time).

Model (theta = log(delta, gamma, k, m0, scale, t0, offset, sigma), as TransfectionSSA): at t0 the cell
holds round(m0) mRNA; mRNA decays at rate delta, is translated at rate k and protein decays at rate
gamma; observed log(scale P + offset) + N(0, sigma^2) at 1, ..., 30 h.
  - individual parameters c_i = log(delta, gamma, k, m0, t0) ~ N(mu, D(tau) Omega D(tau));
  - cell constants kappa = log(scale, offset) and noise xi = log(sigma) (shared in the synthetic
    population), estimated on the log scale;
  - priors: mu ~ uniform on our prior box; tau ~ half-Cauchy(0, 2.5) (their default in every example);
    Omega ~ LKJ(5, 3) (their clock notebook); kappa_pop, xi_pop ~ uniform on the box.

Deviations, each for a reason:
  - Cells: a seeded random subset (n_cells, default 100, of 2565; pepsdi_io.py), because the
    particle-MCMC cost is linear in cells (Haggstrom et al. needed ~149 h for 40 cells of an SDE
    version). The subset is recorded in the npz.
  - Simulator: Extrande rather than their Gillespie filter, because reactions switch on at the
    cell's t0: the Gillespie direct method stops once the total propensity is zero, so it cannot
    represent a switch-on time. Extrande's bound is zero on intervals that end before t0 and the full
    rates otherwise, which is exact. Neither of their SSA filters can correlate particles (rho = 0,
    as in their clock notebook).
  - Initial state: PEPSDI's option 2 computes a cell's initial state once, from its starting
    parameters, and never again, so an initial count that depends on a cell parameter (m0) would stay
    at its starting value. The model therefore starts every particle at (0, 0) with a third,
    never-reacting flag species, and the propensity-bound callback, which Extrande calls before any
    reaction, sets M = round(m0) the first time it sees a particle. This is the same process as
    x0 = (round(m0), 0); PEPSDI's code is not modified.
  - Prior box: proposals (cells, kappa or sigma) outside our model's prior box get likelihood zero
    and are rejected, as our population is restricted to the box (cmc/bayes.py does the same); this
    also bounds the cost of a simulation.
  - Starting values: their default (prior centre: the mean, or the median for Cauchy priors), except
    the noise, which starts at log sigma = -0.5: with cells at the prior centre and sigma near the
    prior centre, every particle's weight underflows to zero, and a pseudo-marginal chain cannot leave
    a zero-likelihood start. (Their examples start near the true values, which we cannot.)
  - Run length: their pilot uses 500-2000 particles and 5000 iterations, their main runs 5000-50000
    iterations. Here the pilot uses n_particles_pilot (default 200) particles and as many iterations
    (between n_pilot_min and n_pilot_max) as a fraction pilot_frac of the remaining budget allows, and
    the main run as many iterations (n_main_min to n_main_max) as fit in the time left before
    `minutes`; the time per iteration is measured with short runs first. All are recorded.
  - Posterior: burn-in burn_frac of the main run (their notebooks discard 10-20%), then evenly spaced
    rows; pop_draws (pepsdi_io.py) use 50 of them, each with n_pop cells drawn from the population and
    the constants at kappa_pop, xi_pop (the weak per-cell variation of option 2 is a device of the
    sampler, not part of the population).
  - Their posterior visual check after the main run is skipped (benchmark=true): it is a plotting
    diagnostic, not part of the inference.
  - Their filters draw random numbers from unseeded generators, so runs are not bit-reproducible.
=#
using Printf, Random, LinearAlgebra, Statistics, DelimitedFiles, Dates

const T_START = time()

function parse_args(a)
    d = Dict{String, String}()
    for i in 1:2:length(a)
        startswith(a[i], "--") || error("unexpected argument $(a[i])")
        d[a[i][3:end]] = a[i + 1]
    end
    return d
end

function read_settings(path)
    d = Dict{String, String}()
    for line in readlines(path)[2:end]
        k, v = split(line, ","; limit=2)
        d[String(k)] = String(v)
    end
    return d
end

const ARG = parse_args(ARGS)
const REPO = abspath(ARG["repo"])
const WORK = abspath(ARG["work"])
const ST = read_settings(joinpath(WORK, "settings.csv"))
geti(k) = parse(Int, ST[k])
getf(k) = parse(Float64, ST[k])
Random.seed!(geti("seed"))
ENV["GKSwstype"] = "100"  # PEPSDI loads Plots: no display on the runner

cd(REPO)  # PEPSDI includes its source files relative to the working directory
push!(LOAD_PATH, joinpath(REPO, "src"))
using Distributions
using PEPSDI
const T_LOADED = time()
@printf("Julia %s, %d threads; PEPSDI loaded in %.0f s\n", VERSION, Threads.nthreads(), T_LOADED - T_START)

# ---- the model (cmc/models.py, TransfectionSSA; box constants copied from there)
const LOWER = [-4.0, -12.0, -3.0, 4.0, -2.0, -3.0, 0.0, -5.0]
const UPPER = [1.5, 0.0, 1.5, 6.5, 3.0, 2.0, 4.0, 0.0]
const C_IDX = [1, 2, 3, 4, 6]  # delta, gamma, k, m0, t0: individual parameters c
const K_IDX = [5, 7]           # scale, offset: cell constants kappa
const S_IDX = 8                # sigma: error parameter xi
const LO_C = exp.(LOWER[C_IDX]); const HI_C = exp.(UPPER[C_IDX])
const LO_K = exp.(LOWER[K_IDX]); const HI_K = exp.(UPPER[K_IDX])
const LO_S = exp(LOWER[S_IDX]); const HI_S = exp(UPPER[S_IDX])

# state u = (M, P, F): F = 0 until the particle is initialised, then 1 (parameters in the box) or 2 (not)
function tf_init!(u, p)
    c, kp = p.c, p.kappa
    ok = true
    @inbounds for i in 1:5
        ok &= LO_C[i] <= c[i] <= HI_C[i]
    end
    @inbounds for i in 1:2
        ok &= LO_K[i] <= kp[i] <= HI_K[i]
    end
    if ok
        u[1] = UInt16(round(c[4]))  # round(m0) mRNA at t0 (at most exp(6.5) = 665 in the box)
        u[3] = 0x0001
    else
        u[1] = 0x0000
        u[3] = 0x0002
    end
    return nothing
end

# Extrande: upper bound of the propensities on [t_start, t_end]; called before every candidate reaction
function tf_h_max!(u, h_vec, p, t_start, t_end)
    u[3] == 0x0000 && tf_init!(u, p)
    c = p.c
    if u[3] != 0x0001 || t_end <= c[5]
        h_vec[1] = 0.0; h_vec[2] = 0.0; h_vec[3] = 0.0
    else
        h_vec[1] = c[1] * u[1]  # mRNA decay
        h_vec[2] = c[3] * u[1]  # translation
        h_vec[3] = c[2] * u[2]  # protein decay
    end
    return nothing
end

function tf_h!(u, h_vec, p, t)
    c = p.c
    if u[3] != 0x0001 || t < c[5]  # nothing happens before transfection at t0
        h_vec[1] = 0.0; h_vec[2] = 0.0; h_vec[3] = 0.0
    else
        h_vec[1] = c[1] * u[1]
        h_vec[2] = c[3] * u[1]
        h_vec[3] = c[2] * u[2]
    end
    return nothing
end

function tf_x0!(u0, p)
    u0 .= 0
    return nothing
end

function tf_obs(y_mod, u, p, t)
    y_mod[1] = u[3] == 0x0001 ? log(p.kappa[1] * u[2] + p.kappa[2]) : NaN
    return nothing
end

function tf_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    s = error_param[1]
    (isnan(y_mod[1]) || !(LO_S <= s <= HI_S)) && return 0.0
    return exp(logpdf(Normal(0.0, s), y_obs[1] - y_mod[1]))
end

# PEPSDI applies x .-= S[r, :]: mRNA decay, translation, protein decay (the flag never changes)
const S_MAT = convert(Array{Int16, 2}, [1 0 0; 0 -1 0; 0 1 0])
const MODEL = ExtrandModel(tf_h!, tf_h_max!, tf_x0!, tf_obs, tf_prob_obs, 3, 1, 3, S_MAT)

# ---- inference options
const N_C = length(C_IDX)
prior_mean = [Uniform(LOWER[i], UPPER[i]) for i in C_IDX]
prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf) for _ in 1:N_C]
prior_corr = LKJ(N_C, 3.0)
prior_kappa = [Uniform(LOWER[i], UPPER[i]) for i in K_IDX]
prior_sigma = [Uniform(LOWER[S_IDX], UPPER[S_IDX])]
init_log_sigma = getf("init_log_sigma")

pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma,
                                     prior_pop_kappa=prior_kappa, prior_pop_corr=prior_corr,
                                     init_pop_sigma=[init_log_sigma],
                                     pos_pop_kappa=false, log_pop_kappa=true,
                                     pos_pop_sigma=false, log_pop_sigma=true)
ind_param_info = init_ind_param_info("mean", N_C, log_scale=true, pos_param=false)
n_part_pilot = geti("n_particles_pilot")
filter_pilot = init_filter(BootstrapExtrand(), n_particles=n_part_pilot)
mcmc_ci = init_mcmc(RamSampler(), ind_param_info, cov_mat=diagm(fill(0.1, N_C)), step_before_update=250)
mcmc_ks = init_mcmc(RamSampler(), pop_param_info, cov_mat=diagm(fill(0.1, 3)), step_before_update=250)
pop_sampler_opt = init_pop_sampler_opt(PopNormalLKJ(), n_warm_up=50)
ks_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances=fill(0.01, 3))

data_csv = joinpath(WORK, "data.csv")
dir_save = joinpath(WORK, "pepsdi")
file_loc = init_file_loc(data_csv, "tf_ssa", dir_save=dir_save, multiple_ind=true)
file_loc_timing = init_file_loc(data_csv, "timing", dir_save=joinpath(WORK, "timing"), multiple_ind=true)

function run_opt2(n, floc, filt; pilot_id=0)
    t = time()
    out = run_PEPSDI_opt2(n, deepcopy(pop_param_info), deepcopy(ind_param_info), floc, MODEL, filt,
                          deepcopy(mcmc_ci), deepcopy(mcmc_ks), pop_sampler_opt, ks_sampler_opt,
                          pilot_id=pilot_id, benchmark=true)
    return out, time() - t
end

deadline = T_START + 60 * getf("minutes")

# ---- time per iteration of the pilot configuration (a first short run compiles everything)
run_opt2(3, file_loc_timing, filter_pilot)
_, t32 = run_opt2(33, file_loc_timing, filter_pilot)
t_iter_pilot = t32 / 32  # includes the fixed set-up (warm-up of NUTS): an overestimate
n_pilot = clamp(floor(Int, getf("pilot_frac") * (deadline - time()) / t_iter_pilot), geti("n_pilot_min"), geti("n_pilot_max"))
@printf("pilot: %.2f s per iteration with %d particles per cell; running %d iterations\n",
        t_iter_pilot, n_part_pilot, n_pilot)

# ---- their pilot run and particle tuning
t_tune0 = time()
tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=n_part_pilot, n_samples_pilot=n_pilot,
                                     rho_list=[0.0], n_times_run_filter=geti("n_times_run_filter"),
                                     init_sigma=[init_log_sigma])
tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, file_loc, MODEL, filter_pilot,
                    mcmc_ci, mcmc_ks, pop_sampler_opt, ks_sampler_opt)
seconds_pilot_tune = time() - t_tune0
pilot_dir = joinpath(dir_save, "Ram_sampler", "Pilot_run_data_alt")
tags = filter(s -> startswith(s, "Exp_tag"), readdir(pilot_dir))
length(tags) == 1 || error("expected one pilot run in $pilot_dir, found $tags")
exp_id = parse(Int, tags[1][length("Exp_tag") + 1:end])
particles = vec(readdlm(joinpath(pilot_dir, tags[1], "Tune_particles", "Rho0d0.csv"), ',', Int, skipstart=1))
@printf("pilot and tuning: %.1f min; tuned particles per cell: median %d, range %d-%d\n",
        seconds_pilot_tune / 60, round(Int, median(particles)), minimum(particles), maximum(particles))

# ---- main run from the pilot's end point with the tuned particles
_, t32m = run_opt2(33, file_loc, filter_pilot, pilot_id=exp_id)
t_iter_main = t32m / 32
n_main_req = geti("n_main_max")
n_main = clamp(floor(Int, (deadline - time() - 300) / t_iter_main), geti("n_main_min"), n_main_req)
@printf("main: %.2f s per iteration; running %d iterations\n", t_iter_main, n_main)
t_main0 = time()
res, _ = run_opt2(n_main, file_loc, filter_pilot, pilot_id=exp_id)
seconds_main = time() - t_main0
ch = res[1]

# ---- thinned draws after burn-in
burn = max(1, floor(Int, getf("burn_frac") * n_main))
n_rows = min(geti("n_keep_rows"), n_main - burn)
rows = unique(round.(Int, LinRange(burn + 1, n_main, n_rows)))
header = vcat(["mu$i" for i in 1:N_C], ["tau$i" for i in 1:N_C],
              ["corr$(i)_$(j)" for j in 1:N_C for i in 1:N_C], ["ks$i" for i in 1:3])
M = hcat(permutedims(ch.mean[:, rows]), permutedims(ch.scale[:, rows]),
         permutedims(reshape(ch.corr[:, :, rows], N_C * N_C, length(rows))), permutedims(ch.kappa_sigma[:, rows]))
open(joinpath(WORK, "draws.csv"), "w") do io
    println(io, join(header, ","))
    writedlm(io, M, ',')
end
# acceptance of the joint (c_i, kappa_i, xi_i) proposals after burn-in, per cell
acc = [mean([any(ch.ind_param[j][:, k] .!= ch.ind_param[j][:, k - 1]) for k in (burn + 1):n_main])
       for j in 1:length(ch.ind_param)]
n_cells = length(ch.ind_param)
n_times = length(unique(readdlm(data_csv, ',', Float64, skipstart=1)[:, 1]))
info = [("seconds", time() - T_START), ("seconds_load", T_LOADED - T_START),
        ("seconds_pilot_tune", seconds_pilot_tune), ("seconds_main", seconds_main),
        ("n_cells", n_cells), ("n_times", n_times), ("n_cells_available", geti("n_cells_available")),
        ("n_particles_pilot", n_part_pilot), ("n_pilot", n_pilot), ("n_pilot_max", geti("n_pilot_max")),
        ("t_iter_pilot", t_iter_pilot), ("t_iter_main", t_iter_main),
        ("n_main", n_main), ("n_main_requested", n_main_req), ("burn_in", burn),
        ("particles_median", median(particles)), ("particles_mean", mean(particles)),
        ("particles_min", minimum(particles)), ("particles_max", maximum(particles)),
        ("accept_mean", mean(acc)), ("accept_min", minimum(acc)),
        ("minutes_budget", getf("minutes")), ("julia_version", string(VERSION)), ("threads", Threads.nthreads())]
open(joinpath(WORK, "info.csv"), "w") do io
    println(io, "key,value")
    for (k, v) in info
        println(io, k, ",", v)
    end
end
@printf("done in %.1f min (main run %.1f min, %d iterations; acceptance %.3f)\n",
        (time() - T_START) / 60, seconds_main / 60, n_main, mean(acc))
