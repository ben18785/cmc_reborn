//! NPAG (or NPOD) with PMcore, the Rust engine of Pmetrics, for cmc's transfection model.
//!
//!     pmcore_tf data.csv settings.json out_dir
//!
//! data.csv: Pmetrics-format observations (written by workflow/scripts/fit_pmetrics.py).
//! settings.json: {"algorithm": "npag"|"npod", "names": [...5], "lower": [...5], "upper": [...5],
//!                 "points": int, "sobol_seed": int, "max_cycles": int, "minutes": float, "sigma0": float}
//! out_dir: theta.csv (support points and their probabilities, PMcore's own writer), cycles.csv
//! (PMcore's cycle log) and summary.json (-2LL, cycles, stop reason, sigma, timings).
//!
//! Model (the same closed form as cmc.models.Transfection, parameters on the log scale):
//!     p(t) = km0 exp(-lo tau) (1 - exp(-(hi - lo) tau)) / (hi - lo),  tau = max(t - t0, 0),
//!     lo = min(delta, gamma), hi = max(delta, gamma),  y = log(p + offset) + N(0, sigma^2).
//! The prediction is computed in the output equation directly from t, so the analytical kernel
//! (a one-compartment structure with a dummy elimination rate and no doses) only carries an unused
//! state. The error model is Pmetrics' additive model with a zero assay polynomial,
//! sd = sqrt(0^2 + lambda^2) = lambda: homoscedastic on the log scale, with lambda (= sigma, shared
//! by all cells) estimated by NPAG's error-model step.
use anyhow::{bail, Context};
use pmcore::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        bail!("usage: pmcore_tf data.csv settings.json out_dir");
    }
    let s: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&args[2]).context("reading settings")?,
    )?;
    let out = std::path::PathBuf::from(&args[3]);
    std::fs::create_dir_all(&out)?;
    Logger::new().stdout(true).init()?;

    let eq = analytical! {
        name: "transfection",
        params: [ldelta, lgamma, lkm, lt0, loff],
        derived: [ke],
        states: [central],
        outputs: [outeq_1],
        routes: [
            bolus(input_1) -> central,
        ],
        structure: one_compartment,
        derive: |_t| {
            ke = 0.0;
        },
        out: |_x, t, y| {
            let d = ldelta.exp();
            let g = lgamma.exp();
            let lo = d.min(g);
            let hi = d.max(g);
            let tau = (t - lt0.exp()).max(0.0);
            let z = (hi - lo) * tau;
            // (1 - exp(-(hi - lo) tau)) / (hi - lo), stable as hi - lo -> 0 (as cmc's _relax)
            let rel = if z.abs() < 1e-8 { tau } else { -(-z).exp_m1() / (hi - lo) };
            let mut p = lkm.exp() * (-lo * tau).exp() * rel;
            if !p.is_finite() {
                p = 0.0;
            }
            y[outeq_1] = (p + loff.exp()).ln();
        },
    };

    let data = data::read_pmetrics(args[1].as_str())?;
    let n_subjects = data.subjects().len();

    let names: Vec<String> = s["names"].as_array().context("names")?.iter()
        .map(|v| v.as_str().unwrap().to_string()).collect();
    let lower: Vec<f64> = s["lower"].as_array().context("lower")?.iter().map(|v| v.as_f64().unwrap()).collect();
    let upper: Vec<f64> = s["upper"].as_array().context("upper")?.iter().map(|v| v.as_f64().unwrap()).collect();
    // the parameter space must list the parameters in the model's order
    let expected = ["ldelta", "lgamma", "lkm", "lt0", "loff"];
    if names.iter().map(String::as_str).ne(expected.iter().copied()) {
        bail!("parameter names {:?} must be {:?}", names, expected);
    }
    let mut space = ParameterSpace::bounded();
    for k in 0..names.len() {
        space = space.add(names[k].as_str(), lower[k], upper[k]);
    }
    let points = s["points"].as_u64().unwrap_or(2028) as usize;
    let sobol_seed = s["sobol_seed"].as_u64().unwrap_or(22) as usize;
    let prior = Theta::sobol_with_seed(&space, points, sobol_seed)?;

    // additive error, zero polynomial: sd = lambda, estimated (Factor::Variable). Registered on the
    // dense output slot 0 (the model's only output): PMcore's error-model optimisation iterates the
    // dense slots, so a model registered under a name label would never have its lambda updated.
    let sigma0 = s["sigma0"].as_f64().unwrap_or(0.37);
    let error_models = AssayErrorModels::new().add(
        0,
        AssayErrorModel::additive(ErrorPoly::new(0.0, 0.0, 0.0, 0.0), sigma0),
    )?;

    let max_cycles = s["max_cycles"].as_u64().unwrap_or(1000) as usize;
    let algorithm = s["algorithm"].as_str().unwrap_or("npag").to_string();
    let algo: NonParametricAlgorithm = match algorithm.as_str() {
        "npag" => NpagConfig::new().max_cycles(max_cycles).progress(false).into(),
        "npod" => NpodConfig::new().max_cycles(max_cycles).progress(false).into(),
        other => bail!("unknown algorithm {other}"),
    };
    let budget = s["minutes"].as_f64().unwrap_or(240.0) * 60.0;

    let problem = EstimationProblem::nonparametric(eq, data, prior, error_models)?;
    let start = Instant::now();
    let mut controller = problem.fit_controller(algo)?;
    let mut stopped_for_time = false;
    let mut longest = 0.0_f64;
    loop {
        let t_cycle = Instant::now();
        let status = controller.step()?;
        let dt = t_cycle.elapsed().as_secs_f64();
        longest = longest.max(dt);
        let sigma = controller.error_models().factor(0).unwrap_or(f64::NAN);
        println!(
            "cycle {} | -2LL {:.4} | support points {} | sigma {:.5} | {:.1} s (total {:.1} min)",
            controller.cycle(), controller.n2ll(), controller.theta().nspp(), sigma, dt,
            start.elapsed().as_secs_f64() / 60.0
        );
        if status.is_stop() {
            break;
        }
        // wall-clock budget: stop if another cycle (as long as the longest so far) would overrun it
        if start.elapsed().as_secs_f64() + 1.2 * longest > budget {
            println!("stopping: wall-clock budget of {:.0} min", budget / 60.0);
            stopped_for_time = true;
            controller.request_stop();
            break;
        }
    }
    let result = controller.into_result()?;
    let seconds = start.elapsed().as_secs_f64();

    result.write_theta(&out.join("theta.csv"))?;
    result.write_cycles(&out.join("cycles.csv"))?;
    let sigma = result.error_models().factor(0)?;
    let summary = serde_json::json!({
        "algorithm": algorithm,
        "neg2ll": result.objf(),
        "cycles": result.cycles(),
        "converged": result.converged(),
        "stopped_for_time": stopped_for_time,
        "sigma": sigma,
        "n_support": result.get_theta().nspp(),
        "n_subjects": n_subjects,
        "seconds": seconds,
        "points": points,
        "max_cycles": max_cycles,
        "pmcore_version": "0.29.0",
        "pharmsol_version": "0.29.4",
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
