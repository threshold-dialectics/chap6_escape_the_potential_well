#chap6_escape_from_potential_well.py

# ---------------------------------------------------------------------
# 0  Imports & output directory
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy
from tqdm.auto import tqdm
from scipy.stats import (
    linregress,
    pearsonr,
    spearmanr,
    t as t_dist,
    skew,
    kurtosis,
)

# ---------------------------------------------------------------------
# 0.1  Paths
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1  Potential, its gradients, and helpers
# ---------------------------------------------------------------------
A_potential = 2.0
B_potential = 1.0
C_potential = 0.1
F_CRIT_STABLE = 10.0

beta_sym, F_crit_sym = sympy.symbols("beta F_crit")
A_s, B_s, C_s, Fcs_s = sympy.symbols("A B C F_crit_stable")

U_sym = (
    0.25 * beta_sym**4
    - (A_s / 2) * beta_sym**2
    - C_s * beta_sym
    + (B_s / 2) * (F_crit_sym - Fcs_s) ** 2
)

grad_U_beta_sym = sympy.diff(U_sym, beta_sym)
grad_U_F_crit_sym = sympy.diff(U_sym, F_crit_sym)

get_grad_U_beta = sympy.lambdify(
    (beta_sym, F_crit_sym, A_s, B_s, C_s, Fcs_s),
    grad_U_beta_sym,
    "numpy",
)
get_grad_U_F_crit = sympy.lambdify(
    (beta_sym, F_crit_sym, A_s, B_s, C_s, Fcs_s),
    grad_U_F_crit_sym,
    "numpy",
)

# ---------------------------------------------------------------------
# 2  Stochastic system definition
# ---------------------------------------------------------------------
class SystemStatePotential:
    """Langevin dynamics in the double-well potential."""

    def __init__(self, beta_initial: float, F_crit_initial: float) -> None:
        self.beta = beta_initial
        self.F_crit = F_crit_initial

    def update_levers_langevin(
        self,
        sigma_beta_noise: float,
        sigma_F_crit_noise: float,
        rho_input: float,
        dt: float = 0.01,
    ) -> None:
        # deterministic forces
        force_beta = -get_grad_U_beta(
            self.beta,
            self.F_crit,
            A_potential,
            B_potential,
            C_potential,
            F_CRIT_STABLE,
        )
        force_F_crit = -get_grad_U_F_crit(
            self.beta,
            self.F_crit,
            A_potential,
            B_potential,
            C_potential,
            F_CRIT_STABLE,
        )

        # correlated noise
        actual_rho = (
            0.0 if sigma_beta_noise == 0 or sigma_F_crit_noise == 0 else rho_input
        )
        z1, z2 = np.random.normal(size=2)
        noise_beta_dt = sigma_beta_noise * z1 * np.sqrt(dt)
        noise_F_crit_dt = (
            sigma_F_crit_noise
            * (actual_rho * z1 + np.sqrt(1 - actual_rho**2) * z2)
            * np.sqrt(dt)
        )

        # Euler–Maruyama step
        self.beta += force_beta * dt + noise_beta_dt
        self.F_crit += force_F_crit * dt + noise_F_crit_dt


# ---------------------------------------------------------------------
# 3  Simulation helpers
# ---------------------------------------------------------------------
def run_single_simulation_potential(params: dict) -> float | None:
    state = SystemStatePotential(params["beta_initial"], params["F_crit_initial"])

    for t_step in range(params["max_sim_steps"]):
        state.update_levers_langevin(
            params["sigma_beta_noise"],
            params["sigma_F_crit_noise"],
            params["rho_input"],
            params["dt"],
        )
        if state.beta > params["beta_escape_threshold"]:
            return (t_step + 1) * params["dt"]
    return None  # right-censored


def run_ensemble_potential(params: dict, N_runs: int) -> dict:
    collapse_times: list[float] = []

    for _ in range(N_runs):
        T_collapse = run_single_simulation_potential(params)
        if T_collapse is not None:
            collapse_times.append(T_collapse)

    return {
        "N_runs": N_runs,
        "N_collapse": len(collapse_times),
        "collapse_times": np.asarray(collapse_times),
    }


# ---------------------------------------------------------------------
# 4  Study set-up
# ---------------------------------------------------------------------
BETA_INITIAL = -1.3
F_CRIT_INITIAL = F_CRIT_STABLE
BETA_ESCAPE_THRESHOLD = 2.0

SIGMA_BETA_NOISE_POT = 0.5
SIGMA_F_CRIT_NOISE_POT = 0.5
NOISE_INTENSITY_PROXY = SIGMA_BETA_NOISE_POT**2 + SIGMA_F_CRIT_NOISE_POT**2

BASE_PARAMS = {
    "beta_initial": BETA_INITIAL,
    "F_crit_initial": F_CRIT_INITIAL,
    "max_sim_steps": 200_000,  # 20 000 time-units
    "dt": 0.1,
    "beta_escape_threshold": BETA_ESCAPE_THRESHOLD,
    "sigma_beta_noise": SIGMA_BETA_NOISE_POT,
    "sigma_F_crit_noise": SIGMA_F_CRIT_NOISE_POT,
}

SIM_TIME_WINDOW = BASE_PARAMS["max_sim_steps"] * BASE_PARAMS["dt"]  # 20000.0

N_ENSEMBLE_RUNS_POT = 500
RHO_INPUT_VALUES_POT = np.linspace(-0.95, 0.95, 21)

rows: list[dict] = []

# ---------------------------------------------------------------------
# 5  Main loop over ρ
# ---------------------------------------------------------------------
for rho in tqdm(RHO_INPUT_VALUES_POT, desc="ρ values"):
    params = BASE_PARAMS.copy()
    params["rho_input"] = rho

    stats = run_ensemble_potential(params, N_ENSEMBLE_RUNS_POT)
    collapse_times = stats["collapse_times"]
    N_coll = stats["N_collapse"]

    # initialise with NaNs
    (
        mean_T,
        std_T,
        se_T,
        gamma,
        gamma_se,
        median_T,
        q25_T,
        q75_T,
        skew_T,
        kurt_T,
        T_CI_low,
        T_CI_high,
        gamma_CI_low,
        gamma_CI_high,
    ) = (np.nan,) * 14

    if N_coll > 0:
        mean_T = collapse_times.mean()
        std_T = collapse_times.std(ddof=1) if N_coll > 1 else 0.0
        se_T = std_T / np.sqrt(N_coll)

        # 95 % CI for ⟨Tθ⟩
        df = N_coll - 1
        t_crit = t_dist.ppf(0.975, df)
        T_CI_low, T_CI_high = mean_T + np.array([-1, 1]) * t_crit * se_T

        gamma = 1.0 / mean_T
        gamma_se = se_T / mean_T**2
        gamma_CI_low, gamma_CI_high = gamma + np.array([-1, 1]) * t_crit * gamma_se

        median_T = np.median(collapse_times)
        q25_T, q75_T = np.percentile(collapse_times, [25, 75])

        skew_T = skew(collapse_times, bias=False)
        kurt_T = kurtosis(collapse_times, bias=False, fisher=True)

    rows.append(
        {
            "rho_input": rho,
            "N_runs": stats["N_runs"],
            "N_collapse": N_coll,
            "collapse_frac": N_coll / stats["N_runs"],
            "mean_T_theta": mean_T,
            "T_theta_SE": se_T,
            "T_theta_CI_low": T_CI_low,
            "T_theta_CI_high": T_CI_high,
            "gamma": gamma,
            "Gamma_SE": gamma_se,
            "Gamma_CI_low": gamma_CI_low,
            "Gamma_CI_high": gamma_CI_high,
            "median_T_theta": median_T,
            "q25_T_theta": q25_T,
            "q75_T_theta": q75_T,
            "skew_T_theta": skew_T,
            "kurt_T_theta": kurt_T,
            "denominator_fac": np.nan if rho >= 0.999 else 1.0 / (1.0 - rho),
        }
    )

# ---------------------------------------------------------------------
# 6  Save per-ρ statistics CSV
# ---------------------------------------------------------------------
stats_df = pd.DataFrame(rows)
csv_path = RESULTS_DIR / "potential_well_escape_stats.csv"
stats_df.to_csv(csv_path, index=False)

# ---------------------------------------------------------------------
# 7  Global linear regression & correlations
# ---------------------------------------------------------------------
valid_mask = stats_df["gamma"].notna() & stats_df["denominator_fac"].notna()

slope = intercept = r_value = p_value = slope_stderr = intercept_stderr = np.nan
slope_CI_low = slope_CI_high = intercept_CI_low = intercept_CI_high = np.nan

if valid_mask.any():
    x = stats_df.loc[valid_mask, "denominator_fac"].values
    y = stats_df.loc[valid_mask, "gamma"].values
    res = linregress(x, y)

    slope, intercept, r_value, p_value, slope_stderr = (
        res.slope,
        res.intercept,
        res.rvalue,
        res.pvalue,
        res.stderr,
    )
    intercept_stderr = getattr(res, "intercept_stderr", np.nan)
    df_reg = len(x) - 2
    t_crit_reg = t_dist.ppf(0.975, df_reg)
    slope_CI_low, slope_CI_high = slope + np.array([-1, 1]) * t_crit_reg * slope_stderr
    if not np.isnan(intercept_stderr):
        intercept_CI_low, intercept_CI_high = (
            intercept + np.array([-1, 1]) * t_crit_reg * intercept_stderr
        )

# correlations (with p-values)
pearson_gamma = pearson_gamma_p = spearman_gamma = spearman_gamma_p = np.nan
pearson_T = pearson_T_p = spearman_T = spearman_T_p = np.nan

if stats_df["gamma"].notna().any():
    pearson_gamma, pearson_gamma_p = pearsonr(
        stats_df["rho_input"], stats_df["gamma"]
    )
    spearman_gamma, spearman_gamma_p = spearmanr(
        stats_df["rho_input"], stats_df["gamma"]
    )
if stats_df["mean_T_theta"].notna().any():
    pearson_T, pearson_T_p = pearsonr(
        stats_df["rho_input"], stats_df["mean_T_theta"]
    )
    spearman_T, spearman_T_p = spearmanr(
        stats_df["rho_input"], stats_df["mean_T_theta"]
    )

# ---------------------------------------------------------------------
# 7.1  Global descriptive numbers
# ---------------------------------------------------------------------
min_gamma, max_gamma, mean_gamma, sd_gamma = (
    np.nanmin(stats_df["gamma"]),
    np.nanmax(stats_df["gamma"]),
    np.nanmean(stats_df["gamma"]),
    np.nanstd(stats_df["gamma"], ddof=1),
)

min_T, max_T, mean_T_global, sd_T_global = (
    np.nanmin(stats_df["mean_T_theta"]),
    np.nanmax(stats_df["mean_T_theta"]),
    np.nanmean(stats_df["mean_T_theta"]),
    np.nanstd(stats_df["mean_T_theta"], ddof=1),
)

# global skewness & kurtosis of per-ρ means (not of raw trajectories)
global_skew_T = skew(stats_df["mean_T_theta"], bias=False)
global_kurt_T = kurtosis(stats_df["mean_T_theta"], bias=False, fisher=True)

total_runs = stats_df["N_runs"].sum()
total_collapses = stats_df["N_collapse"].sum()

# ---------------------------------------------------------------------
# 8  Write plain-text summary file
# ---------------------------------------------------------------------
summary_path = RESULTS_DIR / "potential_well_escape_summary.txt"
fig_path = RESULTS_DIR / "potential_well_escape.png"

with summary_path.open("w", encoding="utf-8") as fh:
    # Parameters ------------------------------------------------------
    fh.write("# ------------------------------------------\n")
    fh.write("# Simulation parameters\n")
    fh.write("# ------------------------------------------\n")
    for k, v in BASE_PARAMS.items():
        fh.write(f"{k:25s}= {v}\n")
    fh.write(f"sim_time_window          = {SIM_TIME_WINDOW}\n")
    fh.write(f"rho_range                = {RHO_INPUT_VALUES_POT.tolist()}\n")
    fh.write(f"noise_intensity_proxy    = {NOISE_INTENSITY_PROXY:.4f}\n")
    fh.write(f"total_runs               = {total_runs}\n")
    fh.write(f"total_escapes            = {total_collapses}\n")

    # Regression ------------------------------------------------------
    fh.write("\n# ------------------------------------------\n")
    fh.write("# Linear regression: Γ vs 1/(1-ρ)\n")
    fh.write("# ------------------------------------------\n")
    fh.write(f"slope                    = {slope:.6e}\n")
    fh.write(f"intercept                = {intercept:.6e}\n")
    fh.write(f"R_squared                = {r_value**2:.4f}\n")
    fh.write(f"p_value                  = {p_value:.3e}\n")
    fh.write(f"slope_stderr             = {slope_stderr:.3e}\n")
    fh.write(f"intercept_stderr         = {intercept_stderr:.3e}\n")
    fh.write(f"slope_95%CI              = [{slope_CI_low:.6e}, {slope_CI_high:.6e}]\n")
    fh.write(
        f"intercept_95%CI          = [{intercept_CI_low:.6e}, {intercept_CI_high:.6e}]\n"
    )

    # Correlations ----------------------------------------------------
    fh.write("\n# ------------------------------------------\n")
    fh.write("# Correlations (ρ_input with response vars)\n")
    fh.write("# ------------------------------------------\n")
    fh.write(
        f"Pearson_r(ρ, Γ)          = {pearson_gamma:.4f}   (p = {pearson_gamma_p:.3e})\n"
    )
    fh.write(
        f"Spearman_ρ(ρ, Γ)         = {spearman_gamma:.4f}   (p = {spearman_gamma_p:.3e})\n"
    )
    fh.write(
        f"Pearson_r(ρ, ⟨T⟩)        = {pearson_T:.4f}   (p = {pearson_T_p:.3e})\n"
    )
    fh.write(
        f"Spearman_ρ(ρ, ⟨T⟩)       = {spearman_T:.4f}   (p = {spearman_T_p:.3e})\n"
    )

    # Global descriptive ---------------------------------------------
    fh.write("\n# ------------------------------------------\n")
    fh.write("# Global descriptive numbers (over all ρ)\n")
    fh.write("# ------------------------------------------\n")
    fh.write(f"min_Γ                    = {min_gamma:.6e}\n")
    fh.write(f"max_Γ                    = {max_gamma:.6e}\n")
    fh.write(f"mean_Γ                   = {mean_gamma:.6e}\n")
    fh.write(f"sd_Γ                     = {sd_gamma:.6e}\n")
    fh.write(f"min_⟨T⟩                  = {min_T:.2f}\n")
    fh.write(f"max_⟨T⟩                  = {max_T:.2f}\n")
    fh.write(f"mean_⟨T⟩                 = {mean_T_global:.2f}\n")
    fh.write(f"sd_⟨T⟩                   = {sd_T_global:.2f}\n")
    fh.write(f"skew_⟨T⟩                 = {global_skew_T:.4f}\n")
    fh.write(f"kurt_⟨T⟩                 = {global_kurt_T:.4f}\n")

    # Per-ρ table -----------------------------------------------------
    fh.write("\n# ------------------------------------------\n")
    fh.write("# Per-ρ descriptive statistics (95 % CI, skew, kurtosis)\n")
    fh.write("# ------------------------------------------\n")
    header = (
        "ρ     N_coll  frac  ⟨Tθ⟩  CI_low  CI_high  Γ  CI_low  CI_high  "
        "skew  kurt  q25  median  q75\n"
    )
    fh.write(header)
    for _, r in stats_df.iterrows():
        fh.write(
            f"{r.rho_input:5.3f} "
            f"{int(r.N_collapse):6d} "
            f"{r.collapse_frac:5.3f} "
            f"{r.mean_T_theta:7.2f} {r.T_theta_CI_low:7.2f} {r.T_theta_CI_high:7.2f} "
            f"{r.gamma:6.6f} {r.Gamma_CI_low:6.6f} {r.Gamma_CI_high:6.6f} "
            f"{r.skew_T_theta:5.2f} {r.kurt_T_theta:6.2f} "
            f"{r.q25_T_theta:6.1f} {r.median_T_theta:7.1f} {r.q75_T_theta:6.1f}\n"
        )

print(f"Summary written ➜ {summary_path}")

# ---------------------------------------------------------------------
# 9  Plot (same layout)
# ---------------------------------------------------------------------
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.set_ylabel(r"$\Gamma = 1/\mathbb{E}[T_\Theta]$", color="tab:red")
ax1.plot(
    stats_df["rho_input"], stats_df["gamma"], color="tab:red", marker="o"
)
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.grid(True, linestyle=":", alpha=0.7)
ax1.set_title(
    rf"Potential-well escape: $\Gamma$ vs. $\rho_{{\mathrm{{input}}}}$"
    rf"\n($\sigma_\beta^2+\sigma_F^2 \approx {NOISE_INTENSITY_PROXY:.2f}$)"
)

ax2 = ax1.twinx()
ax2.set_ylabel(r"$\mathbb{E}[T_\Theta]$", color="tab:blue")
ax2.plot(
    stats_df["rho_input"], stats_df["mean_T_theta"], color="tab:blue", marker="x"
)
ax2.tick_params(axis="y", labelcolor="tab:blue")

valid = stats_df["denominator_fac"].notna() & stats_df["gamma"].notna()
if valid.any():
    denom_sorted = stats_df.loc[valid, "denominator_fac"].values
    gamma_sorted = stats_df.loc[valid, "gamma"].values
    order = denom_sorted.argsort()
    ax3.plot(
        denom_sorted[order],
        gamma_sorted[order],
        color="tab:green",
        marker="s",
    )

ax3.set_xlabel(r"$1/(1-\rho_{\mathrm{input}})$")
ax3.set_ylabel(r"$\Gamma$", color="tab:green")
ax3.tick_params(axis="y", labelcolor="tab:green")
ax3.grid(True, linestyle=":", alpha=0.7)
ax3.set_title(r"$\Gamma$ vs. $1/(1-\rho_{\mathrm{input}})$")

fig.tight_layout()
fig.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Figure saved ➜ {fig_path}\n--- Finished ---")
