# Experiment 6A: Context-Dependency of the Hazard Rate

## 1. Overview

This repository contains the Python code for **Experiment 6A: Context-Dependency of the Hazard Rate: A Computational Test**, as described in Chapter 6, Section \ref{sec:diagnostics_hazard_context} of the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness*.

This computational experiment critically examines the relationship between coupled system dynamics and the hazard rate of collapse ($\Gamma$). It serves as a direct test of early theoretical formulations within the Threshold Dialectics framework, specifically a "Universal Horizon Bound" (UHB)-like model, which posited a simple relationship between the hazard rate, the speed of system drift, and the coupling of its internal levers.

The key finding of this experiment is that while the principle of risk amplification by detrimental coupling is robust, the specific mathematical form of this relationship is highly **context-dependent**. The experiment demonstrates that the simple UHB-like formula is an oversimplification for the specific system modeled here, highlighting the value of computational experiments in refining theory and motivating the need for more nuanced, system-specific, or ensemble-based diagnostic approaches.

## 2. Theoretical Context

A central hypothesis in Threshold Dialectics is that the risk of collapse is amplified not just by the speed at which a system's adaptive capacities drift, but also by how these drifts are coordinated. Early theoretical work suggested a UHB-like formulation for the hazard rate ($\Gamma = 1/\mathbb{E}[T_\Theta]$):

$$ \Gamma(t) \propto \frac{\mathcal{S}_{\text{eff}}^2(t)}{1 - \mathcal{C}(t)} $$

where:
- $\mathcal{S}_{\text{eff}}^2$ is the effective squared **Speed Index**, representing the intensity of drift.
- $\mathcal{C}(t)$ is the **Couple Index**, measuring the synchrony of lever drifts, with $\mathcal{C}(t) \to +1$ representing a particularly detrimental coupling pattern in this model.

This experiment is designed to isolate and test the denominator of this relationship, $1 - \mathcal{C}(t)$, by keeping the effective speed term approximately constant.

### Mapping to the Simulation Model

The experiment uses a canonical model from statistical physics: the escape of a particle from a potential well. This provides an abstract but mathematically tractable testbed for TD principles.

- **Levers**: The system's state is defined by "beta" ($\beta$) and "F_crit" ($F$), which are analogous to the TD levers for **Policy Precision** and **Energetic Slack**.
- **Collapse**: "Collapse" is defined as the particle escaping the well, i.e., when "beta" exceeds a threshold ("beta_escape_threshold"). The time to this event is $T_\Theta$.
- **Speed Index ($\mathcal{S}_{\text{eff}}^2$) Proxy**: The total intensity of the stochastic noise driving the system, $\sigma_\beta^2 + \sigma_F^2$, is held constant throughout the experiment, serving as a proxy for a constant Speed Index.
- **Couple Index ($\mathcal{C}$) Proxy**: The correlation between the noise terms affecting $\dot{\beta}$ and $\dot{F}$ is explicitly controlled by the parameter "rho_input" ($\rho_{\text{input}}$). By varying $\rho_{\text{input}}$ from -0.95 to 0.95, we directly manipulate the "coupling" and observe its effect on the hazard rate $\Gamma$.

The experiment thus tests the hypothesis that $\Gamma$ is linearly proportional to $1 / (1 - \rho_{\text{input}})$.

## 3. The Simulation Model

The script "chap6_escape_from_potential_well.py" implements a Langevin simulation of a particle in a 2D potential landscape $U(\beta, F)$. The particle's motion is governed by two coupled stochastic differential equations, solved numerically using the Euler-Maruyama method.

- **Potential $U(\beta, F)$**: Defined by a double-well potential in the $\beta$ dimension and a harmonic potential in the $F$ dimension, creating a "valley" from which the particle can escape.
- **Dynamics**: The system evolves according to the forces derived from the potential ($-\nabla U$) plus correlated Gaussian white noise terms.
- **Ensemble Simulation**: For each value of "rho_input", an ensemble of 500 independent simulations is run. The time to escape is recorded for each run, and censored if the particle does not escape within the maximum time window.
- **Analysis**: The script calculates the mean escape time $\mathbb{E}[T_\Theta]$, the hazard rate $\Gamma = 1/\mathbb{E}[T_\Theta]$, and other descriptive statistics for each "rho_input" value. It then performs a linear regression of $\Gamma$ vs. $1/(1-\rho_{\text{input}})$ and calculates correlations.

## 4. How to Run the Code

### Prerequisites

You will need Python 3.x and the following libraries:
- "numpy"
- "pandas"
- "matplotlib"
- "sympy"
- "scipy"
- "tqdm"

You can install these dependencies using pip:
"""bash
pip install numpy pandas matplotlib sympy scipy tqdm
"""

### Execution

To run the full simulation and generate the results, simply execute the Python script from your terminal:

"""bash
python chap6_escape_from_potential_well.py
"""

The script will display a progress bar as it iterates through the different values of "rho_input". Upon completion, it will print confirmation messages indicating where the results and figure have been saved.

## 5. Outputs

The script will create a "results" directory and save the following files inside it:

1.  **"potential_well_escape_stats.csv"**: A CSV file containing detailed statistics for each value of "rho_input", including mean escape time, hazard rate ($\Gamma$), confidence intervals, skewness, kurtosis, etc.

2.  **"potential_well_escape.png"**: A two-panel plot visualizing the key results, as seen in the book.
    - The top panel shows the Hazard Rate ($\Gamma$) and Mean Escape Time ($\mathbb{E}[T_\Theta]$) as a function of "rho_input".
    - The bottom panel shows the Hazard Rate ($\Gamma$) plotted against the transformed variable $1/(1-\rho_{\text{input}})$ to visually assess the UHB-like hypothesis.

3.  **"potential_well_escape_summary.txt"**: A plain-text summary of the simulation parameters and key findings. An example output is included below for reference.

### Example "summary.txt" Output

"""
# ------------------------------------------
# Simulation parameters
# ------------------------------------------
beta_initial             = -1.3
F_crit_initial           = 10.0
max_sim_steps            = 200000
dt                       = 0.1
beta_escape_threshold    = 2.0
sigma_beta_noise         = 0.5
sigma_F_crit_noise       = 0.5
...

# ------------------------------------------
# Linear regression: Γ vs 1/(1-ρ)
# ------------------------------------------
slope                    = 9.658805e-07
intercept                = 4.153499e-04
R_squared                = 0.0417
p_value                  = 3.747e-01
...

# ------------------------------------------
# Correlations (ρ_input with response vars)
# ------------------------------------------
Pearson_r(ρ, Γ)          = 0.1896   (p = 4.103e-01)
Spearman_ρ(ρ, Γ)         = 0.2039   (p = 3.753e-01)
...
"""

## 6. Key Findings and Interpretation

The simulation results provide strong evidence that the simple UHB-like hazard model is not a good fit for this specific system, highlighting the context-dependency of the relationship between lever dynamics and systemic risk.

1.  **Weak Linear Relationship**: The linear regression of $\Gamma$ vs. $1/(1-\rho_{\text{input}})$ yields an extremely low **R-squared value of 0.0417**. This indicates that the hypothesized relationship explains only about 4.2% of the variance in the hazard rate. The slope of the regression line is not statistically significant ($p = 0.375$).

2.  **No Significant Correlation**: Both Pearson and Spearman correlations between "rho_input" and the hazard rate $\Gamma$ are weak and not statistically significant. This confirms the absence of a simple, strong monotonic relationship.

3.  **Implications for Threshold Dialectics**:
    - This experiment **successfully refines the theory** by demonstrating that a simple, universal formula for hazard rate is unlikely to exist.
    - It highlights that the **nature of "detrimental coupling" is context-dependent**. A high positive correlation in one system might be highly detrimental, while in another (like this one), its effect may be more complex or less pronounced.
    - It underscores the value of **computational experiments** as a crucial tool for testing and challenging theoretical assumptions.
    - It motivates the need for **more nuanced, system-specific, or ensemble-based approaches** to risk assessment, as explored elsewhere in *Threshold Dialectics*.

In summary, this experiment does not invalidate the core TD principle that coupled lever dynamics are crucial for understanding risk. Instead, it provides a powerful, data-driven argument that the specific *expression* of that risk is a complex, emergent property of the system itself.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.