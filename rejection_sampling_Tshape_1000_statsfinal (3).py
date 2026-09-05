"""
Rejection-sampling / thinning simulation for a T-shaped activation model
with ERK negative feedback, using the same statistical analysis workflow
as the unified Gillespie script.

Main points:
  1. Keep the rejection-sampling / thinning construction:
       - generate activation candidates from a dominating Poisson process;
       - generate death candidates from a homogeneous Poisson process;
       - accept/reject candidates using the active-zone and ERK-zone rules.
  2. Burn-in first: simulate 500 observed deaths but do not use them in statistics.
  3. Then collect and analyze the next 1000 observed deaths.
  4. For each parameter setting, run several independent seeds.
  5. Save the same kind of statistics as the Gillespie file:
       - number of analyzed observed deaths,
       - time needed for the analyzed 1000 deaths after burn-in,
       - mean / median inter-death time after burn-in,
       - 2D spatial histogram,
       - spatial CV,
       - chi-square uniformity diagnostic on a 10 x 10 grid,
       - T-zone density ratio,
       - aggregated mean/std/min/max across seeds.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import math
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

try:
    from scipy.stats import chi2 as scipy_chi2
except Exception:  # pragma: no cover
    scipy_chi2 = None


# ============================================================
# 1. Parameters
# ============================================================


@dataclass(frozen=True)
class Parameters:
    # Reproducibility
    seed: int = 123

    # Spatial domain W = [0, Lx] x [0, Ly]
    Lx: float = 10.0
    Ly: float = 10.0

    # Burn-in and stopping rule.
    # The simulation runs until burn_in_deaths + target_deaths observed deaths.
    # Statistics are computed only on the target_deaths after burn-in.
    burn_in_deaths: int = 500
    target_deaths: int = 1000

    # Activation intensities with fixed T-shaped extension.
    # lambda_a_1 is the dominating intensity and active-zone intensity.
    lambda_a_1: float = 5.00
    lambda_a_T: float = 0.50
    lambda_a_c: float = 0.05

    # Candidate death intensity on W.
    lambda_d: float = 1.0

    # Activation marks: R^a ~ Exp(beta_a_R), T^a ~ Exp(beta_a_T).
    beta_a_R: float = 2.5
    beta_a_T: float = 1.2

    # ERK marks after accepted deaths: R^d ~ Exp(beta_d_R), T^d ~ Exp(beta_d_T).
    # In this rejection-sampling version, T^d is pre-sampled as an exponential lifetime.
    beta_d_R: float = 2.0
    beta_d_T: float = 0.8

    # Plotting/statistics choices.
    bins_time: int = 30
    bins_space: int = 20       # used for the 2D plot and descriptive spatial CV
    bins_chi2: int = 10        # coarser bins for a more reliable chi-square test

    # Safety stop, only to avoid infinite runs if bad parameters are chosen.
    max_proposals: int = 10_000_000

    # Rejection-sampling-specific block length.
    # Candidates are generated block by block from Poisson processes on W x [t0, t1].
    block_duration: float = 1.0
    max_blocks: int = 200_000

    @property
    def area_W(self) -> float:
        return self.Lx * self.Ly

    @property
    def total_deaths_to_simulate(self) -> int:
        return self.burn_in_deaths + self.target_deaths


@dataclass
class SimulationResult:
    scenario: str
    params: Parameters

    # All accepted observed deaths, including burn-in deaths.
    death_x_all: np.ndarray
    death_y_all: np.ndarray
    death_t_all: np.ndarray

    # Analyzed observed deaths, after burn-in.
    death_x: np.ndarray
    death_y: np.ndarray
    death_t: np.ndarray
    death_t_since_burnin: np.ndarray

    # Accepted activation centers, historical.
    activation_x: np.ndarray
    activation_y: np.ndarray
    activation_t: np.ndarray

    final_time: float
    burn_in_time: float
    analysis_time_span: float

    n_observed_deaths_total: int
    n_observed_deaths_analyzed: int
    n_burn_in_deaths: int
    n_accepted_activations: int
    n_activation_candidates: int
    n_death_candidates: int
    n_activation_rejected: int
    n_death_rejected: int
    n_death_rejected_outside_active: int
    n_death_rejected_by_erk: int
    n_proposals: int
    n_blocks: int
    stopped_by_target: bool


# ============================================================
# 2. Geometry: fixed T-shaped zone and current disk unions
# ============================================================


def validate_params(p: Parameters) -> None:
    if not (0.0 <= p.lambda_a_c <= p.lambda_a_T <= p.lambda_a_1):
        raise ValueError("Need 0 <= lambda_a_c <= lambda_a_T <= lambda_a_1 for thinning.")
    if min(p.lambda_d, p.beta_a_R, p.beta_a_T, p.beta_d_R, p.beta_d_T) <= 0:
        raise ValueError("Rates lambda_d, beta_a_R, beta_a_T, beta_d_R, beta_d_T must be positive.")
    if p.Lx <= 0 or p.Ly <= 0:
        raise ValueError("Need positive domain size.")
    if p.burn_in_deaths < 0 or p.target_deaths <= 0:
        raise ValueError("Need burn_in_deaths >= 0 and target_deaths > 0.")
    if p.bins_time <= 0 or p.bins_space <= 0 or p.bins_chi2 <= 0:
        raise ValueError("Need positive histogram bin numbers.")
    if p.block_duration <= 0 or p.max_blocks <= 0 or p.max_proposals <= 0:
        raise ValueError("Need positive block_duration, max_blocks, and max_proposals.")


def t_zone_bounds(p: Parameters) -> Tuple[float, float, float, float]:
    """Return the 3x3-grid cut points defining the fixed T-shaped zone."""
    x1 = p.Lx / 3.0
    x2 = 2.0 * p.Lx / 3.0
    y1 = p.Ly / 3.0
    y2 = 2.0 * p.Ly / 3.0
    return x1, x2, y1, y2


def t_zone_area(p: Parameters) -> float:
    """Area of the fixed T-zone: middle vertical column plus left middle arm."""
    x1, x2, y1, y2 = t_zone_bounds(p)
    middle_column_area = (x2 - x1) * p.Ly
    left_arm_area = x1 * (y2 - y1)
    return middle_column_area + left_arm_area


def is_inside_T_zone(x: float, y: float, p: Parameters) -> bool:
    """Fixed T-zone: middle vertical column plus left middle arm."""
    x1, x2, y1, y2 = t_zone_bounds(p)
    inside_middle_column = (x1 <= x <= x2) and (0.0 <= y <= p.Ly)
    inside_left_arm = (0.0 <= x <= x1) and (y1 <= y <= y2)
    return inside_middle_column or inside_left_arm


def is_inside_current_disks(x: float, y: float, centers_x: List[float], centers_y: List[float], radii: List[float]) -> bool:
    """Return True if (x,y) is inside at least one currently alive disk."""
    for cx, cy, r in zip(centers_x, centers_y, radii):
        if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
            return True
    return False


def prune_expired_disks(t: float, centers_x: List[float], centers_y: List[float], radii: List[float], end_times: List[float]) -> None:
    """
    Remove disks whose lifetime ended strictly before t.

    We keep disks with end_time >= t, corresponding to the convention
        S < t <= S + T.
    Since accepted disks are inserted only after their birth time, all current disks
    automatically have S < t at later candidate events with probability one.
    """
    i = len(end_times) - 1
    while i >= 0:
        if end_times[i] < t:
            centers_x.pop(i)
            centers_y.pop(i)
            radii.pop(i)
            end_times.pop(i)
        i -= 1


def add_t_zone_overlay(ax: plt.Axes, p: Parameters) -> None:
    """Add dashed rectangles showing the fixed T-zone."""
    x1, x2, y1, y2 = t_zone_bounds(p)
    ax.add_patch(Rectangle((x1, 0.0), x2 - x1, p.Ly, fill=False, linewidth=1.2, linestyle="--"))
    ax.add_patch(Rectangle((0.0, y1), x1, y2 - y1, fill=False, linewidth=1.2, linestyle="--"))


# ============================================================
# 3. Rejection sampling / thinning simulation
# ============================================================


def simulate_one(scenario: str, p: Parameters) -> SimulationResult:
    """
    Simulate until p.burn_in_deaths + p.target_deaths observed deaths are accepted.

    In each block [t0, t1], we generate:
        - activation candidates from PPP(lambda_a_1) on W x [t0, t1],
        - death candidates from PPP(lambda_d) on W x [t0, t1].

    Candidate events are processed in chronological order inside the block.

    Activation thinning probability:
        1                                      if x is inside current active zones,
        lambda_a_T / lambda_a_1               if x is in the fixed T-zone but outside active zones,
        lambda_a_c / lambda_a_1               otherwise.

    Death candidates are accepted iff they are inside active zones and outside ERK zones.
    Accepted deaths create ERK protection disks with pre-sampled exponential lifetimes.
    """
    validate_params(p)
    rng = np.random.default_rng(p.seed)

    # Current active disks V^a_t.
    active_x: List[float] = []
    active_y: List[float] = []
    active_r: List[float] = []
    active_end: List[float] = []

    # Current ERK protection disks V^p_t.
    erk_x: List[float] = []
    erk_y: List[float] = []
    erk_r: List[float] = []
    erk_end: List[float] = []

    # Historical accepted activations.
    activation_x: List[float] = []
    activation_y: List[float] = []
    activation_t: List[float] = []

    # Accepted observed deaths, including burn-in.
    death_x: List[float] = []
    death_y: List[float] = []
    death_t: List[float] = []

    n_activation_candidates = 0
    n_death_candidates = 0
    n_activation_rejected = 0
    n_death_rejected_outside_active = 0
    n_death_rejected_by_erk = 0
    n_proposals = 0

    t0 = 0.0
    n_blocks_used = 0

    for block_index in range(p.max_blocks):
        n_blocks_used = block_index + 1
        t1 = t0 + p.block_duration

        # Candidate activations in W x [t0,t1].
        n_act = int(rng.poisson(p.lambda_a_1 * p.area_W * p.block_duration))
        act_t = rng.uniform(t0, t1, size=n_act)
        act_x = rng.uniform(0.0, p.Lx, size=n_act)
        act_y = rng.uniform(0.0, p.Ly, size=n_act)
        act_u = rng.uniform(0.0, 1.0, size=n_act)
        act_r = rng.exponential(scale=1.0 / p.beta_a_R, size=n_act)
        act_tau = rng.exponential(scale=1.0 / p.beta_a_T, size=n_act)

        # Candidate deaths in W x [t0,t1].
        n_death = int(rng.poisson(p.lambda_d * p.area_W * p.block_duration))
        death_cand_t = rng.uniform(t0, t1, size=n_death)
        death_cand_x = rng.uniform(0.0, p.Lx, size=n_death)
        death_cand_y = rng.uniform(0.0, p.Ly, size=n_death)
        death_cand_r = rng.exponential(scale=1.0 / p.beta_d_R, size=n_death)
        death_cand_tau = rng.exponential(scale=1.0 / p.beta_d_T, size=n_death)

        n_activation_candidates += n_act
        n_death_candidates += n_death
        n_proposals += n_act + n_death

        if n_proposals > p.max_proposals:
            raise RuntimeError(
                f"Safety stop: processed {n_proposals} candidates before reaching "
                f"{p.total_deaths_to_simulate} observed deaths. Increase max_proposals."
            )

        # Combine event times and process the two independent candidate PPPs chronologically.
        # type 0 = activation candidate, type 1 = death candidate.
        event_t = np.concatenate([act_t, death_cand_t])
        event_type = np.concatenate([np.zeros(n_act, dtype=np.int8), np.ones(n_death, dtype=np.int8)])
        event_index = np.concatenate([np.arange(n_act, dtype=np.int64), np.arange(n_death, dtype=np.int64)])

        for pos in np.argsort(event_t):
            if len(death_t) >= p.total_deaths_to_simulate:
                break

            t = float(event_t[pos])
            kind = int(event_type[pos])
            idx = int(event_index[pos])
            
            prune_expired_disks(t, active_x, active_y, active_r, active_end)
            prune_expired_disks(t, erk_x, erk_y, erk_r, erk_end)

            if kind == 0:
                x = float(act_x[idx])
                y = float(act_y[idx])
                u = float(act_u[idx])
                r = float(act_r[idx])
                tau = float(act_tau[idx])

                inside_active = is_inside_current_disks(x, y, active_x, active_y, active_r)
                if inside_active:
                    accept_prob = 1.0
                elif is_inside_T_zone(x, y, p):
                    accept_prob = p.lambda_a_T / p.lambda_a_1
                else:
                    accept_prob = p.lambda_a_c / p.lambda_a_1

                if u <= accept_prob:
                    activation_x.append(x)
                    activation_y.append(y)
                    activation_t.append(t)
                    active_x.append(x)
                    active_y.append(y)
                    active_r.append(r)
                    active_end.append(t + tau)
                else:
                    n_activation_rejected += 1

            else:
                x = float(death_cand_x[idx])
                y = float(death_cand_y[idx])
                r = float(death_cand_r[idx])
                tau = float(death_cand_tau[idx])

                inside_active = is_inside_current_disks(x, y, active_x, active_y, active_r)
                inside_erk = is_inside_current_disks(x, y, erk_x, erk_y, erk_r)

                if inside_active and not inside_erk:
                    death_x.append(x)
                    death_y.append(y)
                    death_t.append(t)
                    erk_x.append(x)
                    erk_y.append(y)
                    erk_r.append(r)
                    erk_end.append(t + tau)
                else:
                    if inside_erk:
                        n_death_rejected_by_erk += 1
                    else:
                        n_death_rejected_outside_active += 1

        if len(death_t) >= p.total_deaths_to_simulate:
            break

        t0 = t1
    else:
        raise RuntimeError(
            f"Only {len(death_t)} observed deaths after {p.max_blocks} blocks. "
            "Increase max_blocks, max_proposals, lambda_d, lambda_a_T/lambda_a_c, "
            "or active-zone coverage."
        )

    death_x_all = np.asarray(death_x[:p.total_deaths_to_simulate])
    death_y_all = np.asarray(death_y[:p.total_deaths_to_simulate])
    death_t_all = np.asarray(death_t[:p.total_deaths_to_simulate])

    if len(death_t_all) > p.burn_in_deaths:
        burn_in_time = 0.0 if p.burn_in_deaths == 0 else float(death_t_all[p.burn_in_deaths - 1])
        death_x_an = death_x_all[p.burn_in_deaths:]
        death_y_an = death_y_all[p.burn_in_deaths:]
        death_t_an = death_t_all[p.burn_in_deaths:]
        death_t_since_burnin = death_t_an - burn_in_time
        analysis_time_span = float(death_t_all[-1] - burn_in_time) if len(death_t_an) else 0.0
    else:
        burn_in_time = float(death_t_all[-1]) if len(death_t_all) else 0.0
        death_x_an = np.array([])
        death_y_an = np.array([])
        death_t_an = np.array([])
        death_t_since_burnin = np.array([])
        analysis_time_span = 0.0

    n_death_rejected = n_death_rejected_outside_active + n_death_rejected_by_erk

    return SimulationResult(
        scenario=scenario,
        params=p,
        death_x_all=death_x_all,
        death_y_all=death_y_all,
        death_t_all=death_t_all,
        death_x=death_x_an,
        death_y=death_y_an,
        death_t=death_t_an,
        death_t_since_burnin=death_t_since_burnin,
        activation_x=np.asarray(activation_x),
        activation_y=np.asarray(activation_y),
        activation_t=np.asarray(activation_t),
        final_time=float(death_t_all[-1]) if len(death_t_all) else 0.0,
        burn_in_time=burn_in_time,
        analysis_time_span=analysis_time_span,
        n_observed_deaths_total=len(death_t_all),
        n_observed_deaths_analyzed=len(death_t_an),
        n_burn_in_deaths=min(p.burn_in_deaths, len(death_t_all)),
        n_accepted_activations=len(activation_t),
        n_activation_candidates=n_activation_candidates,
        n_death_candidates=n_death_candidates,
        n_activation_rejected=n_activation_rejected,
        n_death_rejected=n_death_rejected,
        n_death_rejected_outside_active=n_death_rejected_outside_active,
        n_death_rejected_by_erk=n_death_rejected_by_erk,
        n_proposals=n_proposals,
        n_blocks=n_blocks_used,
        stopped_by_target=(len(death_t_all) >= p.total_deaths_to_simulate),
    )


# ============================================================
# 4. Statistics and plots on the observed death process
# ============================================================


def spatial_histogram(result: SimulationResult, bins: int) -> np.ndarray:
    p = result.params
    H, _, _ = np.histogram2d(
        result.death_x,
        result.death_y,
        bins=bins,
        range=[[0.0, p.Lx], [0.0, p.Ly]],
    )
    return H


def chi2_stats_from_histogram(H: np.ndarray) -> Tuple[float, int, float, float]:
    """Return chi2 statistic, df, chi2/df, and p-value if scipy is available."""
    n = float(np.sum(H))
    k = int(H.size)
    expected = n / k if k > 0 else np.nan
    if not np.isfinite(expected) or expected <= 0:
        return np.nan, k - 1, np.nan, np.nan
    chi2_uniform = float(np.sum((H - expected) ** 2 / expected))
    df = k - 1
    chi2_per_df = chi2_uniform / df if df > 0 else np.nan
    p_value = float(scipy_chi2.sf(chi2_uniform, df)) if scipy_chi2 is not None and df > 0 else np.nan
    return chi2_uniform, df, chi2_per_df, p_value


def t_zone_density_stats(result: SimulationResult) -> Dict[str, float | int]:
    p = result.params
    in_T = np.array([is_inside_T_zone(float(x), float(y), p) for x, y in zip(result.death_x, result.death_y)])
    n_T = int(np.sum(in_T))
    n_out = int(len(in_T) - n_T)
    area_T = t_zone_area(p)
    area_out = p.area_W - area_T
    density_T = n_T / area_T if area_T > 0 else np.nan
    density_out = n_out / area_out if area_out > 0 else np.nan
    ratio = density_T / density_out if density_out > 0 else np.nan
    return {
        "t_zone_area": area_T,
        "outside_t_zone_area": area_out,
        "t_zone_observed_deaths": n_T,
        "outside_t_zone_observed_deaths": n_out,
        "t_zone_observed_fraction": n_T / len(in_T) if len(in_T) else np.nan,
        "t_zone_area_fraction": area_T / p.area_W,
        "t_zone_death_density": density_T,
        "outside_t_zone_death_density": density_out,
        "t_zone_density_ratio": ratio,
    }


def split_scenario_replicate(scenario: str) -> Tuple[str, int]:
    """Split names such as 'baseline__rep03' into ('baseline', 3)."""
    if "__rep" not in scenario:
        return scenario, 1
    base_name, rep_part = scenario.rsplit("__rep", 1)
    try:
        return base_name, int(rep_part)
    except ValueError:
        return base_name, 1


def death_statistics(result: SimulationResult) -> Dict[str, float | int | str | bool]:
    p = result.params
    base_scenario, replicate = split_scenario_replicate(result.scenario)

    if len(result.death_t) > 0:
        inter_death_after_burnin = np.diff(np.concatenate([[result.burn_in_time], result.death_t]))
    else:
        inter_death_after_burnin = np.array([])

    H_plot = spatial_histogram(result, p.bins_space)
    expected_plot = result.n_observed_deaths_analyzed / float(p.bins_space * p.bins_space)
    spatial_cv = float(np.std(H_plot) / np.mean(H_plot)) if np.mean(H_plot) > 0 else np.nan
    empty_bins = int(np.sum(H_plot == 0))

    H_chi2 = spatial_histogram(result, p.bins_chi2)
    expected_chi2 = result.n_observed_deaths_analyzed / float(p.bins_chi2 * p.bins_chi2)
    chi2_coarse, df_coarse, chi2_per_df_coarse, p_value_coarse = chi2_stats_from_histogram(H_chi2)

    t_stats = t_zone_density_stats(result)
    k_plot = p.bins_space * p.bins_space
    expected_cv_uniform_plot = np.sqrt((k_plot - 1) / result.n_observed_deaths_analyzed)
    spatial_cv_ratio_to_uniform = spatial_cv / expected_cv_uniform_plot

    activation_acceptance_rate = (
        result.n_accepted_activations / result.n_activation_candidates
        if result.n_activation_candidates > 0 else np.nan
    )
    death_rejection_rate = (
        result.n_death_rejected / result.n_death_candidates
        if result.n_death_candidates > 0 else np.nan
    )
    death_rejection_rate_outside_active = (
        result.n_death_rejected_outside_active / result.n_death_candidates
        if result.n_death_candidates > 0 else np.nan
    )
    death_rejection_rate_by_erk = (
        result.n_death_rejected_by_erk / result.n_death_candidates
        if result.n_death_candidates > 0 else np.nan
    )
    death_acceptance_rate_total = (
        result.n_observed_deaths_total / result.n_death_candidates
        if result.n_death_candidates > 0 else np.nan
    )

    return {
        "scenario": result.scenario,
        "base_scenario": base_scenario,
        "replicate": replicate,
        "seed": p.seed,
        "burn_in_deaths": p.burn_in_deaths,
        "analyzed_target_deaths": p.target_deaths,
        "total_deaths_simulated": result.n_observed_deaths_total,
        "n_observed_deaths_analyzed": result.n_observed_deaths_analyzed,
        "stopped_by_target": result.stopped_by_target,
        "burn_in_time": result.burn_in_time,
        "raw_final_time": result.final_time,
        "total_time_for_analyzed_1000_deaths_after_burnin": result.analysis_time_span,
        "observed_death_rate_after_burnin": result.n_observed_deaths_analyzed / result.analysis_time_span if result.analysis_time_span > 0 else np.nan,
        "mean_inter_death_time_after_burnin": float(np.mean(inter_death_after_burnin)) if len(inter_death_after_burnin) else np.nan,
        "median_inter_death_time_after_burnin": float(np.median(inter_death_after_burnin)) if len(inter_death_after_burnin) else np.nan,
        "accepted_activations_total": result.n_accepted_activations,
        "activation_candidates": result.n_activation_candidates,
        "activation_rejected": result.n_activation_rejected,
        "activation_acceptance_rate": activation_acceptance_rate,
        "death_candidates": result.n_death_candidates,
        "death_rejected": result.n_death_rejected,
        "death_rejected_outside_active": result.n_death_rejected_outside_active,
        "death_rejected_by_erk": result.n_death_rejected_by_erk,
        "death_rejection_rate": death_rejection_rate,
        "death_rejection_rate_outside_active": death_rejection_rate_outside_active,
        "death_rejection_rate_by_erk": death_rejection_rate_by_erk,
        "death_acceptance_rate_total": death_acceptance_rate_total,
        "all_proposals": result.n_proposals,
        "n_blocks": result.n_blocks,
        "spatial_expected_cv_if_uniform_plot_bins": expected_cv_uniform_plot,
        "spatial_cv_ratio_to_uniform_plot_bins": spatial_cv_ratio_to_uniform,
        "spatial_bins_per_axis_for_plot": p.bins_space,
        "spatial_expected_count_per_plot_bin_if_uniform": expected_plot,
        "spatial_coefficient_of_variation_plot_bins": spatial_cv,
        "spatial_empty_bins_plot_bins": empty_bins,
        "spatial_bins_per_axis_for_chi2_test": p.bins_chi2,
        "spatial_expected_count_per_chi2_bin_if_uniform": expected_chi2,
        "spatial_chi2_statistic_against_uniform_counts_chi2_bins": chi2_coarse,
        "spatial_chi2_df_chi2_bins": df_coarse,
        "spatial_chi2_per_df_chi2_bins": chi2_per_df_coarse,
        "spatial_chi2_p_value_chi2_bins": p_value_coarse,
        **t_stats,
        "lambda_a_1": p.lambda_a_1,
        "lambda_a_T": p.lambda_a_T,
        "lambda_a_c": p.lambda_a_c,
        "lambda_a_T_over_lambda_a_c": p.lambda_a_T / p.lambda_a_c if p.lambda_a_c > 0 else np.inf,
        "lambda_d": p.lambda_d,
        "beta_a_R": p.beta_a_R,
        "beta_a_T": p.beta_a_T,
        "beta_d_R": p.beta_d_R,
        "beta_d_T": p.beta_d_T,
    }


# ============================================================
# 5. Output helpers
# ============================================================


def save_death_events_csv(result: SimulationResult, out_dir: Path) -> Path:
    path = out_dir / f"{result.scenario}_observed_deaths_after_burnin.csv"
    p = result.params
    in_T = [is_inside_T_zone(float(x), float(y), p) for x, y in zip(result.death_x, result.death_y)]
    raw_indices = np.arange(p.burn_in_deaths + 1, p.burn_in_deaths + result.n_observed_deaths_analyzed + 1)
    analysis_indices = np.arange(1, result.n_observed_deaths_analyzed + 1)
    data = np.column_stack([
        analysis_indices,
        raw_indices,
        result.death_t,
        result.death_t_since_burnin,
        result.death_x,
        result.death_y,
        np.asarray(in_T, dtype=int),
    ])
    header = "analysis_death_index,raw_death_index,raw_death_time,time_since_burnin,x,y,in_T_zone"
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", "%.10f", "%.10f", "%.10f", "%.10f", "%d"],
    )
    return path


def wrap_underscore_name(name: str, width: int = 55) -> str:
    """Wrap a long scenario name at underscores so that figure titles are not clipped."""
    base_name, replicate = split_scenario_replicate(name)
    parts = base_name.split("_")
    lines: List[str] = []
    current = ""
    for part in parts:
        candidate = part if current == "" else current + "_" + part
        if len(candidate) > width and current != "":
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    if "__rep" in name:
        lines.append(f"rep{replicate:02d}")
    return "\n".join(lines)


def plot_time_histogram(result: SimulationResult, out_dir: Path) -> Path:
    p = result.params
    path = out_dir / f"{result.scenario}_time_histogram_after_burnin.png"
    scenario_title = wrap_underscore_name(result.scenario, width=55)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.hist(result.death_t_since_burnin, bins=p.bins_time, range=(0.0, result.analysis_time_span))
    ax.set_xlabel("time after burn-in")
    ax.set_ylabel("number of observed deaths")
    ax.set_title(
        f"Observed cell death times after burn-in\n"
        f"{scenario_title}\n"
        f"burn-in = {p.burn_in_deaths}, N = {result.n_observed_deaths_analyzed}, "
        f"time span = {result.analysis_time_span:.3f}",
        fontsize=11,
        pad=12,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    return path


def plot_spatial_histogram_2d(result: SimulationResult, out_dir: Path, colorbar_vmax: int) -> Path:
    p = result.params
    path = out_dir / f"{result.scenario}_space_histogram_2d_after_burnin.png"
    H = spatial_histogram(result, p.bins_space)
    scenario_title = wrap_underscore_name(result.scenario, width=55)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(
        H.T,
        origin="lower",
        extent=[0.0, p.Lx, 0.0, p.Ly],
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=colorbar_vmax,
    )
    add_t_zone_overlay(ax, p)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"2D histogram of observed death locations\n"
        f"{scenario_title}\n"
        f"after burn-in {p.burn_in_deaths}, {p.bins_space} x {p.bins_space} bins, "
        f"N = {result.n_observed_deaths_analyzed}",
        fontsize=11,
        pad=12,
    )
    cbar = fig.colorbar(image, ax=ax, label="number of observed deaths", shrink=0.88)
    cbar.set_ticks(np.arange(0, colorbar_vmax + 1, max(1, colorbar_vmax // 7)))
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    return path


def plot_cumulative_deaths(result: SimulationResult, out_dir: Path) -> Path:
    path = out_dir / f"{result.scenario}_cumulative_deaths_after_burnin.png"
    scenario_title = wrap_underscore_name(result.scenario, width=65)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.step(result.death_t_since_burnin, np.arange(1, result.n_observed_deaths_analyzed + 1), where="post")
    ax.set_xlabel("time after burn-in")
    ax.set_ylabel("cumulative observed deaths after burn-in")
    ax.set_title(
        f"Cumulative observed deaths after burn-in\n{scenario_title}",
        fontsize=11,
        pad=12,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    return path


def save_summary_csv(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "parameter_sweep_summary_after_burnin.csv"
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _mean_std_min_max(values: List[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, std, float(np.min(arr)), float(np.max(arr))


def aggregate_replicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate replicate-level rows by base_scenario."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        base_name = str(row.get("base_scenario", row.get("scenario", "unknown")))
        groups.setdefault(base_name, []).append(row)

    metrics = [
        "total_time_for_analyzed_1000_deaths_after_burnin",
        "observed_death_rate_after_burnin",
        "spatial_coefficient_of_variation_plot_bins",
        "spatial_cv_ratio_to_uniform_plot_bins",
        "spatial_chi2_per_df_chi2_bins",
        "spatial_chi2_p_value_chi2_bins",
        "t_zone_density_ratio",
        "death_rejected",
        "death_rejection_rate",
        "death_rejection_rate_outside_active",
        "death_rejection_rate_by_erk",
        "accepted_activations_total",
        "activation_acceptance_rate",
    ]

    aggregated_rows: List[Dict[str, Any]] = []
    for base_name, group in groups.items():
        first = group[0]
        out: Dict[str, Any] = {
            "base_scenario": base_name,
            "n_replicates": len(group),
            "n_finished_target": sum(bool(r.get("stopped_by_target", False)) for r in group),
            "lambda_a_1": first.get("lambda_a_1", np.nan),
            "lambda_a_T": first.get("lambda_a_T", np.nan),
            "lambda_a_c": first.get("lambda_a_c", np.nan),
            "lambda_a_T_over_lambda_a_c": first.get("lambda_a_T_over_lambda_a_c", np.nan),
            "lambda_d": first.get("lambda_d", np.nan),
            "beta_a_R": first.get("beta_a_R", np.nan),
            "beta_a_T": first.get("beta_a_T", np.nan),
            "beta_d_R": first.get("beta_d_R", np.nan),
            "beta_d_T": first.get("beta_d_T", np.nan),
        }

        p_values = [_safe_float(r.get("spatial_chi2_p_value_chi2_bins", np.nan)) for r in group]
        out["fraction_chi2_p_value_below_0p05"] = (
            float(np.mean([p < 0.05 for p in p_values if np.isfinite(p)]))
            if any(np.isfinite(p) for p in p_values) else np.nan
        )

        for metric in metrics:
            values = [_safe_float(r.get(metric, np.nan)) for r in group]
            mean, std, min_v, max_v = _mean_std_min_max(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_min"] = min_v
            out[f"{metric}_max"] = max_v

        aggregated_rows.append(out)

    return aggregated_rows


def save_aggregate_summary_csv(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    path = out_dir / "parameter_sweep_summary_by_parameter_setting_after_burnin.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_analysis_report(rows: List[Dict[str, Any]], out_dir: Path, colorbar_vmax: int, aggregated_rows: List[Dict[str, Any]] | None = None) -> Path:
    path = out_dir / "spatial_uniformity_analysis_report.md"
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return path

    first = rows[0]
    n = int(first["n_observed_deaths_analyzed"])
    bins_plot = int(first["spatial_bins_per_axis_for_plot"])
    bins_chi2 = int(first["spatial_bins_per_axis_for_chi2_test"])
    k_plot = bins_plot * bins_plot
    k_chi2 = bins_chi2 * bins_chi2
    df_chi2 = k_chi2 - 1
    expected_plot = n / k_plot
    expected_chi2 = n / k_chi2
    cv_ref = math.sqrt((k_plot - 1) / n)
    sd_reduced = math.sqrt(2.0 / df_chi2)

    lines: List[str] = []
    lines.append("# Spatial uniformity analysis after burn-in: rejection-sampling version\n\n")
    lines.append(f"All 2D histograms use the same colorbar scale: 0 to {colorbar_vmax} observed deaths per bin.\n\n")
    lines.append(
        f"Burn-in removes the initial transient period: the first {int(first['burn_in_deaths'])} "
        f"observed deaths are simulated but not used for statistics. The reported statistics use "
        f"the next {n} observed deaths.\n\n"
    )
    lines.append("## Interpretation rules\n\n")
    lines.append("- `spatial_coefficient_of_variation_plot_bins = std(H)/mean(H)`: close to 0 means more uniform; larger values mean more spatial heterogeneity.\n")
    lines.append("- `spatial_chi2_per_df_chi2_bins`: close to 1 is compatible with uniform counts; much larger than 1 indicates over-dispersion / non-uniformity.\n")
    lines.append("- `spatial_chi2_p_value_chi2_bins`: small p-value, for example < 0.05, rejects spatial uniformity on the chosen 10 x 10 grid.\n")
    lines.append("- `t_zone_density_ratio`: values > 1 mean deaths are denser in the T-zone than outside; values < 1 mean the opposite.\n")
    lines.append("- `death_rejection_rate`: fraction of death candidates rejected, split into outside-active rejection and ERK rejection.\n\n")

    lines.append("## Spatial coefficient of variation\n\n")
    lines.append("The spatial coefficient of variation is\n\n")
    lines.append("```latex\nCV_{space}=\\frac{sd(H_1,\\ldots,H_k)}{mean(H_1,\\ldots,H_k)}.\n```\n\n")
    lines.append(
        f"For the {bins_plot} x {bins_plot} plot grid, k = {k_plot} and n = {n}, so "
        f"CV_uniform ≈ sqrt(({k_plot}-1)/{n}) = {cv_ref:.3f}.\n\n"
    )

    lines.append("## Chi-square uniformity diagnostic\n\n")
    lines.append(
        f"The {bins_plot} x {bins_plot} grid has expected count {expected_plot:.2f} per bin. "
        f"The formal diagnostic uses the coarser {bins_chi2} x {bins_chi2} grid, "
        f"where expected count is {expected_chi2:.2f} per bin.\n\n"
    )
    lines.append("```latex\nX^2 = \\sum_{i=1}^{k} \\frac{(O_i-E_i)^2}{E_i}.\n```\n\n")
    lines.append(
        f"Under spatial uniformity, X^2/df should fluctuate around 1. "
        f"Here df = {df_chi2}, so sqrt(2/df) = {sd_reduced:.3f}.\n\n"
    )

    lines.append("## Replicate-level results\n\n")
    lines.append(
        "| scenario | time for analyzed 1000 deaths | CV 20x20 | chi2/df 10x10 | p-value 10x10 | T-zone density ratio | death rejection | empty bins 20x20 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for r in rows:
        lines.append(
            f"| {r['scenario']} | "
            f"{float(r['total_time_for_analyzed_1000_deaths_after_burnin']):.3f} | "
            f"{float(r['spatial_coefficient_of_variation_plot_bins']):.3f} | "
            f"{float(r['spatial_chi2_per_df_chi2_bins']):.3f} | "
            f"{float(r['spatial_chi2_p_value_chi2_bins']):.3g} | "
            f"{float(r['t_zone_density_ratio']):.3f} | "
            f"{float(r['death_rejection_rate']):.3f} | "
            f"{int(r['spatial_empty_bins_plot_bins'])} |\n"
        )

    if aggregated_rows:
        lines.append("\n## Aggregated results by parameter setting\n\n")
        lines.append(
            "| base scenario | n | fraction p<0.05 | time mean | chi2/df mean | p-value mean | T-zone ratio mean | death rejection mean |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for r in aggregated_rows:
            lines.append(
                f"| {r['base_scenario']} | "
                f"{int(r['n_replicates'])} | "
                f"{float(r['fraction_chi2_p_value_below_0p05']):.3f} | "
                f"{float(r['total_time_for_analyzed_1000_deaths_after_burnin_mean']):.3f} | "
                f"{float(r['spatial_chi2_per_df_chi2_bins_mean']):.3f} | "
                f"{float(r['spatial_chi2_p_value_chi2_bins_mean']):.3g} | "
                f"{float(r['t_zone_density_ratio_mean']):.3f} | "
                f"{float(r['death_rejection_rate_mean']):.3f} |\n"
            )

    path.write_text("".join(lines), encoding="utf-8")
    return path


# ============================================================
# 6. Parameter sweep: same settings as the Gillespie analysis
# ============================================================


def make_scenarios(base: Parameters) -> Dict[str, Parameters]:
    """
    Build a replicated parameter sweep.

    Important idea:
        One run with one seed is not enough for a stochastic simulation.
        For each parameter setting, we run several independent replications
        with different random seeds.

    The scenario name has the form:
        parameter_setting__rep01
        parameter_setting__rep02
        ...

    The part before "__rep" is the parameter setting.
    The rep number is the independent random replication.
    """

    n_replicates = 5
    replicate_seeds = [1001 + i for i in range(n_replicates)]

    parameter_sets: Dict[str, Parameters] = {
        "baseline": base,

        "higher_death_rate": replace(base, lambda_d=1.5),

        "larger_ERK_radius": replace(base, beta_d_R=1.0),

        "shorter_ERK_duration": replace(base, beta_d_T=2.0),

        "stronger_T_zone_activation": replace(base, lambda_a_T=1.0),

        "strong_visible_T_zone": replace(
            base,
            lambda_a_T=2.0,
            lambda_a_c=0.005,
            beta_a_R=5.0,
            beta_a_T=3.0,
        ),

        # Sweep A: keep lambda_a_T close to baseline, only decrease lambda_a_c.
        "T0p5_c0p005_ratio_1e2": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=5e-3,
        ),
        "T0p5_c0p0005_ratio_1e3": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=5e-4,
        ),
        "T0p5_c0p00005_ratio_1e4": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=5e-5,
        ),
        "T0p5_c0p025_ratio_20": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=2.5e-2,
        ),
        "T0p5_c0p01_ratio_50": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=1e-2,
        ),

        # Sweep B: absolute order sweep.
        "T1e-1_c1e-2_ratio_1e1": replace(
            base,
            lambda_a_T=1e-1,
            lambda_a_c=1e-2,
        ),
        "T1e-1_c1e-3_ratio_1e2": replace(
            base,
            lambda_a_T=1e-1,
            lambda_a_c=1e-3,
        ),
        "T1e-1_c1e-4_ratio_1e3": replace(
            base,
            lambda_a_T=1e-1,
            lambda_a_c=1e-4,
        ),
        "T1e-2_c1e-3_ratio_1e1": replace(
            base,
            lambda_a_T=1e-2,
            lambda_a_c=1e-3,
            max_proposals=30_000_000,
        ),
        "T1e-2_c1e-4_ratio_1e2": replace(
            base,
            lambda_a_T=1e-2,
            lambda_a_c=1e-4,
            max_proposals=30_000_000,
        ),
        "T1e-3_c1e-4_ratio_1e1": replace(
            base,
            lambda_a_T=1e-3,
            lambda_a_c=1e-4,
            max_proposals=50_000_000,
        ),

        # Sweep C: same lambda contrast, but make activation zones more local.
        "T0p5_c0p005_ratio_1e2_local_activation": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=5e-3,
            beta_a_R=5.0,
            beta_a_T=3.0,
            max_proposals=50_000_000,
        ),
        "T0p5_c0p0005_ratio_1e3_local_activation": replace(
            base,
            lambda_a_T=5e-1,
            lambda_a_c=5e-4,
            beta_a_R=5.0,
            beta_a_T=3.0,
            max_proposals=50_000_000,
        ),
        "T1e-1_c1e-4_ratio_1e3_local_activation": replace(
            base,
            lambda_a_T=1e-1,
            lambda_a_c=1e-4,
            beta_a_R=5.0,
            beta_a_T=3.0,
            max_proposals=80_000_000,
        ),
    }

    def pstr(x: float) -> str:
        """Convert numbers to safe strings for scenario names."""
        return str(x).replace(".", "p").replace("-", "m")

    beta_a_R_values = [2.5, 3.5, 5.0, 7.5]
    beta_a_T_values = [1.2, 2.0, 3.0, 5.0]

    # Case D1: ratio = 100, lambda_a_T remains close to baseline.
    for beta_R in beta_a_R_values:
        for beta_T in beta_a_T_values:
            name = (
                f"T0p5_c0p005_ratio_1e2"
                f"_betaR{pstr(beta_R)}"
                f"_betaT{pstr(beta_T)}"
            )
            parameter_sets[name] = replace(
                base,
                lambda_a_T=5e-1,
                lambda_a_c=5e-3,
                beta_a_R=beta_R,
                beta_a_T=beta_T,
                max_proposals=80_000_000,
            )

    # Case D2: ratio = 1000, lower absolute activation intensity.
    for beta_R in beta_a_R_values:
        for beta_T in beta_a_T_values:
            name = (
                f"T1e-1_c1e-4_ratio_1e3"
                f"_betaR{pstr(beta_R)}"
                f"_betaT{pstr(beta_T)}"
            )
            parameter_sets[name] = replace(
                base,
                lambda_a_T=1e-1,
                lambda_a_c=1e-4,
                beta_a_R=beta_R,
                beta_a_T=beta_T,
                max_proposals=120_000_000,
            )

    scenarios: Dict[str, Parameters] = {}
    for parameter_name, p0 in parameter_sets.items():
        for rep_idx, seed in enumerate(replicate_seeds, start=1):
            scenario_name = f"{parameter_name}__rep{rep_idx:02d}"
            scenarios[scenario_name] = replace(p0, seed=seed)

    return scenarios


# ============================================================
# 7. Main
# ============================================================


def main() -> None:
    out_dir = Path("simulation_outputs_1000_rejection_burnin")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Parameters(burn_in_deaths=500, target_deaths=1000)
    scenarios = make_scenarios(base)

    print("Running T-shaped rejection-sampling simulations: burn-in 500 deaths, then analyze 1000 deaths...\n")

    results: List[SimulationResult] = []
    for name, params in scenarios.items():
        print(f"Scenario: {name}")
        result = simulate_one(name, params)
        results.append(result)
        print(f"  total observed deaths simulated: {result.n_observed_deaths_total}")
        print(f"  analyzed observed deaths:        {result.n_observed_deaths_analyzed}")
        print(f"  burn-in time:                    {result.burn_in_time:.6f}")
        print(f"  time for analyzed 1000 deaths:   {result.analysis_time_span:.6f}")
        print("")

    global_colorbar_max = int(max(np.max(spatial_histogram(result, result.params.bins_space)) for result in results))
    global_colorbar_max = max(global_colorbar_max, 1)

    summary_rows: List[Dict[str, Any]] = []
    for result in results:
        stats = death_statistics(result)
        summary_rows.append(stats)

        save_death_events_csv(result, out_dir)
        plot_time_histogram(result, out_dir)
        plot_spatial_histogram_2d(result, out_dir, global_colorbar_max)
        plot_cumulative_deaths(result, out_dir)

        print(f"Scenario: {result.scenario}")
        print(f"  activation acceptance: {stats['activation_acceptance_rate']:.4f}")
        print(f"  death rejection rate:  {stats['death_rejection_rate']:.4f}")
        print(f"    outside active:      {stats['death_rejection_rate_outside_active']:.4f}")
        print(f"    by ERK protection:   {stats['death_rejection_rate_by_erk']:.4f}")
        print(f"  spatial CV 20x20:      {stats['spatial_coefficient_of_variation_plot_bins']:.4f}")
        print(f"  chi2/df 10x10:         {stats['spatial_chi2_per_df_chi2_bins']:.4f}")
        print(f"  chi2 p-value 10x10:    {stats['spatial_chi2_p_value_chi2_bins']:.4g}")
        print(f"  T-zone density ratio:  {stats['t_zone_density_ratio']:.4f}")
        print("")

    summary_path = save_summary_csv(summary_rows, out_dir)
    aggregated_rows = aggregate_replicate_rows(summary_rows)
    aggregate_summary_path = save_aggregate_summary_csv(aggregated_rows, out_dir)
    report_path = write_analysis_report(summary_rows, out_dir, global_colorbar_max, aggregated_rows)

    print("Done.")
    print(f"Common colorbar vmax: {global_colorbar_max}")
    print(f"Replicate-level summary CSV: {summary_path}")
    print(f"Aggregated-by-parameter summary CSV: {aggregate_summary_path}")
    print(f"Analysis report: {report_path}")
    print(f"Plots and death-event CSV files are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
