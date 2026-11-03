"""An exact-expectile, fully in-support linear IQL counterexample.

Two states, deterministic next state 0, reward .1. The atomless behavior action
distribution is U[.99,1.01] with probability 31/32 and U[9.99,10.01] otherwise.
Q_theta(s,a)=theta*phi(s)*a with phi=(1,2), d_state=(.99,.01).
Both intervals are in support. Critic regression is integrated and solved
exactly. Expectile regression is also solved exactly, so zeta=0.
High expectiles can nevertheless destabilize the projected Bellman iteration.
This is a constructive example, not an identification of OGBench's failure cause.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    phi = np.asarray([1., 2.])
    half_width = .01
    # These are repeated group means, used to integrate the FOC exactly: the
    # expectiles for both tested taus lie in the gap between the two intervals.
    action = np.asarray([1.]*31+[10.])
    x = phi[:, None]*action[None]
    distribution = np.asarray([.99, .01])[:, None]*np.ones((1, 32))/32
    feature_mean = np.sum(distribution*x)
    feature_square = np.sum(distribution*x*x)+np.sum(np.asarray([.99, .01])*phi**2)*half_width**2/3
    rows, summaries = [], []
    for tau in (.9, .99):
        kappa = (31*(1-tau)+tau)/32
        expectile_factor = (31*(1-tau)+10*tau)/(31*(1-tau)+tau)
        for gamma in (.99, .999):
            for method, zeta, ridge in (("exact_V", 0., 0.),
                                         ("tiny_residual", 1e-5, 0.),
                                         ("exact_V_ridge_B", 0., 8.),
                                         ("exact_V_bounded_target", 0., 0.)):
                theta = .01
                gain = gamma*expectile_factor*feature_mean/(feature_square+ridge)
                max_normalized_score = 0.
                for step in range(101):
                    q = theta*x
                    # Positive theta is invariant here. The two-interval expectile
                    # has this closed form, without any learned V network.
                    v_exact = theta*phi*expectile_factor
                    v = v_exact+zeta/kappa
                    adv = q-v_exact[:, None]
                    score = np.mean(np.where(adv >= 0, tau, 1-tau)*adv, axis=-1)
                    normalized_score = np.max(np.abs(score))/max(np.max(np.abs(q)), 1.)
                    max_normalized_score = max(max_normalized_score, normalized_score)
                    rows.append({"tau": tau, "gamma": gamma, "method": method,
                                 "step": step, "theta": theta, "q_max": theta*phi.max()*(10+half_width),
                                 "true_value": .1/(1-gamma), "kappa": kappa,
                                 "normalized_exact_foc_residual": normalized_score,
                                 "injected_zeta": zeta})
                    target = .1+gamma*v[0]
                    if method == "exact_V_bounded_target":
                        target = np.clip(target, 0., .1/(1-gamma))
                    theta = feature_mean*target/(feature_square+ridge)
                summaries.append({"tau": tau, "gamma": gamma, "method": method,
                                  "atomless_action_half_width": half_width,
                                  "unclipped_projected_gain": gain, "kappa": kappa,
                                  "theta_after_100": rows[-1]["theta"],
                                  "max_normalized_exact_foc_residual": max_normalized_score})
    with (out / "trajectories.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    # A partial critic fit has an additional dependence on its starting model.
    # Measuring target sensitivity while fixing that model omits this carry term.
    warm = []
    for tau in (.9, .99):
        c = (31*(1-tau)+10*tau)/(31*(1-tau)+tau)
        assert 1+half_width < c < 10-half_width
        for gamma in (.99, .999):
            projected = gamma*c*feature_mean/feature_square
            for relaxation in (1., .5, .1, .01):
                target_path = relaxation*projected
                carry = 1-relaxation
                actual = carry+target_path
                gamma_b_inf = gamma*relaxation*phi.max()*(10+half_width)*feature_mean/feature_square
                def iteration(theta):
                    y = .1+gamma*c*theta
                    return (1-relaxation)*theta+relaxation*feature_mean*y/feature_square
                eps = 1e-5
                fd = (iteration(2.+eps)-iteration(2.-eps))/(2*eps)
                assert abs(fd-actual) < 1e-9
                warm.append({"tau": tau, "gamma": gamma, "fit_fraction": relaxation,
                             "exact_fit_gain": projected,
                             "target_path_only_gain": target_path,
                             "starting_model_carry_gain": carry,
                             "actual_error_propagation_gain": actual,
                             "gamma_times_target_B_operator_inf_norm": gamma_b_inf,
                             "finite_difference_actual_gain": fd})
    assert any(r["gamma_times_target_B_operator_inf_norm"] < 1 and
               r["actual_error_propagation_gain"] > 1 for r in warm)
    (out / "warm_start_summary.json").write_text(json.dumps(warm, indent=2))
    # Separate reduction of V error at fixed Q from stability of the coupled
    # iteration. w is the critic-fit contribution gamma*c_reg*V, whose exact
    # expectile target is m*theta. One V relaxation uses fraction beta.
    coupled = []
    for tau in (.9, .99):
        c_expectile = (31*(1-tau)+10*tau)/(31*(1-tau)+tau)
        for gamma in (.99, .999):
            m = gamma*c_expectile*feature_mean/feature_square
            for alpha in (.1, .5):
                for beta in (.1, .5):
                    for fits in (1, 2, 4, 20):
                        v_carry = (1-beta)**fits
                        fitted = 1-v_carry
                        for backup in ("old_V", "fresh_V"):
                            matrix = (np.asarray([[1-alpha, alpha], [fitted*m, v_carry]])
                                      if backup == "old_V" else
                                      np.asarray([[1-alpha+alpha*fitted*m, alpha*v_carry],
                                                  [fitted*m, v_carry]]))
                            def iteration(z):
                                theta, w = z
                                next_w = v_carry*w+fitted*m*theta
                                next_theta = (1-alpha)*theta+alpha*(
                                    feature_mean*.1/feature_square +
                                    (w if backup == "old_V" else next_w))
                                return np.asarray([next_theta, next_w])
                            eps = 1e-5
                            center = np.asarray([2., 3.])
                            fd = np.column_stack([(iteration(center+eps*e)-iteration(center-eps*e))/(2*eps)
                                                  for e in np.eye(2)])
                            np.testing.assert_allclose(fd, matrix, atol=1e-9, rtol=1e-9)
                            radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
                            assert (radius < 1) == (m < 1)
                            np.testing.assert_allclose(np.linalg.det(np.eye(2)-matrix),
                                                       alpha*fitted*(1-m), atol=1e-12)
                            coupled.append({"tau": tau, "gamma": gamma, "full_fit_gain_m": m,
                                "critic_fit_fraction_alpha": alpha, "value_fit_fraction_beta": beta,
                                "value_fits_K": fits, "backup": backup,
                                "fixed_Q_value_error_reduction_factor": v_carry,
                                "coupled_error_matrix": matrix.tolist(), "spectral_radius": radius})
    (out / "coupled_value_fitting_summary.json").write_text(json.dumps({
        "scope": "Exact atomless linear example with explicit critic/value relaxation, not a fit of OGBench Adam dynamics",
        "identity": "det(I-M)=alpha*(1-(1-beta)^K)*(1-m) for old and fresh backups",
        "consequence": "More V fits reduce fixed-Q value error, but cannot stabilize this coupled system when m>1",
        "measurements": coupled}, indent=2))
    assert all(s["max_normalized_exact_foc_residual"] < 1e-14 for s in summaries)
    for gamma in (.99, .999):
        ordinary = next(s for s in summaries if s["tau"] == .9 and s["gamma"] == gamma and s["method"] == "exact_V")
        high = next(s for s in summaries if s["tau"] == .99 and s["gamma"] == gamma and s["method"] == "exact_V")
        assert ordinary["unclipped_projected_gain"] < 1
        assert high["unclipped_projected_gain"] > 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, gamma in zip(axes, (.99, .999)):
        for tau in (.9, .99):
            for method in ("exact_V", "tiny_residual", "exact_V_ridge_B", "exact_V_bounded_target"):
                group = [r for r in rows if r["tau"] == tau and r["gamma"] == gamma and r["method"] == method]
                axis.plot([r["step"] for r in group], [r["q_max"] for r in group], label=f"tau={tau} {method}")
        axis.set_yscale("log")
        axis.set_title(f"gamma={gamma}; exact conditional expectile")
        axis.set_xlabel("Exact fitted Bellman iterations")
        axis.set_ylabel("Maximum Q")
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "exact_residual_counterexample.png", dpi=180)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, tau, gamma in ((axes[0], .9, .99), (axes[1], .99, .999)):
        for backup in ("old_V", "fresh_V"):
            group = [r for r in coupled if r["tau"] == tau and r["gamma"] == gamma
                     and r["critic_fit_fraction_alpha"] == .1 and r["value_fit_fraction_beta"] == .1
                     and r["backup"] == backup]
            axis.plot([r["value_fits_K"] for r in group], [r["spectral_radius"] for r in group],
                      "-o", label=backup)
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"tau={tau}, gamma={gamma}; full-fit m={group[0]['full_fit_gain_m']:.3f}")
        axis.set_xlabel("V fitting steps K per critic step")
        axis.set_ylabel("Coupled spectral radius")
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle("Fixed-Q V error shrinks by (1-beta)^K; coupled stability is a separate condition")
    fig.tight_layout()
    fig.savefig(out / "coupled_value_fitting.png", dpi=180)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    print("Verified", len(coupled), "coupled critic/V matrices against finite differences")
    for summary in summaries:
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
