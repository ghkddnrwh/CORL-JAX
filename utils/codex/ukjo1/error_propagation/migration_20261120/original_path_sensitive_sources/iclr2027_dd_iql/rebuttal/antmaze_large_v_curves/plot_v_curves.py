"""Plot saved fixed-probe V trajectories for AM-L IQL and DD-IQL through 1M."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "utils/iclr2027_dd_iql/rebuttal/value_explosion_audit/scan_logs.py"
spec = importlib.util.spec_from_file_location("audit", AUDIT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
OUT = ROOT / "logs/iclr2027_dd_iql/rebuttal/antmaze_large_v_curves"
ENV = "antmaze-large-navigate-singletask-v0"
COLORS = {"IQL": "#1f77b4", "DD-IQL": "#ff7f0e"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(ax, data, tau, gamma):
    for method in ("IQL", "DD-IQL"):
        selected = data[
            (data.method == method) & (data.expectile == tau) & (data.gamma == gamma)
        ]
        pivot = selected.pivot(index="step", columns="seed", values="v")
        assert set(pivot.columns) == {0, 1, 2, 3}
        for seed in pivot:
            observed = pivot[seed].dropna()
            ax.plot(
                observed.index / 1e6, observed, color=COLORS[method], alpha=0.25, lw=0.7
            )
        # Average only actual shared observations from all four seeds.
        common = pivot.dropna()
        assert common.index[-1] == 1_000_000
        ax.plot(
            common.index / 1e6,
            common.mean(1),
            color=COLORS[method],
            lw=2,
            linestyle="-" if method == "IQL" else "--",
            label=method,
        )
    if tau == 0.99:
        ax.set_yscale("symlog", linthresh=100)
    ax.set(xlabel="Updates (millions)", ylabel="Mean V", xlim=(0, 1))
    ax.text(
        0.03,
        0.96,
        rf"$\tau={tau},\ \gamma={gamma}$",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )
    ax.grid(alpha=0.2)
    if tau == 0.99:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.88))
    else:
        ax.legend(frameon=False, loc="lower right")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    table = pd.read_csv(
        ROOT / "logs/iclr2027_dd_iql/rebuttal/value_explosion_audit/run_summary.csv"
    )
    selected = table[(table.env == ENV) & table.method.isin(["IQL", "DD-IQL"])]
    assert len(selected) == 32
    frames, manifest, endpoints = [], [], []
    for _, row in selected.iterrows():
        path, cfg_path = ROOT / row.data_path, ROOT / row.config_path
        cfg = audit.config(cfg_path)
        assert cfg["env"] == ENV and cfg["tau"] == 0.005
        if row.method == "IQL":
            steps, values = audit.history_values(path)
        else:
            assert cfg["ensemble_size"] == 20 and cfg["delayed_update_period"] == 100
            with np.load(path, allow_pickle=False) as h:
                steps, values = h["timestep"], h["diag/v_mean"]
        keep = steps <= 1_000_000
        steps, values = steps[keep], values[keep]
        assert steps[-1] == 1_000_000 and np.isfinite(values).all()
        # The summary CSV can round its decimal representation by a few ulps.
        np.testing.assert_allclose(values[-1], row.v_at_1000000, rtol=1e-14, atol=1e-10)
        ids = dict(
            method=row.method,
            gamma=float(cfg["discount"]),
            expectile=float(cfg["iql_tau"]),
            seed=int(cfg["seed"]),
        )
        frames.append(pd.DataFrame(dict(**ids, step=steps, v=values)))
        endpoints.append(dict(**ids, end_step=int(steps[-1]), v=float(values[-1])))
        manifest.append(
            dict(
                **ids,
                data_path=row.data_path,
                config_path=row.config_path,
                data_sha256=sha256(path),
                config_sha256=sha256(cfg_path),
                metric="diag/v_mean",
                end_step=int(steps[-1]),
            )
        )
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT / "v_trajectories.csv.gz", index=False)
    endpoint = pd.DataFrame(endpoints)
    endpoint.to_csv(OUT / "per_seed_endpoints.csv", index=False)
    summary = (
        endpoint.groupby(["expectile", "gamma", "method"])
        .v.agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(OUT / "endpoint_summary.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "pdf.fonttype": 42,
        }
    )
    with PdfPages(OUT / "comparison.pdf") as booklet:
        for tau in (0.9, 0.99):
            for gamma in (0.99, 0.999):
                fig, ax = plt.subplots(figsize=(5.6, 3.8))
                panel(ax, data, tau, gamma)
                fig.tight_layout()
                stem = f"tau{tau}_gamma{gamma}"
                for extension in ("pdf", "png"):
                    fig.savefig(
                        OUT / "figures" / (stem + "." + extension),
                        dpi=220,
                        bbox_inches="tight",
                    )
                plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
            for ax, gamma in zip(axes, (0.99, 0.999)):
                panel(ax, data, tau, gamma)
            fig.tight_layout()
            booklet.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            dict(
                env=ENV,
                runs=manifest,
                code_sha256=sha256(Path(__file__)),
                reader_sha256=sha256(AUDIT),
                all_32_runs_end_at_1M=True,
                curve=(
                    "Thick: four-seed mean on shared recorded steps; "
                    "thin: individual seeds"
                ),
                scale="tau=.99: symlog with linear threshold100; tau=.9: linear",
            ),
            indent=2,
        )
        + "\n"
    )
    (OUT / "notes.txt").write_text(
        "AM-L, IQL rho=.005 versus DD-IQL N20/D100/rho=.005.\n"
        "Metric: fixed-probe diag/v_mean, not training-minibatch v_mean.\n"
        "All 32 original histories were read directly and have finite V at 1M.\n"
        "Thick lines: mean across four seeds on their shared recorded grid.\n"
        "Thin lines: individual seed observations; no extrapolation or smoothing.\n"
        "No SD band: individual curves show the positive-tail seed spread directly.\n"
        "tau=.99 uses symlog (linear near zero, logarithmic outside +/-100).\n"
        "comparison.pdf: page1 tau=.9; page2 tau=.99, both gamma=.99/.999.\n"
        "The tau=.99 logs show large positive value explosion for all IQL seeds;\n"
        "the tau=.9 logs do not show that behavior within 1M. This is an observed\n"
        "finite-budget trajectory claim, not an oracle Q/V bias or infinity proof.\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
