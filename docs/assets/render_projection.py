# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Render the projection case study from saved CPU/MPS measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    assets = Path(__file__).resolve().parent
    reports = {device: json.loads((assets.parents[1] / f"results/projection-{device}-matrix.json").read_text())
               for device in ("mps", "cpu")}
    if not all(report["complete"] for report in reports.values()):
        raise ValueError("Both benchmark matrices must finish before publishing the figure")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.27, top=0.68, wspace=0.15)
    fig.text(0.055, 0.9, "Compute only the logits generation needs", fontsize=21, weight="bold", color="#172b42")
    fig.text(0.055, 0.82, "Change in whole-request latency versus the full-logits baseline", fontsize=13, color="#4b596b")
    hardware = reports["mps"]["hardware"] or "Mac"
    fig.text(0.055, 0.755, f"{hardware} · Qwen2.5-0.5B · float32 · KV cache enabled for both variants", fontsize=10, color="#4b596b")
    all_values = []
    for ax, device in zip(axes, ("mps", "cpu")):
        for output, color in [(16, "#087e8b"), (64, "#c57529")]:
            cases = sorted((c for c in reports[device]["cases"] if c["output_tokens"] == output), key=lambda c: c["prompt_tokens"])
            values = [c["median_generation_latency_reduction_percent"] for c in cases]
            all_values.extend(values)
            ax.plot([c["prompt_tokens"] for c in cases], values, marker="o", markersize=6,
                    color=color, linewidth=2, label=f"{output} output tokens")
        ax.axhline(0, color="#6b7888", linewidth=1, linestyle="--")
        ax.set_xticks([16, 64, 128, 256])
        ax.set_xlabel("Input tokens", labelpad=8)
        ax.set_title("Apple GPU (MPS)" if device == "mps" else "CPU (one thread)", fontsize=12, weight="bold", pad=12)
        ax.grid(axis="y", color="#e5eaf0")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Generation latency reduction (%)\nPositive is faster", labelpad=12)
    axes[0].set_ylim(min(-5, min(all_values) - 5), max(all_values) + 12)
    fig.text(0.055, 0.12, "Five runs per variant after two warmups; points compare medians. Small changes may be run-to-run noise.", fontsize=9, color="#4b596b")
    fig.text(0.055, 0.07, "One fixed prompt per length, one request at a time. Separate phase diagnostics are in the linked report.", fontsize=9, color="#4b596b")
    for extension in ("png", "svg"):
        output = assets / f"last-token-projection.{extension}"
        fig.savefig(output, dpi=180, facecolor="white")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
