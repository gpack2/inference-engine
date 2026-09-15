# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Render the portfolio figure from saved results without rerunning inference.

Run: uv run --no-project --python 3.12 docs/assets/render_results.py
Dependencies live in uv's isolated script environment, not the engine lockfile.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    assets = Path(__file__).resolve().parent
    root = assets.parents[1]
    report = json.loads((root / "results/qwen-mps-generation-smoke.json").read_text())
    cached = report["results"]["cached"]["median_seconds"] * 1000
    uncached = report["results"]["uncached"]["median_seconds"] * 1000
    reduction = (1 - cached / uncached) * 100

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#ffffff")
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.28, top=0.66)
    fig.text(0.06, 0.9, "A first look at KV caching", fontsize=22, weight="bold", color="#172b42")
    fig.text(0.06, 0.82, f"{reduction:.1f}% lower median generation latency on this workload", fontsize=13, color="#087e8b")
    fig.text(0.06, 0.745, f"Qwen2.5-0.5B  ·  Apple MPS  ·  float32  ·  {len(report['prompt_ids'])} input / {report['new_tokens']} output tokens", color="#4b596b", fontsize=10)

    for row, mode, label, color in [(1, "uncached", "Full-prefix\nrecomputation", "#b5c4d5"),
                                    (0, "cached", "KV cache", "#087e8b")]:
        result = report["results"][mode]
        median = result["median_seconds"] * 1000
        samples = [value * 1000 for value in result["seconds"]]
        ax.barh(row, median, height=0.48, color=color, zorder=2)
        jitter = [(i - (len(samples) - 1) / 2) * 0.045 for i in range(len(samples))]
        ax.scatter(samples, [row + offset for offset in jitter], s=28, color="#172b42",
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.text(max(samples) + 18, row, f"{median:.1f} ms", va="center", weight="bold", color="#172b42")
    ax.set_yticks([0, 1], ["KV cache", "Full-prefix\nrecomputation"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, max(uncached, cached) * 1.24)
    ax.set_xlabel("Complete generation latency (ms) — lower is better", labelpad=10, color="#4b596b")
    ax.grid(axis="x", color="#e5eaf0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelcolor="#4b596b")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.06, 0.11, "Bars: medians. Dots: five measured runs per path, after two warmups. Synchronization at timing boundaries.", fontsize=9, color="#4b596b")
    fig.text(0.06, 0.06, "Includes prefill, decode, Python overhead and cache allocation; excludes model loading. Preliminary single-request result.", fontsize=9, color="#4b596b")
    for extension in ("png", "svg"):
        path = assets / f"kv-cache-baseline.{extension}"
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
