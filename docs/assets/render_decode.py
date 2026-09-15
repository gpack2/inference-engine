# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Render decode case-study figures and tables from completed raw reports."""

import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ASSETS = Path(__file__).resolve().parent
DEVICES = ("mps", "cpu")
COLORS = {"recompute_rope": "#be7638", "shared_rope": "#087e8b"}
LABELS = {"recompute_rope": "Recompute each Q/K", "shared_rope": "Share once per forward"}
COMPONENTS = {
    "attention_linear": "Attention projections", "mlp_linear": "MLP projections",
    "vocabulary_projection": "Vocabulary projection", "normalization": "Normalization",
    "rope": "RoPE (including shared setup)", "attention": "Attention scores / softmax / values",
    "other": "Cache, embedding, selection and residual",
}


def group_components(trace):
    values = trace["component_ms_per_step"]
    result = {key: values.get(key, 0) for key in COMPONENTS if key != "other"}
    result["rope"] += values.get("shared_rope_setup", 0)
    result["other"] = trace["decode_mean_step_ms"] - sum(result.values())
    return result


def style(ax):
    ax.grid(axis="y", color="#e5eaf0")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    for extension in ("png", "svg"):
        path = ASSETS / f"{name}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


def render_sweep(reports):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=.09, right=.97, top=.82, bottom=.15, hspace=.57, wspace=.28)
    hardware = reports["mps"]["hardware"]
    steps = reports["mps"]["settings"]["steps"]
    fig.text(.06, .945, "Fewer operations can still run slower", fontsize=22, weight="bold", color="#172b42")
    fig.text(.06, .9, f"{hardware} · Qwen2.5-0.5B · float32 · {steps} decode steps after prefill", fontsize=12, color="#4b596b")
    for ax, device in zip(axes[0], DEVICES):
        cases = reports[device]["cases"]
        xs = [c["context_tokens"] for c in cases]
        for name, color in COLORS.items():
            values = [c["variants"][name]["normal_mean_decode_step_ms"] for c in cases]
            ax.plot(xs, [v["median"] for v in values], color=color, marker="o", label=LABELS[name])
            ax.fill_between(xs, [v["min"] for v in values], [v["max"] for v in values], color=color, alpha=.13)
        ax.set_title("Normal decode · " + ("MPS" if device == "mps" else "CPU, one thread"), weight="bold")
        ax.set_ylabel("Mean decode step (ms)")
        ax.set_xlabel("Initial cached context (tokens)")
        ax.set_xticks(xs)
        ax.legend(frameon=False, fontsize=8, loc="best")
        style(ax)
    ax = axes[1, 0]
    for device, color in (("mps", "#087e8b"), ("cpu", "#815aa0")):
        cases = reports[device]["cases"]
        ax.plot([c["context_tokens"] for c in cases],
                [c["variants"]["recompute_rope"]["component_median_ms_per_step"]["attention"] for c in cases],
                color=color, marker="o", label=device.upper())
    ax.set_title("Attention · synchronized diagnostic", weight="bold")
    ax.set_ylabel("Attention wall time per step (ms)")
    ax.set_xlabel("Initial cached context (tokens)")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    ax = axes[1, 1]
    cases = reports["mps"]["cases"]
    xs = [c["context_tokens"] for c in cases]
    runs = [c["variants"]["recompute_rope"]["normal_runs"][0] for c in cases]
    for field, label, color in (("cache_allocated_bytes", "Allocated capacity", "#087e8b"),
                                ("cache_valid_bytes", "Valid prefix after decode", "#be7638")):
        ax.plot(xs, [r[field] / 2**20 for r in runs], marker="o", color=color, label=label)
    ax.set_title("KV tensor storage · both variants", weight="bold")
    ax.set_ylabel("MiB (not peak memory)")
    ax.set_xlabel("Initial cached context (tokens)")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.text(.06, .075, "Top: medians of five run means; shading is observed min–max, not a confidence interval. Context grows each step.", fontsize=9, color="#4b596b")
    fig.text(.06, .04, "Bottom-left: separate intrusive baseline diagnostics, not GPU kernel times or a breakdown of the top panels.", fontsize=9, color="#4b596b")
    save(fig, "decode-context-sweep")


def render_components(reports):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=.22, right=.96, bottom=.37, top=.72)
    context = reports["mps"]["cases"][-1]["context_tokens"]
    fig.text(.06, .93, "A closer look at completed component work", fontsize=22, weight="bold", color="#172b42")
    fig.text(.06, .86, f"Initial context {context} tokens · synchronization around each component · diagnostic only", fontsize=12, color="#4b596b")
    labels, groups = [], []
    for device in DEVICES:
        case = reports[device]["cases"][-1]
        for mode in COLORS:
            traces = [group_components(t) for t in case["variants"][mode]["diagnostic_runs"]]
            groups.append({k: statistics.mean(t[k] for t in traces) for k in COMPONENTS})
            labels.append(f"{device.upper()} · {'recompute' if mode == 'recompute_rope' else 'shared'}")
    left = [0.0] * len(groups)
    colors = ["#264653", "#4e91a0", "#94c9c5", "#b9a1ce", "#e6ab62", "#d17b65", "#d6dde5"]
    for (key, label), color in zip(COMPONENTS.items(), colors):
        values = [g[key] for g in groups]
        ax.barh(range(len(groups)), values, left=left, color=color, height=.6, label=label)
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Instrumented wall time per decode step (ms)", labelpad=10)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-.22, -.3), fontsize=9)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e5eaf0")
    fig.text(.06, .09, "Means of two diagnostic run means. These totals include instrumentation and can greatly exceed normal decode latency.", fontsize=9, color="#4b596b")
    fig.text(.06, .045, "RoPE groups rotation plus shared setup. The residual includes unsplit activations, Python work and instrumentation.", fontsize=9, color="#4b596b")
    save(fig, "decode-components")


def write_tables(reports):
    lines = ["# Decode and RoPE measurements", "",
             "Generated from completed raw reports by `docs/assets/render_decode.py`. Lower latency is better; positive reductions mean faster. Each request generates 33 tokens: one from prefill, then 32 decode steps. Five paired runs per variant after two warmups; float32, one request, one CPU thread, EOS disabled.", ""]
    for device in DEVICES:
        report = reports[device]
        lines += [f"**{report['hardware']} · {device.upper()}**", "",
                  f"[Raw samples, source hashes and environment](../results/decode-rope-{device}.json)", "",
                  "| Initial context | Decode ms/step, recompute → shared | Decode reduction | Request ms, recompute → shared | Request reduction | Positive paired decode / request runs |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for c in report["cases"]:
            base, opt = c["variants"].values()
            d = [v["normal_mean_decode_step_ms"]["median"] for v in (base, opt)]
            g = [v["generation_ms"]["median"] for v in (base, opt)]
            wins = [sum(x > 0 for x in c[field]) for field in ("paired_decode_latency_reduction_percent", "paired_generation_latency_reduction_percent")]
            lines.append(f"| {c['context_tokens']} | {d[0]:.2f} → {d[1]:.2f} | {c['median_decode_latency_reduction_percent']:.1f}% | {g[0]:.2f} → {g[1]:.2f} | {c['median_generation_latency_reduction_percent']:.1f}% | {wins[0]}/5 · {wins[1]}/5 |")
        case = report["cases"][-1]
        lines += ["", f"**Separate component diagnostics: {case['context_tokens']} initial tokens**", "",
                  "Mean of two diagnostic run means, milliseconds per decode step. Synchronized components include host overhead; these are not normal execution percentages or pure GPU kernel times.", "",
                  "| Component | Recompute | Shared |", "| --- | ---: | ---: |"]
        groups = []
        for variant in case["variants"].values():
            traces = [group_components(t) for t in variant["diagnostic_runs"]]
            groups.append({k: statistics.mean(t[k] for t in traces) for k in COMPONENTS})
        for key, label in COMPONENTS.items():
            lines.append(f"| {label} | {groups[0][key]:.3f} | {groups[1][key]:.3f} |")
        lines += ["", "**Host profiler: first four decode steps at this context**", "",
                  "| Operator | Calls, recompute → shared | Exclusive CPU self time ms, recompute → shared |",
                  "| --- | ---: | ---: |"]
        events = [{e["name"]: e for e in v["host_profile"]["host_operator_events"]} for v in case["variants"].values()]
        for name in ("aten::cos", "aten::sin", "aten::_local_scalar_dense", "aten::linear", "aten::bmm"):
            a, b = [e.get(name, {"calls": 0, "self_cpu_ms": 0}) for e in events]
            lines.append(f"| `{name}` | {a['calls']} → {b['calls']} | {a['self_cpu_ms']:.3f} → {b['self_cpu_ms']:.3f} |")
        lines += ["", "Scalar-read time can include previous queued GPU work. It must not be interpreted as wholly removable validation cost.", ""]
    lines += ["**KV storage is unchanged by RoPE sharing**", "",
              "| Initial context | Allocated capacity (tokens) | Valid tokens after decode | Allocated MiB | Valid MiB |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for case in reports["mps"]["cases"]:
        run = case["variants"]["recompute_rope"]["normal_runs"][0]
        lines.append(f"| {case['context_tokens']} | {case['context_tokens'] + case['decode_steps'] + 1} | {run['final_cache_length']} | {run['cache_allocated_bytes'] / 2**20:.3f} | {run['cache_valid_bytes'] / 2**20:.3f} |")
    lines += ["", "One output token has not yet been inserted into the cache. The engine allocates one extra position for that token; valid storage is therefore one token smaller than capacity at the end. These are exact KV tensor sizes, not peak memory.", "",
              "[Method, implementation and limitations](decode-performance.md)", ""]
    output = ASSETS.parent / "decode-results.md"
    output.write_text("\n".join(lines))
    print(output)


def main():
    reports = {d: json.loads((ASSETS.parents[1] / f"results/decode-rope-{d}.json").read_text()) for d in DEVICES}
    if not all(r["complete"] and all(c["all_trial_token_ids_equal"] for c in r["cases"]) for r in reports.values()):
        raise ValueError("Both matrices must finish and pass parity before rendering")
    if not all(r["settings"]["steps"] == 32 and r["settings"]["repeats"] == 5
               and r["settings"]["diagnostic_repeats"] == 2 and r["settings"]["warmup"] == 2 for r in reports.values()):
        raise ValueError("This case-study renderer expects the documented 32-step, 5-repeat, 2-warmup, 2-diagnostic matrix")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path"})
    render_sweep(reports)
    render_components(reports)
    write_tables(reports)


if __name__ == "__main__":
    main()
