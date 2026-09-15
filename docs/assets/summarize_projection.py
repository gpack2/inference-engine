"""Create compact result tables from completed benchmark JSON reports."""

import json
from pathlib import Path


def main():
    docs = Path(__file__).resolve().parents[1]
    reports = {device: json.loads((docs.parent / f"results/projection-{device}-matrix.json").read_text())
               for device in ("mps", "cpu")}
    if not all(report["complete"] for report in reports.values()):
        raise ValueError("Refusing to summarize an incomplete matrix")
    lines = [
        "# Projection experiment: measured results", "",
        "Generated from saved measurements. All times are milliseconds; arrows show full-position → last-position logits. Both variants use KV caching. Positive reduction means faster normal generation.", "",
        "Generation values are medians of five ordinary timed runs after two warmups. Phase columns come from three separate synchronized diagnostic runs. Decode reports the median of each run's mean step time. Phase values do not add up to ordinary generation time.", "",
        "All cases use Qwen2.5-0.5B, float32, batch size one and one CPU thread; generated token IDs matched exactly. See the [method and interpretation](performance-measurement.md) for scope and limitations.", "",
    ]
    for device, report in reports.items():
        lines += [f"**{report['hardware']} · {'Apple GPU / MPS' if device == 'mps' else 'CPU / one thread'}**", "",
                  "| Input / output tokens | Generation ms | Reduction | Diagnostic prefill + first token ms | Diagnostic decode step ms |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for case in report["cases"]:
            assert all(case["parity"][field] for field in ("generated_token_ids_equal", "diagnostic_token_ids_equal"))
            full, last = [case["variants"][name] for name in ("full_logits", "last_token_logits")]
            values = []
            for field in ("generation_ms", "diagnostic_prefill_first_token_ms", "diagnostic_mean_decode_step_ms"):
                values.append(f"{full[field]['median']:.2f} → {last[field]['median']:.2f}")
            lines.append(f"| {case['prompt_tokens']} / {case['output_tokens']} | {values[0]} | {case['median_generation_latency_reduction_percent']:+.1f}% | {values[1]} | {values[2]} |")
        lines += ["", f"[Raw {device.upper()} samples, token IDs, tensor sizes, and environment](../results/projection-{device}-matrix.json)", ""]
    lines += ["**Storage example**", ""]
    example = next(case for case in reports["mps"]["cases"] if case["prompt_tokens"] == 256 and case["output_tokens"] == 16)
    full, last = [example["variants"][name]["diagnostic_runs"][0] for name in ("full_logits", "last_token_logits")]
    lines += [f"For 256 input / 16 output tokens, the allocated KV buffers remain {full['cache_allocated_bytes'] / 2**20:.3f} MiB in both variants. The prefill logits tensor shrinks from {full['prefill_logits_tensor_bytes'] / 2**20:.3f} MiB to {last['prefill_logits_tensor_bytes'] / 2**20:.3f} MiB. These are tensor storage sizes, not measured peak memory.", "",
              "Regenerate this file with `uv run python docs/assets/summarize_projection.py`.", ""]
    output = docs / "projection-results.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
