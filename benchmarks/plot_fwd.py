"""Turn bench_fwd CSV output into the README chart and markdown table.

  python -m benchmarks.bench_fwd --tokens 1 --csv benchmarks/results/fwd.csv
  python -m benchmarks.plot_fwd --csv benchmarks/results/fwd.csv --dim 64 --heads 8

Writes the chart to docs/figures/kernel-throughput.png (override with --out) and
prints the execution-time table and the throughput summary line for the README.
"""
import argparse, csv, math, os


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "D": int(r["D"]), "N": int(r["N"]), "B": int(r["B"]), "H": int(r["H"]),
                "causal": r["causal"] == "True",
                "ext_ms": float(r["ext_ms"]), "ext_tf": float(r["ext_tflops"]),
                "sdpa_ms": float(r["sdpa_ms"]), "sdpa_tf": float(r["sdpa_tflops"]),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--out", default="docs/figures/kernel-throughput.png")
    a = p.parse_args()

    rows = load(a.csv)
    sel = [r for r in rows if r["D"] == a.dim and r["H"] == a.heads]
    if not sel:
        raise SystemExit(f"no rows for D={a.dim}, H={a.heads} in {a.csv}")
    seqlens = sorted({r["N"] for r in sel})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teal, blue = "#13877a", "#2767b0"
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5), constrained_layout=True)
    w = 0.38
    for ax, causal, title in ((axes[0], False, "Dense"), (axes[1], True, "Causal")):
        by_n = {r["N"]: r for r in sel if r["causal"] == causal}
        xs = list(range(len(seqlens)))
        ext = [by_n[n]["ext_tf"] if n in by_n else 0.0 for n in seqlens]
        sdpa = [by_n[n]["sdpa_tf"] if n in by_n and not math.isnan(by_n[n]["sdpa_tf"]) else 0.0
                for n in seqlens]
        ax.bar([x - w / 2 for x in xs], ext, w, label="Custom", color=teal)
        ax.bar([x + w / 2 for x in xs], sdpa, w, label="SDPA", color=blue)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n:,}" for n in seqlens])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("TFLOP/s")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}\n")

    print(f"Execution time (ms), D={a.dim}, H={a.heads}, B={sel[0]['B']}:\n")
    print("| Mask | Sequence length | Custom | SDPA |")
    print("|---|---:|---:|---:|")
    for causal, name in ((False, "Dense"), (True, "Causal")):
        for n in seqlens:
            m = [r for r in sel if r["N"] == n and r["causal"] == causal]
            if m:
                print(f"| {name} | {n} | {m[0]['ext_ms']:.3f} | {m[0]['sdpa_ms']:.3f} |")

    dense = [r["ext_tf"] for r in rows if not r["causal"]]
    causal_ = [r["ext_tf"] for r in rows if r["causal"]]
    print(f"\nThe full sweep covers {len(rows)} cases.")
    if dense:
        print(f"Custom dense throughput:  {min(dense):.2f}-{max(dense):.2f} TFLOP/s")
    if causal_:
        print(f"Custom causal throughput: {min(causal_):.2f}-{max(causal_):.2f} TFLOP/s")


if __name__ == "__main__":
    main()
