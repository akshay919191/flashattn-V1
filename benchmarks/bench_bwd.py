import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import flash_acc_reg_ext as ext


# ------------------------------------------------------------
# Your new API
# ------------------------------------------------------------

def flash_fwd(q, k, v, causal):
    return ext.flash_fwd(q, k, v, causal)


def flash_bwd(q, k, v, o, do, L, causal):
    return ext.flash_bwd(
        q,
        k,
        v,
        o,
        do,
        L,
        causal,
    )


# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------

def time_ms(fn, warmup=10, iters=50):

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    times = []

    for _ in range(iters):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        fn()

        end.record()

        end.synchronize()

        times.append(
            start.elapsed_time(end)
        )

    times.sort()

    return times[len(times) // 2]


# ------------------------------------------------------------
# Backward FLOPs
# ------------------------------------------------------------

def bwd_flops(B, H, N, D, causal):

    flops = 10.0 * B * H * N * N * D

    if causal:
        flops *= 0.5

    return flops


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128],
    )

    p.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
    )

    p.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 2],
    )

    p.add_argument(
        "--heads",
        type=int,
        default=16,
    )

    p.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
    )

    p.add_argument(
        "--causal",
        choices=["false", "true", "both"],
        default="both",
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    p.add_argument(
        "--iters",
        type=int,
        default=50,
    )

    p.add_argument(
        "--csv",
        type=str,
        default=None,
    )

    args = p.parse_args()

    assert torch.cuda.is_available()

    if args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    if args.causal == "false":
        causals = [False]
    elif args.causal == "true":
        causals = [True]
    else:
        causals = [False, True]


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("CUSTOM FLASH BACKWARD vs PYTORCH SDPA FLASH BACKEND")
    print("=" * 110)

    print("GPU     :", torch.cuda.get_device_name())
    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.version.cuda)
    print("dtype   :", args.dtype)
    print("heads   :", args.heads)

    print("=" * 110)

    header = (
        f"{'B':>3} "
        f"{'H':>3} "
        f"{'N':>6} "
        f"{'D':>4} "
        f"{'Causal':>6} | "
        f"{'Custom ms':>11} "
        f"{'Custom TF':>10} | "
        f"{'SDPA ms':>10} "
        f"{'SDPA TF':>10} | "
        f"{'Speedup':>8}"
    )

    print(header)
    print("-" * len(header))


    rows = []


    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    for B in args.batch:

        for H in [args.heads]:

            for N in args.seqlens:

                for D in args.dims:

                    for causal in causals:

                        print(
                            f"Testing "
                            f"B={B} H={H} "
                            f"N={N} D={D} "
                            f"causal={causal} ...",
                            end="\r",
                            flush=True,
                        )

                        try:

                            # ------------------------------------------------
                            # Q K V dO
                            # ------------------------------------------------

                            q = torch.randn(
                                B, H, N, D,
                                device="cuda",
                                dtype=dtype,
                            )

                            k = torch.randn(
                                B, H, N, D,
                                device="cuda",
                                dtype=dtype,
                            )

                            v = torch.randn(
                                B, H, N, D,
                                device="cuda",
                                dtype=dtype,
                            )

                            do = torch.randn(
                                B, H, N, D,
                                device="cuda",
                                dtype=dtype,
                            )


                            # =================================================
                            # YOUR FLASHATTN
                            # =================================================

                            o, L = flash_fwd(
                                q,
                                k,
                                v,
                                causal,
                            )


                            def custom_backward():

                                return flash_bwd(
                                    q,
                                    k,
                                    v,
                                    o,
                                    do,
                                    L,
                                    causal,
                                )


                            # =================================================
                            # PYTORCH SDPA
                            #
                            # Force Flash Attention implementation.
                            # =================================================

                            qs = (
                                q.detach()
                                .clone()
                                .requires_grad_(True)
                            )

                            ks = (
                                k.detach()
                                .clone()
                                .requires_grad_(True)
                            )

                            vs = (
                                v.detach()
                                .clone()
                                .requires_grad_(True)
                            )


                            with sdpa_kernel(
                                SDPBackend.FLASH_ATTENTION
                            ):

                                sdpa_o = (
                                    F.scaled_dot_product_attention(
                                        qs,
                                        ks,
                                        vs,
                                        is_causal=causal,
                                    )
                                )


                            def sdpa_backward():

                                return torch.autograd.grad(
                                    sdpa_o,
                                    (qs, ks, vs),
                                    do,
                                    retain_graph=True,
                                )


                            # =================================================
                            # TIME
                            # =================================================

                            t_custom = time_ms(
                                custom_backward,
                                args.warmup,
                                args.iters,
                            )

                            t_sdpa = time_ms(
                                sdpa_backward,
                                args.warmup,
                                args.iters,
                            )


                            # =================================================
                            # METRICS
                            # =================================================

                            flops = bwd_flops(
                                B,
                                H,
                                N,
                                D,
                                causal,
                            )

                            custom_tflops = (
                                flops /
                                t_custom /
                                1e9
                            )

                            sdpa_tflops = (
                                flops /
                                t_sdpa /
                                1e9
                            )

                            speedup = (
                                t_sdpa /
                                t_custom
                            )


                            print(
                                f"{B:>3} "
                                f"{H:>3} "
                                f"{N:>6} "
                                f"{D:>4} "
                                f"{str(causal):>6} | "
                                f"{t_custom:>11.3f} "
                                f"{custom_tflops:>10.2f} | "
                                f"{t_sdpa:>10.3f} "
                                f"{sdpa_tflops:>10.2f} | "
                                f"{speedup:>7.2f}x"
                            )


                            rows.append([
                                B,
                                H,
                                N,
                                D,
                                causal,
                                t_custom,
                                custom_tflops,
                                t_sdpa,
                                sdpa_tflops,
                                speedup,
                            ])


                        except RuntimeError as e:

                            if "out of memory" in str(e).lower():

                                print(
                                    f"{B:>3} "
                                    f"{H:>3} "
                                    f"{N:>6} "
                                    f"{D:>4} "
                                    f"{str(causal):>6} | "
                                    f"OOM - skipped"
                                )

                                torch.cuda.empty_cache()

                            else:

                                print(
                                    f"{B:>3} "
                                    f"{H:>3} "
                                    f"{N:>6} "
                                    f"{D:>4} "
                                    f"{str(causal):>6} | "
                                    f"FAILED: {e}"
                                )


                        finally:

                            for name in [
                                "q",
                                "k",
                                "v",
                                "do",
                                "o",
                                "L",
                                "qs",
                                "ks",
                                "vs",
                                "sdpa_o",
                            ]:

                                if name in locals():
                                    del locals()[name]

                            torch.cuda.empty_cache()


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if args.csv:

        with open(
            args.csv,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "B",
                "H",
                "N",
                "D",
                "causal",
                "custom_ms",
                "custom_tflops",
                "sdpa_ms",
                "sdpa_tflops",
                "speedup",
            ])

            writer.writerows(rows)

        print()
        print("Saved:", args.csv)


if __name__ == "__main__":
    main()