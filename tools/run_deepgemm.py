#!/usr/bin/env python3
"""DeepGEMM bf16 GEMM benchmark bridge for the C++ testbed.

Usage: python run_deepgemm.py -M 2048 -N 2048 -K 2048 -warmup 2 -iter 10

Prints a single JSON line with benchmark results that the C++ testbed parses.
"""

import argparse
import json
import sys

import torch
import deep_gemm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=2048)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("-K", type=int, default=2048)
    parser.add_argument("-warmup", type=int, default=1)
    parser.add_argument("-iter", type=int, default=10)
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    d = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    try:
        for _ in range(args.warmup):
            deep_gemm.bf16_gemm_nt(a, b, d)
        torch.cuda.synchronize()
    except Exception as e:
        result = {"error": str(e)}
        print(json.dumps(result))
        sys.exit(0)

    times_ms = []
    for _ in range(args.iter):
        start.record()
        deep_gemm.bf16_gemm_nt(a, b, d)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    avg_ms = sum(times_ms) / len(times_ms)

    total_flops = 2.0 * M * N * K
    gflops = total_flops / (avg_ms / 1000.0) / 1e9

    print(json.dumps({
        "time_ms":   round(avg_ms, 4),
        "gflops":    round(gflops, 2),
        "iterations": args.iter,
        "M": M, "N": N, "K": K,
    }))


if __name__ == "__main__":
    main()
