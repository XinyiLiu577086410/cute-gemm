#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <sstream>
#include <functional>
#include <memory>
#include <cstdio>
#include <cstdlib>

#include "testbed.hpp"
#include "data_utils.hpp"
#include "timing.hpp"
#include "verify.hpp"

#ifdef HAS_DEEPGEMM
static std::string run_deepgemm_python(int M, int N, int K,
                                         int warmup, int iterations) {
    std::ostringstream cmd;
    cmd << "python3 tools/run_deepgemm.py"
        << " -M " << M << " -N " << N << " -K " << K
        << " -warmup " << warmup << " -iter " << iterations
        << " 2>/dev/null";
    std::string result;
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) return "";
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    if (result.empty()) return "";
    auto start = result.find('{');
    auto end   = result.rfind('}');
    if (start == std::string::npos || end == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

static KernelResult run_deepgemm(const TestConfig& cfg,
                                  const DeviceInfo& info) {
    KernelResult res;
    res.name = "DeepGEMM";

    std::string json = run_deepgemm_python(cfg.M, cfg.N, cfg.K,
                                             cfg.warmup, cfg.iterations);
    if (json.empty() || json.find("\"error\"") != std::string::npos) {
        std::cerr << "  DeepGEMM: not available on this GPU (requires SM90+).\n";
        return res;
    }

    auto get_val = [&](const std::string& key) -> double {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0.0;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        char* end = nullptr;
        double val = std::strtod(json.c_str() + pos, &end);
        return val;
    };

    res.time_ms = get_val("time_ms");
    res.gflops = get_val("gflops");
    res.bandwidth_gbs = compute_bandwidth_gbs(cfg.M, cfg.N, cfg.K, res.time_ms);
    res.compute_util_pct = compute_utilization(res.gflops,
                                                info.theoretical_peak_gflops);
    res.bandwidth_util_pct = compute_bandwidth_utilization(
        res.bandwidth_gbs, info.theoretical_peak_bandwidth_gbs);
    return res;
}
#endif

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -M <int>         M dimension (rows of A/C)        [default: 4096]\n"
              << "  -N <int>         N dimension (cols of B/C)        [default: 4096]\n"
              << "  -K <int>         K dimension (cols of A, rows of B)[default: 4096]\n"
              << "  -warmup <int>    Warmup iterations                [default: 1]\n"
              << "  -iter <int>      Measurement iterations           [default: 10]\n"
              << "  --skip-cublas    Skip cuBLAS baseline\n"
              << "  --skip-deepgemm  Skip DeepGEMM baseline\n"
              << "  --skip-user      Skip user kernel\n"
              << "  --threshold <f>  Error threshold for correctness   [default: 1e-2]\n"
              << "  -h               Print this help\n";
}

static TestConfig parse_args(int argc, char* argv[]) {
    TestConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) {
            cfg.M = std::stoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            cfg.N = std::stoi(argv[++i]);
        } else if (arg == "-K" && i + 1 < argc) {
            cfg.K = std::stoi(argv[++i]);
        } else if (arg == "-warmup" && i + 1 < argc) {
            cfg.warmup = std::stoi(argv[++i]);
        } else if (arg == "-iter" && i + 1 < argc) {
            cfg.iterations = std::stoi(argv[++i]);
        } else if (arg == "--skip-cublas") {
            cfg.skip_cublas = true;
        } else if (arg == "--skip-deepgemm") {
            cfg.skip_deepgemm = true;
        } else if (arg == "--skip-user") {
            cfg.skip_user = true;
        } else if (arg == "--threshold" && i + 1 < argc) {
            cfg.error_threshold = std::stof(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return cfg;
}

static void print_header(const TestConfig& cfg, const DeviceInfo& info) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "  GEMM Benchmark Testbed\n";
    std::cout << "================================================================================\n";
    std::cout << "  Configuration: M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K << "\n";
    std::cout << "  Warmup: " << cfg.warmup << ", Iterations: " << cfg.iterations << "\n";
    std::cout << "  Device: " << info.name << "\n";
    std::cout << "  SMs: " << info.sm_count
              << ", Clock: " << (info.clock_rate_khz / 1000) << " MHz\n";
    std::cout << "  Memory: " << (info.memory_clock_khz / 1000) << " MHz"
              << ", Bus: " << info.memory_bus_width << " bits\n";
    std::cout << "  Theoretical Peak BF16: "
              << std::fixed << std::setprecision(1)
              << info.theoretical_peak_gflops << " GFLOPS\n";
    std::cout << "  Theoretical Peak BW:   "
              << std::fixed << std::setprecision(1)
              << info.theoretical_peak_bandwidth_gbs << " GB/s\n";
    std::cout << "================================================================================\n\n";
}

static KernelResult run_cublas(const TestConfig& cfg,
                                const DeviceInfo& info,
                                const DeviceBuffers& bufs,
                                cublasHandle_t handle,
                                std::vector<__nv_bfloat16>& hC_ref) {
    KernelResult res;
    res.name = "cuBLAS";
    float alpha = 1.0f;
    float beta  = 0.0f;

    auto launch = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            cfg.M, cfg.N, cfg.K,
            &alpha,
            bufs.dA, CUDA_R_16BF, cfg.M,
            bufs.dB, CUDA_R_16BF, cfg.K,
            &beta,
            bufs.dC, CUDA_R_16BF, cfg.M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };

    res.time_ms = measure_kernel_time_ms(launch, cfg.warmup, cfg.iterations);
    res.gflops = compute_gflops(cfg.M, cfg.N, cfg.K, res.time_ms);
    res.bandwidth_gbs = compute_bandwidth_gbs(cfg.M, cfg.N, cfg.K, res.time_ms);
    res.compute_util_pct = compute_utilization(res.gflops,
                                                info.theoretical_peak_gflops);
    res.bandwidth_util_pct = compute_bandwidth_utilization(
        res.bandwidth_gbs, info.theoretical_peak_bandwidth_gbs);

    hC_ref = copy_to_host(bufs.dC, bufs.size_C);
    return res;
}

static KernelResult run_user_kernel(const TestConfig& cfg,
                                     const DeviceInfo& info,
                                     const DeviceBuffers& bufs,
                                     std::vector<__nv_bfloat16>& hC_user) {
    KernelResult res;
    res.name = "User GEMM";

    auto launch = [&]() {
        user_gemm(bufs.dA, bufs.dB, bufs.dC, cfg.M, cfg.N, cfg.K);
    };

    for (int w = 0; w < cfg.warmup; ++w) {
        zero_device(bufs.dC, bufs.size_C);
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    res.time_ms = measure_kernel_time_ms_prepare(
        [&]() { zero_device(bufs.dC, bufs.size_C); },
        launch,
        cfg.iterations);

    res.gflops = compute_gflops(cfg.M, cfg.N, cfg.K, res.time_ms);
    res.bandwidth_gbs = compute_bandwidth_gbs(cfg.M, cfg.N, cfg.K, res.time_ms);
    res.compute_util_pct = compute_utilization(res.gflops,
                                                info.theoretical_peak_gflops);
    res.bandwidth_util_pct = compute_bandwidth_utilization(
        res.bandwidth_gbs, info.theoretical_peak_bandwidth_gbs);

    hC_user = copy_to_host(bufs.dC, bufs.size_C);
    return res;
}

static void print_results_table(const std::vector<KernelResult>& results) {
    std::cout << "  Benchmark Results:\n";
    std::cout << "  " << std::left << std::setw(16) << "Name"
              << std::right
              << std::setw(12) << "Time(ms)"
              << std::setw(14) << "GFLOPS"
              << std::setw(14) << "BW(GB/s)"
              << std::setw(14) << "CompUtil(%)"
              << std::setw(14) << "BWUtil(%)\n";
    std::cout << "  " << std::string(84, '-') << "\n";

    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(16) << r.name
                  << std::right << std::fixed
                  << std::setprecision(4) << std::setw(12) << r.time_ms
                  << std::setprecision(2) << std::setw(14) << r.gflops
                  << std::setprecision(2) << std::setw(14) << r.bandwidth_gbs
                  << std::setprecision(1) << std::setw(13) << r.compute_util_pct
                  << "%"
                  << std::setprecision(1) << std::setw(13) << r.bandwidth_util_pct
                  << "%\n";
    }
    std::cout << "\n";
}

static void print_speedup(const std::vector<KernelResult>& results) {
    const KernelResult* user_res = nullptr;
    const KernelResult* cublas_res = nullptr;
    const KernelResult* deepgemm_res = nullptr;

    for (const auto& r : results) {
        if (r.name == "User GEMM")  user_res = &r;
        if (r.name == "cuBLAS")     cublas_res = &r;
        if (r.name == "DeepGEMM")   deepgemm_res = &r;
    }

    if (!user_res || user_res->time_ms <= 0.0) return;

    std::cout << "  Speedup:\n";
    if (cublas_res && cublas_res->time_ms > 0.0) {
        double speedup = cublas_res->time_ms / user_res->time_ms;
        std::cout << "    User GEMM vs cuBLAS:   "
                  << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    if (deepgemm_res && deepgemm_res->time_ms > 0.0) {
        double speedup = deepgemm_res->time_ms / user_res->time_ms;
        std::cout << "    User GEMM vs DeepGEMM: "
                  << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    std::cout << "\n";
}

static void print_correctness(const CorrectnessResult& cres,
                               const TestConfig& cfg) {
    std::cout << "  Correctness (User GEMM vs cuBLAS):\n";
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "    Max Absolute Error: " << cres.max_absolute_error << "\n";
    std::cout << "    Max Relative Error: " << cres.max_relative_error << "\n";
    std::cout << std::fixed;
    std::cout << "    Threshold: " << cfg.error_threshold << "\n";
    std::cout << "    Result: "
              << (cres.passed ? "\033[1;32mPASS\033[0m"
                               : "\033[1;31mFAIL\033[0m")
              << "\n";
    std::cout << "================================================================================\n\n";
}

int main(int argc, char* argv[]) {
    TestConfig cfg = parse_args(argc, argv);

    try {
        DeviceInfo info = query_device_info(0);
        print_header(cfg, info);

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        std::vector<__nv_bfloat16> hA = generate_random_bf16(
            static_cast<size_t>(cfg.M) * cfg.K, 42);
        std::vector<__nv_bfloat16> hB = generate_random_bf16(
            static_cast<size_t>(cfg.K) * cfg.N, 123);

        DeviceBuffers bufs;
        bufs.allocate(cfg.M, cfg.N, cfg.K);
        copy_to_device(bufs.dA, hA);
        copy_to_device(bufs.dB, hB);

        std::vector<KernelResult> results;
        std::vector<__nv_bfloat16> hC_ref;
        std::vector<__nv_bfloat16> hC_user;

        if (!cfg.skip_cublas) {
            std::cout << "  Running cuBLAS baseline ..." << std::flush;
            results.push_back(
                run_cublas(cfg, info, bufs, cublas_handle, hC_ref));
            std::cout << " done.\n";
        }

#ifndef HAS_DEEPGEMM
        if (!cfg.skip_deepgemm) {
            std::cout << "  DeepGEMM: not enabled (build with DEEPGEMM=1). Skipping.\n";
        }
#else
        if (!cfg.skip_deepgemm) {
            std::cout << "  Running DeepGEMM baseline ..." << std::flush;
            auto dg_res = run_deepgemm(cfg, info);
            if (dg_res.time_ms > 0.0) {
                results.push_back(dg_res);
                std::cout << " done.\n";
            }
        }
#endif

        if (!cfg.skip_user) {
            std::cout << "  Running User GEMM kernel ..." << std::flush;
            results.push_back(run_user_kernel(cfg, info, bufs, hC_user));
            std::cout << " done.\n";
        }

        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        bufs.release();

        print_results_table(results);
        print_speedup(results);

        if (!cfg.skip_user && !cfg.skip_cublas &&
            !hC_ref.empty() && !hC_user.empty()) {
            size_t count = static_cast<size_t>(cfg.M) * cfg.N;
            CorrectnessResult cres = verify_correctness(
                hC_user, hC_ref, count, cfg.error_threshold);
            print_correctness(cres, cfg);
        } else if (!cfg.skip_user && cfg.skip_cublas) {
            std::cout << "  Correctness: cuBLAS baseline skipped. Cannot verify.\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
