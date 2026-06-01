#pragma once

#include <cuda_bf16.h>
#include <vector>
#include <cmath>
#include <algorithm>

#include "testbed.hpp"

inline CorrectnessResult verify_correctness(
    const std::vector<__nv_bfloat16>& computed,
    const std::vector<__nv_bfloat16>& reference,
    size_t count,
    float threshold) {

    CorrectnessResult result;
    result.max_absolute_error = 0.0f;
    result.max_relative_error = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float c_val = __bfloat162float(computed[i]);
        float r_val = __bfloat162float(reference[i]);

        float abs_err = std::abs(c_val - r_val);
        result.max_absolute_error = std::max(result.max_absolute_error, abs_err);

        if (std::abs(r_val) > 1e-8f) {
            float rel_err = abs_err / std::abs(r_val);
            result.max_relative_error =
                std::max(result.max_relative_error, rel_err);
        }
    }

    result.passed = (result.max_absolute_error <= threshold);
    return result;
}
