#ifndef KERNEL_HIP
#define KERNEL_HIP

#include <hip/hip_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cfloat>
#include "main.h"

#define TPB 256  // threads per block
#define MIN_DENOM 1e-12f

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = call;                                            \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,  \
                    hipGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Kernel: each block computes a block-local maximum and writes to block_max[blockIdx.x]
__global__ void max_kernel(const float* __restrict__ input, float* __restrict__ block_max, int N) {
    __shared__ double sdata[TPB];  // shared memory in double
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    double local_max = -DBL_MAX;

    // Strided loop over input
    for (int idx = gid; idx < N; idx += gridDim.x * blockDim.x) {
        double v = (double)input[idx];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // reduction in shared memory for max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double o = sdata[tid + s];
            if (o > sdata[tid]) sdata[tid] = o;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = (float)sdata[0];  // cast back to float
    }
}

// Kernel: compute t_i = exp(input[i] - m), write to temp[i], and compute per-block sum -> block_sum[blockIdx.x]
__global__ void sum_exp_kernel(const float* __restrict__ input, float* __restrict__ temp, float* __restrict__ block_sum, float m, int N) {
    __shared__ double sdata[TPB];  // shared memory in double
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    double local_sum = 0.0;

    for (int idx = gid; idx < N; idx += gridDim.x * blockDim.x) {
        double v = (double)input[idx];
        double t = exp(v - (double)m);
        temp[idx] = (float)t;  // keep temp as float
        local_sum += t;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // reduction in shared memory for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = (float)sdata[0];  // cast back to float
    }
}

// Kernel: normalize temp by S -> output
__global__ void normalize_kernel(const float* __restrict__ temp, float* __restrict__ output, float S, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = gid; idx < N; idx += gridDim.x * blockDim.x) {
        double val = (double)temp[idx] / (double)S;
        output[idx] = (float)val;
    }
}

extern "C" void solve(const float* input_host, float* output_host, int N) {
    if (N <= 0) return;

    // Device pointers
    float *d_input = nullptr, *d_temp = nullptr, *d_output = nullptr;
    float *d_block_max = nullptr, *d_block_sum = nullptr;

    size_t bytes = sizeof(float) * (size_t)N;

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_temp, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));

    // copy input to device
    HIP_CHECK(hipMemcpy(d_input, input_host, bytes, hipMemcpyHostToDevice));

    // choose grid size
    int blocks = (N + TPB - 1) / TPB;
    // cap blocks to a reasonable maximum
    int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;

    // allocate block arrays sized by blocks
    HIP_CHECK(hipMalloc(&d_block_max, sizeof(float) * blocks));
    HIP_CHECK(hipMalloc(&d_block_sum, sizeof(float) * blocks));

    dim3 grid(blocks);
    dim3 block(TPB);

    // 1 compute per-block maxima
    hipLaunchKernelGGL(max_kernel, grid, block, 0, 0, d_input, d_block_max, N);
    HIP_CHECK(hipGetLastError());

    // copy block_max to host and reduce to get global max
    std::vector<float> h_block_max(blocks);
    HIP_CHECK(hipMemcpy(h_block_max.data(), d_block_max, sizeof(float) * blocks, hipMemcpyDeviceToHost));
    double m_d = -DBL_MAX;
    for (int i = 0; i < blocks; ++i) if ((double)h_block_max[i] > m_d) m_d = (double)h_block_max[i];
    float m = (float)m_d;

    // 2 compute exp(x - m) and per-block sums
    hipLaunchKernelGGL(sum_exp_kernel, grid, block, 0, 0, d_input, d_temp, d_block_sum, m, N);
    HIP_CHECK(hipGetLastError());

    // copy block_sum back and reduce to S
    std::vector<float> h_block_sum(blocks);
    HIP_CHECK(hipMemcpy(h_block_sum.data(), d_block_sum, sizeof(float) * blocks, hipMemcpyDeviceToHost));
    double S_d = 0.0;
    for (int i = 0; i < blocks; ++i) S_d += (double)h_block_sum[i];
    float S = (float)S_d;
    if (S < MIN_DENOM) S = MIN_DENOM;

    // 3 normalize
    hipLaunchKernelGGL(normalize_kernel, grid, block, 0, 0, d_temp, d_output, S, N);
    HIP_CHECK(hipGetLastError());

    // copy output back
    HIP_CHECK(hipMemcpy(output_host, d_output, bytes, hipMemcpyDeviceToHost));

    // free device memory
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_temp));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_block_max));
    HIP_CHECK(hipFree(d_block_sum));
}

#endif // KERNEL_HIP
