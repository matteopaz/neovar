// Copyright 2020 California Institute of Technology. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Author: Ethan Jaszewski

#include "aov.h"

#include <algorithm>

#include "cuda_runtime.h"
#include "math.h"

#include "errchk.cuh"

//
// Simple AOV Function Definitions
//

AOV::AOV(size_t n_bins, size_t bin_overlap) {
    num_bins = n_bins;
    num_overlap = bin_overlap;

    bin_size = 1.0 / static_cast<float>(n_bins);
}

__host__ __device__ size_t AOV::NumPhaseBins() const {
    return num_bins;
}

__host__ __device__ size_t AOV::NumPhaseBinOverlap() const {
    return num_overlap;
}

__host__ __device__ size_t AOV::PhaseBin(float phase_val) const {
    return static_cast<size_t>(phase_val / bin_size);
}

//
// CUDA Kernels
//

extern __shared__ uint32_t shared_bytes[];

__global__ void FoldBinKernel(const float* times,
                              const float* mags,
                              const size_t length,
                              const float* periods,
                              const float* period_dts,
                              const AOV aov,
                              AOVData* data) {
    uint32_t* sh_count = &shared_bytes[0];
    float* sh_sums = (float*)&shared_bytes[aov.NumPhaseBins()];
    float* sh_sq_sums = (float*)&shared_bytes[2 * aov.NumPhaseBins()];

    for (size_t idx = threadIdx.x; idx < aov.NumPhaseBins();
         idx += blockDim.x) {
        sh_count[idx] = 0;
        sh_sums[idx] = 0;
        sh_sq_sums[idx] = 0;
    }

    __syncthreads();

    // Period and period time derivative for this block.
    const float period = periods[blockIdx.x];
    const float period_dt = period_dts[blockIdx.y];

    // Time derivative correction factor.
    const float pdt_corr = (period_dt / period) / 2;

    float i_part;  // Only used for modff.

    // Compute the histogram statistics.
    for (size_t idx = threadIdx.x; idx < length; idx += blockDim.x) {
        float t = times[idx];
        float t_corr = t - pdt_corr * t * t;
        float folded = fabsf(modff(t_corr / period, &i_part));

        float mag = mags[idx];

        size_t bin = aov.PhaseBin(folded);

        for (size_t i = 0; i < aov.NumPhaseBinOverlap(); i++) {
            size_t idx = (bin + i) % aov.NumPhaseBins();

            atomicAdd(&sh_count[bin], 1);
            atomicAdd(&sh_sums[bin], mag);
            atomicAdd(&sh_sq_sums[bin], mag * mag);
        }
    }

    __syncthreads();

    size_t block_id = blockIdx.x * gridDim.y + blockIdx.y;

    for (size_t idx = threadIdx.x; idx < aov.NumPhaseBins();
         idx += blockDim.x) {
        data[block_id * aov.NumPhaseBins() + idx] = {
            sh_count[idx], sh_sums[idx], sh_sq_sums[idx]};
    }
}

__global__ void AOVKernel(const AOVData* data,
                          const size_t num_hists,
                          const float length,
                          const float avg,
                          const AOV aov,
                          float* aovs) {
    size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id >= num_hists)
        return;

    float s1 = 0;
    float s2 = 0;

    for (size_t idx = 0; idx < aov.NumPhaseBins(); idx++) {
        AOVData a = data[thread_id * aov.NumPhaseBins() + idx];
        float n = static_cast<float>(a.count);
        float sum = a.sum;
        float sq_sum = a.sq_sum;

        if (n != 0) {
            float aux = sum / n;
            float residual = aux - avg;
            s1 += n * residual * residual;
            s2 += sq_sum - n * aux * aux;
        }
    }

    aovs[thread_id] = (static_cast<float>(length - aov.NumPhaseBins())
                       / static_cast<float>(aov.NumPhaseBins() - 1))
                      * (s1 / s2);
}

//
// Helper Functions
//

float ArrayMean(const float* data, const size_t length) {
    float sum = 0;

    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }

    return sum / static_cast<float>(length);
}

//
// Wrapper Functions
//

AOVData* AOV::DeviceFoldAndBin(const float* times,
                               const float* mags,
                               const size_t length,
                               const float* periods,
                               const float* period_dts,
                               const size_t num_periods,
                               const size_t num_p_dts) const {
    // Number of bytes of global memory required to store output
    size_t bytes = NumPhaseBins() * sizeof(AOVData) * num_periods * num_p_dts;

    // Allocate and zero global memory for output histograms
    AOVData* dev_hists;
    gpuErrchk(cudaMalloc(&dev_hists, bytes));

    // Number of threads and corresponding shared memory usage
    const size_t num_threads = 256;
    const size_t shared_bytes = NumPhaseBins() * sizeof(AOVData);

    // Grid to search over periods and time derivatives
    const dim3 grid_dim = dim3(num_periods, num_p_dts);

    // NOTE: An AOV object is small enough that we can pass it in
    //       the registers by dereferencing it.
    FoldBinKernel<<<grid_dim, num_threads, shared_bytes>>>(
        times, mags, length, periods, period_dts, *this, dev_hists);

    return dev_hists;
}

AOVData* AOV::FoldAndBin(const float* times,
                         const float* mags,
                         const size_t length,
                         const float* periods,
                         const float* period_dts,
                         const size_t num_periods,
                         const size_t num_p_dts) const {
    // Number of bytes of input data
    const size_t data_bytes = length * sizeof(float);

    // Allocate device pointers
    float* dev_times;
    float* dev_mags;
    float* dev_periods;
    float* dev_period_dts;
    gpuErrchk(cudaMalloc(&dev_times, data_bytes));
    gpuErrchk(cudaMalloc(&dev_mags, data_bytes));
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));

    // Copy data to device memory
    gpuErrchk(cudaMemcpy(dev_times, times, data_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_mags, mags, data_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
                         cudaMemcpyHostToDevice));

    AOVData* dev_hists =
        DeviceFoldAndBin(dev_times, dev_mags, length, dev_periods,
                         dev_period_dts, num_periods, num_p_dts);

    // Allocate host histograms and copy from device
    size_t bytes = NumPhaseBins() * num_periods * num_p_dts * sizeof(AOVData);
    AOVData* hists = (AOVData*)malloc(bytes);
    gpuErrchk(cudaMemcpy(hists, dev_hists, bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpuErrchk(cudaFree(dev_times));
    gpuErrchk(cudaFree(dev_mags));
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    gpuErrchk(cudaFree(dev_hists));

    return hists;
}

float* AOV::DeviceCalcAOVFromHists(const AOVData* hists,
                                   const size_t num_hists,
                                   const float length,
                                   const float avg) const {
    // Allocate global memory for output conditional entropy values
    float* dev_aovs;
    gpuErrchk(cudaMalloc(&dev_aovs, num_hists * sizeof(float)));

    const size_t n_t = 512;
    const size_t n_b = (num_hists / n_t) + 1;

    // NOTE: An AOV object is small enough that we can pass it in
    //       the registers by dereferencing it.
    AOVKernel<<<n_b, n_t>>>(hists, num_hists, length, avg, *this, dev_aovs);

    return dev_aovs;
}

float* AOV::CalcAOVFromHists(const AOVData* hists,
                             const size_t num_hists,
                             const float length,
                             const float avg) const {
    // Number of bytes in the histogram
    const size_t bytes = num_hists * NumPhaseBins() * sizeof(AOVData);

    // Allocate device memory for histograms and copy over
    AOVData* dev_hists;
    gpuErrchk(cudaMalloc(&dev_hists, bytes));
    gpuErrchk(cudaMemcpy(dev_hists, hists, bytes, cudaMemcpyHostToDevice));

    float* dev_ces = DeviceCalcAOVFromHists(dev_hists, num_hists, length, avg);

    // Copy CEs to host
    float* ces = (float*)malloc(num_hists * sizeof(float));
    gpuErrchk(cudaMemcpy(ces, dev_ces, num_hists * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpuErrchk(cudaFree(dev_hists));
    gpuErrchk(cudaFree(dev_ces));

    return ces;
}

void AOV::CalcAOVVals(float* times,
                      float* mags,
                      size_t length,
                      const float* periods,
                      const float* period_dts,
                      const size_t num_periods,
                      const size_t num_p_dts,
                      float* aov_out) const {
    CalcAOVValsBatched(std::vector<float*>{times}, std::vector<float*>{mags},
                       std::vector<size_t>{length}, periods, period_dts,
                       num_periods, num_p_dts, aov_out);
}

float* AOV::CalcAOVVals(float* times,
                        float* mags,
                        size_t length,
                        const float* periods,
                        const float* period_dts,
                        const size_t num_periods,
                        const size_t num_p_dts) const {
    return CalcAOVValsBatched(std::vector<float*>{times},
                              std::vector<float*>{mags},
                              std::vector<size_t>{length}, periods, period_dts,
                              num_periods, num_p_dts);
}

void AOV::CalcAOVValsBatched(const std::vector<float*>& times,
                             const std::vector<float*>& mags,
                             const std::vector<size_t>& lengths,
                             const float* periods,
                             const float* period_dts,
                             const size_t num_periods,
                             const size_t num_p_dts,
                             float* aov_out) const {
    // TODO: Use async memory transferring
    // TODO: Look at ways of batching data transfer.

    // Size of one AOV out array, and total AOV output size.
    size_t aov_out_size = num_periods * num_p_dts * sizeof(float);
    size_t aov_size_total = aov_out_size * lengths.size();

    // Copy trial information over
    float* dev_periods;
    float* dev_period_dts;
    gpuErrchk(cudaMalloc(&dev_periods, num_periods * sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_period_dts, num_p_dts * sizeof(float)));
    gpuErrchk(cudaMemcpy(dev_periods, periods, num_periods * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_period_dts, period_dts, num_p_dts * sizeof(float),
                         cudaMemcpyHostToDevice));

    // Intermediate histogram memory
    size_t num_hists = num_periods * num_p_dts;
    size_t hist_bytes = NumPhaseBins() * sizeof(AOVData) * num_hists;
    AOVData* dev_hists;
    gpuErrchk(cudaMalloc(&dev_hists, hist_bytes));

    // Intermediate conditional entropy memory
    float* dev_aovs;
    gpuErrchk(cudaMalloc(&dev_aovs, aov_out_size));

    // Kernel launch information for the fold & bin step
    const size_t num_threads_fb = 256;
    const size_t shared_bytes_fb = NumPhaseBins() * sizeof(AOVData);
    const dim3 grid_dim_fb = dim3(num_periods, num_p_dts);

    // Kernel launch information for the ce calculation step
    const size_t num_threads_aov = 256;
    const size_t num_blocks_aov = (num_hists / num_threads_aov) + 1;
    const size_t shared_bytes_aov = num_threads_aov * sizeof(float);

    // Buffer size (large enough for longest light curve)
    auto max_length = std::max_element(lengths.begin(), lengths.end());
    const size_t buffer_length = *max_length;
    const size_t buffer_bytes = sizeof(float) * buffer_length;

    float* dev_times_buffer;
    float* dev_mags_buffer;
    gpuErrchk(cudaMalloc(&dev_times_buffer, buffer_bytes));
    gpuErrchk(cudaMalloc(&dev_mags_buffer, buffer_bytes));

    for (size_t i = 0; i < lengths.size(); i++) {
        float mean_mag = ArrayMean(mags[i], lengths[i]);

        // Copy light curve into device buffer
        const size_t curve_bytes = lengths[i] * sizeof(float);
        cudaMemcpy(dev_times_buffer, times[i], curve_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(dev_mags_buffer, mags[i], curve_bytes,
                   cudaMemcpyHostToDevice);

        // Zero AOV output
        gpuErrchk(cudaMemset(dev_aovs, 0, aov_out_size));

        // NOTE: An AOV object is small enough that we can pass it
        //       in the registers by dereferencing it.

        FoldBinKernel<<<grid_dim_fb, num_threads_fb, shared_bytes_fb>>>(
            dev_times_buffer, dev_mags_buffer, lengths[i], dev_periods,
            dev_period_dts, *this, dev_hists);

        AOVKernel<<<num_blocks_aov, num_threads_aov, shared_bytes_aov>>>(
            dev_hists, num_hists, lengths[i], mean_mag, *this, dev_aovs);

        // Copy AOV data back to host
        cudaMemcpy(&aov_out[i * num_hists], dev_aovs, aov_out_size,
                   cudaMemcpyDeviceToHost);
    }

    // Free all of the GPU memory
    gpuErrchk(cudaFree(dev_periods));
    gpuErrchk(cudaFree(dev_period_dts));
    gpuErrchk(cudaFree(dev_hists));
    gpuErrchk(cudaFree(dev_aovs));
    gpuErrchk(cudaFree(dev_times_buffer));
    gpuErrchk(cudaFree(dev_mags_buffer));
}

float* AOV::CalcAOVValsBatched(const std::vector<float*>& times,
                               const std::vector<float*>& mags,
                               const std::vector<size_t>& lengths,
                               const float* periods,
                               const float* period_dts,
                               const size_t num_periods,
                               const size_t num_p_dts) const {
    // Size of one AOV out array, and total AOV output size.
    size_t aov_out_size = num_periods * num_p_dts * sizeof(float);
    size_t aov_size_total = aov_out_size * lengths.size();

    // Allocate the output AOV array so we can copy to it.
    float* aov_out = (float*)malloc(aov_size_total);

    // Perform AOV calculation.
    CalcAOVValsBatched(times, mags, lengths, periods, period_dts, num_periods,
                       num_p_dts, aov_out);

    return aov_out;
}
