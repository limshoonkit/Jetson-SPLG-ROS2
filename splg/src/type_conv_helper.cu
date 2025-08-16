#include "type_conv_helper.cuh"

__global__ void hwcToNchwKernel(const cv::cuda::PtrStepSz<float> left_mat,
                                const cv::cuda::PtrStepSz<float> right_mat,
                                float *out,
                                int height, int width)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = height * width;

    if (tid >= total_pixels)
        return;

    const int y = tid / width;
    const int x = tid % width;

    // Coalesced writes to output buffer
    out[tid] = left_mat.ptr(y)[x];                 // Batch 0
    out[total_pixels + tid] = right_mat.ptr(y)[x]; // Batch 1
}

__global__ void convertFP16ToFP32Kernel(const __half *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = __half2float(in[i]);
    }
}

__global__ void convertFP32ToFP16Kernel(const float *in, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = __float2half(in[i]);
    }
}

__global__ void fillConstantKernel(float *output, float value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = value;
    }
}

void fillBufferWithConstant(float *d_output, float value, int size, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    fillConstantKernel<<<blocks, threads, 0, stream>>>(d_output, value, size);
}

void launchConvertFP16ToFP32(const __half *in, float *out, int n, cudaStream_t stream)
{
    if (n <= 0)
        return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convertFP16ToFP32Kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launchConvertFP32ToFP16(const float *in, __half *out, int n, cudaStream_t stream)
{
    if (n <= 0)
        return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convertFP32ToFP16Kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launchHWCToNCHWConversion(const cv::cuda::GpuMat &left_mat,
                               const cv::cuda::GpuMat &right_mat,
                               float *d_output,
                               int height, int width,
                               cudaStream_t stream)
{
    const int total_pixels = height * width;
    const int block_size = 256;
    const int grid_size = (total_pixels + block_size - 1) / block_size;

    hwcToNchwKernel<<<grid_size, block_size, 0, stream>>>(
        left_mat, right_mat, d_output, height, width);
}