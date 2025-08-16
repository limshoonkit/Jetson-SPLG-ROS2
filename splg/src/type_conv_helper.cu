#include "type_conv_helper.cuh"

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