#include "type_conv_helper.cuh"

__global__ void hwcToNchwKernel(const cv::cuda::PtrStepSz<float> left_mat,
                               const cv::cuda::PtrStepSz<float> right_mat,
                               float* out,
                               int height,
                               int width) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Calculate indices
    const int input_idx = y * left_mat.step / sizeof(float) + x;
    const int output_idx = y * width + x;
    const int plane_size = height * width;

    // Copy left image (batch 0)
    out[output_idx] = left_mat.ptr(y)[x];
    
    // Copy right image (batch 1)
    out[plane_size + output_idx] = right_mat.ptr(y)[x];
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

__global__ void fillConstantKernel(float* output, float value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = value;
    }
}

void fillBufferWithConstant(float* d_output, float value, int size, cudaStream_t stream)
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

void launchHWCToNCHWConversion(const cv::cuda::PtrStepSz<float> left_mat,
                              const cv::cuda::PtrStepSz<float> right_mat,
                              float* output,
                              int height,
                              int width,
                              cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    hwcToNchwKernel<<<grid, block, 0, stream>>>(left_mat, right_mat, output, height, width);
}