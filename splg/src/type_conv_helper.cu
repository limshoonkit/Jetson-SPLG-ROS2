#include "type_conv_helper.cuh"

// https://www.dotndash.net/2023/03/09/using-tensorrt-with-opencv-cuda.html
__global__ void toNCHWKernel(
    cv::cuda::PtrStepSz<uint8_t> left_img,
    cv::cuda::PtrStepSz<uint8_t> right_img, 
    float* out,
    int height, 
    int width)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int pixel_idx = y * width + x;
    const int channel_size = height * width;
    
    // Left image -> batch 0, channel 0
    out[pixel_idx] = static_cast<float>(left_img(y, x)) / 255.0f;
    
    // Right image -> batch 1, channel 0  
    out[channel_size + pixel_idx] = static_cast<float>(right_img(y, x)) / 255.0f;
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

void launchToNCHW(cv::cuda::PtrStepSz<uint8_t> d_left, cv::cuda::PtrStepSz<uint8_t> d_right,
                  void *d_input_, int input_height_, int input_width_)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((input_width_ + blockSize.x - 1) / blockSize.x,
                  (input_height_ + blockSize.y - 1) / blockSize.y);

    toNCHWKernel<<<gridSize, blockSize>>>(
        d_left, d_right,
        static_cast<float *>(d_input_),
        input_height_, input_width_);
}