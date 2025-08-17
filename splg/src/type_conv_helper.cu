#include "../include/type_conv_helper.cuh"

// https://www.dotndash.net/2023/03/09/using-tensorrt-with-opencv-cuda.html
// Stack and normalize
__global__ void toNCHWKernel(
    cv::cuda::PtrStepSz<uint8_t> left_img,
    cv::cuda::PtrStepSz<uint8_t> right_img,
    // half *out,
    float *out,
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
    // out[pixel_idx] = __float2half((left_img(y, x)) / 255.0f);
    out[pixel_idx] = static_cast<float>(left_img(y, x)) / 255.0f;

    // Right image -> batch 1, channel 0
    // out[channel_size + pixel_idx] = __float2half((right_img(y, x)) / 255.0f);
    out[channel_size + pixel_idx] = static_cast<float>(right_img(y, x)) / 255.0f;
}

void launchToNCHW(cv::cuda::PtrStepSz<uint8_t> d_left, cv::cuda::PtrStepSz<uint8_t> d_right,
                  void *d_input_, int input_height_, int input_width_)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((input_width_ + blockSize.x - 1) / blockSize.x,
                  (input_height_ + blockSize.y - 1) / blockSize.y);

    toNCHWKernel<<<gridSize, blockSize>>>(
        d_left, d_right,
        // static_cast<half *>(d_input_),
        static_cast<float *>(d_input_),
        input_height_, input_width_);
}