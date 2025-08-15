#include "type_conv_helper.cuh"

__global__ void convertFP16ToFP32Kernel(const __half *input, float *output, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        output[idx] = __half2float(input[idx]);
    }
}

void launchConvertFP16ToFP32(const __half *input, float *output, int total_elements, cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize = (total_elements + blockSize - 1) / blockSize;
    convertFP16ToFP32Kernel<<<gridSize, blockSize, 0, stream>>>(input, output, total_elements);
}