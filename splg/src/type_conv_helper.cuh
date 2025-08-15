#ifndef TYPE_CONV_HELPER_CUH
#define TYPE_CONV_HELPER_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launchConvertFP16ToFP32(const __half *input, float *output, int total_elements, cudaStream_t stream);

#endif // TYPE_CONV_HELPER_CUH