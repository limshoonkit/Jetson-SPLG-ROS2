#ifndef TYPE_CONV_HELPER_CUH
#define TYPE_CONV_HELPER_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void fillBufferWithConstant(float *d_output, float value, int size, cudaStream_t stream);
void launchConvertFP16ToFP32(const __half *in, float *out, int n, cudaStream_t stream);
void launchConvertFP32ToFP16(const float *in, __half *out, int n, cudaStream_t stream);

#endif // TYPE_CONV_HELPER_CUH