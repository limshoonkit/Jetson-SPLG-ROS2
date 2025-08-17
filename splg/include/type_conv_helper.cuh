#ifndef TYPE_CONV_HELPER_CUH
#define TYPE_CONV_HELPER_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/core/cuda.hpp>

void launchToNCHW(cv::cuda::PtrStepSz<uint8_t> d_left, cv::cuda::PtrStepSz<uint8_t> d_right, void *d_input_, int input_height_, int input_width_);

#endif // TYPE_CONV_HELPER_CUH