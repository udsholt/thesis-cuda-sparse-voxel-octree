#ifndef _RESTLESS_CUDA_UTIL_H
#define _RESTLESS_CUDA_UTIL_H

#include "cuda.h"
#include "cuda_runtime.h"

namespace restless
{
	bool cudasafe(cudaError_t err, const char * str);
	bool cudasafe_post_kernel(const char * kernelName);
}

#endif