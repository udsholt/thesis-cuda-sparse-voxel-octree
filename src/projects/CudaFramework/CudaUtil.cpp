#include "CudaUtil.h"

#include <Framework/Util/Log.h>

#include "cuda.h"
#include "cuda_runtime.h"

namespace restless
{
	bool cudasafe(cudaError_t err, const char * str)
	{
		if (err != cudaSuccess) {
			L_ERROR << str << ", error: " << (int) err << " '" << cudaGetErrorString(err) << "'";
			return false;
		}
		return true;
	}

	bool cudasafe_post_kernel(const char * kernelName)
	{
		cudaThreadSynchronize();
		return cudasafe(cudaGetLastError(), kernelName);
	}

}
