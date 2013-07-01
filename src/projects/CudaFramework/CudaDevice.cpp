#include "CudaDevice.h"

#include <Framework/GL.h>
#include <Framework/Util/Log.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "CudaUtil.h"

namespace restless
{
	CudaDevice::CudaDevice() :
		_deviceIndex(0)
	{
	}


	CudaDevice::~CudaDevice()
	{
	}

	CudaMemoryInfo CudaDevice::getMemoryInfo() const
	{
		CudaMemoryInfo info;

		if (cudasafe( cudaMemGetInfo(&info.freeBytes, &info.totalBytes), "cudaMemGetInfo" ) == false) {
			return info;
		}

		info.usedBytes = info.totalBytes - info.freeBytes;

		return info;
	}

	bool CudaDevice::chooseDevice(const int majorCapability, const int minorCapability)
	{
		cudaDeviceProp deviceProp;
		deviceProp.major = majorCapability;
		deviceProp.minor = minorCapability;

		return chooseDevice(deviceProp);
	}

	bool CudaDevice::chooseDevice(const cudaDeviceProp & deviceProp)
	{
		if (cudasafe( cudaChooseDevice(& _deviceIndex, & deviceProp), "cudaChooseDevice" ) == false) {
			return false;
		}

		if (cudasafe( cudaGLSetGLDevice(_deviceIndex), "cudaGLSetGLDevice" ) == false) {
			return false;
		}

		L_INFO << "CUDA Device " << _deviceIndex << " choosen for OpenGL";

		return true;
	}

	void CudaDevice::listDevices()
	{
		int deviceCount = 0;

		if (cudasafe( cudaGetDeviceCount(& deviceCount), "cudaGetDeviceCount" ) == false) {
			return;
		}

		if (deviceCount == 0) {
			L_ERROR << "There are no available device(s) that support CUDA";
			return;
		}

		L_INFO << "Detected " << deviceCount << " CUDA Capable device";

		for (int devIndex = 0; devIndex < deviceCount; ++devIndex) {

			int driverVersion = 0;
			int runtimeVersion = 0;

			cudaDeviceProp deviceProp;

			cudasafe( cudaSetDevice(devIndex), "cudaSetDevice" );
			cudasafe( cudaGetDeviceProperties(& deviceProp, devIndex), "cudaGetDeviceProperties" );

			cudasafe( cudaDriverGetVersion(& driverVersion), "cudaDriverGetVersion" );
			cudasafe( cudaRuntimeGetVersion(& runtimeVersion), "cudaRuntimeGetVersion" );

			L_INFO << "Device " << devIndex << ": " << deviceProp.name;
			L_INFO << "  CUDA Driver Version / Runtime Version:         " << driverVersion/1000 << "." << (driverVersion%100)/10 << " / " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10;
			L_INFO << "  CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor;
			L_INFO << "  Total amount of global memory:                 " << (float) deviceProp.totalGlobalMem/1048576.0f << "MBytes (" << (unsigned long long) deviceProp.totalGlobalMem << " bytes)";
			L_INFO << "  Multiprocessors:                               " << deviceProp.multiProcessorCount;
			L_INFO << "  GPU Clock rate:                                " << deviceProp.memoryClockRate * 1e-3f << " MHz";
			L_INFO << "  Memory Clock rate:                             " << deviceProp.clockRate * 1e-3f << " MHz (" <<  deviceProp.clockRate * 1e-6f << " GHz)";
			L_INFO << "  Memory Bus Width:                              " << deviceProp.memoryBusWidth << "-bit";
			L_INFO << "  L2 Cache Size:                                 " << deviceProp.l2CacheSize << " bytes";
			L_INFO << "  Max Texture Dimension Size (x,y,z):            1D=(" << deviceProp.maxTexture1D << "), 2D=(" << deviceProp.maxTexture2D[0] << "," << deviceProp.maxTexture2D[1] << "), 3D=(" << deviceProp.maxTexture3D[0] << "," << deviceProp.maxTexture3D[1] << "," << deviceProp.maxTexture3D[2] << ")";
			L_INFO << "  Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes";
			L_INFO << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes";
			L_INFO << "  Total number of registers available per block: " << deviceProp.regsPerBlock;
			L_INFO << "  Warp size:                                     " << deviceProp.warpSize;
			L_INFO << "  Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor;
			L_INFO << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock;
			L_INFO << "  Maximum number of threads per block (dim):     " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2];
			L_INFO << "  Maximum number of blocks per grid (dim):       " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2];
		}
	}

}