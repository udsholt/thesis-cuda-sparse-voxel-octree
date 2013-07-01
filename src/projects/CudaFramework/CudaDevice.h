#ifndef _RESTLESS_CUDA_DEVICE_H
#define _RESTLESS_CUDA_DEVICE_H

struct cudaDeviceProp;

namespace restless
{
	struct CudaMemoryInfo
	{
		unsigned int freeBytes;
		unsigned int totalBytes;
		unsigned int usedBytes;

		
		float getFreeMegabytes() const
		{
			return freeBytes * 9.53674316e-07F;
		}

		float getTotalMegabytes() const
		{
			return totalBytes * 9.53674316e-07F;
		}

		float getUsedMegabytes() const
		{
			return usedBytes * 9.53674316e-07F;
		}

	};

	class CudaDevice
	{
	public:
		CudaDevice();
		virtual ~CudaDevice();

		void listDevices();

		bool chooseDevice(const int major, const int minor);
		bool chooseDevice(const cudaDeviceProp & deviceProp);

		CudaMemoryInfo getMemoryInfo() const;

	protected:
		int _deviceIndex;
	};
}

#endif
