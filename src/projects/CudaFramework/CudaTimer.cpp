#include "CudaTimer.h"

namespace restless
{

	CudaTimer::CudaTimer() 
	{
		elapsedTime = 0.0f;
		eventStart = nullptr;
		eventStop = nullptr;
	}

	CudaTimer::~CudaTimer()
	{
		if (eventStart != nullptr) {
			cudaEventDestroy(eventStart);
			eventStart = nullptr;
		}

		if (eventStop != nullptr) {
			cudaEventDestroy(eventStop);
			eventStop = nullptr;
		}
	}

	void CudaTimer::intialize()
	{
		cudasafe( cudaEventCreate(&eventStart), "cudaEventCreate(&eventStart)" );
		cudasafe( cudaEventCreate(&eventStop), "cudaEventCreate(&eventStop)" );
	}

	void CudaTimer::start()
	{
		cudasafe( cudaEventRecord(eventStart, 0), "cudaEventRecord(eventStart, 0)" );
	}

	void CudaTimer::stop()
	{
		cudasafe( cudaEventRecord(eventStop, 0), "cudaEventRecord(eventStop, 0)" );
		cudasafe( cudaEventSynchronize(eventStop), "cudaEventSynchronize(eventStop)" );
		cudasafe( cudaEventElapsedTime(&elapsedTime, eventStart, eventStop), "cudaEventElapsedTime(&elapsedTime, eventStart, eventStop)");
	}

	float CudaTimer::getElapsed()
	{
		return elapsedTime;
	}

}