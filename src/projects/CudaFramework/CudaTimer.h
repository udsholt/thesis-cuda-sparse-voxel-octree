#ifndef _RESTLESS_CUDA_TIMER_H
#define _RESTLESS_CUDA_TIMER_H

#include "CudaUtil.h"
#include "cuda_runtime.h"

namespace restless
{
	class CudaTimer
	{
	public:
	
		CudaTimer();

		~CudaTimer();

		void intialize();

		void start();

		void stop();

		float getElapsed();

	protected:

		cudaEvent_t eventStart;
		cudaEvent_t eventStop;

		float elapsedTime;

	};

}


#endif