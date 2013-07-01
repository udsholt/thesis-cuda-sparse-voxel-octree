#ifndef _RESTLESS_CUDA_TIMERCOLLECTION_H
#define _RESTLESS_CUDA_TIMERCOLLECTION_H

#include "CudaTimer.h"

#include <string>
#include <vector>

namespace restless
{

	struct CudaTimerBin
	{
		bool enabled;
		bool summed;
		CudaTimer timer;
		std::string name;
		float elapsed;
		float maxElasped;

		CudaTimerBin()
		{
			enabled = false;
			summed = false;
			elapsed = 0.0f;
			maxElasped = 0.0f;
			name = "unknown";
		}

	};

	class CudaTimerCollection
	{
	public:
		CudaTimerCollection(void);
		~CudaTimerCollection(void);

		void initialize(unsigned int size);
		void reset();

		void registerBin(unsigned int index, const char * name, const bool summed = false);

		unsigned int getBinCount() const;
		unsigned int getEnabledBinCount() const;

		CudaTimerBin & getBin(unsigned int index);

		void start(unsigned int index);
		void stop(unsigned int index);

		void setElapsed(unsigned int index, const float elapsed);

		bool enabled;

	protected:

		std::vector<CudaTimerBin> bins;

	};

}

#endif