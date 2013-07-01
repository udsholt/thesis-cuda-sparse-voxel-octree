#include "CudaTimerCollection.h"

using namespace std;

namespace restless
{

	CudaTimerCollection::CudaTimerCollection(void) :
		enabled(true)
	{
		bins.reserve(10);
	}


	CudaTimerCollection::~CudaTimerCollection(void)
	{
	}

	void CudaTimerCollection::initialize(unsigned int size)
	{
		bins.resize(size);
	}

	void CudaTimerCollection::reset()
	{
		unsigned int size = getBinCount();

		for (unsigned int index  = 0; index < size; ++index) {
			CudaTimerBin & bin = bins.at(index);
			bin.elapsed = 0.0f;
		}
	}

	unsigned int CudaTimerCollection::getBinCount() const
	{
		return bins.size();
	}

	unsigned int CudaTimerCollection::getEnabledBinCount() const
	{
		unsigned int size = getBinCount();
		unsigned int count = 0;

		for (unsigned int index  = 0; index < size; ++index) {
			if (bins.at(index).enabled) {
				++count;
			}
		}

		return count;
	}

	CudaTimerBin & CudaTimerCollection::getBin(unsigned int index)
	{
		return bins.at(index);
	}

	void CudaTimerCollection::registerBin(unsigned int index, const char * name, const bool summed)
	{
		CudaTimerBin & bin = bins.at(index);
		bin.name = string(name);
		bin.summed = summed;
		bin.elapsed = 0.0f;
		bin.maxElasped = 0.0f;
		bin.timer.intialize();
		bin.enabled = true;
	}

	void CudaTimerCollection::start(unsigned int index)
	{
		if (!enabled) {
			return;
		}

		CudaTimerBin & bin = bins.at(index);

		if (bin.enabled) {
			bin.timer.start();
		}
	}

	void CudaTimerCollection::stop(unsigned int index)
	{
		if (!enabled) {
			return;
		}

		CudaTimerBin & bin = bins.at(index);

		if (bin.enabled) {
			bin.timer.stop();
			bin.elapsed = bin.timer.getElapsed();
			bin.maxElasped = max(bin.maxElasped, bin.elapsed);
		}
	}

	void CudaTimerCollection::setElapsed(unsigned int index, const float elapsed)
	{
		CudaTimerBin & bin = bins.at(index);

		bin.elapsed = elapsed;
		bin.maxElasped = max(bin.maxElasped, bin.elapsed);
	}

}