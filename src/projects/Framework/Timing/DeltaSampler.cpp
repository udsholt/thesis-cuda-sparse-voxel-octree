#include "DeltaSampler.h"

namespace restless
{
	DeltaSampler::DeltaSampler(unsigned int size) : 
		_size(size),
		_samples(nullptr),
		currentSample(0)
	{
		_samples = new float[size];
		for(unsigned int i = 0; i < size; i++) {
			_samples[i] = 0.0f;
		}
	}

	DeltaSampler::~DeltaSampler() 
	{
		delete[] _samples;
	}

	void DeltaSampler::addSample(const float & sample)
	{
		_samples[currentSample++] = sample;
		if (currentSample >= _size) {
			currentSample = 0;
		}
	}

	const float DeltaSampler::getAvarage() const
	{
		float avarage = 0.0f; 
		for(unsigned int i = 0; i < _size; i++) {
			avarage += _samples[i];
		}
		avarage = avarage / _size;
		return avarage;
	}


}