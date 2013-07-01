#ifndef _RESTLESS_TIMING_DELTASAMPLER_H
#define _RESTLESS_TIMING_DELTASAMPLER_H

namespace restless
{
	class DeltaSampler
	{
	protected:
		float * _samples;
		const unsigned int _size;
		unsigned int currentSample;

	public:
		DeltaSampler(unsigned int size);
		~DeltaSampler();

		void addSample(const float & sample);
		const float getAvarage() const;
	};
}


#endif