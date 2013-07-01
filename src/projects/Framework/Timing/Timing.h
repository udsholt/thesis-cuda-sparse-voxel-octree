#ifndef _RESTLESS_TIMING_TIMING_H
#define _RESTLESS_TIMING_TIMING_H

#include "DeltaSampler.h"

namespace restless
{
	class Timing
	{
	public:
		Timing();
		~Timing();
		
		void initialize();
		void reset();
		void update();

		const float getTimeDelta() const;
		const float getTime() const;

		const float getTimeDeltaAvarage() const;
		const float getFPS() const;

		const unsigned int getFrame() const;

	protected:
		unsigned int frame;

		double timeLast;
		double timeCurrent;
		double timeDelta;
		
		DeltaSampler deltaSampler;
	};
}
#endif