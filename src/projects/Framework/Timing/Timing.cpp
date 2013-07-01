#include "Timing.h"

#include "Time.h"

namespace restless
{
	Timing::Timing() :
		timeLast(0),
		timeDelta(0),
		timeCurrent(0.0f),
		frame(0),
		deltaSampler(10)
	{
	}

	Timing::~Timing()
	{
	}

	void Timing::initialize()
	{
		reset();
	}

	void Timing::reset()
	{
		timeDelta = 0.0f;
		timeCurrent = 0.0f;
		frame = 0;
	}

	void Timing::update()
	{
		if (timeLast == 0) {
			timeLast = Time::getTime();
			return;
		}

		double currentTime = Time::getTime();

		frame++;

		timeDelta = (currentTime - timeLast);
		timeCurrent += timeDelta;

		if (timeDelta > 0) {
			deltaSampler.addSample((float) timeDelta);
		}

		timeLast = currentTime;
	}

	const float Timing::getTimeDelta() const
	{
		return (float) timeDelta;
	}

	const float Timing::getTime() const
	{
		return (float) timeCurrent;
	}

	const float Timing::getTimeDeltaAvarage() const
	{
		return deltaSampler.getAvarage();
	}

	const float Timing::getFPS() const
	{
		return 1.0f / deltaSampler.getAvarage();
	}

	const unsigned int Timing::getFrame() const
	{
		return frame;
	}
}