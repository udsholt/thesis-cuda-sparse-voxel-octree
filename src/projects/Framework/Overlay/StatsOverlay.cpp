#include "StatsOverlay.h"

#include "OverlayRenderer.h"
#include "TextFont.h"

#include "../Timing/Timing.h"

#include <sstream>
#include <string>


namespace restless
{
	StatsOverlay::StatsOverlay(Timing & _timing) :
		fpsText(),
		timing(_timing),
		refreshInterval(0.1f),
		refreshTimer(0.1f)
	{

	}


	StatsOverlay::~StatsOverlay()
	{

	}

	void StatsOverlay::initialize(TextFont & textFont)
	{
		fpsText.initialize(textFont);
		fpsText.setPosition(Vec2f(textFont.getCharScreenWidthWithSpacing(), textFont.getCharScreenHeight()) / 2.0f);
		fpsText.setText("0.0");
	}

	void StatsOverlay::update(const float deltaTime)
	{
		refreshTimer -= timing.getTimeDelta();
		if (refreshTimer < 0.0f) {
			std::ostringstream o;
			o.precision(2);
			o << std::fixed << timing.getFPS();
			fpsText.setText(o.str().c_str());
			refreshTimer = refreshInterval;
		}
	
	}

	void StatsOverlay::render(OverlayRenderer & renderer)
	{
		renderer.drawText(fpsText);
	}

	void StatsOverlay::onViewportResize(const unsigned int width, const unsigned int height)
	{

	}
}