#ifndef _RESTLESS_OVERLAY_STATSOVERLAY_H
#define _RESTLESS_OVERLAY_STATSOVERLAY_H

#include "TextObject.h"
#include "Overlay.h"

namespace restless
{
	class Timing;
}

namespace restless
{
	class StatsOverlay : public Overlay
	{
	public:
		StatsOverlay(Timing & _timing);
		virtual ~StatsOverlay();

		virtual void initialize(TextFont & font);
		virtual void update(const float deltaTime);
		virtual void render(OverlayRenderer & renderer);
		virtual void onViewportResize(const unsigned int width, const unsigned int height);
		virtual OverlayType getOverlayType() { return Overlay::TYPE_STATS; }

	protected:
		TextObject fpsText;
		Timing & timing;

		float refreshTimer;
		const float refreshInterval;
	};
}

#endif