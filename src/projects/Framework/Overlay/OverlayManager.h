#ifndef _RESTLESS_OVERLAY_OVERLAYMANAGER_H
#define _RESTLESS_OVERLAY_OVERLAYMANAGER_H

#include "OverlayRenderer.h"
#include "ConsoleOverlay.h"
#include "StatsOverlay.h"
#include "TextFont.h"

#include <vector>

namespace restless
{
	class Core;
	class Overlay;
}

namespace restless
{

	class OverlayManager
	{
	public:
		OverlayManager(Core & _core);
		~OverlayManager();

		void initialize();
		void update();
		void render();

		void toggleStats();

		void addOverlay(Overlay & overlay);

		void onViewportResize(const unsigned int width, const unsigned int height);
		
	protected:

		Core & core;

		TextFont textFont;

		OverlayRenderer renderer;
		ConsoleOverlay consoleOverlay;
		StatsOverlay statsOverlay;

		std::vector<Overlay*> overlays;

		unsigned int viewportWidth;
		unsigned int viewportHeight;

		bool statsEnabled;

	};

}

#endif 