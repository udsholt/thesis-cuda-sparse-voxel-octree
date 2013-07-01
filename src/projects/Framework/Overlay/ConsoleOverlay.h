#ifndef _RESTLESS_OVERLAY_CONSOLEOVERLAY_H
#define _RESTLESS_OVERLAY_CONSOLEOVERLAY_H

#include "TextObject.h"
#include "../Debug/DebugQuad.h"

#include "Overlay.h"

namespace restless
{
	class Console;
}

namespace restless
{
	class ConsoleOverlay : public Overlay
	{
	public:
		ConsoleOverlay(Console & console);
		virtual ~ConsoleOverlay();

		virtual void initialize(TextFont & font);
		virtual void update(const float deltaTime);
		virtual void render(OverlayRenderer & renderer);
		virtual OverlayType getOverlayType() { return Overlay::TYPE_OTHER; }

		virtual void onViewportResize(const unsigned int width, const unsigned int height);

	protected:

		Console & console;
		
		DebugQuad backdrop;

		TextObject log;
		TextObject line;
		TextObject cursor;

		unsigned int margin;
		unsigned int padding;
		unsigned int viewportWidth;
		unsigned int viewportHeight;

		unsigned int lineTicks;
		unsigned int cursorTicks;
		unsigned int logTicks;

		float logHeight;
		unsigned int logLines;

		float charWidth;
		float charHeight;

		bool viewportResized;

		float cycle;
	};
}

#endif 