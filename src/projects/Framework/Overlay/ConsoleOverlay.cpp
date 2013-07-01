#include "ConsoleOverlay.h"

#include "OverlayRenderer.h"
#include "TextFont.h"

#include "../Console/Console.h"
#include "../Util/Log.h"

#include <vector>
#include <string>

using namespace std;

namespace restless
{

	ConsoleOverlay::ConsoleOverlay(Console & _console) :
		console(_console),
		line(),
		log(),
		cursor(),
		margin(50),
		padding(10),
		viewportWidth(640),
		viewportHeight(480),
		lineTicks(0),
		logTicks(0),
		cursorTicks(0),
		viewportResized(true),
		cycle(0.0),
		logHeight(0.0f),
		logLines(0),
		charWidth(0.0f),
		charHeight(0.0f),
		backdrop()
	{

	}


	ConsoleOverlay::~ConsoleOverlay()
	{

	}

	void ConsoleOverlay::initialize(TextFont & font)
	{
		charWidth = font.getCharScreenWidthWithSpacing();
		charHeight = font.getCharScreenHeight();

		line.initialize(font);
		line.setText("");
		line.setColor(Vec4f(1.0f, 0.6f, 0.6f, 1.0f));

		log.initialize(font);
		log.setText("");

		cursor.initialize(font);
		cursor.setText("_");
	
		backdrop.initialize();
	}

	void ConsoleOverlay::update(const float deltaTime)
	{
		cycle += deltaTime;

		cursor.setColor(Vec4f(1.0f, 1.0f, 1.0f, sinf(cycle * 10.0f)));

		// Calculate the new height of the log, and the number of lines to display
		if (viewportResized) {
			logHeight = viewportHeight - (margin * 2) - (padding * 2);
			logLines =  clampi((logHeight / charHeight) - 1, 0, 100);
		}

		// Move the backdrop, this could have been done with a transformation instead
		if (viewportResized) {
			backdrop.initialize(Vec2f(margin, margin), Vec2f(viewportWidth - margin, viewportHeight - margin), -1.0f);
		}

		// Update the log text
		if (viewportResized || logTicks != console.getLogTicks()) {
			logTicks = console.getLogTicks();
			string logText = console.getLog(logLines);
			log.setText(logText.c_str());
			log.setPosition(Vec2f(margin + padding, margin + padding));
		}

		// Update the line text
		if (viewportResized || lineTicks != console.getLineTicks()) {
			lineTicks = console.getLineTicks();

			string lineText = console.getLine();
			line.setText(lineText.c_str());
			line.setPosition(Vec2f(margin + padding, margin + padding + logHeight - charHeight));
		}

		// Update the cursor
		if (viewportResized || cursorTicks != console.getCursorTicks()) {
			cursorTicks = console.getCursorTicks();
			cursor.setPosition(Vec2f(margin + padding + (console.getCursor() * charWidth), margin + padding + logHeight - charHeight));
		}

		viewportResized = false;
	}

	void ConsoleOverlay::render(OverlayRenderer & renderer)
	{
		if (console.isEnabled()) {
			renderer.drawQuad(backdrop, Vec4f(0.7f, 0.7f, 0.7, 0.2f));
			renderer.drawText(line);
			renderer.drawText(cursor);
			renderer.drawText(log);
		}
	}

	void ConsoleOverlay::onViewportResize(const unsigned int width, const unsigned int height)
	{
		viewportWidth = width;
		viewportHeight = height;
		viewportResized = true;
	}

}