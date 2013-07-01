#include "OverlayManager.h"

#include "../Core.h"
#include "../Texture/Texture2D.h"
#include "../Texture/TextureLoader.h"

#include "Overlay.h"
#include "ConsoleOverlay.h"
#include "StatsOverlay.h"

namespace restless
{
	OverlayManager::OverlayManager(Core & _core) :
		core(_core),
		consoleOverlay(_core.console()),
		statsOverlay(_core.timing()),
		statsEnabled(true)
	{
		
	}

	OverlayManager::~OverlayManager()
	{
	}

	void OverlayManager::addOverlay(Overlay & overlay)
	{
		overlay.initialize(textFont);
		overlay.onViewportResize(viewportWidth, viewportHeight);
		overlays.push_back(&overlay);
	}

	void OverlayManager::toggleStats()
	{
		statsEnabled = !statsEnabled;
	}

	void OverlayManager::initialize()
	{
		Texture2D fontTexture = TextureLoader::load2DTexture(core.files().path("fonts/font_pretty.png").c_str(), true);

		textFont.initialize(fontTexture, 16, 16);
		textFont.setSpacing(-6.0f);
		textFont.setScale(0.5f);

		renderer.initialize();

		addOverlay(consoleOverlay);
		addOverlay(statsOverlay);
	}

	void OverlayManager::update()
	{
		const float deltaTime = core.timing().getTimeDelta();

		for(unsigned int i = 0; i < overlays.size(); ++i) {
			if (overlays[i]->getOverlayType() != Overlay::TYPE_STATS || statsEnabled) {
				overlays[i]->update(deltaTime);
			}
		}
	}

	void OverlayManager::render()
	{
		for(unsigned int i = 0; i < overlays.size(); ++i) {
			if (overlays[i]->getOverlayType() != Overlay::TYPE_STATS || statsEnabled) {
				overlays[i]->render(renderer);
			}
		}
	}

	void OverlayManager::onViewportResize(const unsigned int width, const unsigned int height)
	{
		viewportWidth = width;
		viewportHeight = height;

		renderer.onViewportResize(viewportWidth, viewportHeight);

		for(unsigned int i = 0; i < overlays.size(); ++i) {
			overlays[i]->onViewportResize(viewportWidth, viewportHeight);
		}
	}
}