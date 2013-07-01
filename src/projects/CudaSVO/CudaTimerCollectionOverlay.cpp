#include "CudaTimerCollectionOverlay.h"

#include <Framework/Overlay/TextFont.h>
#include <Framework/Overlay/OverlayRenderer.h>
#include <CudaFramework/CudaTimerCollection.h>
#include "cuda/defines.h"

#include <sstream>
#include <string>

using namespace restless;

CudaTimerCollectionOverlay::CudaTimerCollectionOverlay(CudaTimerCollection & cudaTimerCollection) :
	timerCollection(cudaTimerCollection), 
	refreshInterval(0.2f),
	refreshTimer(0.2f),
	enabled(true)
{
}


CudaTimerCollectionOverlay::~CudaTimerCollectionOverlay(void)
{
}

void CudaTimerCollectionOverlay::initialize(TextFont & textFont)
{
	textObject.initialize(textFont);
	textObject.setPosition(Vec2f(textFont.getCharScreenWidthWithSpacing() / 2.0f, textFont.getCharScreenHeight() * 2.0f));
	textObject.setText("");

	unsigned int enabledBinCount = timerCollection.getEnabledBinCount();

	Vec2f topRight = textObject.getPosition();
	Vec2f bottomLeft = textObject.getPosition();
	bottomLeft[0] += textObject.getFont().getCharScreenWidthWithSpacing() * (2 + 35 + 1 + 9 + 9 + 1);
	bottomLeft[1] += textObject.getFont().getCharScreenHeight() * (enabledBinCount + 5);

	backdrop.initialize(topRight, bottomLeft, -1.0f);
}

void CudaTimerCollectionOverlay::update(const float deltaTime)
{
	refreshTimer -= deltaTime;

	if (refreshTimer > 0.0f) {
		return;
	}

	refreshTimer = refreshInterval;

	if (!timerCollection.enabled) {
		return;
	}


	std::ostringstream o;
	o.setf(std::ios::fixed);

	std::string rowSep = std::string(2 + 35 + 1 + 9 + 9, '=') + "\n";

	o << rowSep;
	o << std::setw(35) << std::left  << "CUDA Binname";
	o << " ";
	o << " ";
	o << std::setw(9) << std::right << "Time (ms)";
	o << " ";
	o << std::setw(9) << std::right << "Max (ms)";
	o << "\n";
	o << rowSep;

	unsigned int binCount = timerCollection.getBinCount();

	unsigned int row = 0;

	float all = timerCollection.getBin(BIN_ALL).elapsed;
	float sum = 0.0f;

	for (unsigned int b = 0; b < binCount; ++b) {

		CudaTimerBin & bin = timerCollection.getBin(b);
		
		if (!bin.enabled) {
			continue;
		}

		if (!bin.summed) {
			sum += bin.elapsed;
		}

		o << std::setw(35) << std::left << bin.name;
		o << (bin.summed ? "S" : "N");
		o << " ";
		o << std::setw(9) << std::right << std::setprecision(3) << bin.elapsed;
		o << " ";
		o << std::setw(9) << std::right << std::setprecision(3) << bin.maxElasped;
		o << "\n";
	}

	o << std::setw(35) << std::left << "OVERHEAD";
	o << "D";
	o << " ";
	o << std::setw(9) << std::right << std::setprecision(3) << (all - sum);
	o << " ";
	o << std::string(12, ' ');
	o << "\n";
	o << rowSep;

	textObject.setText(o.str().c_str());
	 
	
}

void CudaTimerCollectionOverlay::render(OverlayRenderer & renderer)
{
	if (enabled) {
		renderer.drawQuad(backdrop, Vec4f(0.7f, 0.3f, 0.3f, 0.7f));
		renderer.drawText(textObject);
	}
}

void CudaTimerCollectionOverlay::onViewportResize(const unsigned int width, const unsigned int height)
{

}