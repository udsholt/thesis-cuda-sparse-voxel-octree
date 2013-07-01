#ifndef _INC_CUDA_TIMERCOLLECTIONOVERLAY_H
#define _INC_CUDA_TIMERCOLLECTIONOVERLAY_H

#include <Framework/Overlay/Overlay.h>
#include <Framework/Overlay/TextObject.h>
#include <Framework/Debug/DebugQuad.h>

namespace restless
{
	class CudaTimerCollection;
	class TextFont;
	class OverlayRenderer;
}

class CudaTimerCollectionOverlay : public restless::Overlay
{
public:
	CudaTimerCollectionOverlay(restless::CudaTimerCollection & cudaTimerCollection);
	~CudaTimerCollectionOverlay();

	virtual void initialize(restless::TextFont & textFont);
	virtual void update(const float deltaTime);
	virtual void render(restless::OverlayRenderer & renderer);
	virtual void onViewportResize(const unsigned int width, const unsigned int height);
	virtual OverlayType getOverlayType() { return Overlay::TYPE_STATS; }

	bool enabled;

protected:

	restless::CudaTimerCollection & timerCollection;

	restless::TextObject textObject;
	restless::DebugQuad backdrop;

	float refreshTimer;
	const float refreshInterval;

};

#endif