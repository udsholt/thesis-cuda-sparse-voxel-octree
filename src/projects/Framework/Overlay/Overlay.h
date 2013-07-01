#ifndef _RESTLESS_OVERLAY_OVERLAY_H
#define _RESTLESS_OVERLAY_OVERLAY_H

namespace restless
{
	class TextFont;
	class OverlayRenderer;
}

namespace restless
{
	class Overlay
	{
	public:

		enum OverlayType
		{
			TYPE_OTHER,
			TYPE_STATS
		};

	public:
		Overlay();
		virtual ~Overlay();

		virtual void initialize(TextFont & textFont) = 0;
		virtual void update(const float deltaTime) = 0;
		virtual void render(OverlayRenderer & renderer) = 0;
		virtual void onViewportResize(const unsigned int width, const unsigned int height) = 0;
		virtual OverlayType getOverlayType() = 0;
	};

}

#endif