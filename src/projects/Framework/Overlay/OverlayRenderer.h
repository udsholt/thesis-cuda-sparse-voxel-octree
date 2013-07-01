#ifndef _RESTLESS_OVERLAY_HELPER_H
#define _RESTLESS_OVERLAY_HELPER_H

#include <string>

#include <Mathlib/Mat4x4f.h>
#include <Mathlib/Vec4f.h>

#include "../Debug/DebugQuad.h"
#include "../Shader/ShaderProgram.h"

namespace restless
{
	class TextObject;
}

namespace restless
{
	class OverlayRenderer
	{
	public:
		OverlayRenderer();
		~OverlayRenderer();

		void initialize();
		void setSpacing(int spacing);
		void onViewportResize(const int width, const int height);
		void drawText(const TextObject & text);
		void drawQuad(const DebugQuad & quad, const Vec4f color);

	protected:
		ShaderProgram textShader;
		ShaderProgram primitveShader;
		Mat4x4f projection;

	};
}

#endif