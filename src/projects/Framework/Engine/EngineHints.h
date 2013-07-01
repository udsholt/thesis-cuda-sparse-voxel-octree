#ifndef _RESTLESS_ENGINE_ENGINEHINTS_H
#define _RESTLESS_ENGINE_ENGINEHINTS_H

namespace restless
{
	struct EngineHints
	{
		enum OpenglProfile {
			OPENGL_PROFILE_CORE,
			OPENGL_PROFILE_COMPAT
		};

		EngineHints() :
			windowAllowResize(true),
			windowWidth(800),
			windowHeight(600),
			openglFFSAMultisamples(4),
			openglProfile(EngineHints::OPENGL_PROFILE_CORE),
			openglVersionMajor(4),
			openglVersionMinor(3),
			openglRedBits(8),
			openglGreenBits(8),
			openglBlueBits(8),
			openglAlphaBits(8),
			openglStencilBits(8)
		{

		}

		unsigned int windowWidth;
		unsigned int windowHeight;

		bool windowAllowResize;

		OpenglProfile openglProfile;

		unsigned int openglVersionMajor;
		unsigned int openglVersionMinor;

		unsigned int openglFFSAMultisamples;

		unsigned int openglRedBits;
		unsigned int openglGreenBits;
		unsigned int openglBlueBits;
		unsigned int openglAlphaBits;
		unsigned int openglDepthBits;
		unsigned int openglStencilBits;

	};
}

#endif