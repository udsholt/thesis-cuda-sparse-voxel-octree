#ifndef _RESTLESS_TEXTURE_TEXTETURELOADER_H
#define _RESTLESS_TEXTURE_TEXTETURELOADER_H

#include "Texture2D.h"

namespace restless
{
	class TextureLoader
	{
	public:
		static Texture2D load2DTexture(const char * filename, const bool flipVertical = false);

	private:
		TextureLoader();
		TextureLoader(const TextureLoader & other);
		~TextureLoader();
	};
}

#endif