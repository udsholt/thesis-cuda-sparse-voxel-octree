#ifndef _RESTLESS_TEXTURE_TEXTURE2D_H
#define _RESTLESS_TEXTURE_TEXTURE2D_H

#include "TextureInterface.h"

namespace restless
{
	class Texture2D : public TextureInterface
	{
	public:
		Texture2D();
		Texture2D(const unsigned int textureId, const unsigned int width, const unsigned height);
		Texture2D(const Texture2D & other);
		virtual ~Texture2D();

		virtual void bind(const unsigned int channel = 0) const;
		virtual void unbind() const;

		const unsigned int getTextureId() const { return _textureId; };
		const unsigned int getWidth() const { return _width; };
		const unsigned int getHeight() const { return _height; };

	protected:
		unsigned int _textureId;
		unsigned int _width;
		unsigned int _height;
	};

}
#endif