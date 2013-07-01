#ifndef _RESTLESS_TEXTURE_TEXTURE3D_H
#define _RESTLESS_TEXTURE_TEXTURE3D_H

#include "TextureInterface.h"

namespace restless
{
	class Texture3D : public TextureInterface
	{
	public:
		Texture3D();
		virtual ~Texture3D();

		const int getInternalFormat() const { return _internalFormat; };

		const unsigned int getTextureId() const { return _textureId; };
		const unsigned int getWidth() const { return _width; };
		const unsigned int getHeight() const { return _height; };
		const unsigned int getDepth() const { return _depth; };

		virtual void bind(const unsigned int channel = 0) const;
		virtual void unbind() const;

		bool initialize(const int interalFormat, const unsigned int width, const unsigned int height, const unsigned depth);
		bool resize(const unsigned int width, const unsigned int height, const unsigned depth);

	protected:

		int _internalFormat;

		unsigned int _textureId;
		unsigned int _width;
		unsigned int _height;
		unsigned int _depth;
	};
}

#endif