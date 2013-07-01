#include "Texture3D.h"

#include "../GL.h"
#include "../Util/Log.h"
#include "../Util/Helper.h"

namespace restless
{

	Texture3D::Texture3D() :
		_internalFormat(0)
	{
	}


	Texture3D::~Texture3D()
	{
	}

	void Texture3D::bind(const unsigned int channel) const
	{
		glActiveTexture(GL_TEXTURE0 + channel);
		glBindTexture(GL_TEXTURE_3D, _textureId);
	}

	void Texture3D::unbind() const
	{
		glBindTexture(GL_TEXTURE_3D, 0);
	}

	bool Texture3D::initialize(const int internalFormat , const unsigned int width, const unsigned int height, const unsigned depth)
	{
		_internalFormat = internalFormat;
		return resize(width, height, depth);
	}

	bool Texture3D::resize(const unsigned int width, const unsigned int height, const unsigned depth)
	{
		if (_internalFormat != GL_RGBA8 && _internalFormat != GL_RGBA32F) {
			L_ERROR << "invalid internal format, maybe the 3d texture is not initialized";
			return false;
		}

		if (_textureId) {
			glDeleteTextures(1, & _textureId);
		}

		_width = width;
		_height = height;
		_depth = depth;

		glGenTextures(1, & _textureId);
		glBindTexture(GL_TEXTURE_3D, _textureId);

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);

		glTexImage3D(GL_TEXTURE_3D, 0, _internalFormat, _width, _height, _depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glBindTexture(GL_TEXTURE_3D, 0);

		return true;
	}

}