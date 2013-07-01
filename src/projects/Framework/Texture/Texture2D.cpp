#include "Texture2D.h"

#include "../GL.h"

#include "../Util/Log.h"

namespace restless {

	Texture2D::Texture2D() :
		_textureId(0),
		_width(0),
		_height(0)
	{

	}

	Texture2D::Texture2D(const unsigned int textureId, const unsigned int width, const unsigned height) :
		_textureId(textureId),
		_width(width),
		_height(height)
	{

	}

	Texture2D::Texture2D(const Texture2D & other) :
		_textureId(other.getTextureId()),
		_width(other.getWidth()),
		_height(other.getHeight())
	{
		
	}


	Texture2D::~Texture2D()
	{
		// TODO: might have to add some cleanup here
		//       but since multiple texture can map
		//       to the same id, it cannot just be 
		//       deleted
	}

	void Texture2D::bind(const unsigned int channel) const
	{
		glActiveTexture(GL_TEXTURE0 + channel);
		glBindTexture(GL_TEXTURE_2D, _textureId);
	}

	void Texture2D::unbind() const
	{	
		// this is apprently not kosher in opengl >= 3.3 
		glBindTexture(GL_TEXTURE_2D, 0);
	}

}