#include "TextFont.h"

#include "../GL.h"
#include "../Util/Log.h"

namespace restless
{

	TextFont::TextFont() :
		_spacing(0.0f),
		_scale(1.0f)
	{
	}


	TextFont::~TextFont()
	{
	}

	void TextFont::setSpacing(const float spacing)
	{
		_spacing = spacing;
	}

	void TextFont::setScale(const float scale)
	{
		_scale = scale;
	}

	const Vec2f TextFont::getTexcoord(const char character) const
	{
		// Find the row, col for the character
		unsigned int uchar = character;
		unsigned int row = uchar / _columns;
		unsigned int col = uchar - (row * _columns);

		float texOffsetX = _texCharWidth * col;
		float texOffsetY = _texCharHeight * row;

		// Find offset in texture
		return Vec2f(texOffsetX, texOffsetY);
	}

	const Texture2D & TextFont::getTexture() const
	{
		return _texture;
	}

	const float TextFont::getSpacing() const
	{
		return _spacing;
	}

	const float TextFont::getTexCharWidth() const
	{
		return _texCharWidth;
	}

	const float TextFont::getTexCharHeight() const
	{
		return _texCharHeight;
	}

	const float TextFont::getCharScreenWidth() const
	{
		return (float) _charWidth * _scale;
	}

	const float TextFont::getCharScreenWidthWithSpacing() const
	{
		return getCharScreenWidth() + _spacing;
	}

	const float TextFont::getCharScreenHeight() const
	{
		return (float) _charHeight * _scale;
	}

	void TextFont::initialize(const Texture2D & fontTexture, const unsigned int rows, const unsigned int columns)
	{
		_texture = fontTexture;

		// Setup some font settings
		_rows    = rows;
		_columns = columns;

		// Width and height of a char in the texture
		_texCharWidth  = 1.0f / _columns;
		_texCharHeight = 1.0f / _rows;

		// Width and height of a char on the screen
		_charWidth = (unsigned int)((_texture.getWidth() / _columns));
		_charHeight = (unsigned int)((_texture.getHeight() / _rows));

		
		_texture.bind(0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		_texture.unbind();
	}

}