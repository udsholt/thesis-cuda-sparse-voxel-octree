#ifndef _RESTLESS_OVERLAY_TEXTFONT_H
#define _RESTLESS_OVERLAY_TEXTFONT_H

#include "../Texture/Texture2D.h"

#include <Mathlib/Vec2f.h>

namespace restless 
{

	class TextFont
	{
	public:
		TextFont();
		~TextFont();

		void initialize(const Texture2D & fontTexture, const unsigned int rows, const unsigned int columns);
		void setSpacing(const float spacing);
		void setScale(const float scale);

		const Texture2D & getTexture() const;

		const float getSpacing() const;

		const Vec2f getTexcoord(const char character) const;
		const float getTexCharWidth() const;
		const float getTexCharHeight() const;

		const float getCharScreenWidth() const;
		const float getCharScreenWidthWithSpacing() const;
		const float getCharScreenHeight() const; 

	protected:

		Texture2D _texture;

		unsigned int _rows;
		unsigned int _columns;

		float _texCharWidth;
		float _texCharHeight;

		unsigned int _charWidth; 
		unsigned int _charHeight;

		float _spacing;
		float _scale;
	};

}

#endif