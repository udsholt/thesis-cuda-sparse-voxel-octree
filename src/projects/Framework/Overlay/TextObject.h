#ifndef _RESTLESS_OVERLAY_TEXTOBJECT_H
#define _RESTLESS_OVERLAY_TEXTOBJECT_H

#include "../Shader/ShaderProgram.h"

#include <Mathlib/Vec2f.h>
#include <Mathlib/Vec4f.h>

namespace restless
{
	class TextFont;
}

namespace restless
{
	class TextObject
	{
	public:
		TextObject();
		~TextObject();

		const TextFont & getFont() const;

		void initialize(TextFont & font);
		void setText(const char * text);
		void draw() const;

		const Vec2f getPosition() const;
		void setPosition(const Vec2f & position);

		const Vec4f getColor() const;
		void setColor(const Vec4f color);

	protected:

		Vec2f _position;
		Vec4f _color;

		unsigned int _vertexBufferObject;
		unsigned int _vertexArrayObject;
		unsigned int _vertexCount;

		TextFont * _font;
	};
}

#endif