#include "TextObject.h"

#include "../GL.h"
#include "../Util/Log.h"
#include "../Shader/ShaderProgram.h"
#include "TextFont.h"


#include <Mathlib/Vec3f.h>
#include <Mathlib/Vec2f.h>

#include <string>
#include <algorithm>

using namespace std;

namespace restless
{

	struct VertexData
	{
		Vec3f position;
		Vec2f texCoord;
	};


	TextObject::TextObject() :
		_vertexCount(0),
		_font(nullptr),
		_position(0,0),
		_color(Vec4f(1.0f, 1.0f, 1.0f, 1.0f))
	{
	}


	TextObject::~TextObject()
	{

	}

	const Vec2f TextObject::getPosition() const
	{
		return _position;
	}

	void TextObject::setPosition(const Vec2f & position)
	{
		_position = position;
	}

	const Vec4f TextObject::getColor() const
	{
		return _color;
	}

	void TextObject::setColor(const Vec4f color)
	{
		_color = color;
	}

	void TextObject::draw() const
	{
		//L_DEBUG << "draw text " << _vertexCount;
		if (_vertexCount == 0) {
			return;
		}
		
		glBindVertexArray(_vertexArrayObject);
		glDrawArrays(GL_TRIANGLES, 0, _vertexCount);
		glBindVertexArray(0);
	}

	void TextObject::initialize(TextFont & font)
	{
		// Store a pointer to the font
		_font = & font;

		// Create the vertex buffer (VBO)
		glGenBuffers(1, & _vertexBufferObject);
		glGenVertexArrays(1, & _vertexArrayObject);

		// Configure VAO, VBO
		glBindVertexArray(_vertexArrayObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(ShaderProgram::ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (const void *)0);
		glVertexAttribPointer(ShaderProgram::ATTRIB_MULTITEX_COORD_0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (const void *)sizeof(Vec3f));
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_VERTEX);
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_MULTITEX_COORD_0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	const TextFont & TextObject::getFont() const
	{
		assert(_font != nullptr);
		return *_font;
	}

	void TextObject::setText(const char * text)
	{
		if (_font == nullptr) {
			return;
		}

		float charScreenWidth  = _font->getCharScreenWidth();
		float charScreenHeight = _font->getCharScreenHeight();
		float texCharWidth     = _font->getTexCharWidth();
		float texCharHeight    = _font->getTexCharHeight();
		float spacing          = _font->getSpacing();

		string _text = string(text);

		int newlineCount = std::count(_text.begin(), _text.end(), '\n');
		int vertexCount = (_text.length() - newlineCount) * 6;

		VertexData * vertices = new VertexData[vertexCount];

		float xPos = 0.0f;
		float yPos = 0.0f;

		unsigned int v = 0;

		// Build vertices
		for (unsigned int i = 0; i < _text.length(); i++) {

			// Jump to a newline and ignore the character
			if (_text[i] == '\n') {
				xPos  = 0.0f;
				yPos += charScreenHeight;
				continue;
			}

			// Get the texcoord from the font
			Vec2f texcoord = _font->getTexcoord(_text[i]);

			// First triangle positions
			vertices[v+0].position = Vec3f(xPos,                   yPos,                    -1);
			vertices[v+1].position = Vec3f(xPos + charScreenWidth, yPos + charScreenHeight, -1);
			vertices[v+2].position = Vec3f(xPos + charScreenWidth, yPos,                    -1);
			
			// Second triangle positions
			vertices[v+3].position = Vec3f(xPos,                   yPos,                    -1);
			vertices[v+4].position = Vec3f(xPos,                   yPos + charScreenHeight, -1);
			vertices[v+5].position = Vec3f(xPos + charScreenWidth, yPos + charScreenHeight, -1);

			// First triangle texcoords
			vertices[v+0].texCoord = Vec2f(texcoord[0],                texcoord[1]);
			vertices[v+1].texCoord = Vec2f(texcoord[0] + texCharWidth, texcoord[1] + texCharHeight);
			vertices[v+2].texCoord = Vec2f(texcoord[0] + texCharWidth, texcoord[1]);

			// Second triangle texcoords
			vertices[v+3].texCoord = Vec2f(texcoord[0],                texcoord[1]);
			vertices[v+4].texCoord = Vec2f(texcoord[0],                texcoord[1] + texCharHeight);
			vertices[v+5].texCoord = Vec2f(texcoord[0] + texCharWidth, texcoord[1] + texCharHeight);

			// Move one char + spacing in x
			xPos += charScreenWidth + spacing;

			// Create the next 6 vertices
			v = v + 6;
		}

		// Put the vertex data in the vbo
		glBindVertexArray(_vertexArrayObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(VertexData), vertices, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		// Cleanup
		delete [] vertices;

		_vertexCount = vertexCount;
	}
}