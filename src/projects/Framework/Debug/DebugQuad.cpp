#include "DebugQuad.h"

#include "../GL.h"
#include "../Shader/ShaderProgram.h"

#include <Mathlib/Vec2f.h>
#include <Mathlib/Vec3f.h>

#include "../Util/Log.h"

namespace restless
{
	struct QuadVertexData
	{
		Vec3f position;
		Vec2f texCoord;
	};

	DebugQuad::DebugQuad() :
		_vertexBufferObject(0),
		_vertexArrayObject(0)
	{

	}


	DebugQuad::~DebugQuad()
	{

	}

	void DebugQuad::draw() const
	{
		glBindVertexArray(_vertexArrayObject);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		glBindVertexArray(0);
	}

	/**
	 *  Vertex coords:      Tex coords:
	 * (0,1)------(1,1)  (0,0)------(1,0)
	 *   |          |      |          |
	 *   |          |      |          |
	 * (0,0)------(1,0)  (0,1)------(1,1)
	 *
	 * Vertex.z is -1.0f
	 */
	void DebugQuad::initialize()
	{
		initialize(Vec2f(0.0f, 1.0f), Vec2f(1.0f, 0.0f), -1.0f);
	}

	void DebugQuad::initialize(const Vec2f topLeft, const Vec2f bottomRight, const float z)
	{
		if (_vertexArrayObject) {
			glDeleteVertexArrays(1, & _vertexArrayObject);
		}

		if (_vertexBufferObject) {
			glDeleteBuffers(1, & _vertexBufferObject);
		}

		const Vec3f v0 = Vec3f(topLeft[0],     topLeft[1],     z);
		const Vec3f v1 = Vec3f(topLeft[0],     bottomRight[1], z);
		const Vec3f v2 = Vec3f(bottomRight[0], topLeft[1],     z);
		const Vec3f v3 = Vec3f(bottomRight[0], bottomRight[1], z);

		const Vec2f topRight = Vec2f(bottomRight[0], topLeft[1]);
		const Vec2f bottomLeft = Vec2f(topLeft[0], bottomRight[1]);

		QuadVertexData vertices[] = {
			{v0, Vec2f(0.0f, 0.0f)}, // 0
			{v1, Vec2f(0.0f, 1.0f)}, // 1
			{v2, Vec2f(1.0f, 0.0f)}, // 2
			{v3, Vec2f(1.0f, 1.0f)}  // 3
		};

		// Create the vertex buffer (VBO)
		glGenBuffers(1, & _vertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(QuadVertexData), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Create the vertex array object (VAO)
		glGenVertexArrays(1, & _vertexArrayObject);
		glBindVertexArray(_vertexArrayObject);

		// Connect the VBO to the VAO and configure the generic vertex attributes
		// so that the shader program knows where to find vertex position and texcoord
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glVertexAttribPointer(ShaderProgram::ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertexData), (const void *)0);
		glVertexAttribPointer(ShaderProgram::ATTRIB_MULTITEX_COORD_0, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertexData), (const void *)sizeof(Vec3f));

		// Enable the attributes that where just configured
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_VERTEX);
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_MULTITEX_COORD_0);

		glBindVertexArray(0);
	}
}