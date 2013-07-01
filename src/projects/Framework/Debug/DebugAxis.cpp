#include "DebugAxis.h"

#include "../GL.h"
#include "../Shader/ShaderProgram.h"

#include <Mathlib/Vec3f.h>

namespace restless
{
	DebugAxis::DebugAxis() :
		_vertexBufferObject(0),
		_vertexArrayObject(0)
	{
	}


	DebugAxis::~DebugAxis()
	{
	}

	void DebugAxis::initialize(const float scale)
	{
		struct VertexData {
			Vec3f position;
			Vec3f color;
		};

		const VertexData vertices[] = {
			{ Vec3f(-0.01f,   0.0f,  0.0f ) * scale, Vec3f(1.0f, 0.0f, 0.0f) }, // X axis: red
			{ Vec3f(  1.0f,   0.0f,  0.0f ) * scale, Vec3f(1.0f, 0.0f, 0.0f) }, // ...
			{ Vec3f(  0.0f, -0.01f,  0.0f ) * scale, Vec3f(0.0f, 1.0f, 0.0f) }, // Y axis: green
			{ Vec3f(  0.0f,   1.0f,  0.0f ) * scale, Vec3f(0.0f, 1.0f, 0.0f) }, // ...
			{ Vec3f(  0.0f,   0.0f, -0.01f) * scale, Vec3f(0.0f, 0.0f, 1.0f) }, // Z axis: blue
			{ Vec3f(  0.0f,   0.0f,  1.0f ) * scale, Vec3f(0.0f, 0.0f, 1.0f) }  // ...
		};

		// Create the vertex buffer (VBO)
		glGenBuffers(1, & _vertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(VertexData), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Create the vertex array object (VAO)
		glGenVertexArrays(1, & _vertexArrayObject);
		glBindVertexArray(_vertexArrayObject);

		// Connect the VBO to the VAO and configure the generic vertex attributes
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glVertexAttribPointer(ShaderProgram::ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (const void *)0);
		glVertexAttribPointer(ShaderProgram::ATTRIB_COLOR, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (const void *)sizeof(Vec3f));
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_VERTEX);
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_COLOR);

		glBindVertexArray(0);
	}

	void DebugAxis::draw()
	{
		glLineWidth(2.0f);
		glBindVertexArray(_vertexArrayObject);
		glDrawArrays(GL_LINES, 0, 6);
		glBindVertexArray(0);
		glLineWidth(1.0f);
	}

}