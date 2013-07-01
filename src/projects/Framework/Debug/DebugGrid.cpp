#include "DebugGrid.h"

#include "../GL.h"
#include "../Shader/ShaderProgram.h"

#include <Mathlib/Vec3f.h>

namespace restless
{
	DebugGrid::DebugGrid() :
		_vertexCount(0),
		_vertexBufferObject(0),
		_vertexArrayObject(0)
	{
	}


	DebugGrid::~DebugGrid()
	{
	}

	void DebugGrid::initialize(const int gridDimension, const float cellSize)
	{
		_vertexCount = gridDimension * 4;

		const float halfLength = ((float)((gridDimension - 1) * cellSize) / 2.0f);
		unsigned int offset = 0;

		Vec3f * vertices = new Vec3f[_vertexCount];

		for (int x = 0; x < gridDimension; ++x) {
			const float xPos = (x * cellSize) - halfLength;
			vertices[offset++] = Vec3f(xPos, 0.0f, - halfLength);
			vertices[offset++] = Vec3f(xPos, 0.0f,   halfLength);
		}

		for (int z = 0; z < gridDimension; ++z) {
			const float zPos = (z * cellSize) - halfLength;
			vertices[offset++] = Vec3f(- halfLength, 0.0f, zPos);
			vertices[offset++] = Vec3f(  halfLength, 0.0f, zPos);
		}

		// Create the vertex buffer (VBO)
		glGenBuffers(1, & _vertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, _vertexCount * sizeof(Vec3f), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Create the vertex array object (VAO)
		glGenVertexArrays(1, & _vertexArrayObject);
		glBindVertexArray(_vertexArrayObject);

		// Connect the VBO to the VAO and configure the generic vertex attributes
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glVertexAttribPointer(ShaderProgram::ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3f), (const void *)0);
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_VERTEX);

		glBindVertexArray(0);

		delete[] vertices;
	}

	void DebugGrid::draw()
	{
		glBindVertexArray(_vertexArrayObject);
		glDrawArrays(GL_LINES, 0, _vertexCount);
		glBindVertexArray(0);
	}

}