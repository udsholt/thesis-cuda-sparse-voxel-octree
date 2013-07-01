#include "DebugCube.h"

#include <Mathlib/Vec3f.h>

#include "../Shader/ShaderProgram.h"
#include "../GL.h"

namespace restless
{

	DebugCube::DebugCube() : 
		_boundingboxMin(0),
		_boundingboxMax(0)
	{
	}


	DebugCube::~DebugCube()
	{
	}

	void DebugCube::drawSolid()
	{
		glBindVertexArray(_vertexArrayObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _solidIndexBufferObject);
		glDrawElements(GL_TRIANGLE_STRIP, 14, GL_UNSIGNED_SHORT, 0);
		glBindVertexArray(0);
	}

	void DebugCube::drawWire()
	{
		glBindVertexArray(_vertexArrayObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _lineIndexBufferObject);
		glDrawElements(GL_LINES, 24, GL_UNSIGNED_SHORT, 0);
		glBindVertexArray(0);
	}

	void DebugCube::initialize(float scale)
	{
		initialize(Vec3f(-1.0, -1.0, -1.0) * scale, Vec3f(1.0,  1.0,  1.0) * scale);
	}

	void DebugCube::initialize(const Vec3f bboxMin, const Vec3f bboxMax)
	{
		Vec3f vertices[] = {
			Vec3f(bboxMin[0], bboxMin[1], bboxMax[2]), // 0
			Vec3f(bboxMax[0], bboxMin[1], bboxMax[2]), // 1
			Vec3f(bboxMin[0], bboxMax[1], bboxMax[2]), // 2
			Vec3f(bboxMax[0], bboxMax[1], bboxMax[2]), // 3 (max)
			Vec3f(bboxMin[0], bboxMin[1], bboxMin[2]), // 4 (min)
			Vec3f(bboxMax[0], bboxMin[1], bboxMin[2]), // 5 
			Vec3f(bboxMin[0], bboxMax[1], bboxMin[2]), // 6
			Vec3f(bboxMax[0], bboxMax[1], bboxMin[2])  // 7
		};

		_boundingboxMin = vertices[4];
		_boundingboxMax = vertices[3];

		GLushort solidIndices[] = {
			0, 1, 2, 3, 7, 1, 5, 4, 7, 6, 2, 4, 0, 1
		};

		GLushort lineIndices[] = {
			0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 4, 5, 4, 6, 7, 3, 7, 5, 7, 6
		};

		// Setup index buffer
		glGenBuffers(1, & _solidIndexBufferObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _solidIndexBufferObject);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 14 * sizeof(GLushort), solidIndices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// Setup index buffer
		glGenBuffers(1, & _lineIndexBufferObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _lineIndexBufferObject);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24 * sizeof(GLushort), lineIndices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// Setup vertex buffer
		glGenBuffers(1, & _vertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(Vec3f), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Setup vertex array buffer
		glGenVertexArrays(1, & _vertexArrayObject);
		glBindVertexArray(_vertexArrayObject);

		// Describe the data to the vertex array buffer
		//    vertex [ATTRIB_VERTEX] is 3 [3] floats [GL_FLOAT]
		//    its is not normalized [GL_FALSE]
		//    the offset between the vertices is [sizeof(Vec3f)]
		//    the first element starts at 0 [(const void *) 0] in the buffer
		glBindBuffer(GL_ARRAY_BUFFER, _vertexBufferObject);
		glVertexAttribPointer(ShaderProgram::ATTRIB_VERTEX, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3f), (const void *) 0);
		glEnableVertexAttribArray(ShaderProgram::ATTRIB_VERTEX);

		// Describe indices to the vertex array buffer
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBufferObject);

		// And we are done
		glBindVertexArray(0);
	}

}