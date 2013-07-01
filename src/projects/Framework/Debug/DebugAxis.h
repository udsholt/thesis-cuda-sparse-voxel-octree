#ifndef _RESTLESS_DEBUG_DEBUGAXIS_H
#define _RESTLESS_DEBUG_DEBUGAXIS_H

#include <Mathlib/Vec3f.h>

namespace restless
{

	class DebugAxis
	{
	public:
		DebugAxis();
		~DebugAxis();

		void initialize(const float scale = 1.0f);
		void draw();

	protected:

		unsigned int _vertexBufferObject; // Vertex buffer object
		unsigned int _vertexArrayObject;  // Vertex array object
	};

}

#endif