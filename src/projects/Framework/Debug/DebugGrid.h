#ifndef _RESTLESS_DEBUG_DEBUGGRID_H
#define _RESTLESS_DEBUG_DEBUGGRID_H

namespace restless
{
	class DebugGrid
	{
	public:
		DebugGrid();
		~DebugGrid();

		void initialize(const int dimension, const float resolution);
		void draw();

	protected:

		int _vertexCount;

		unsigned int _vertexBufferObject; // Vertex buffer object
		unsigned int _vertexArrayObject;  // Vertex array object
	};
}

#endif