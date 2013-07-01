#ifndef _RESTLESS_DEBUG_DEBUGCUBE_H
#define _RESTLESS_DEBUG_DEBUGCUBE_H

#include <Mathlib/Vec3f.h>

namespace restless
{

	class DebugCube
	{
	public:
		DebugCube();
		~DebugCube();

		const Vec3f & getBoundingBoxMin() { return _boundingboxMin; }
		const Vec3f & getBoundingBoxMax() { return _boundingboxMax; }

		void initialize(const float scale = 1.0f);
		void initialize(const Vec3f bboxMin, const Vec3f bboxMax);
		void drawSolid();
		void drawWire();

	protected:

		Vec3f _boundingboxMin;
		Vec3f _boundingboxMax;

		unsigned int _solidIndexBufferObject; // Index buffer object
		unsigned int _lineIndexBufferObject;  // Index buffer object
		unsigned int _vertexBufferObject;     // Vertex buffer object
		unsigned int _vertexArrayObject;      // Vertex array object
	};

}

#endif