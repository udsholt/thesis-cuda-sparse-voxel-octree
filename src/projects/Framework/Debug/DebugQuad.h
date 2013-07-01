#ifndef _RESTLESS_DEBUG_DEBUGQUAD_H
#define _RESTLESS_DEBUG_DEBUGQUAD_H

namespace restless
{
	class Vec2f;
}

namespace restless
{
	class DebugQuad
	{
	public:
		DebugQuad();
		virtual ~DebugQuad();

		void initialize();
		void initialize(const Vec2f topLeft, const Vec2f bottomRight, const float z);
		void draw() const;

	protected:

		unsigned int _vertexBufferObject; // Vertex buffer object
		unsigned int _vertexArrayObject;  // Vertex array object
	};
}

#endif

