#ifndef _RESTLESS_CAMERA_FRUSTUM_H
#define _RESTLESS_CAMERA_FRUSTUM_H

#include <Mathlib/Mat4x4f.h>

namespace restless
{

	class Frustum
	{
	public:
		Frustum();
		~Frustum();

		const float getLeft() const { return _left; };
		const float getRight() const { return _right; };
		const float getBottom() const { return _bottom; };
		const float getTop() const { return _top; };
		const float getNear() const { return _near; };
		const float getFar() const { return _far; };

		void setPerspective(const float fov, const float aspect, const float near, const float far);
		void setFrustrum(const float left, const float right, const float bottom, const float top, const float near, const float far);

		const Mat4x4f & getProjectionMatrix() const;
		const Mat4x4f & getInverseProjectionMatrix() const;

	protected:

		float _left;
		float _right;
		float _bottom;
		float _top;
		float _near;
		float _far;

		Mat4x4f _projectionMatrix;
		Mat4x4f _inverseProjectionMatrix;
	};

}

#endif