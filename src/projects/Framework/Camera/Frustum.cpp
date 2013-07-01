#include "Frustum.h"

#include <Mathlib/Mathlib.h>

namespace restless
{
	Frustum::Frustum() :
		_left(0),
		_right(0),
		_bottom(0),
		_top(0),
		_near(0),
		_far(0),
		_projectionMatrix(0),
		_inverseProjectionMatrix(0)
	{
	}


	Frustum::~Frustum()
	{
	}

	// http://www.opengl.org/archives/resources/faq/technical/transformations.htm
	void Frustum::setPerspective(const float fov, const float aspect, const float near, const float far)
	{
		const float top = tan(fov * DEG_TO_RAD * 0.5f) * near;
		const float bottom = -top;
		const float left = aspect * bottom;
		const float right = aspect * top;

		setFrustrum(left, right, bottom, top, near, far);
	}

	void Frustum::setFrustrum(const float left, const float right, const float bottom, const float top, const float near, const float far)
	{
		_left   = left;
		_right  = right;
		_bottom = bottom;
		_top    = top;
		_near   = near;
		_far    = far;

		_projectionMatrix = Mat4x4f::frustum(_left, _right, _bottom, _top, _near, _far);
		_inverseProjectionMatrix = _projectionMatrix.inverse();
	}

	const Mat4x4f & Frustum::getProjectionMatrix() const
	{
		return _projectionMatrix;
	}

	const Mat4x4f & Frustum::getInverseProjectionMatrix() const
	{
		return _inverseProjectionMatrix;
	}
}
