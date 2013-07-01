#ifndef __RESTLESS_MATH_MAT4X4F_H
#define __RESTLESS_MATH_MAT4X4F_H

#include "Mat4x4.h"
#include "Vec4f.h"

namespace restless
{
	class Mat4x4f: public restless::Mat4x4<float>
	{
	public:
		Mat4x4f();
		Mat4x4f(const float value);
		Mat4x4f(const Vec4f & v1, const Vec4f &  v2, const Vec4f &  v3, const Vec4f & v4);
		Mat4x4f(const Mat4x4<float> & matrix) :
			Mat4x4(matrix) {}
	};
}

#endif