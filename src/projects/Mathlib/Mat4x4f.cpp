#include "Mat4x4f.h"
#include <iostream>

namespace restless
{

	Mat4x4f::Mat4x4f() :
		Mat4x4() {}

	Mat4x4f::Mat4x4f(const float value) :
		Mat4x4(value) {}

	Mat4x4f::Mat4x4f(const Vec4f & v1, const Vec4f &  v2, const Vec4f &  v3, const Vec4f & v4) :
		Mat4x4(v1, v2, v3, v4) {}
}