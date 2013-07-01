#include "Vec4f.h"

namespace restless 
{
	Vec4f::Vec4f() : VecN() {}

	Vec4f::Vec4f(const float value) : VecN(value) {}

	Vec4f::Vec4f(const float x, const float y, const float z, const float w) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
		(*this)[3] = w;
	}

	Vec4f::Vec4f(const VecN<4,float> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
		(*this)[2] = vector[2];
		(*this)[3] = vector[3];
	}
}
