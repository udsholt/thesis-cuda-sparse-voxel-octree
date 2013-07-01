#include "Vec2f.h"

namespace restless 
{
	Vec2f::Vec2f() : VecN() {}
	Vec2f::Vec2f(const float value) : VecN(value) {}

	Vec2f::Vec2f(const float x, const float y) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
	}

	Vec2f::Vec2f(const VecN<2,float> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
	}
}