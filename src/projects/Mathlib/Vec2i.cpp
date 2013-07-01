#include "Vec2i.h"

namespace restless 
{
	Vec2i::Vec2i() : VecN() {}
	Vec2i::Vec2i(const int value) : VecN(value) {}

	Vec2i::Vec2i(const int x, const int y) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
	}

	Vec2i::Vec2i(const VecN<2,int> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
	}
}