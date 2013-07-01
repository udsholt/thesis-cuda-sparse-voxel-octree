#include "Vec4i.h"

namespace restless 
{
	Vec4i::Vec4i() : VecN() {}

	Vec4i::Vec4i(const int value) : VecN(value) {}

	Vec4i::Vec4i(const int x, const int y, const int z, const int w) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
		(*this)[3] = w;
	}

	Vec4i::Vec4i(const VecN<4,int> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
		(*this)[2] = vector[2];
		(*this)[3] = vector[3];
	}
}
