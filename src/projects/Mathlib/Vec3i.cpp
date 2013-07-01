#include "Vec3i.h"

namespace restless 
{
	Vec3i::Vec3i() : VecN() {}

	Vec3i::Vec3i(const int value) : VecN(value) {}

	Vec3i::Vec3i(const int x, const int y, const int z) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
	}

	Vec3i::Vec3i(const VecN<3,int> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
		(*this)[2] = vector[2];
	}
}
