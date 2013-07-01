#include "Vec3f.h"

namespace restless 
{
	Vec3f::Vec3f() : VecN() {}

	Vec3f::Vec3f(const float value) : VecN(value) {}

	Vec3f::Vec3f(const float x, const float y, const float z) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
	}

	Vec3f::Vec3f(const VecN<3,float> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
		(*this)[2] = vector[2];
	}

	Vec3f::Vec3f(const VecN<3,int> & vector)
	{
		(*this)[0] = (float) vector[0];
		(*this)[1] = (float) vector[1];
		(*this)[2] = (float) vector[2];
	}

	Vec3f Vec3f::cross(const VecN<3,float> & vector) const
	{
		Vec3f result(
			(*this)[1] * vector[2] - (*this)[2] * vector[1],
			(*this)[2] * vector[0] - (*this)[0] * vector[2],
			(*this)[0] * vector[1] - (*this)[1] * vector[0]
		);

		return result;
	}
}
