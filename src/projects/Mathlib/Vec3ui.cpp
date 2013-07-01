#include "Vec3ui.h"

namespace restless 
{
	Vec3ui::Vec3ui() : VecN() {}

	Vec3ui::Vec3ui(const unsigned int value) : VecN(value) {}

	Vec3ui::Vec3ui(const unsigned int x, const unsigned int y, const unsigned int z) : VecN() 
	{
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
	}

	Vec3ui::Vec3ui(const VecN<3,unsigned int> & vector)
	{
		(*this)[0] = vector[0];
		(*this)[1] = vector[1];
		(*this)[2] = vector[2];
	}
}
