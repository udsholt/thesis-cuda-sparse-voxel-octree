#ifndef __RESTLESS_MATH_VEC3I_H
#define __RESTLESS_MATH_VEC3I_H

#include "VecN.h"

namespace restless
{
	class Vec3i : public restless::VecN<3,int>
	{
	public:
		Vec3i();
		Vec3i(const int value);
		Vec3i(const int x, const int y, const int z);	
		Vec3i(const VecN<3,int> & vector);	
	};
}

#endif