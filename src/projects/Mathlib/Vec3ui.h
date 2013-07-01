#ifndef __RESTLESS_MATH_VEC3UI_H
#define __RESTLESS_MATH_VEC3UI_H

#include "VecN.h"

namespace restless
{
	class Vec3ui : public restless::VecN<3,unsigned int>
	{
	public:
		Vec3ui();
		Vec3ui(const unsigned int value);
		Vec3ui(const unsigned int x, const unsigned int y, const unsigned int z);	
		Vec3ui(const VecN<3,unsigned int> & vector);	
	};
}

#endif