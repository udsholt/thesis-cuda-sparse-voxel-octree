#ifndef __RESTLESS_MATH_VEC4I_H
#define __RESTLESS_MATH_VEC4I_H

#include "VecN.h"

namespace restless
{
	class Vec4i : public restless::VecN<4,int>
	{
	public:
		Vec4i();
		Vec4i(const int value);
		Vec4i(const int x, const int y, const int z, const int w);	
		Vec4i(const VecN<4,int> & vector);	

		// Cross product
		Vec4i cross(const VecN<4,int> & vector) const;
	};
}

#endif