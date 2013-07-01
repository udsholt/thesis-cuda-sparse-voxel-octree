#ifndef __RESTLESS_MATH_VEC2I_H
#define __RESTLESS_MATH_VEC2I_H

#include "VecN.h"

namespace restless
{
	class Vec2i : public restless::VecN<2,int>
	{
	public:
		Vec2i();
		Vec2i(const int value);
		Vec2i(const int x, const int y);	
		Vec2i(const VecN<2,int> & vector);
	};
}

#endif