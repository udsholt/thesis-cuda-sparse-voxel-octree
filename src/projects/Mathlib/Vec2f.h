#ifndef __RESTLESS_MATH_VEC2F_H
#define __RESTLESS_MATH_VEC2F_H

#include "VecN.h"

namespace restless
{
	class Vec2f : public restless::VecN<2,float>
	{
	public:
		Vec2f();
		Vec2f(const float value);
		Vec2f(const float x, const float y);	
		Vec2f(const VecN<2,float> & vector);
	};
}

#endif