#ifndef __RESTLESS_MATH_VEC4F_H
#define __RESTLESS_MATH_VEC4F_H

#include "VecN.h"

namespace restless
{
	class Vec4f : public restless::VecN<4,float>
	{
	public:
		Vec4f();
		Vec4f(const float value);
		Vec4f(const float x, const float y, const float z, const float w);	
		Vec4f(const VecN<4,float> & vector);	

		// Cross product
		Vec4f cross(const VecN<4,float> & vector) const;
	};
}

#endif