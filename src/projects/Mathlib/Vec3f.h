#ifndef __RESTLESS_MATH_VEC3F_H
#define __RESTLESS_MATH_VEC3F_H

#include "VecN.h"

namespace restless
{
	class Vec3f : public restless::VecN<3,float>
	{
	public:
		Vec3f();
		Vec3f(const float value);
		Vec3f(const float x, const float y, const float z);	
		Vec3f(const VecN<3,float> & vector);
		Vec3f(const VecN<3,int> & vector);	

		// Cross product
		Vec3f cross(const VecN<3,float> & vector) const;	
	};
}

#endif