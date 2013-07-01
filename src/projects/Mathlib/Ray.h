#ifndef _RESTLESS_MATH_RAY_H
#define _RESTLESS_MATH_RAY_H

#include "Vec3f.h"
#include "Vec4f.h"

namespace restless
{
	class Ray
	{
	public:
		Ray(const Vec3f & origin, const Vec3f & direction);
		Ray(const Vec4f & origin, const Vec4f & direction);
		~Ray();

		bool intersectBox(const Vec3f bboxMin, const Vec3f bboxMax, float & tnear, float & tfar);

		Vec3f o;
		Vec3f d;
	};
}

#endif