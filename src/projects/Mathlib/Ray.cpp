#include "Ray.h"

#include "Mathlib.h"

namespace restless
{
	Ray::Ray(const Vec3f & origin, const Vec3f & direction) :
		o(origin),
		d(direction)
	{

	}

	Ray::Ray(const Vec4f & origin, const Vec4f & direction) :
		o(origin[0], origin[1], origin[2]),
		d(direction[0], direction[1], direction[2])
	{

	}

	Ray::~Ray()
	{
	}

	bool Ray::intersectBox(const Vec3f bboxMin, const Vec3f bboxMax, float & tnear, float & tfar)
	{
		// compute intersection of ray with all six bbox planes
		Vec3f invR = Vec3f(1.0f, 1.0f, 1.0f) / d;
		Vec3f tbot = invR * (bboxMin - o);
		Vec3f ttop = invR * (bboxMax - o);

		// re-order intersections to find smallest and largest on each axis
		Vec3f tmin = vminf(ttop, tbot);
		Vec3f tmax = vmaxf(ttop, tbot);

		// find the largest tmin and the smallest tmax
		float largest_tmin = maxf(maxf(tmin[0], tmin[1]), maxf(tmin[0], tmin[2]));
		float smallest_tmax = minf(minf(tmax[0], tmax[1]), minf(tmax[0], tmax[2]));

		tnear = largest_tmin;
		tfar = smallest_tmax;

		return smallest_tmax > largest_tmin;
	}

	/*
	__device__
	int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
	{
		// compute intersection of ray with all six bbox planes
		float3 invR = make_float3(1.0f, 1.0f, 1.0f) / r.d;
		float3 tbot = invR * (boxmin - r.o);
		float3 ttop = invR * (boxmax - r.o);

		// re-order intersections to find smallest and largest on each axis
		float3 tmin = fminf(ttop, tbot);
		float3 tmax = fmaxf(ttop, tbot);

		// find the largest tmin and the smallest tmax
		float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
		float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

		*tnear = largest_tmin;
		*tfar = smallest_tmax;

		return smallest_tmax > largest_tmin;
	}
	*/
	
}