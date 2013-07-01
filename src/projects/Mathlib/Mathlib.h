#ifndef _RESTLESS_MATH_MATHLIB_H
#define _RESTLESS_MATH_MATHLIB_H

#include <cmath>

#include "Vec3f.h"

// Define M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON     1.192092896e-07F
#endif

#define DEG_CIRCLE 360
#define DEG_TO_RAD (M_PI / (DEG_CIRCLE / 2))
#define RAD_TO_DEG ((DEG_CIRCLE / 2) / M_PI)

#define PI_OVER_360 (M_PI/DEG_CIRCLE)

namespace restless
{

	enum Axis {
		AXIS_X = 0,
		AXIS_Y = 1,
		AXIS_Z = 2
	};

	inline double deg2rad(double degrees)
	{
		return degrees * DEG_TO_RAD;
	}

	inline double rad2deg(double radians)
	{
		return radians * RAD_TO_DEG;
	}

	inline float maxf(const float a, const float b)
	{
		return a >= b ? a : b;
	}

	inline float minf(const float a, const float b)
	{
		return a <= b ? a : b;
	}

	inline Vec3f vmaxf(const Vec3f a, const Vec3f b)
	{
		return Vec3f(maxf(a[0], b[0]), maxf(a[1], b[1]), maxf(a[2], b[2]));
	}

	inline Vec3f vminf(const Vec3f a, const Vec3f b)
	{
		return Vec3f(minf(a[0], b[0]), minf(a[1], b[1]), minf(a[2], b[2]));
	}

	inline float clampf(const float a, const float low, const float high)
	{
		return minf(maxf(a, low), high);
	}

	inline int maxi(const int a, const int b)
	{
		return a >= b ? a : b;
	}

	inline int mini(const int a, const int b)
	{
		return a <= b ? a : b;
	}

	inline int clampi(const int a, const int low, const int high)
	{
		return mini(maxi(a, low), high);
	}

	inline float wrapf(const float x, const float y)
	{
		if (0 == y) {
			return x;
		}

		return x - y * floor(x/y);
	}

	// http://isezen.com/2012/01/15/quadratic-interpolation-three-point/
	inline float qinterpf(float xInterp, float x0, float y0, float x1, float y1, float x2, float y2)
	{
		float a0 = y0 / ((x0 - x1) * (x0 - x2));
		float a1 = y1 / ((x1 - x0) * (x1 - x2));
		float a2 = y2 / ((x2 - x0) * (x2 - x1));

		float A = a0 + a1 + a2;
		float B = -(a0 * (x1 + x2) + a1 * (x0 + x2) + a2 * (x0 + x1));
		float C = a0 * x1 * x2 + a1 * x0 * x2 + a2 * x0 * x1;

		return A * xInterp * xInterp + B * xInterp + C;
	}

}

#endif