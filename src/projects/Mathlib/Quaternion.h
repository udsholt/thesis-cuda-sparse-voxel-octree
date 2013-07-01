#ifndef __RESTLESS_MATH_QUATERNION_H
#define __RESTLESS_MATH_QUATERNION_H

#include "Mathlib.h"
#include "Mat4x4f.h"
#include "Vec3f.h"

namespace restless
{
	class Quaternion
	{
	  public:
		float W, X, Y, Z;

		Quaternion();
		Quaternion(const float w, const float x = 0.0, const float y = 0.0, const float z = 0.0);
		Quaternion(const float w, const Vec3f v = Vec3f(0,0,0));

		friend Quaternion operator * (const Quaternion&, const Quaternion&);
		const Quaternion& operator *= (const Quaternion&);
		const Quaternion& operator ~ ();
		const Quaternion& operator - ();

		void Quaternion::normalize();

		Quaternion conjugate() const;

		static Quaternion fromAxis(const float Angle, Vec3f v);
		static Quaternion fromEuler(float x, float y, float z);
		static Quaternion fromEuler(Vec3f v);

		static Quaternion interpolate(const Quaternion & a, const Quaternion & b, const float t); // uses slerp

		void toMatrix(Mat4x4f& matrix) const;
		Mat4x4f asMatrix() const;

		Mat4x4f getMatrix() const;

		void slerp(const Quaternion & a, const Quaternion & b, const float);

		const Quaternion& exp();
		const Quaternion& log();

		friend std::ostream & operator<<(std::ostream & os, const restless::Quaternion & q);

	};
	std::ostream & operator<<(std::ostream & os, const restless::Quaternion & q);
}
#endif