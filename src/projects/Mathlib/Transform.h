#ifndef _RESTLESS_MATH_TRANSFORM_H
#define _RESTLESS_MATH_TRANSFORM_H

#include "Vec3f.h"
#include "Quaternion.h"
#include "Mat4x4f.h"

namespace restless
{
	class Transform
	{
	public:
		Transform();
		~Transform();


		const Mat4x4f & getLocalToWorld();
		const Mat4x4f & getWorldToLocal();

		const Mat4x4f & Transform::getLocal();
		const Mat4x4f & Transform::getLocalInverse();

		const Vec3f & getLocalPosition() const;
		const Quaternion & getLocalRotation() const;
		const Vec3f & getLocalScale() const;

		void setLocalPosition(const Vec3f & position);
		void setLocalRotation(const Vec3f & rotation);
		void setLocalRotation(const Quaternion& rotation);
		void setLocalScale(const Vec3f & scale);
		void setLocalScale(float scale);

		void translate(const Vec3f & translation);
		void rotate(const Vec3f & rotation);
		void scale(const Vec3f & scalar);
		void scale(float scale);

	protected:
		Transform * _transformParent; // Not supported now, i dont need it

		Vec3f _localPosition;
		Vec3f _localScale;
		Quaternion _localRotation;

		Mat4x4f _mLocal;		// Stored local translation, rotation and scale
		Mat4x4f _mLocalInv;		// Stored inverse local translation, rotation and scale

		Mat4x4f _mWorld;		// Local to world
		Mat4x4f _mWorldInv;		// World to local

		bool _mLocal_invalid;
		bool _mLocalInv_invalid;
		bool _mWorld_invalid;
		bool _mWorldInv_invalid;

		void invalidateLocalMatrices();
		void invalidateWorldMatrices();
	};
}

#endif