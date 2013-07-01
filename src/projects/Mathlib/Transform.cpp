#include "Transform.h"

namespace restless 
{

	Transform::Transform(void) :
		_localPosition(0,0,0),
		_localRotation(1,0,0,0),
		_localScale(1,1,1),
		_mLocal_invalid(true),
		_mLocalInv_invalid(true),
		_mWorld_invalid(true),
		_mWorldInv_invalid(true),
		_transformParent(nullptr)
	{

	}


	Transform::~Transform(void)
	{

	}

	const Mat4x4f & Transform::getLocalToWorld()
	{
		if (_mWorld_invalid) {
			// If a parent transform exists that should be the base of the new _mWorld matrix
			// Otherwise we just use the local matrix
			if (_transformParent) {
				_mWorld  = Mat4x4f(_transformParent->getLocalToWorld());
				_mWorld *= getLocal();
			} else {
				_mWorld = Mat4x4f(getLocal());
			}
			_mWorld_invalid = false;
		}
		return _mWorld;
	}

	const Mat4x4f & Transform::getWorldToLocal()
	{
		if (_mWorldInv_invalid) {

			_mWorldInv = Mat4x4f(getLocalInverse());

			// Multiply with parents inverse if a parent exists
			if (_transformParent) {
				_mWorldInv *= _transformParent->getWorldToLocal();
			}

			_mWorldInv_invalid = false;
		}
		return _mWorldInv;
	}

	void Transform::invalidateLocalMatrices()
	{
		_mLocal_invalid    = true;
		_mLocalInv_invalid = true;

		invalidateWorldMatrices();
	}

	void Transform::invalidateWorldMatrices()
	{
		_mWorld_invalid    = true;
		_mWorldInv_invalid = true;
	}

	const Mat4x4f & Transform::getLocal()
	{
		if (_mLocal_invalid) {
			// The order is Translation * Rotation * Scale
			_mLocal  = Mat4x4f(Mat4x4f::translation(_localPosition[0],_localPosition[1],_localPosition[2]));
			_mLocal *= _localRotation.getMatrix();
			_mLocal *= Mat4x4f::scale(_localScale[0],_localScale[1],_localScale[2]);
			_mLocal_invalid = false;
		}
		return _mLocal;
	}

	const Mat4x4f & Transform::getLocalInverse()
	{
		if (_mLocalInv_invalid) {
			// The order is -Scale * -Rotation * -Translation
			_mLocalInv  = Mat4x4f(Mat4x4f::scale(-_localScale[0],-_localScale[1],-_localScale[2]));
			_mLocal    *= _localRotation.conjugate().getMatrix();
			_mLocalInv *= Mat4x4f::translation(-_localPosition[0],-_localPosition[1],-_localPosition[2]);
			_mLocalInv_invalid = false;
		}
		return _mLocalInv;
	}

	const Vec3f & Transform::getLocalPosition() const
	{
		return _localPosition;
	}

	const Quaternion & Transform::getLocalRotation() const
	{
		return _localRotation;
	}

	const Vec3f & Transform::getLocalScale() const
	{
		return _localScale;
	}

	void Transform::setLocalPosition(const Vec3f & position)
	{		
		_localPosition = position;
		invalidateLocalMatrices();
	}

	void Transform::setLocalRotation(const Vec3f & rotation)
	{
		_localRotation = Quaternion::fromEuler(rotation[0],rotation[1],rotation[2]);
		invalidateLocalMatrices();
	}

	void Transform::setLocalRotation(const Quaternion& rotation)
	{
		_localRotation = rotation;
		invalidateLocalMatrices();
	}

	void Transform::setLocalScale(const Vec3f & scale)
	{
		_localScale = scale;
		invalidateLocalMatrices();
	}

	void Transform::setLocalScale(float scalar)
	{
		setLocalScale(Vec3f(scalar));
	}

	void Transform::translate(const Vec3f & translation)
	{
		_localPosition += translation;
		invalidateLocalMatrices();
	}

	void Transform::rotate(const Vec3f & rotation)
	{
		_localRotation *= Quaternion::fromEuler(rotation[0],rotation[1],rotation[2]);
		invalidateLocalMatrices();
	}

	void Transform::scale(const Vec3f & scalar)
	{
		_localScale *= scalar;
		invalidateLocalMatrices();
	}

	void Transform::scale(float scalar)
	{
		scale(Vec3f(scalar));
	}

}