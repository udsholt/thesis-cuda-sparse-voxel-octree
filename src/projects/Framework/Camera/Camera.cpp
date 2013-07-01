#include "Camera.h"

#include "../Util/Log.h"
#include "../Util/Helper.h"

namespace restless
{
	Camera::Camera() :
		_pitch(0.0),
		_yaw(0.0),
		_viewMatrixInvalid(true),
		_viewMatrixInverseInvalid(true),
		_frustum(),
		_fov(60.0f),
		_near(0.2f),
		_far(100.0f),
		_viewportHeight(800.0f),
		_viewportWidth(600.0f)
	{
	}

	Camera::~Camera()
	{

	}

	const Frustum & Camera::getFrustum() const
	{
		return _frustum;
	}

	Vec3f Camera::getPosition() const
	{
		return _transform.getLocalPosition();
	}

	void Camera::setPosition(const Vec3f & position)
	{
		_transform.setLocalPosition(position);
	}

	void Camera::forward(const float distance)
	{
		Vec3f dir = _transform.getLocalToWorld() * Vec3f(0.0f, 0.0f, 1.0f);
		_transform.translate(dir * distance);

		_viewMatrixInvalid = true;
		_viewMatrixInverseInvalid = true;
	}

	void Camera::up(const float distance)
	{
		Vec3f up  = _transform.getLocalToWorld() * Vec3f(0.0f, 1.0f, 0.0f);
		_transform.translate(up * distance);

		_viewMatrixInvalid = true;
		_viewMatrixInverseInvalid = true;
	}

	void Camera::strafe(const float distance)
	{
		Vec3f dir = _transform.getLocalToWorld() * Vec3f(0.0f, 0.0f, 1.0f);
		Vec3f up  = _transform.getLocalToWorld() * Vec3f(0.0f, 1.0f, 0.0f);
		Vec3f right = dir.cross(up);

		_transform.translate(right * distance);

		_viewMatrixInvalid = true;
		_viewMatrixInverseInvalid = true;
	}

	void Camera::rotate(const float pitch, const float yaw)
	{
		setRotation(_pitch + pitch, _yaw + yaw);
	}

	float Camera::getPitch() const
	{
		return _pitch;
	}

	float Camera::getYaw() const
	{
		return _yaw;
	}

	void Camera::setPitch(const float pitch)
	{
		setRotation(pitch, _yaw);
	}

	void Camera::setYaw(const float yaw)
	{
		setRotation(_pitch, yaw);
	}

	void Camera::setRotation(const float pitch, const float yaw)
	{
		_pitch = wrapf(pitch, 2*M_PI);
		_yaw = wrapf(yaw, 2*M_PI);

		Quaternion xQuaternion = Quaternion::fromAxis(_pitch, Vec3f(0,1,0));
		Quaternion yQuaternion = Quaternion::fromAxis(_yaw, Vec3f(1,0,0));
		
		_transform.setLocalRotation(xQuaternion * yQuaternion);

		_viewMatrixInvalid = true;
		_viewMatrixInverseInvalid = true;
	}

	void Camera::onViewportResize(const int width, const int height)
	{
		_viewportWidth = width;
		_viewportHeight = height;

		updateFrustum();
	}

	void Camera::setNear(const float near)
	{
		_near = near;
		updateFrustum();
	}

	void Camera::setFar(const float far)
	{
		_far = far;
		updateFrustum();
	}

	void Camera::setFov(const float fov)
	{
		_fov = fov;
		updateFrustum();
	}

	float Camera::getNear() const
	{
		return _near;
	}

	float Camera::getFar() const
	{
		return _far;
	}

	float Camera::getFov() const
	{
		return _fov;
	}

	void Camera::updateFrustum()
	{
		const float apectRatio = ((float)_viewportWidth/_viewportHeight);

		_frustum.setPerspective(_fov, apectRatio, _near, _far);
	}

	const Mat4x4f & Camera::getViewMatrix()
	{
		if (_viewMatrixInvalid) {
			const Mat4x4f & localToWorld = _transform.getLocalToWorld();

			Vec3f dir = localToWorld * Vec3f(0.0f, 0.0f, 1.0f);
			Vec3f up  = localToWorld * Vec3f(0.0f, 1.0f, 0.0f);
			Vec4f pos = localToWorld * Vec4f(0.0f, 0.0f, 0.0f, 1.0f);
			Vec3f eye = Vec3f(pos[0], pos[1], pos[2]);
			Vec3f target = eye + dir;

			_viewMatrix = Mat4x4f::lookat(eye, target, up);
			_viewMatrixInvalid = false;
		}

		return _viewMatrix;
	}

	const Mat4x4f & Camera::getViewMatrixInverse()
	{
		if (_viewMatrixInverseInvalid) {
			_viewMatrixInverse = getViewMatrix().inverse();
			_viewMatrixInverseInvalid = false;
		}
		return _viewMatrixInverse;
	}

	const Mat4x4f & Camera::getProjectionMatrix() const
	{
		return _frustum.getProjectionMatrix();
	}

	const Mat4x4f & Camera::getProjectionMatrixInverse() const
	{
		return _frustum.getInverseProjectionMatrix();
	}
}