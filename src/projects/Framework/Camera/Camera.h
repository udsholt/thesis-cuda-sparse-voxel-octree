#ifndef _RESTLESS_CAMERA_CAMERA_H
#define _RESTLESS_CAMERA_CAMERA_H

#include <Mathlib/Transform.h>
#include <Mathlib/Mat4x4f.h>
#include <Mathlib/Vec3f.h>
#include <Mathlib/Vec2f.h>

#include "Frustum.h"

namespace restless
{
	class Camera 
	{
	public:

		Camera();
		virtual ~Camera();

		virtual void setNear(const float near);
		virtual void setFar(const float far);
		virtual void setFov(const float fov);

		virtual float getNear() const;
		virtual float getFar() const;
		virtual float getFov() const;

		virtual void onViewportResize(const int width, const int height);

		virtual const Frustum & getFrustum() const;

		virtual const Mat4x4f & getViewMatrix();
		virtual const Mat4x4f & getProjectionMatrix() const;

		virtual const Mat4x4f & getViewMatrixInverse();
		virtual const Mat4x4f & getProjectionMatrixInverse() const;

		virtual Vec3f getPosition() const;
		virtual void setPosition(const Vec3f & position);

		virtual void forward(const float distance);
		virtual void up(const float distance);
		virtual void strafe(const float distance);
		virtual void rotate(const float pitch, const float yaw);

		virtual float getPitch() const;
		virtual float getYaw() const;

		virtual void setPitch(const float pitch);
		virtual void setYaw(const float yaw);
		virtual void setRotation(const float pitch, const float yaw);

	protected:

		void updateFrustum();

		Frustum _frustum;

		float _viewportWidth;
		float _viewportHeight;
		float _fov;
		float _near;
		float _far;

		float _pitch;
		float _yaw;

		bool _viewMatrixInvalid;
		bool _viewMatrixInverseInvalid;

		Transform _transform;

		Mat4x4f _viewMatrix;
		Mat4x4f _viewMatrixInverse;

		Mat4x4f _projectionMatrix;
		Mat4x4f _projectionMatrixInverse;
	};
}

#endif