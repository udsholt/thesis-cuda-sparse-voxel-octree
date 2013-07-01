#ifndef _INC_CAMERAKEYFRAME_H
#define _INC_CAMERAKEYFRAME_H

#include <Mathlib/Vec3f.h>
#include <Mathlib/Mathlib.h>

struct CameraKeyframe
{
	restless::Vec3f position;
	float pitch;
	float yaw;

	CameraKeyframe() : position(0), pitch(0), yaw(0) { }

	CameraKeyframe(const restless::Vec3f position, float pitch, float yaw)
	{
		this->position = position;
		this->pitch = pitch;
		this->yaw = yaw;
	}

	CameraKeyframe lerp(const CameraKeyframe & next, const float interp)
	{
		CameraKeyframe k;
		k.position = position + (next.position - position) * interp;
		k.pitch = pitch + (next.pitch - pitch) * interp;
		k.yaw = yaw + (next.yaw - yaw) * interp;
		return k;
	}

	CameraKeyframe qlerp(const CameraKeyframe & prev, const CameraKeyframe & next, const float interp)
	{
		CameraKeyframe k;

		k.position = restless::Vec3f(
			restless::qinterpf(interp, -1.0f, prev.position[0], 0.0f, position[0], 1.0f, next.position[0]),
			restless::qinterpf(interp, -1.0f, prev.position[1], 0.0f, position[1], 1.0f, next.position[1]),
			restless::qinterpf(interp, -1.0f, prev.position[2], 0.0f, position[2], 1.0f, next.position[2])
		);

		k.pitch = restless::qinterpf(interp, -1.0f, prev.pitch, 0.0f, pitch, 1.0f, next.pitch); 
		k.pitch = restless::qinterpf(interp, -1.0f, prev.yaw, 0.0f, yaw, 1.0f, next.yaw); 
		
		return k;
	}
};

#endif