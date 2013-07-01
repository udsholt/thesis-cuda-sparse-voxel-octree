#ifndef _RESTLESS_CUDA_CUDA_VIEWFRUSTUM_H
#define _RESTLESS_CUDA_CUDA_VIEWFRUSTUM_H

struct ViewFrustum
{ 
	float left;
	float right;
	float top;
	float bottom;
	float near;
	float far;
};

inline __host__ __device__
ViewFrustum make_frustum(const float * data)
{
	ViewFrustum f;
	f.left   = data[0];
	f.right  = data[1];
	f.top    = data[2];
	f.bottom = data[3];
	f.near   = data[4];
	f.far    = data[5];

	return f;
}

inline __host__ __device__
ViewFrustum make_frustum(const float left, const float right, const float top, const float bottom, const float near, const float far)
{
	ViewFrustum f;
	f.left   = left;
	f.right  = right;
	f.top    = top;
	f.bottom = bottom;
	f.near   = near;
	f.far    = far;
	return f;
}

// Convert from restless::Frustum to Frustum
#if !defined(__CUDACC__)

	#include <Framework/Camera/Frustum.h>

	inline
	ViewFrustum make_frustum(const restless::Frustum & frustum) 
	{
		ViewFrustum f;
		f.left   = frustum.getLeft();
		f.right  = frustum.getRight();
		f.top    = frustum.getTop();
		f.bottom = frustum.getBottom();
		f.near   = frustum.getNear();
		f.far    = frustum.getFar();
		return f;
	}

#endif

#endif