#ifndef _RESTLESS_CUDA_CUDA_MATRIX_H
#define _RESTLESS_CUDA_CUDA_MATRIX_H

#include "Math.h"

struct Matrix4x4
{ 
	float m[16]; 
};

inline __host__ __device__
Matrix4x4 make_matrix4x4(const float * data)
{
	Matrix4x4 M;
	for (unsigned int i = 0; i < 16; ++i) {
		M.m[i] = data[i];
	}
	return M;
}

inline __host__ __device__
float4 mul(const Matrix4x4 & M, const float4 & v)
{
    float4 r = make_float4(0, 0, 0, 0);

	r.x += M.m[ 0] * v.x;
	r.x += M.m[ 4] * v.y;
	r.x += M.m[ 8] * v.z;
	r.x += M.m[12] * v.w;

	r.y += M.m[ 1] * v.x;
	r.y += M.m[ 5] * v.y;
	r.y += M.m[ 9] * v.z;
	r.y += M.m[13] * v.w;

	r.z += M.m[ 2] * v.x;
	r.z += M.m[ 6] * v.y;
	r.z += M.m[10] * v.z;
	r.z += M.m[14] * v.w;

	r.w += M.m[ 3] * v.x;
	r.w += M.m[ 7] * v.y;
	r.w += M.m[11] * v.z;
	r.w += M.m[15] * v.w;

    return r;
}

#endif