#ifndef _RESTLESS_CUDA_CUDA_RAY_H
#define _RESTLESS_CUDA_CUDA_RAY_H

#include "Math.h"

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

inline __host__ __device__
Ray make_ray(const float3 & origin, const float3 direction)
{
	Ray ray;
	ray.o = origin;
	ray.d = direction;
	return ray;
}

inline __host__ __device__
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

#endif