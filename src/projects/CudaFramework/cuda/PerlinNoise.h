#ifndef _RESTLESS_CUDA_CUDA_H
#define _RESTLESS_CUDA_CUDA_H

/**
 * Originally taken from:
 * https://code.google.com/p/agt6-cuda/source/browse/trunk/Source/CudaFiles/Perlin.cu?r=2
 */

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Math.h"

#define PERLIN_SIZE 256
#define PERLIN_MASK 0xFF

// Permutation table
__constant__ unsigned char __perlin_p[PERLIN_SIZE];

// Gradients
__constant__ float __perlin_gx[PERLIN_SIZE];
__constant__ float __perlin_gy[PERLIN_SIZE];
__constant__ float __perlin_gz[PERLIN_SIZE];

__device__ 
float noise1(float x)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & PERLIN_MASK;
	qx1 = qx1 & PERLIN_MASK;

	// Compute the dotproduct between the vectors and the gradients
	float v0 = __perlin_gx[qx0]*tx0;
	float v1 = __perlin_gx[qx1]*tx1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v = v0 - wx*(v0 - v1);

	return v;
}

__device__
float noise2(float x, float y)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	int qy0 = (int)floorf(y);
	int qy1 = qy0 + 1;
	float ty0 = y - (float)qy0;
	float ty1 = ty0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & PERLIN_MASK;
	qx1 = qx1 & PERLIN_MASK;

	qy0 = qy0 & PERLIN_MASK;
	qy1 = qy1 & PERLIN_MASK;

	// Permutate values to get pseudo randomly chosen gradients
	int q00 = __perlin_p[(qy0 + __perlin_p[qx0]) & PERLIN_MASK];
	int q01 = __perlin_p[(qy0 + __perlin_p[qx1]) & PERLIN_MASK];

	int q10 = __perlin_p[(qy1 + __perlin_p[qx0]) & PERLIN_MASK];
	int q11 = __perlin_p[(qy1 + __perlin_p[qx1]) & PERLIN_MASK];

	// Compute the dotproduct between the vectors and the gradients
	float v00 = __perlin_gx[q00]*tx0 + __perlin_gy[q00]*ty0;
	float v01 = __perlin_gx[q01]*tx1 + __perlin_gy[q01]*ty0;

	float v10 = __perlin_gx[q10]*tx0 + __perlin_gy[q10]*ty1;
	float v11 = __perlin_gx[q11]*tx1 + __perlin_gy[q11]*ty1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v0 = v00 - wx*(v00 - v01);
	float v1 = v10 - wx*(v10 - v11);

	float wy = (3 - 2*ty0)*ty0*ty0;
	float v = v0 - wy*(v0 - v1);

	return v;	
}

__inline__ __device__ float noise2(float2 pos)
{
	return noise2(pos.x, pos.y);
}

__device__
float noise3(float x, float y, float z)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	int qy0 = (int)floorf(y);
	int qy1 = qy0 + 1;
	float ty0 = y - (float)qy0;
	float ty1 = ty0 - 1;

	int qz0 = (int)floorf(z);
	int qz1 = qz0 + 1;
	float tz0 = z - (float)qz0;
	float tz1 = tz0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & PERLIN_MASK;
	qx1 = qx1 & PERLIN_MASK;

	qy0 = qy0 & PERLIN_MASK;
	qy1 = qy1 & PERLIN_MASK;

	qz0 = qz0 & PERLIN_MASK;
	qz1 = qz1 & PERLIN_MASK;

	// Permutate values to get pseudo randomly chosen gradients
	int q000 = __perlin_p[(qz0 + __perlin_p[(qy0 + __perlin_p[qx0]) & PERLIN_MASK]) & PERLIN_MASK];
	int q001 = __perlin_p[(qz0 + __perlin_p[(qy0 + __perlin_p[qx1]) & PERLIN_MASK]) & PERLIN_MASK];

	int q010 = __perlin_p[(qz0 + __perlin_p[(qy1 + __perlin_p[qx0]) & PERLIN_MASK]) & PERLIN_MASK];
	int q011 = __perlin_p[(qz0 + __perlin_p[(qy1 + __perlin_p[qx1]) & PERLIN_MASK]) & PERLIN_MASK];

	int q100 = __perlin_p[(qz1 + __perlin_p[(qy0 + __perlin_p[qx0]) & PERLIN_MASK]) & PERLIN_MASK];
	int q101 = __perlin_p[(qz1 + __perlin_p[(qy0 + __perlin_p[qx1]) & PERLIN_MASK]) & PERLIN_MASK];

	int q110 = __perlin_p[(qz1 + __perlin_p[(qy1 + __perlin_p[qx0]) & PERLIN_MASK]) & PERLIN_MASK];
	int q111 = __perlin_p[(qz1 + __perlin_p[(qy1 + __perlin_p[qx1]) & PERLIN_MASK]) & PERLIN_MASK];

	// Compute the dotproduct between the vectors and the gradients
	float v000 = __perlin_gx[q000]*tx0 + __perlin_gy[q000]*ty0 + __perlin_gz[q000]*tz0;
	float v001 = __perlin_gx[q001]*tx1 + __perlin_gy[q001]*ty0 + __perlin_gz[q001]*tz0;  

	float v010 = __perlin_gx[q010]*tx0 + __perlin_gy[q010]*ty1 + __perlin_gz[q010]*tz0;
	float v011 = __perlin_gx[q011]*tx1 + __perlin_gy[q011]*ty1 + __perlin_gz[q011]*tz0;

	float v100 = __perlin_gx[q100]*tx0 + __perlin_gy[q100]*ty0 + __perlin_gz[q100]*tz1;
	float v101 = __perlin_gx[q101]*tx1 + __perlin_gy[q101]*ty0 + __perlin_gz[q101]*tz1;  

	float v110 = __perlin_gx[q110]*tx0 + __perlin_gy[q110]*ty1 + __perlin_gz[q110]*tz1;
	float v111 = __perlin_gx[q111]*tx1 + __perlin_gy[q111]*ty1 + __perlin_gz[q111]*tz1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v00 = v000 - wx*(v000 - v001);
	float v01 = v010 - wx*(v010 - v011);
	float v10 = v100 - wx*(v100 - v101);
	float v11 = v110 - wx*(v110 - v111);

	float wy = (3 - 2*ty0)*ty0*ty0;
	float v0 = v00 - wy*(v00 - v01);
	float v1 = v10 - wy*(v10 - v11);

	float wz = (3 - 2*tz0)*tz0*tz0;
	float v = v0 - wz*(v0 - v1);

	return v;	
}

__inline__ __device__ float noise3(float3 pos)
{
	return noise3(pos.x, pos.y, pos.z);
}


__device__
float perlinNoise2d(float x, float y, int octaves, float frequency, float amplitude)
{
	float result = 0.0f;

	x *= frequency;
	y *= frequency;

	for(int i=0; i < octaves; i++) {
		result += noise2(x, y) * amplitude;
		x *= 2.0f;
		y *= 2.0f;
		amplitude *= 0.5f;
	}

	return result;
}

__device__
float perlinNoise3d(float x, float y, float z, int octaves, float frequency, float amplitude)
{
	float result = 0.0f;

	x *= frequency;
	y *= frequency;
	z *= frequency;

	for(int i=0; i < octaves; i++) {
		result += noise3(x, y, z) * amplitude;
		x *= 2.0f;
		y *= 2.0f;
		z *= 2.0f;
		amplitude *= 0.5f;
	}

	return result;
}

__host__
cudaError_t perlinNoiseInitialize(int seed) 
{
	// Permutation table
	unsigned char temp_p[PERLIN_SIZE];

	// Gradients
	float temp_gx[PERLIN_SIZE];
	float temp_gy[PERLIN_SIZE];
	float temp_gz[PERLIN_SIZE];

	int i, j, nSwap;

	srand(seed);

	// Initialize the permutation table
	for(i = 0; i < PERLIN_SIZE; i++) {
		temp_p[i] = i;
	}

	for(i = 0; i < PERLIN_SIZE; i++) {
		j = rand() & PERLIN_MASK;

		nSwap = temp_p[i];
		temp_p[i]  = temp_p[j];
		temp_p[j]  = nSwap;
	}

	// Generate the gradient lookup tables
	for(i = 0; i < PERLIN_SIZE; i++) {
		
		// Ken Perlin proposes that the gradients are taken from the unit 
		// circle/sphere for 2D/3D, but there are no noticable difference 
		// between that and what I'm doing here. For the sake of generality 
		// I will not do that.

		temp_gx[i] = float(rand())/(RAND_MAX/2) - 1.0f; 
		temp_gy[i] = float(rand())/(RAND_MAX/2) - 1.0f;
		temp_gz[i] = float(rand())/(RAND_MAX/2) - 1.0f;
	}

	cudaError_t cudaErrorId;
	cudaErrorId = cudaMemcpyToSymbol(__perlin_p, temp_p, sizeof(unsigned char) * PERLIN_SIZE);
	if (cudaErrorId != cudaSuccess) {
		return cudaErrorId;
	}

	cudaErrorId = cudaMemcpyToSymbol(__perlin_gx, temp_gx, sizeof(float) * PERLIN_SIZE);
	if (cudaErrorId != cudaSuccess) {
		return cudaErrorId;
	}

	cudaErrorId = cudaMemcpyToSymbol(__perlin_gy, temp_gy, sizeof(float) * PERLIN_SIZE);
	if (cudaErrorId != cudaSuccess) {
		return cudaErrorId;
	}

	cudaErrorId = cudaMemcpyToSymbol(__perlin_gz, temp_gz, sizeof(float) * PERLIN_SIZE);
	if (cudaErrorId != cudaSuccess) {
		return cudaErrorId;
	}

	return cudaSuccess;
}

#endif