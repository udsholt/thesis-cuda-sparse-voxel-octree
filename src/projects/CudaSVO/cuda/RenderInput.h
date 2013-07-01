#ifndef _INC_CUDA_RENDER_INPUT_H
#define _INC_CUDA_RENDER_INPUT_H

#include <CudaFramework/cuda/ViewFrustum.h>
#include <CudaFramework/cuda/Matrix.h>

struct PixelBuffer
{
	unsigned int width;
	unsigned int height;

	float4 * ptr;
};

struct Light
{
	float3 worldPosition;
	float shininess;
	unsigned char enabled;
};

struct RenderInput
{
	Matrix4x4 viewMatrixInverse;
	ViewFrustum viewFrustum;
	PixelBuffer buffer;

	Light light;

	float stepSize;

	float pixelSizeOnNearPlane;

	int maxDepth;

	int enableSubdivide;

	unsigned char renderMode;
};


#endif