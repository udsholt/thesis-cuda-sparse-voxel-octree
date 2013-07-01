#include "CudaPixelBufferResource.h"

#include <Framework/GL.h>
#include <Framework/Util/Log.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "CudaUtil.h"

namespace restless 
{
	CudaPixelBufferResource::CudaPixelBufferResource() :
		cudaResource(nullptr),
		pixelBufferObject(0),
		textureId(0),
		textureWidth(970),
		textureHeight(600)
	{
	}

	CudaPixelBufferResource::~CudaPixelBufferResource()
	{
		cudaError_t cudaErrorId;

		if (cudaResource) {
			if (cudasafe( cudaGraphicsUnregisterResource(cudaResource), "cudaGraphicsUnregisterResource" ) == false) {
				return;
			}
			cudaResource = nullptr;
		}

		if (textureId) {
			glDeleteTextures(1, & textureId);
			textureId = 0;
		}

		if (pixelBufferObject) {
			glDeleteBuffers(1, & pixelBufferObject);
			pixelBufferObject = 0;
		}
	}

	const unsigned int CudaPixelBufferResource::getWidth() const
	{
		return textureWidth;
	}

	const unsigned int CudaPixelBufferResource::getHeight() const
	{
		return textureHeight;
	}
	void CudaPixelBufferResource::bindBufferTexture(const unsigned int channel) const
	{
		glActiveTexture(channel);
		glBindTexture(GL_TEXTURE_BUFFER, textureId);
	}

	void CudaPixelBufferResource::unbindBufferTexture() const
	{	
		// this is apprently not kosher in opengl >= 3.3 
		glBindTexture(GL_TEXTURE_BUFFER, 0);
	}

	float4 * CudaPixelBufferResource::mapDevicePointer()
	{
		float4 * devPtr;
		size_t size;

		if (cudasafe( cudaGraphicsMapResources(1, & cudaResource, NULL), "cudaGraphicsMapResources" ) == false) {
			return nullptr;
		}

		if (cudasafe( cudaGraphicsResourceGetMappedPointer( (void**) &devPtr, &size, cudaResource), "cudaGraphicsResourceGetMappedPointer" ) == false) {
			cudasafe(cudaGraphicsUnmapResources(1, & cudaResource, NULL), "cudaGraphicsUnmapResources");
			return nullptr;
		}

		return devPtr;
	}

	void CudaPixelBufferResource::unmapDevicePointer()
	{
		cudasafe( cudaGraphicsUnmapResources(1, & cudaResource, NULL), "cudaGraphicsUnmapResources" );
	}


	void CudaPixelBufferResource::initialize(const unsigned int width, const unsigned int height)
	{
		resize(width, height);
	}

	void CudaPixelBufferResource::resize(const unsigned int width, const unsigned int height)
	{
		textureWidth = width;
		textureHeight = height;

		if (cudaResource) {
			if (cudasafe( cudaGraphicsUnregisterResource(cudaResource), "cudaGraphicsUnregisterResource" ) == false) {
				return;
			}
			cudaResource = nullptr;
		}

		if (textureId) {
			glDeleteTextures(1, & textureId);
			textureId = 0;
		}

		if (pixelBufferObject) {
			glDeleteBuffers(1, & pixelBufferObject);
			pixelBufferObject = 0;
		}
	
		glGenBuffers(1, & pixelBufferObject);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObject);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, textureWidth * textureHeight * sizeof(float4), NULL, GL_DYNAMIC_DRAW);

		glGenTextures(1, & textureId);
		glBindTexture(GL_TEXTURE_BUFFER, textureId);
		glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, pixelBufferObject);
		glBindTexture(GL_TEXTURE_BUFFER, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		cudasafe( cudaGraphicsGLRegisterBuffer(&cudaResource, pixelBufferObject, cudaGraphicsMapFlagsNone), "cudaGraphicsGLRegisterBuffer" );
	}
}