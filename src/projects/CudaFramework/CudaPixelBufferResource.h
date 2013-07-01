#ifndef _RESTLESS_CUDA_PIXELBUFFERRESOURCE_H
#define _RESTLESS_CUDA_PIXELBUFFERRESOURCE_H

// forward declare cudaGraphicsResource, float4
struct cudaGraphicsResource;
struct float4;

namespace restless
{
	class CudaPixelBufferResource
	{
	public:
		CudaPixelBufferResource();
		~CudaPixelBufferResource();

		void bindBufferTexture(const unsigned int channel) const;
		void unbindBufferTexture() const;

		const unsigned int getWidth() const;
		const unsigned int getHeight() const;

		float4 * mapDevicePointer();
		void unmapDevicePointer();

		void initialize(const unsigned int width, const unsigned int height);
		void resize(const unsigned int width, const unsigned int height);

	protected:

		cudaGraphicsResource * cudaResource;

		unsigned int pixelBufferObject;
		unsigned int textureId;
		unsigned int textureWidth;
		unsigned int textureHeight;
	};
}

#endif