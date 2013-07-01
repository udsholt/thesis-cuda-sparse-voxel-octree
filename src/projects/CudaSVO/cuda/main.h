#include "Defines.h"
#include "Octree.h"
#include "RenderInput.h"

#define OK_DONE             0
#define OK_MORE_WORK        1
#define ERROR_OUT_OF_BRICKS 2
#define ERROR_OUT_OF_TILES  3
#define ERROR_KERNEL        4

namespace restless
{
	class CudaTimerCollection;
}

extern "C" 
{
	typedef float  VoxelDensityType;
	typedef uchar4 VoxelColorType;

	void cudaInitializePerlinNoise(unsigned int seed);
	//void cudaInitializeConstants();

	void cudaSetTimerCollection(restless::CudaTimerCollection & timerCollection);
	void cudaOctreeSet(Octree & octree);

	void cudaVolumeRenderToPixelBuffer(const RenderInput & input, const bool verbose);
	void cudaOctreeUpdateUsageMasks(const bool verbose);
	void cudaOctreeInvalidateNodes(const bool verbose);
	void cudaOctreeProcessRequests(const bool verbose);

	const surfaceReference * cudaVolumeGetDensitySurfaceReference();
	const surfaceReference * cudaVolumeGetColorSurfaceReference();
	const textureReference * cudaVolumeGetDensityTextureReference();
	const textureReference * cudaVolumeGetColorTextureReference();
}