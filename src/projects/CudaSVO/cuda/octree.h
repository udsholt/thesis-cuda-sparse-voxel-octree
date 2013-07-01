#ifndef _INC_CUDA_OCTREE_H
#define _INC_CUDA_OCTREE_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <CudaFramework/cuda/Math.h>

#define FLAG_DATA_TYPE_CONSTANT 0
#define FLAG_DATA_TYPE_BRICK    1

#define FLAG_CAN_SUBDIVIDE      0
#define FLAG_CANNOT_SUBDIVIDE   1

// Octree device pointer wrapper
struct Octree
{
	// Node data
	int  * dev_childData;
	int  * dev_brickData;

	// Localization
	int  * dev_localizationCode;
	char * dev_localizationDepth;

	// Node timestamp and usage
	int * dev_tileUsage;
	int * dev_tileTimestamp;

	// Brick timestamp and usage
	int * dev_brickUsage;
	int * dev_brickTimestamp;

	// Requests
	int * dev_raymarcherRequests;
	int * dev_dataRequests;

	// Volumne size
	cudaExtent volumeSize;

	// Current timestamp
	int timestamp;

	// Texture/surface arrays
	cudaArray * dev_densityArray;
	cudaArray * dev_colorArray;

	// Node lenght and pixel size at depths
	float * dev_nodeLenght;
	float * dev_nodeVoxelPixelSize;
};

// Unpacked node childata
struct NodeChildData
{
	int flagMaxSubdivision;
    int flagDataType;
    int childAddress;
} ;

// Unpacked node constant color
typedef float4 NodeConstantColor;

// Unpacked pointer into brick texture
typedef int3 NodeBrickPointer;

// Unpacked node localization code
typedef int3 NodeLocalizationCode;

// Unpacked node localzation
struct NodeLocalization
{
	int depth;
	float3 bboxMin;
	float3 bboxMax;
};

struct NodeLocateResult
{
	int current;
	int parent;
	float3 pos;
	float3 parentPos;
};

struct TileUsage
{
	int tileAddress;
	int flag;
};

struct BrickUsage
{
	int3 brickPointer;
	int flag;
};

// Pack NodeChildData into a 32bit int
inline __host__ __device__ 
int packNodeChildData(const NodeChildData & data) 
{
	return (data.flagMaxSubdivision << 31) | (data.flagDataType << 30) | data.childAddress;
}

// Unpack NodeChildData from 32bit int
inline __host__ __device__ 
NodeChildData unpackNodeChildData(int packed) 
{
	NodeChildData data;
	data.childAddress = (packed << 2) >> 2;
	packed = packed >> 30;
	data.flagDataType = (packed & 1);
	packed = packed >> 1;
	data.flagMaxSubdivision = (packed & 1);
	return data;
}

inline __host__ __device__ 
int packTileUsage(const TileUsage & data)
{
	return (data.flag << 31) | data.tileAddress;
}

inline __host__ __device__ 
TileUsage unpackTileUsage(int packed)
{
	TileUsage usage;
	usage.tileAddress = (packed << 2) >> 2;
	packed = packed >> 31;
	usage.flag = (packed & 1);
	return usage;
}

// Pack brick pointer into 32bit int
inline __host__ __device__ 
int packBrickUsage(const BrickUsage & data)
{
	return (data.flag << 31) | (data.brickPointer.z << 20) | (data.brickPointer.y << 10)  | data.brickPointer.x;
}

// Unpack brick pointer code from 32bit int
inline __host__ __device__ 
BrickUsage unpackBrickUsage(int packed)
{
	BrickUsage usage;

	usage.brickPointer.x = packed & 1023; // 10 bits = z code
	packed >>= 10;
	usage.brickPointer.y = packed & 1023; // 10 bits = y code
	packed >>= 10;
	usage.brickPointer.z = packed & 1023; // 10 bits = x code
	packed >>= 11;
	usage.flag = (packed & 1);

	return usage;
}

inline __host__ __device__
int getTileTimestampIndexFromNodeTileAddress(int nodeTileAddress)
{
	return nodeTileAddress / 8;
}

// Convert a brick pointer (x,y,z) into a address into the brickUsage list
inline __host__ __device__
int getBrickTimestampIndexFromBrickPointer(int x, int y, int z)
{
	x /= BRICK_SIZE;
	y /= BRICK_SIZE;
	z /= BRICK_SIZE;
	return x * MAX_BRICKS_PER_SIDE * MAX_BRICKS_PER_SIDE + y * MAX_BRICKS_PER_SIDE + z; 
}

// Convert a brick pointer (x,y,z) into a address into the brickUsage list
inline __host__ __device__
int getBrickTimestampIndexFromBrickPointer(int3 pointer)
{
	return getBrickTimestampIndexFromBrickPointer(pointer.x, pointer.y, pointer.z); 
}

// Convert a brick pointer (x,y,z) into a address into the brickUsage list
inline __host__ __device__
int getBrickTimestampIndexFromBrickPointer(float3 pointer)
{
	return getBrickTimestampIndexFromBrickPointer((int)pointer.x, (int)pointer.y, (int)pointer.z); 
}



// Pack NodeConstantColor into 32bit int
inline __host__ __device__ 
int packNodeConstantColor(const NodeConstantColor & data)
{
	const unsigned char r = (char) (data.x * 255); // convert normalized float to char
	const unsigned char g = (char) (data.y * 255);
	const unsigned char b = (char) (data.z * 255);
	const unsigned char a = (char) (data.w * 255);

	return r << 24 | g << 16 | b << 8 | a; 
}

// Unpack NodeConstantColor from 32bit int
inline __host__ __device__ 
NodeConstantColor unpackNodeConstantColor(int packed)
{
	NodeConstantColor color;

	color.w = (packed & 255) / 255.0f; // first 8 bits = alpha
	packed = packed >> 8;
	color.z = (packed & 255) / 255.0f; // next 8 bits = blue
	packed = packed >> 8;
	color.y = (packed & 255) / 255.0f; // next 8 bits = green
	packed = packed >> 8;
	color.x = (packed & 255) / 255.0f; // next 8 bits = red

	return color;
}

// Pack brick pointer into 32bit int
inline __host__ __device__ 
int packNodeBrickPointer(const NodeBrickPointer & pointer)
{
	return pointer.x | (pointer.y << 10) | (pointer.z << 20);
}

// Unpack brick pointer code from 32bit int
inline __host__ __device__ 
NodeBrickPointer unpackNodeBrickPointer(int packed)
{
	NodeBrickPointer pointer;

	pointer.x = packed & 1023; // 10 bits = z code
	packed >>= 10;
	pointer.y = packed & 1023; // 10 bits = y code
	packed >>= 10;
	pointer.z = packed & 1023; // 10 bits = x code

	// leaves 2 bit un-used
	return pointer;
}
// Pack localization code into 32bit int
inline __host__ __device__ 
int packNodeLocalizationCode(const NodeLocalizationCode & code)
{
	return code.x | (code.y << 10) | (code.z << 20);
}

// Unpack localization code from 32bit int
inline __host__ __device__ 
NodeLocalizationCode unpackNodeLocalizationCode(int packed)
{
	NodeLocalizationCode code;

	code.x = packed & 1023; // 10 bits = z code
	packed >>= 10;
	code.y = packed & 1023; // 10 bits = y code
	packed >>= 10;
	code.z = packed & 1023; // 10 bits = x code

	// leaves 2 bit un-used

	return code;
}

// Pack depth from int to char
// ... this is only to keep things regular and not really needed
inline __host__ __device__ 
char packNodeLocalizationDepth(const int & depth)
{
	return (char) depth;
}

// Unpack depth from char to int 
// ... this is only to keep things regular and not really needed
inline __host__ __device__ 
int unpackNodeLocalizationDepth(const char & packed)
{
	return (int) packed;
}

// Set the 3 bits in the localization code that corrosponds
// to the given choice at the depth
// 
// see more: http://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit-in-c
//
inline __host__ __device__ 
void setLocalizationChoiceAtDepth(const int & depth, NodeLocalizationCode & code, const int3 & choice)
{
	const int mask = 1 << depth;

	code.x = choice.x ? (code.x | mask) : (code.x & ~(mask));
	code.y = choice.y ? (code.y | mask) : (code.y & ~(mask));
	code.z = choice.z ? (code.z | mask) : (code.z & ~(mask));
}

// Get the localization choice at the give depth as a int3
inline __host__ __device__ 
int3 getLocalizationChoiceAtDepth(const int depth, const NodeLocalizationCode & code)
{
	const int mask = 1 << depth;

	int3 choice;
	choice.x = (code.x & mask) ? 1 : 0;
	choice.y = (code.y & mask) ? 1 : 0;
	choice.z = (code.z & mask) ? 1 : 0;

	return choice;
}

// Calculate the nodes bounding box from a localization code and depth
//
// TODO: i'm not too happy about inlining this... but i want it 
//       in both cuda and cpp
//
inline __host__ __device__ 
NodeLocalization getLocalizationFromDepthAndCode(const int depth, const NodeLocalizationCode & code)
{
	NodeLocalization local;
	local.bboxMin = make_float3(0.0f, 0.0f, 0.0f);
	local.bboxMax = make_float3(1.0f, 1.0f, 1.0f);
	local.depth   = 0;

	float3 bboxDim;
	int3 choice;

	for (local.depth = 0; local.depth < depth; ++local.depth) {

		// Next dimension is half the size of this one
		bboxDim = (local.bboxMax - local.bboxMin) * 0.5f;

		// Get the choice at the current depth
		choice = getLocalizationChoiceAtDepth(local.depth, code);

		// Min is is changed in positive direction for every positive choice
		// Max is is changed in negative direction for every non positive choice
		local.bboxMin += bboxDim * make_float3(choice);
		local.bboxMax -= bboxDim * make_float3(choice.x ? 0.0f : 1.0f, choice.y ? 0.0f : 1.0f, choice.z ? 0.0f : 1.0f);
	}
	
	return local;
}

// These functions should not be inlined, and is only available to the device
#if defined(__CUDACC__)


// Consider
// http://stackoverflow.com/questions/10492590/how-to-calculate-the-viewing-cone-radius-ie-size-of-a-pixel-at-a-distance-in
// http://www.rc-astro.com/resources/reducer.html
// 
// A plot of the intersections:
// http://fooplot.com/plot/7hbtxb7sv1
//

inline __device__ 
NodeLocateResult locateNodeId(const Octree & octree, const float3 xpos, const int maxDepth, const float pixelSizeADepth)
{
	//const float pixelSize = 0.01f;
	//const float pixelSize = 0.05f;
	//const float pixelSize = 0.00125f;

	//const float pixelSize = 0.5f;

	NodeLocateResult r;
	r.current = 0;
	r.parent = 0;
	r.pos = xpos;
	r.parentPos = xpos;

	//int N = 0;

	
	for (int depth = 0; depth <= maxDepth; ++depth) {

		// Unpack the node child data
		NodeChildData nodeChildData = unpackNodeChildData(octree.dev_childData[r.current]);

		// Update the timestamp for the tile, ie. flag it as used this frame
		octree.dev_tileTimestamp[getTileTimestampIndexFromNodeTileAddress(r.current)] = octree.timestamp;

		if (nodeChildData.flagMaxSubdivision == FLAG_CANNOT_SUBDIVIDE) { 
            break;
        }

		// Test the voxels pixels size against the pixel
		// size at the current depth
		const float voxelPixelSize = __powf(0.5f, depth) / BRICK_SIZE;
		//const float voxelPixelSize = octree.dev_nodeVoxelPixelSize[depth];
		if (voxelPixelSize < pixelSizeADepth) {
			break;
		}

		// If the node can be sudivided, but there is no 
		// childAddress, a subdivided request is emitted
		if (nodeChildData.childAddress == 0) {
			octree.dev_raymarcherRequests[r.current] = r.current;
            break;
        }

		// Offsets must not exceed 1
		// [0.6, 0.1, 0.7] -> [1.2, 0.2, 1.4] -> [1 ,0, 1]
		int3 offset = make_int3(min((int)(r.pos.x * 2), 1), min((int)(r.pos.y * 2), 1), min((int)(r.pos.z * 2), 1)); 
	
		// Update next search node
		// [1 ,0, 1] -> (1 + 2 * 0 + 4 * 1) = 5
		r.parent = r.current;
		r.current = nodeChildData.childAddress + offset.x + 2 * offset.y + 4 * offset.z; 

		r.parentPos = r.pos;

		// Update position for next interation
		// [1.2, 0.2, 1.4] - [1 ,0, 1] -> [0.2, 0.2, 0.4]
		r.pos = r.pos * 2 - make_float3(offset.x, offset.y, offset.z);
	}

	return r;
}

#endif

#endif