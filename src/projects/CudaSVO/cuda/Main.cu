#include "main.h"

#include <stdio.h>
#include <iostream>

#include <Framework/GL.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/fill.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

#include <CudaFramework/cuda/Math.h>
#include <CudaFramework/cuda/ViewFrustum.h>
#include <CudaFramework/cuda/Matrix.h>
#include <CudaFramework/cuda/Ray.h>
#include <CudaFramework/cuda/PerlinNoise.h>


#include <CudaFramework/CudaUtil.h>
#include <CudaFramework/CudaTimerCollection.h>

#include "Predicates.h"
#include "Timer.h"

surface<void, cudaSurfaceType3D> densitySurface;	                                   // Writable 3d surface
surface<void, cudaSurfaceType3D> colorSurface;										   // Writable 3d surface

texture<VoxelDensityType, cudaTextureType3D, cudaReadModeElementType> densityTexture;  // Readable texture with hardware interpolation
texture<VoxelColorType, cudaTextureType3D, cudaReadModeNormalizedFloat> colorTexture;  // Readable texture with hardware interpolation

Octree                        * octree;
restless::CudaTimerCollection * timer;


__device__ 
float3 lookup_gradient_texture_central_difference(const float3 p)
{
	const float step = 1.0f / VOLUME_DIMENSION;

	float x0 = tex3D(densityTexture, p.x - step, p.y,        p.z       );
	float x1 = tex3D(densityTexture, p.x + step, p.y,        p.z       );
	float y0 = tex3D(densityTexture, p.x,        p.y - step, p.z       );
	float y1 = tex3D(densityTexture, p.x,        p.y + step, p.z       );
	float z0 = tex3D(densityTexture, p.x,        p.y,        p.z - step);
	float z1 = tex3D(densityTexture, p.x,        p.y,        p.z + step);

	return normalize(make_float3(
		(x0 - x1), 
		(y0 - y1), 
		(z0 - z1)  
	));
}

__inline__ __device__
float4 evaluate_terrain_voxel(float3 pos, const int depth)
{
	float4 voxel = make_float4(0);

	float3 warpPos = pos;
	warpPos.x = noise3(pos * 2.44f);
	warpPos.y = noise3(pos * 1.2f);
	warpPos.z = noise3(pos * 1.04f);

	float3 warpPos2 = pos;
	warpPos2.x = noise3(pos * 32.44f);
	warpPos2.y = noise3(pos * 42.2f);
	warpPos2.z = noise3(pos * 42.04f);

	voxel.w -= pos.y * 2.0f - 1.0f;													// [0..1] -> [-1..1] -> [1..-1]
	voxel.w -= -0.3f;																// put the "ground" -0.3 below the middle of the volume
	voxel.w += noise3(warpPos * 2.53f)  * 0.40f;									// Some low frequency noise
	voxel.w += noise3(warpPos * 10.03f) * 0.20f;									// Some higher frequency noise
	voxel.w -= perlinNoise3d(pos.x, pos.y, pos.z, 3, 50.0f, 0.01f);					// Three octaves of volume noise (high frequency)
	voxel.w -= perlinNoise3d(warpPos2.x, warpPos2.y, warpPos2.z, 4, 1.0f, 0.01f);	// Four octaves of warped volume noise (low frequency)

	float marbA = __sinf((warpPos.y + 3.0f * perlinNoise3d(warpPos.x, warpPos.y, warpPos.z, 5, 20.4f, 1.0f)) * 3.1415926f);
	float marbB = __sinf((pos.z + 3.0f * perlinNoise3d(pos.x, pos.y, pos.z, 5, 10.4f, 1.0f)) * 3.1415926f);
	float marbC = __sinf((pos.z + 3.0f * perlinNoise3d(pos.x, pos.y, pos.z, 4, 1.2f, 1.0f)) * 3.1415926f);

	voxel.w -= marbA * 0.002f;
	voxel.w -= marbB * 0.005f;
	voxel.w -= marbC * 0.004f;

	voxel.x = 0.77f + 0.1f * clamp(marbA * marbB * marbC, 0.0f, 1.0f);
	voxel.y = 0.62f + 0.1f * clamp(marbB, 0.0f, 1.0f);
	voxel.z = 0.44f + 0.1f * clamp(marbC, 0.0f, 1.0f);

	float warpNoise = perlinNoise3d(warpPos2.x, warpPos2.y, warpPos2.z, 6, 5.0f, 0.3f);
	voxel.x -= warpNoise;
	voxel.y -= warpNoise;
	voxel.z -= warpNoise;

	voxel.x = clamp(voxel.x, 0.0f, 1.0f);
	voxel.y = clamp(voxel.y, 0.0f, 1.0f);
	voxel.z = clamp(voxel.z, 0.0f, 1.0f);
	voxel.w -= warpNoise * 0.02f;

	return voxel;
}

__global__ void kernel_render_pixelbuffer(const Octree octree, const RenderInput input)
{
	TIMER_TIC(timer_raymarcher_all)

	// These are correct, this is infact the x and y coordinates of this thread
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Stay inside the image
	if (x >= input.buffer.width || y >= input.buffer.height ) {
		TIMER_TOC(timer_raymarcher_all)
		return;
	}

	TIMER_TIC(timer_raymarcher_raysetup)

	// Offset in pixelbuffer
	const int offset = x + y * input.buffer.width;

	// Bounding box of the volume, changing this requries 
	// changes in how the ray is mapped inside the octree
	const float3 boxMin = make_float3( 0.0f, 0.0f, 0.0f);
	const float3 boxMax = make_float3( 1.0f, 1.0f, 1.0f);

	// Find the point on the near plane where the ray hits
	const float u =  input.viewFrustum.right * ((x / (float) input.buffer.width) * 2.0f - 1.0f);
	const float v = -input.viewFrustum.top * ((y / (float) input.buffer.height) * 2.0f - 1.0f);

	// Setup a ray 
	Ray ray;
	ray.o = make_float3(mul(input.viewMatrixInverse, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
	ray.d = make_float3(mul(input.viewMatrixInverse, make_float4(u, v, -input.viewFrustum.near, 0.0f)));
	ray.d = normalize(ray.d);

	TIMER_TOC(timer_raymarcher_raysetup)

	// Determine if, and where, the ray hits the bounding volume
	// of the octree
	float tNear, tFar;
	TIMER_TIC(timer_raymarcher_global_intesect)
	if (!intersectBox(ray, boxMin, boxMax, & tNear, & tFar)) {
		input.buffer.ptr[offset].x = 0.0f;
		input.buffer.ptr[offset].y = 0.0f;
		input.buffer.ptr[offset].z = 0.0f;
		input.buffer.ptr[offset].w = 0.0f;
		TIMER_TOC(timer_raymarcher_all)
		TIMER_TOC(timer_raymarcher_global_intesect)
		return;
	}
	TIMER_TOC(timer_raymarcher_global_intesect)

	// Clamp to near plane
	if (tNear < 0.0f) { 
		tNear = 0.0f;
	}

	// Start at the near intersection point
	float t = tNear;

	// Brickspace accounts for borders on every side of the node
	const float3 brickSpace = make_float3(BRICK_SIZE - 2 * BRICK_BORDER, BRICK_SIZE - 2 * BRICK_BORDER, BRICK_SIZE - 2 * BRICK_BORDER);
	const float3 brickOffset = make_float3(BRICK_BORDER, BRICK_BORDER, BRICK_BORDER);
	// Volume dim used to map to texture space
	const float3 volumeDim = make_float3(octree.volumeSize.width, octree.volumeSize.height, octree.volumeSize.depth);
	// step size inside brick [0..BRICK_SIZE-2] -> [0..1]
	const float  stepLength  = input.stepSize; 

	// The color will be summed into this float
	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// The numer of steps here is the maximum number of nodes
	// to visit along the ray. This should not be necessary 
	// to have this as a simple while loop should do, but 
	// there are sometimes issues where the step does not 
	// progress along the ray...
	//for (unsigned int step = 0; step < 100; ++step) {
	TIMER_TIC(timer_raymarcher_inner)
	while (true) {

		// Outside the volume
		if (t >= tFar) {
			break;
		}

		// Opaque
		if (sum.w > 0.95f) {
			break;
		}

		// Determine the position along the ray according the the 
		// current step = t. At the first iteration this is the 
		// intesection with the bounding box of the entire volume.
		// Later on the it the intersection with the next node 
		// along the ray
		float3 pos = ray.o + ray.d * t;

		// Calculate the pixel size at distance t
		const float tPixelSize = (t * input.pixelSizeOnNearPlane) / input.viewFrustum.near;

		// Find the node in the octree that matches the position and 
		// fits the LOD needed at this distance
		TIMER_TIC(timer_raymarcher_lookup)
		NodeLocateResult r = locateNodeId(octree, pos, input.maxDepth, tPixelSize);
		TIMER_TOC(timer_raymarcher_lookup)

		TIMER_TIC(timer_raymarcher_localize)
		// Determine the nodes bounding box from depth and localization code
		int localDepth = unpackNodeLocalizationDepth(octree.dev_localizationDepth[r.current]);
		NodeLocalizationCode localCode = unpackNodeLocalizationCode(octree.dev_localizationCode[r.current]);
		NodeLocalization local = getLocalizationFromDepthAndCode(localDepth, localCode);
		TIMER_TOC(timer_raymarcher_localize)
		

		// Find the intersection of the ray with the bounding box of the node
		// ... this should it as the position has already been proven to be
		//     inside the node, but it might miss due to numeric errors
		//     in which case the color becomes yellow to single an error
		float tLocalNear, tLocalFar;
		TIMER_TIC(timer_raymarcher_local_intesect)
		if (!intersectBox(ray, local.bboxMin, local.bboxMax, & tLocalNear, & tLocalFar)) {
			//sum.x = 1.0f;
			//sum.y = 0.0f;
			//sum.z = 0.0f;
			//sum.w = 1.0f;
			t = t + ESCAPE_EPSILON;
			TIMER_TOC(timer_raymarcher_local_intesect)
			continue;
		}
		TIMER_TOC(timer_raymarcher_local_intesect)

		// In case the intersection with the nodes bounding volume is actually
		// behind (or very close to) the current step along the ray, something 
		// fishy has happened. Most likely a grazing angle hit on a bounding 
		// box.
		// In this case the far intersection is moved along the ray by a small
		// amount. This should result in the next intersection being the next 
		// node along the ray (or take us outside the volume)
		if (tLocalFar - t < ESCAPE_EPSILON) {
			t = t + ESCAPE_EPSILON; // TODO: 20? ... 2? .... reason?
			continue;
		}

		// Don't start the march behind the 
		// the original step position
		if (tLocalNear < t) {
			tLocalNear = t;
		}

		NodeChildData nodeChildData = unpackNodeChildData(octree.dev_childData[r.current]);

		// If the node is constant, we ignore it and just step out and 
		// proceed to the next node
		if (nodeChildData.flagDataType == FLAG_DATA_TYPE_CONSTANT) {

			TIMER_TIC(timer_raymarcher_march_constant)

			NodeConstantColor color = unpackNodeConstantColor(octree.dev_brickData[r.current]);

			color.x *= color.w;
			color.y *= color.w;
			color.z *= color.w;
			sum = sum + color * (1.0f - sum.w);

			t = tLocalFar;

			TIMER_TOC(timer_raymarcher_march_constant)

			continue;
		}

		TIMER_TIC(timer_raymarcher_selected_section)

		// Get the pointer to the brick and update the timestamp 
		// for the brick, ie. flag it as used this frame
		float3 nodeBrickPointer = make_float3(unpackNodeBrickPointer(octree.dev_brickData[r.current]));
		octree.dev_brickTimestamp[getBrickTimestampIndexFromBrickPointer(nodeBrickPointer)] = octree.timestamp;

		TIMER_TOC(timer_raymarcher_selected_section)

		TIMER_TIC(timer_raymarcher_march_brick)

		if (input.renderMode == 2) {

			float3 localPos = r.pos;
			float4 color = make_float4(r.pos.x, r.pos.y, r.pos.z, 1.0f);

			sum = sum + color * (1.0f - sum.w);

			if (sum.w > 0.95f) {
				break;
			}

		} else {
		
			// For some reason c_nodeLength[localDepth] != local.bboxMax.x - local.bboxMin.x
			// TODO: this requires investigation... for now use diagonal of bounding box
			const float nodeLenght = __powf(0.5f, localDepth);
			const float tDist = (tLocalFar - tLocalNear) / nodeLenght;
		

			for (float tLocal = 0; tLocal < tDist; tLocal += stepLength) {

				const float3 localPos = r.pos + ray.d * tLocal;
			
				if (localPos.x < 0.0f || localPos.x > 1.0f || 
					localPos.y < 0.0f || localPos.y > 1.0f || 
					localPos.z < 0.0f || localPos.z > 1.0f) {
					break;
				}
			

				const float3 nodeTexPos = (localPos * brickSpace + brickOffset + nodeBrickPointer) / volumeDim;

				float density = tex3D(densityTexture, nodeTexPos.x, nodeTexPos.y, nodeTexPos.z);

				if (density > DENSITY_THRESHOLD_AIR) {

					const float opacity = 1.0f;

					float4 color = input.renderMode == 0 ? tex3D(colorTexture, nodeTexPos.x, nodeTexPos.y, nodeTexPos.z)
														 : make_float4(localPos.x, localPos.y, localPos.z, 1.0f);

					float lightIntisity = 0.7f;

					if (input.light.enabled) {

						const float  ambient  = 0.1f;
						const float3 normal   = lookup_gradient_texture_central_difference(nodeTexPos);
						const float3 lightDir = normalize(input.light.worldPosition - localPos); // This is not correct and most like the  source of lighting errors, should be world pos not local pos.
						const float  diffuse  = max(dot(lightDir, normal), 0.0f); 

						// Specular disabled
						// const float3 viewDir    = normalize(ray.o - localPos);
						// const float3 halfVector = normalize(lightDir + viewDir);
						// const float specular = pow(clamp(dot(halfVector, normal), 0.0f, 1.0f), input.light.shininess);
						
						lightIntisity = min(diffuse + ambient, 1.0f);

					}

					color.w = opacity;
					color.x *= color.w * lightIntisity;
					color.y *= color.w * lightIntisity;
					color.z *= color.w * lightIntisity;
				

					sum = sum + color * (1.0f - sum.w);

					if (sum.w > 0.95f) {
						break;
					}

				}
			}
		}

		TIMER_TOC(timer_raymarcher_march_brick)

		// Step out of the node with a small escape epsilon 
		// this should take the ray into the next node
		t = tLocalFar + ESCAPE_EPSILON;
	}
	TIMER_TOC(timer_raymarcher_inner)

	input.buffer.ptr[offset].x = sum.x;
	input.buffer.ptr[offset].y = sum.y;
	input.buffer.ptr[offset].z = sum.z;
	input.buffer.ptr[offset].w = sum.w;

	TIMER_TOC(timer_raymarcher_all)
}

__global__ void kernel_update_tile_usage_mask(Octree octree)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int usageIndex = x + y * blockDim.x * gridDim.x; 


	if (usageIndex >= MAX_NODE_TILES) {
		return;
	}

	// Get the tile usage information
	TileUsage tileUsage = unpackTileUsage(octree.dev_tileUsage[usageIndex]);

	// Get the tile last used timestamp
	// NOTE: tileUsage.tileAddress contains tile addresses node 
	//       buffer, ie. multiples of 8. However we need to 
	//       convert this to a index into the timestamp buffer 
	//       which only index contains the tiles, so the address
	//       is divided by eight.
	int tileTs = octree.dev_tileTimestamp[getTileTimestampIndexFromNodeTileAddress(tileUsage.tileAddress)];

	// The tile is flag as used if the if it was touched during this frame
	
	//tileUsage.flag = tileTs >= octree.timestamp;
	tileUsage.flag = tileTs > 0;

	// Pack the usage data back
	octree.dev_tileUsage[usageIndex] = packTileUsage(tileUsage);
}

__global__ void kernel_update_brick_usage_mask(Octree octree)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int usageIndex = x + y * blockDim.x * gridDim.x; 

	if (usageIndex >= MAX_BRICKS) {
		return;
	}

	BrickUsage brickUsage = unpackBrickUsage(octree.dev_brickUsage[usageIndex]); 

	int brickTs = octree.dev_brickTimestamp[getBrickTimestampIndexFromBrickPointer(brickUsage.brickPointer)];

	//brickUsage.flag = brickTs >= octree.timestamp;
	brickUsage.flag = brickTs > 0;

	// Pack the usage data back
	octree.dev_brickUsage[usageIndex] = packBrickUsage(brickUsage);
}

__global__ void kernel_invalidate_timestamps_lru(Octree octree)
{
	int rx = threadIdx.x + blockIdx.x * blockDim.x;
	int ry = threadIdx.y + blockIdx.y * blockDim.y;
	int requestIndex = rx + ry * blockDim.x * gridDim.x;

	if (requestIndex >= MAX_SUBDIVIDE_REQUESTS_PER_UPDATE) {
		return;
	}

	int N = octree.dev_raymarcherRequests[requestIndex];
	if (N < 0) {
		return;
	}

	// Mark the tile as recycled by setting its timestamp to 0
	TileUsage tileUsage = unpackTileUsage(octree.dev_tileUsage[requestIndex]);
	octree.dev_tileTimestamp[getTileTimestampIndexFromNodeTileAddress(tileUsage.tileAddress)] = 0;

	// Mark all brick pointers that will be recycled with a 0 timestamp
	for (int offset = 0; offset < 8; ++offset) {
		BrickUsage brickUsage = unpackBrickUsage(octree.dev_brickUsage[requestIndex + 8 * offset]);
		octree.dev_brickTimestamp[getBrickTimestampIndexFromBrickPointer(brickUsage.brickPointer)] = 0;
	}

}

__global__ void kernel_invalidate_nodes_lru(Octree octree)
{
	int nx = threadIdx.x + blockIdx.x * blockDim.x;
	int ny = threadIdx.y + blockIdx.y * blockDim.y;
	int nodeIndex = nx + ny * blockDim.x * gridDim.x;

	if (nodeIndex >= MAX_NODES) {
		return;
	}

	NodeChildData nodeChildData = unpackNodeChildData(octree.dev_childData[nodeIndex]);

	if (nodeChildData.flagDataType == FLAG_DATA_TYPE_BRICK) {

		NodeBrickPointer nodeBrickPointer = unpackNodeBrickPointer(octree.dev_brickData[nodeIndex]);

		int brickTimestamp = octree.dev_brickTimestamp[getBrickTimestampIndexFromBrickPointer(nodeBrickPointer)];

	}

	
}

__global__ void kernel_handle_subdivide_requests_lru(Octree octree)
{
	int rx = threadIdx.x + blockIdx.x * blockDim.x;
	int ry = threadIdx.y + blockIdx.y * blockDim.y;
	int requestIndex = rx + ry * blockDim.x * gridDim.x;

	if (requestIndex >= MAX_SUBDIVIDE_REQUESTS_PER_UPDATE) {
		return;
	}

	// Get the node address from the request index, negative requests 
	// are empty
	int N = octree.dev_raymarcherRequests[requestIndex];
	if (N < 0) {
		return;
	}

	// Get the depth of the node
	int nodeDepth = unpackNodeLocalizationDepth(octree.dev_localizationDepth[N]);

	// Stop at depth > MAX_DEPTH
	if (nodeDepth > MAX_DEPTH) {
		return;
	}

	// Get the child data and localization code for node N
	NodeChildData nodeChildData = unpackNodeChildData(octree.dev_childData[N]);
	NodeLocalizationCode nodeLocalizationCode = unpackNodeLocalizationCode(octree.dev_localizationCode[N]);

	// Node already has children
	if (nodeChildData.childAddress != 0) {
		return;
	}

	// No subdivision available
	if (nodeChildData.flagMaxSubdivision == FLAG_CANNOT_SUBDIVIDE) {
		return;
	}

	// Grab the tile offset from somewhere in the front of the usage cache
	TileUsage tileUsage = unpackTileUsage(octree.dev_tileUsage[requestIndex]);
	int childTileOffset = tileUsage.tileAddress;

	// Update child address and repack the data
	nodeChildData.childAddress = childTileOffset;
	octree.dev_childData[N] = packNodeChildData(nodeChildData);

	// HACK: mark the tile as used
	octree.dev_tileTimestamp[getTileTimestampIndexFromNodeTileAddress(nodeChildData.childAddress)] = octree.timestamp;

	for (int offset = 0; offset < 8; ++offset) {

		int x = (offset & 1) == 1;
		int y = (offset & 2) == 2;
		int z = (offset & 4) == 4;

		int childOffset = childTileOffset + offset;

		 // one level deeper than parent
		int childDepth = nodeDepth + 1;

		// Copy parents localization choices
		// .... and update with current choice
		// !!!! NOTE: the choice is made at parent level, not child level!!!!!
		NodeLocalizationCode childLocalCode;
		childLocalCode.x = nodeLocalizationCode.x;
		childLocalCode.y = nodeLocalizationCode.y;
		childLocalCode.z = nodeLocalizationCode.z;
		setLocalizationChoiceAtDepth(nodeDepth, childLocalCode, make_int3(x, y, z));

		// All nodes are bricks until proven otherwise
		NodeChildData childChildData;
		childChildData.childAddress       = 0;
		childChildData.flagDataType       = FLAG_DATA_TYPE_BRICK;		
		childChildData.flagMaxSubdivision = nodeDepth == MAX_DEPTH;

		// Grab a brick address (it might not be used, but this will be 
		// evalvuated in another kernel)
		//int childBrickData = octree.dev_brickUsage[requestIndex * 8 + offset];
		BrickUsage childBrickUsage = unpackBrickUsage(octree.dev_brickUsage[requestIndex * 8 + offset]);
		NodeBrickPointer childBrickPointer = childBrickUsage.brickPointer;

		// Store child data
		octree.dev_childData[childOffset]         = packNodeChildData(childChildData);
		octree.dev_brickData[childOffset]         = packNodeBrickPointer(childBrickPointer);
		octree.dev_localizationCode[childOffset]  = packNodeLocalizationCode(childLocalCode);
		octree.dev_localizationDepth[childOffset] = packNodeLocalizationDepth(childDepth);

		// Emit a request for child brick data
		octree.dev_dataRequests[requestIndex * 8 + offset] = childOffset;
	}
}

__global__ void kernel_handle_data_requests_lru(const int requestsPerSide, Octree octree)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	int rx = x / BRICK_SIZE;   // x=17 and BRICK_SIZE=8 -> 17/8 = 2 (ie. the third brick request)
	int ry = y / BRICK_SIZE;   // y=4  and BRICK_SIZE=8 -> 4/8  = 0 (ie. the first brick request)
	int rz = z / BRICK_SIZE;   // z=56 and BRICK_SIZE=8 -> 56/8 = 7 (ie. the seventh brick request)

	int requestIndex = (rx * requestsPerSide * requestsPerSide + ry * requestsPerSide + rz);
	
	int N = octree.dev_dataRequests[requestIndex];
	
	if (N < 0) {
		return;
	}

	// Find the "position" inside the brick [0..1]^3
	float3 fPos = make_float3(
		((float) x / BRICK_SIZE) - rx,		// x=17 -> 0.125
		((float) y / BRICK_SIZE) - ry,		// y=4  -> 0.5
		((float) z / BRICK_SIZE) - rz		// z=56 -> 0.0
	);

	int localizationDepth = unpackNodeLocalizationDepth(octree.dev_localizationDepth[N]);
	NodeLocalizationCode localizationCode = unpackNodeLocalizationCode(octree.dev_localizationCode[N]);
	NodeLocalization localization = getLocalizationFromDepthAndCode(localizationDepth, localizationCode);
	
	NodeBrickPointer brickPointer = unpackNodeBrickPointer(octree.dev_brickData[N]);
	
	// Account for brick borders
	const float3 bboxDim = localization.bboxMax - localization.bboxMin;
	const float3 bboxStep = bboxDim / (BRICK_SIZE - (BRICK_BORDER * 2));
	const float3 wrapBoxMin = localization.bboxMin - bboxStep * BRICK_BORDER;
	const float3 wrapBoxMax = localization.bboxMax + bboxStep * BRICK_BORDER;
	const float3 wrapBoxDim = wrapBoxMax - wrapBoxMin;

	// Find the position in the volume
	float3 volumePosition = wrapBoxMin + wrapBoxDim * fPos;

	float4 voxel = evaluate_terrain_voxel(volumePosition, localizationDepth);
	// Calcuate the offset into the volume texture
	int3 brickPosition = brickPointer + make_int3(fPos * BRICK_SIZE);
	
	float density = voxel.w;
	uchar4 color  = make_uchar4(voxel.x * 255, voxel.y * 255, voxel.z * 255, 0);

	surf3Dwrite(density, densitySurface, brickPosition.x * sizeof(float), brickPosition.y, brickPosition.z);
	surf3Dwrite(color, colorSurface, brickPosition.x * sizeof(uchar4), brickPosition.y, brickPosition.z);
}

__global__ void kernel_handle_finalize_data_requests_lru(Octree octree)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int requestIndex = x + y * blockDim.x * gridDim.x;

	if (requestIndex >= MAX_DATA_REQUESTS_PER_UPDATE) {
		return;
	}

	int N = octree.dev_dataRequests[requestIndex];
	if (N < 0) {
		return;
	}

	NodeChildData childData = unpackNodeChildData(octree.dev_childData[N]);
	NodeBrickPointer brickPointer = unpackNodeBrickPointer(octree.dev_brickData[N]);

	float maxDensity = -1000.0f;
	float minDensity =  1000.0f;

	float4 avarageColor = make_float4(0);

	for (int bx = 0; bx < BRICK_SIZE; ++bx) {
		for (int by = 0; by < BRICK_SIZE; ++by) {
			for (int bz = 0; bz < BRICK_SIZE; ++bz) {

				int3 brickPosition = brickPointer + make_int3(bx, by, bz);

				float3 texPos = make_float3(
					(float) brickPosition.x / octree.volumeSize.width,
					(float) brickPosition.y / octree.volumeSize.height,
					(float) brickPosition.z / octree.volumeSize.depth
				);

				float density = tex3D(densityTexture, texPos.x, texPos.y, texPos.z);
				maxDensity = max(maxDensity, density);
				minDensity = min(minDensity, density);

				avarageColor += tex3D(colorTexture, texPos.x, texPos.y, texPos.z);
			}
		}
	}

	

	if (maxDensity < DENSITY_THRESHOLD_AIR) {
		childData.flagDataType = FLAG_DATA_TYPE_CONSTANT;
		childData.flagMaxSubdivision = FLAG_CANNOT_SUBDIVIDE;

		NodeConstantColor constantColor;
		constantColor.x = 0.0f;
		constantColor.y = 0.0f;
		constantColor.z = 0.0f;
		constantColor.w = 0.0f;

		octree.dev_brickData[N] = packNodeConstantColor(constantColor);
		octree.dev_childData[N] = packNodeChildData(childData);
		return;
	}

	if (minDensity > DENSITY_THRESHOLD_SOLID) {
		childData.flagDataType = FLAG_DATA_TYPE_CONSTANT;
		childData.flagMaxSubdivision = FLAG_CANNOT_SUBDIVIDE;

		avarageColor /= BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;

		NodeConstantColor constantColor;
		constantColor.x = avarageColor.x;
		constantColor.y = avarageColor.y;
		constantColor.z = avarageColor.z;
		constantColor.w = 1.0f;

		octree.dev_brickData[N] = packNodeConstantColor(constantColor);
		octree.dev_childData[N] = packNodeChildData(childData);
		return;
	}

	// Signal that the brick is used (it will not be claimed emmidiately
	octree.dev_brickTimestamp[getBrickTimestampIndexFromBrickPointer(brickPointer)] = octree.timestamp;
}

extern "C"
{
	void cudaInitializePerlinNoise(unsigned int seed)
	{
		cudaError_t cudaErrorId = perlinNoiseInitialize(seed);
		if (cudaErrorId != cudaSuccess) {
			printf("perlinNoiseInitialize %d\n-> %s\n", (int)cudaErrorId, cudaGetErrorString(cudaErrorId));
			return;
		}
	}

	void cudaVolumeRenderToPixelBuffer(const RenderInput & input, const bool verbose)
	{
		timer->reset();

		timer->start(BIN_ALL);

		if (input.enableSubdivide) {

			timer->start(BIN_FILL_REQUEST_ARRAYS);

			thrust::device_ptr<int> dev_tRaymarcherRequests(octree->dev_raymarcherRequests);
			thrust::device_ptr<int> dev_tDataRequests(octree->dev_dataRequests);
			thrust::fill_n(dev_tRaymarcherRequests, MAX_NODES, -1);
			thrust::fill_n(dev_tDataRequests, MAX_DATA_REQUESTS_PER_UPDATE, -1);

			timer->stop(BIN_FILL_REQUEST_ARRAYS);

		}

		// Increase the timestamp by one
		octree->timestamp = octree->timestamp + 1;

		dim3 threads(8, 8);
		dim3 blocks((input.buffer.width + threads.x - 1)/threads.x, (input.buffer.height + threads.y - 1)/threads.y);

		#ifdef KERNEL_TIMERS_ENABLE
			cudaResetKernelTimers();
		#endif

		timer->start(BIN_RENDER);
		kernel_render_pixelbuffer<<<blocks,threads>>>(*octree, input);
		timer->stop(BIN_RENDER);
		if (!restless::cudasafe_post_kernel("kernel_render_pixelbuffer")) {
			return;
		}

		#ifdef KERNEL_TIMERS_ENABLE
			float timers[timer_count];
			cudaReadKernelTimers(timers);

			const float clockCycleToMsRatio = 1.0f / ((3000000000 / 1000) * (blocks.x * blocks.y));

			float tRaymarchAll = timers[timer_raymarcher_all] * clockCycleToMsRatio;
			float tRaymarchInner = timers[timer_raymarcher_inner] * clockCycleToMsRatio;
			float tRaymarchRaysetup = timers[timer_raymarcher_raysetup] * clockCycleToMsRatio;
			float tRaymarchGlobalIntersect = timers[timer_raymarcher_global_intesect] * clockCycleToMsRatio;
			float tRaymarchLookup = timers[timer_raymarcher_lookup] * clockCycleToMsRatio;
			float tRaymarchLocalize = timers[timer_raymarcher_localize] * clockCycleToMsRatio;
			float tRaymarchLocalIntersect = timers[timer_raymarcher_local_intesect] * clockCycleToMsRatio;
			float tRaymarchMarchConstant = timers[timer_raymarcher_march_constant] * clockCycleToMsRatio;
			float tRaymarchMarchBrick = timers[timer_raymarcher_march_brick] * clockCycleToMsRatio;
			float tRaymarchMarchSelectedSection = timers[timer_raymarcher_selected_section] * clockCycleToMsRatio;

			timer->setElapsed(BIN_KERNEL_ALL, tRaymarchAll);
			timer->setElapsed(BIN_KERNEL_INNER, tRaymarchInner);
			timer->setElapsed(BIN_KERNEL_RAYSETUP, tRaymarchRaysetup);
			timer->setElapsed(BIN_KERNEL_GLOBAL_INTERSECT, tRaymarchGlobalIntersect);
			timer->setElapsed(BIN_KERNEL_LOOKUP, tRaymarchLookup);
			timer->setElapsed(BIN_KERNEL_LOCALIZE, tRaymarchLocalize);
			timer->setElapsed(BIN_KERNEL_LOCAL_INTERSECT, tRaymarchLocalIntersect);
			timer->setElapsed(BIN_KERNEL_MARCH_CONSTANT, tRaymarchMarchConstant);
			timer->setElapsed(BIN_KERNEL_MARCH_BRICK, tRaymarchMarchBrick);
			timer->setElapsed(BIN_KERNEL_SELECTED_SECTION, tRaymarchMarchSelectedSection);
		#endif

		if (input.enableSubdivide) {

			// Compact raymarcher requests by partition, ie. put all non negative 
			// requests in the front of the request buffer
			thrust::device_ptr<int> dev_tRaymarcherRequests(octree->dev_raymarcherRequests);
			timer->start(BIN_SUBDIVIDE_REQUESTS_COMPACT);
			thrust::device_ptr<int> middle = thrust::partition(dev_tRaymarcherRequests, dev_tRaymarcherRequests + MAX_NODES-1,  is_not_negative());
			timer->stop(BIN_SUBDIVIDE_REQUESTS_COMPACT);

			// If the middle of the partition is not the front, there are some 
			// requests waiting to be processed
			if (middle != dev_tRaymarcherRequests) {

				int requestCount = min(middle - dev_tRaymarcherRequests, MAX_SUBDIVIDE_REQUESTS_PER_UPDATE);
				timer->setElapsed(BIN_COUNT_SUBDIVIDE_REQUEST, requestCount);

				timer->start(BIN_SUBDIVIDE_ALL);
				cudaOctreeUpdateUsageMasks(verbose);
				//cudaOctreeInvalidateNodes(verbose);
				cudaOctreeProcessRequests(verbose);
				timer->stop(BIN_SUBDIVIDE_ALL);
			}
		}

		timer->stop(BIN_ALL);
	}

	void cudaOctreeUpdateUsageMasks(const bool verbose)
	{
		
		{
			const int dimThread = 8;
			const int dimBlock = (ceil(sqrt((float)MAX_NODE_TILES)) + (dimThread - 1)) / dimThread;

			dim3 tileThreads(dimThread, dimThread);
			dim3 tileBlocks(dimBlock, dimBlock);

			if (verbose) {
				printf("\n");
				printf("\ttileThreads:     %d, %d\n", tileThreads.x, tileThreads.y);
				printf("\ttileBlocks:      %d, %d\n", tileBlocks.x, tileBlocks.y);
				printf("\n");
			}

			// Process the node tile usage mask by comparing the timestamp of the node tile
			// with the current timestamp. 
			timer->start(BIN_SUBDIVIDE_USAGE_MASK_TILE);
			kernel_update_tile_usage_mask<<<tileBlocks,tileThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_USAGE_MASK_TILE);

			if (!restless::cudasafe_post_kernel("kernel_update_tile_usage_mask")) {
				return;
			}
		}

		{
			const int dimThread = 8;
			const int dimBlock = (ceil(sqrt((float)MAX_BRICKS)) + (dimThread - 1)) / dimThread;

			dim3 brickThreads(dimThread, dimThread);
			dim3 brickBlocks(dimBlock, dimBlock);

			if (verbose) {
				printf("\n");
				printf("\tbrickThreads:     %d, %d\n", brickThreads.x, brickThreads.y);
				printf("\tbrickBlocks:      %d, %d\n", brickBlocks.x, brickBlocks.y);
				printf("\n");
			}

			// Process the brick usage mask by comparing the timestamp of the node tile
			// with the current timestamp. 
			timer->start(BIN_SUBDIVIDE_USAGE_MASK_BRICK);
			kernel_update_brick_usage_mask<<<brickBlocks,brickThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_USAGE_MASK_BRICK);

			if (!restless::cudasafe_post_kernel("kernel_update_brick_usage_mask")) {
				return;
			}
		}

		// Perform stream compaction by partitioning the tile usage and 
		// brick usage into [unused this frame] + [used the frame]. This
		// means that the first elements in dev_tileUsage and 
		// dev_brickUsage will be the elements that have been used least
		// recently.
		// TODO: stable_partion might be to much work, partition is ok
		timer->start(BIN_SUBDIVIDE_USAGE_MASK_COMPACT);
		thrust::device_ptr<int> dev_tTileUsage(octree->dev_tileUsage);
		thrust::device_ptr<int> dev_tBrickUsage(octree->dev_brickUsage);
		//thrust::stable_partition(dev_tTileUsage, dev_tTileUsage + MAX_NODE_TILES-1,  tile_usage_mask_is_zero());
		//thrust::stable_partition(dev_tBrickUsage, dev_tBrickUsage + MAX_BRICKS-1,  brick_usage_mask_is_zero());
		thrust::partition(dev_tTileUsage, dev_tTileUsage + MAX_NODE_TILES-1,  tile_usage_mask_is_zero());
		thrust::partition(dev_tBrickUsage, dev_tBrickUsage + MAX_BRICKS-1,  brick_usage_mask_is_zero());
		timer->stop(BIN_SUBDIVIDE_USAGE_MASK_COMPACT);

	}

	void cudaOctreeInvalidateNodes(const bool verbose)
	{

		{
			int dimThread = 8;
			int dimBlock  = (pow(MAX_SUBDIVIDE_REQUESTS_PER_UPDATE, 1.0/2.0) + (dimThread - 1)) / dimThread; // sqrt?

			dim3 invalidaTimestampThreads(dimThread, dimThread);
			dim3 invalidaTimestampBlocks(dimBlock, dimBlock);

			if (true) {
				printf("\n");
				printf("\tinvalidaTimestampThreads:     %d, %d\n", invalidaTimestampThreads.x, invalidaTimestampThreads.y);
				printf("\tinvalidaTimestampBlocks:      %d, %d\n", invalidaTimestampBlocks.x, invalidaTimestampBlocks.y);
				printf("\n");
			}

			timer->start(BIN_SUBDIVIDE_INVALIDATE_TIMESTAMPS);
			kernel_invalidate_timestamps_lru<<<invalidaTimestampBlocks,invalidaTimestampThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_INVALIDATE_TIMESTAMPS);
		}

		{
			int dimThread = 8;
			int dimBlock  = (pow(MAX_NODES, 1.0/2.0) + (dimThread - 1)) / dimThread; // sqrt?

			dim3 invalidateThreads(dimThread, dimThread);
			dim3 invalidateBlocks(dimBlock, dimBlock);

			if (true) {
				printf("\n");
				printf("\tinvalidateThreads:     %d, %d\n", invalidateThreads.x, invalidateThreads.y);
				printf("\tinvalidateBlocks:      %d, %d\n", invalidateBlocks.x, invalidateBlocks.y);
				printf("\n");
			}

			timer->start(BIN_SUBDIVIDE_INVALIDATE_NODES);
			kernel_invalidate_nodes_lru<<<invalidateBlocks,invalidateThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_INVALIDATE_NODES);
		}
	}

	void cudaOctreeProcessRequests(const bool verbose)
	{
		thrust::device_ptr<int> dev_tRaymarcherRequests(octree->dev_raymarcherRequests);
		thrust::device_ptr<int> dev_tDataRequests(octree->dev_dataRequests);

		// Handle subdivides
		{
			int dimThread = 8;
			int dimBlock  = (pow(MAX_SUBDIVIDE_REQUESTS_PER_UPDATE, 1.0/2.0) + (dimThread - 1)) / dimThread; // sqrt?

			dim3 subdivideThreads(dimThread,dimThread);
			dim3 subdivideBlocks(dimBlock, dimBlock);

			if (verbose) {
				printf("\n");
				printf("\tsubdivideThreads:     %d, %d, %d\n", subdivideThreads.x, subdivideThreads.y, subdivideThreads.z);
				printf("\tsubdivideBlocks:      %d, %d, %d\n", subdivideBlocks.x, subdivideBlocks.y, subdivideBlocks.z);
				printf("\n");
			}

			timer->start(BIN_SUBDIVIDE_REQUESTS_HANDLE);
			kernel_handle_subdivide_requests_lru<<<subdivideBlocks,subdivideThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_REQUESTS_HANDLE);
			if (!restless::cudasafe_post_kernel("kernel_handle_subdivide_requests_lru")) {
				return;
			}
		}

		// Handle data requests
		{
			int totalRequests   = MAX_DATA_REQUESTS_PER_UPDATE;				// for MAX_SUBDIVIDE_REQUESTS_PER_UPDATE=64 this is 512
			int requestsPerSide = ceil(pow(totalRequests, 1.0/3.0));		// for MAX_SUBDIVIDE_REQUESTS_PER_UPDATE=64 this is 8 (note double is used here for precession)
		
			// Calculate dimensions of the "cube" that will be evaluated
			int sideLength = BRICK_SIZE * requestsPerSide;					// MAX_SUBDIVIDE_REQUESTS_PER_UPDATE=64 and BRICK_SIZE=8 this is 64
			dim3 dataThreads(8,8,8);
			dim3 dataBlocks((sideLength+dataThreads.x-1)/dataThreads.x, (sideLength+dataThreads.y-1)/dataThreads.y, (sideLength+dataThreads.z-1)/dataThreads.z);

			if (verbose) {
				printf("\n");
				printf("\ttotalRequests:   %d\n", totalRequests);
				printf("\trequestsPerSide: %d\n", requestsPerSide);
				printf("\tsideLength:      %d\n", sideLength);
				printf("\tdataThreads:     %d, %d, %d\n", dataThreads.x, dataThreads.y, dataThreads.z);
				printf("\tdataBlocks:      %d, %d, %d\n", dataBlocks.x, dataBlocks.y, dataBlocks.z);
				printf("\n");
			}

			timer->start(BIN_SUBDIVIDE_REQUESTS_DATA);
			kernel_handle_data_requests_lru<<<dataBlocks,dataThreads>>>(requestsPerSide, *octree);
			timer->stop(BIN_SUBDIVIDE_REQUESTS_DATA);

			if (!restless::cudasafe_post_kernel("kernel_handle_data_requests_lru")) {
				return;
			}
			
		}

		// Finalize the results of the data requests
		{
			int dimThread = 16;
			int dimBlock  = (ceil(pow(MAX_DATA_REQUESTS_PER_UPDATE, 1.0/2.0)) + (dimThread - 1)) / dimThread;

			dim3 finalizeThreads(dimThread,dimThread);
			dim3 finalizeBlocks(dimBlock, dimBlock);

			if (verbose) {
				printf("\n");
				printf("\tdataThreads:     %d, %d\n", finalizeThreads.x, finalizeThreads.y);
				printf("\tdataBlocks:      %d, %d\n", finalizeBlocks.x, finalizeBlocks.y);
				printf("\n");
			}

			// Filter mode point, might not be necessary
			cudaTextureFilterMode oldFilterMode = densityTexture.filterMode;
			densityTexture.filterMode = cudaFilterModePoint;

			timer->start(BIN_SUBDIVIDE_REQUESTS_FINALIZE);
			kernel_handle_finalize_data_requests_lru<<<finalizeBlocks,finalizeThreads>>>(*octree);
			timer->stop(BIN_SUBDIVIDE_REQUESTS_FINALIZE);

			if (!restless::cudasafe_post_kernel("kernel_handle_finalize_data_requests_lru")) {
				return;
			}
			
			// Reset filter mode
			densityTexture.filterMode = oldFilterMode;
		}

	}

	void cudaOctreeSet(Octree & _octree)
	{
		octree = &(_octree);
	}

	void cudaSetTimerCollection(restless::CudaTimerCollection & timerCollection)
	{
		timer = &(timerCollection);
	}

}

//
// Getters for texture and surface references
//
extern "C" 
{
	const surfaceReference * cudaVolumeGetDensitySurfaceReference()
	{
		const surfaceReference * constSurfaceRefPtr;
		cudaError_t cudaErrorId = cudaGetSurfaceReference(& constSurfaceRefPtr, & densitySurface);
		if (cudaErrorId != cudaSuccess) {
			printf("cudaGetSurfaceReference(constSurfaceRefPtr, volumeSurface) returned %d\n-> %s\n", (int)cudaErrorId, cudaGetErrorString(cudaErrorId));
			return nullptr;
		}
		return constSurfaceRefPtr;
	}

	const surfaceReference * cudaVolumeGetColorSurfaceReference()
	{
		const surfaceReference * constSurfaceRefPtr;
		cudaError_t cudaErrorId = cudaGetSurfaceReference(& constSurfaceRefPtr, & colorSurface);
		if (cudaErrorId != cudaSuccess) {
			printf("cudaGetSurfaceReference(constSurfaceRefPtr, volumeSurface) returned %d\n-> %s\n", (int)cudaErrorId, cudaGetErrorString(cudaErrorId));
			return nullptr;
		}
		return constSurfaceRefPtr;
	}

	const textureReference * cudaVolumeGetDensityTextureReference()
	{
		const textureReference* constTexRefPtr = nullptr;
		cudaError_t cudaErrorId = cudaGetTextureReference(& constTexRefPtr, & densityTexture);
		if (cudaErrorId != cudaSuccess) {
			printf("cudaGetTextureReference(constTexRefPtr, volumeTexture) returned %d\n-> %s\n", (int)cudaErrorId, cudaGetErrorString(cudaErrorId));
			return nullptr;
		}
		return constTexRefPtr;
	}

	const textureReference * cudaVolumeGetColorTextureReference()
	{
		const textureReference* constTexRefPtr = nullptr;
		cudaError_t cudaErrorId = cudaGetTextureReference(& constTexRefPtr, & colorTexture);
		if (cudaErrorId != cudaSuccess) {
			printf("cudaGetTextureReference(constTexRefPtr, volumeTexture) returned %d\n-> %s\n", (int)cudaErrorId, cudaGetErrorString(cudaErrorId));
			return nullptr;
		}
		return constTexRefPtr;
	}

}
