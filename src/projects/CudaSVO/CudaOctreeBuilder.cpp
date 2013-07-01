#include "CudaOctreeBuilder.h"

#include <Framework/Util/Log.h>
#include <Framework/Util/Conversion.h>

#include <Mathlib/Mathlib.h>

#include <texture_types.h>
#include <channel_descriptor.h>
#include <vector>

#include <CudaFramework/CudaUtil.h>

#include "Util.h"

#include "cuda/main.h"

using namespace restless;

CudaOctreeBuilder::CudaOctreeBuilder() 
{
	done = false;

	octree.dev_childData         = nullptr;
	octree.dev_brickData         = nullptr;
	octree.dev_localizationCode  = nullptr;
	octree.dev_localizationDepth = nullptr;
}


CudaOctreeBuilder::~CudaOctreeBuilder()
{
	destroy();
}


void CudaOctreeBuilder::destroy()
{
	if (octree.dev_childData != nullptr) {
		cudasafe( cudaFree(octree.dev_childData), "cudaFree(octree.dev_childData)");
	}

	if (octree.dev_brickData != nullptr) {
		cudasafe( cudaFree(octree.dev_brickData), "cudaFree(octree.dev_childData)");
	}

	if (octree.dev_localizationCode != nullptr) {
		cudasafe( cudaFree(octree.dev_localizationCode), "cudaFree(octree.dev_childData)");
	}

	if (octree.dev_localizationDepth != nullptr) {
		cudasafe( cudaFree(octree.dev_localizationDepth), "cudaFree(octree.dev_childData)");
	}

	if (octree.dev_tileUsage != nullptr) {
		cudasafe( cudaFree(octree.dev_tileUsage), "cudaFree(octree.dev_tileUsage)");
	}

	if (octree.dev_tileTimestamp != nullptr) {
		cudasafe( cudaFree(octree.dev_tileTimestamp), "cudaFree(octree.dev_tileTimestamp)");
	}

	if (octree.dev_brickUsage != nullptr) {
		cudasafe( cudaFree(octree.dev_brickUsage), "cudaFree(octree.dev_brickUsage)");
	}

	if (octree.dev_brickTimestamp != nullptr) {
		cudasafe( cudaFree(octree.dev_brickTimestamp), "cudaFree(octree.dev_brickTimestamp)");
	}

	if (octree.dev_raymarcherRequests != nullptr) {
		cudasafe( cudaFree(octree.dev_raymarcherRequests), "cudaFree(octree.dev_raymarcherRequests)");
	}

	if (octree.dev_dataRequests != nullptr) {
		cudasafe( cudaFree(octree.dev_dataRequests), "cudaFree(octree.dev_dataRequests)");
	}

	const textureReference * constDensityTextureRefPtr = cudaVolumeGetDensityTextureReference();
	if (constDensityTextureRefPtr != nullptr) {
		cudasafe( cudaUnbindTexture(constDensityTextureRefPtr), "cudaUnbindTexture(constDensityTextureRefPtr)" );
	}

	const textureReference * constColorTextureRefPtr = cudaVolumeGetColorTextureReference();
	if (constColorTextureRefPtr != nullptr) {
		cudasafe( cudaUnbindTexture(constColorTextureRefPtr), "cudaUnbindTexture(constColorTextureRefPtr)" );
	}

	if (octree.dev_densityArray != nullptr) {
		cudasafe( cudaFreeArray(octree.dev_densityArray), "cudaFreeArray(octree.dev_densityArray)");
		octree.dev_densityArray = nullptr;
	}

	if (octree.dev_colorArray != nullptr) {
		cudasafe( cudaFreeArray(octree.dev_colorArray), "cudaFreeArray(octree.dev_colorArray)");
		octree.dev_colorArray = nullptr;
	}
}

void CudaOctreeBuilder::initialize()
{
	cudaInitializePerlinNoise(6745);
	
	initializeOctree();
	initializeVolume();
	initializeRootNode();

	cudaOctreeSet(octree);
	
	rebuild();
}

void CudaOctreeBuilder::rebuild()
{
	octree.timestamp = 1;

	// Clear all device arrays  (i think this is a good idea)
	cudasafe( cudaMemset(octree.dev_childData, 0, MAX_NODES * sizeof(int)), "memset (clear) dev_childData");
	cudasafe( cudaMemset(octree.dev_brickData, 0, MAX_NODES * sizeof(int)), "memset (clear) dev_brickData");
	cudasafe( cudaMemset(octree.dev_localizationCode, 0, MAX_NODES * sizeof(int)), "memset (clear) dev_localizationCode");
	cudasafe( cudaMemset(octree.dev_localizationDepth, 0, MAX_NODES * sizeof(char)), "memset (clear) dev_localizationDepth");
	cudasafe( cudaMemset(octree.dev_tileTimestamp, 0, MAX_NODE_TILES * sizeof(int)), "memset (clear) dev_tileTimestamp");
	cudasafe( cudaMemset(octree.dev_brickTimestamp, 0, MAX_BRICKS * sizeof(int)), "memset (clear) dev_brickTimestamp");
	//cudasafe( cudaMemset(octree.dev_subdivideRequests, -1, MAX_NODES * sizeof(int)), "memset (clear) dev_subdivideRequests");

	// Freshen the usage list
	initializeUsageLists();

	// Add the root node to the tree
	initializeRootNode();
}

void CudaOctreeBuilder::initializeVolume()
{
	cudaChannelFormatDesc densityChannelDesc = cudaCreateChannelDesc<VoxelDensityType>();
	cudaChannelFormatDesc colorChannelDesc = cudaCreateChannelDesc<VoxelColorType>();

	octree.volumeSize.width  = VOLUME_DIMENSION;
	octree.volumeSize.height = VOLUME_DIMENSION;
	octree.volumeSize.depth  = VOLUME_DIMENSION;

	cudasafe( cudaMalloc3DArray(& octree.dev_densityArray, &densityChannelDesc, octree.volumeSize, cudaArraySurfaceLoadStore), "cudaMalloc3DArray: density array" );
	cudasafe( cudaMalloc3DArray(& octree.dev_colorArray, &colorChannelDesc, octree.volumeSize, cudaArraySurfaceLoadStore), "cudaMalloc3DArray: color array" );

	cudasafe( cudaBindTextureToArray(cudaVolumeGetDensityTextureReference(), octree.dev_densityArray, & densityChannelDesc), "cudaBindTextureToArray: density" );
	cudasafe( cudaBindTextureToArray(cudaVolumeGetColorTextureReference(), octree.dev_colorArray, & colorChannelDesc), "cudaBindTextureToArray: color" );

	cudasafe( cudaBindSurfaceToArray(cudaVolumeGetDensitySurfaceReference(), octree.dev_densityArray, & densityChannelDesc), "cudaBindSurfaceToArray: density" );
	cudasafe( cudaBindSurfaceToArray(cudaVolumeGetColorSurfaceReference(), octree.dev_colorArray, & colorChannelDesc), "cudaBindSurfaceToArray: color" );

	textureReference* densityTexRef = const_cast<textureReference*>( cudaVolumeGetDensityTextureReference() );
	densityTexRef->addressMode[0] = cudaAddressModeBorder;
	densityTexRef->addressMode[1] = cudaAddressModeBorder;
	densityTexRef->addressMode[2] = cudaAddressModeBorder;
	densityTexRef->filterMode = cudaFilterModeLinear;
	densityTexRef->normalized = true;

	textureReference* colorTexRef = const_cast<textureReference*>( cudaVolumeGetColorTextureReference() );
	colorTexRef->addressMode[0] = cudaAddressModeBorder;
	colorTexRef->addressMode[1] = cudaAddressModeBorder;
	colorTexRef->addressMode[2] = cudaAddressModeBorder;
	colorTexRef->filterMode = cudaFilterModeLinear;
	colorTexRef->normalized = true;

	L_DEBUG << "";
	L_DEBUG << "Allocated volume array: ";
	L_DEBUG << "\toctree.volumeSize:       " << octree.volumeSize.width << "x" << octree.volumeSize.height << "x" << octree.volumeSize.depth;
	L_DEBUG << "\toctree.dev_densityArray: " << (void*) octree.dev_densityArray << " ~ " << bytesToMegabytes(sizeof(VoxelDensityType) * pow((float)VOLUME_DIMENSION, 3)) << " MB";
	L_DEBUG << "\toctree.dev_colorArray:   " << (void*) octree.dev_colorArray << " ~ " << bytesToMegabytes(sizeof(VoxelColorType) * pow((float)VOLUME_DIMENSION, 3)) << " MB";
	L_DEBUG << "";
}

void CudaOctreeBuilder::initializeOctree()
{
	size_t childDataSize          = MAX_NODES * sizeof(int);
	size_t brickDataSize          = MAX_NODES * sizeof(int);
	size_t localizationCodeSize   = MAX_NODES * sizeof(int);
	size_t localizationDepthSize  = MAX_NODES * sizeof(char);
	size_t tileTimestampSize      = MAX_NODE_TILES * sizeof(int);
	size_t tileUsageSize          = MAX_NODE_TILES * sizeof(int);
	size_t brickUsageSize         = MAX_BRICKS * sizeof(int);
	size_t brickTimestampSize     = MAX_BRICKS * sizeof(int);
	size_t raymarcherRequestsSize = MAX_NODES * sizeof(int);
	size_t dataRequestsSize       = MAX_DATA_REQUESTS_PER_UPDATE * sizeof(int);
	size_t nodeLenghtSize         = MAX_DEPTH * sizeof(float);
	size_t nodeVoxelPixelSizeSize = MAX_DEPTH * sizeof(float);

	cudasafe( cudaMalloc((void**) & octree.dev_childData, childDataSize), "malloc dev_childData");
	cudasafe( cudaMalloc((void**) & octree.dev_brickData, brickDataSize) , "malloc dev_brickData");
	cudasafe( cudaMalloc((void**) & octree.dev_localizationCode, localizationCodeSize), "malloc dev_localizationCode");
	cudasafe( cudaMalloc((void**) & octree.dev_localizationDepth, localizationDepthSize), "malloc dev_localizationDepth");
	cudasafe( cudaMalloc((void**) & octree.dev_tileTimestamp, tileTimestampSize), "malloc dev_tileTimestamp");
	cudasafe( cudaMalloc((void**) & octree.dev_tileUsage, tileUsageSize), "malloc dev_tileUsage");
	cudasafe( cudaMalloc((void**) & octree.dev_brickUsage, brickUsageSize), "malloc dev_brickUsage");
	cudasafe( cudaMalloc((void**) & octree.dev_brickTimestamp, brickTimestampSize), "malloc dev_brickTimestamp");
	cudasafe( cudaMalloc((void**) & octree.dev_raymarcherRequests, raymarcherRequestsSize), "malloc dev_raymarcherRequests");
	cudasafe( cudaMalloc((void**) & octree.dev_dataRequests, dataRequestsSize), "malloc dev_dataRequests");
	cudasafe( cudaMalloc((void**) & octree.dev_nodeLenght, nodeLenghtSize), "malloc dev_nodeLenght");
	cudasafe( cudaMalloc((void**) & octree.dev_nodeVoxelPixelSize, nodeVoxelPixelSizeSize), "malloc dev_nodeVoxelPixelSize");

	L_DEBUG << "";
	L_DEBUG << "Allocated device arrays for otcree: ";
	L_DEBUG << "\toctree.dev_childData:          " << (void*) octree.dev_childData << " " <<  bytesToMegabytes(childDataSize) << " MB ";
	L_DEBUG << "\toctree.dev_brickData:          " << (void*) octree.dev_brickData << " " << bytesToMegabytes(brickDataSize) << " MB ";
	L_DEBUG << "\toctree.dev_localizationCode:   " << (void*) octree.dev_localizationCode << " " << bytesToMegabytes(localizationCodeSize) << " MB ";
	L_DEBUG << "\toctree.dev_localizationDepth:  " << (void*) octree.dev_localizationDepth << " " << bytesToMegabytes(localizationDepthSize) << " MB ";
	L_DEBUG << "\toctree.dev_tileTimestamp:      " << (void*) octree.dev_tileTimestamp << " " << bytesToMegabytes(tileTimestampSize) << " MB ";
	L_DEBUG << "\toctree.dev_tileUsage:          " << (void*) octree.dev_tileUsage << " " << bytesToMegabytes(tileUsageSize) << " MB ";
	L_DEBUG << "\toctree.dev_brickUsage:         " << (void*) octree.dev_brickUsage << " " << bytesToMegabytes(brickUsageSize) << " MB ";
	L_DEBUG << "\toctree.dev_brickTimestamp:     " << (void*) octree.dev_brickTimestamp << " " << bytesToMegabytes(brickTimestampSize) << " MB ";
	L_DEBUG << "\toctree.dev_subdivideRequests:  " << (void*) octree.dev_raymarcherRequests << " " << bytesToMegabytes(raymarcherRequestsSize) << " MB ";
	L_DEBUG << "\toctree.dev_subdivideRequests:  " << (void*) octree.dev_dataRequests << " " << bytesToMegabytes(dataRequestsSize) << " MB ";
	L_DEBUG << "\toctree.dev_nodeLenght:         " << (void*) octree.dev_nodeLenght << " " << bytesToMegabytes(nodeLenghtSize) << " MB ";
	L_DEBUG << "\toctree.dev_nodeVoxelPixelSize: " << (void*) octree.dev_nodeVoxelPixelSize << " " << bytesToMegabytes(nodeVoxelPixelSizeSize) << " MB ";
	L_DEBUG << "";
	L_DEBUG << "\t  total::    " << bytesToMegabytes(childDataSize +
		                                             brickDataSize +
													 localizationCodeSize +
													 localizationDepthSize +
													 tileTimestampSize +
													 tileUsageSize +
													 brickUsageSize +
													 brickTimestampSize +
													 raymarcherRequestsSize +
													 dataRequestsSize +
													 nodeLenghtSize +
													 nodeVoxelPixelSizeSize) << " MB ";

	L_DEBUG << "";

	float host_nodeLength[MAX_DEPTH];
	float host_nodePixelSize[MAX_DEPTH];
	for (int depth = 0; depth < MAX_DEPTH; ++depth) {
		host_nodeLength[depth] = pow(0.5f, depth);
		host_nodePixelSize[depth] = host_nodeLength[depth] / BRICK_SIZE;
		L_DEBUG << "\tdepth: " << depth << "\t" <<  host_nodeLength[depth] << "\t" << host_nodePixelSize[depth];
	}
	L_DEBUG << "";

	cudasafe( cudaMemcpy(octree.dev_nodeLenght, host_nodeLength, MAX_DEPTH * sizeof(float), cudaMemcpyHostToDevice), "memcopy host_nodeLength to dev_nodeLenght");
	cudasafe( cudaMemcpy(octree.dev_nodeVoxelPixelSize, host_nodePixelSize, MAX_DEPTH * sizeof(float), cudaMemcpyHostToDevice), "memcopy host_nodePixelSize to dev_nodeVoxelPixelSize");
}

void CudaOctreeBuilder::initializeUsageLists()
{
	L_DEBUG << "";
	L_DEBUG << "Initializing usages lists...";

	// Setup the tile usage list, this populates the usage list with 
	// addresses of all node tiles with the usage flag set to zero
	int * host_tileUsageList = new int[MAX_NODE_TILES];
	for (unsigned int t = 0; t < MAX_NODE_TILES; ++t) {
		TileUsage usage;
		usage.tileAddress = t * 8;
		usage.flag = 0;
		host_tileUsageList[t] = packTileUsage(usage);
	}
	cudasafe( cudaMemcpy(octree.dev_tileUsage, host_tileUsageList, sizeof(int) * MAX_NODE_TILES, cudaMemcpyHostToDevice), "memcopy host_tileUsageList to dev_tileUsage");
	delete[] host_tileUsageList;

	// Setup the brick usage list, this populates the usage list with 
	// addresses of all bricks and the usage flag set to zero
	int * host_brickUsageList = new int[MAX_BRICKS];
	for (unsigned int bx = 0; bx < MAX_BRICKS_PER_SIDE; ++bx) {
		for (unsigned int by = 0; by < MAX_BRICKS_PER_SIDE; ++by) {
			for (unsigned int bz = 0; bz < MAX_BRICKS_PER_SIDE; ++bz) {

				//int offset = bx * MAX_BRICKS_PER_SIDE * MAX_BRICKS_PER_SIDE + by * MAX_BRICKS_PER_SIDE + bz;

				BrickUsage usage;
				usage.brickPointer.x = bx * BRICK_SIZE;
				usage.brickPointer.y = by * BRICK_SIZE;
				usage.brickPointer.z = bz * BRICK_SIZE;
				usage.flag = 0;

				int offset = getBrickTimestampIndexFromBrickPointer(usage.brickPointer);

				host_brickUsageList[offset] = packBrickUsage(usage);


				//L_DEBUG << bx << ", " << by << ", " << bz << " = " << offset;
				//L_DEBUG << usage.brickPointer.x << ", " << usage.brickPointer.y << ", " << usage.brickPointer.z << " = " << offset;
			}
		}
	}
	cudasafe( cudaMemcpy(octree.dev_brickUsage, host_brickUsageList, sizeof(int) * MAX_BRICKS, cudaMemcpyHostToDevice), "memcopy host_brickUsageList to dev_brickUsage");
	delete[] host_brickUsageList;

	L_DEBUG << "... done!";
	L_DEBUG << "";
}

void CudaOctreeBuilder::initializeRootNode()
{
	NodeChildData rootChildData;
	rootChildData.childAddress = 0;
	rootChildData.flagDataType = FLAG_DATA_TYPE_CONSTANT;
	rootChildData.flagMaxSubdivision = FLAG_CAN_SUBDIVIDE;
	
	NodeConstantColor rootColor;
	rootColor.x = 0.0f;
	rootColor.y = 0.0f;
	rootColor.z = 0.0f;
	rootColor.w = 0.0f;

	NodeLocalizationCode rootLocalizationCode;
	rootLocalizationCode.x = 0;
	rootLocalizationCode.y = 0;
	rootLocalizationCode.z = 0;

	int  * host_childData = new int[1];
	int  * host_brickData = new int[1];
	int  * host_localizationCode = new int[1];
	char * host_localizationDepth = new char[1];

	host_childData[0] = packNodeChildData(rootChildData);
	host_brickData[0] = packNodeConstantColor(rootColor);
	host_localizationCode[0] = packNodeLocalizationCode(rootLocalizationCode);
	host_localizationDepth[0] = packNodeLocalizationDepth(0);

	cudasafe( cudaMemcpy(octree.dev_childData, host_childData, sizeof(int), cudaMemcpyHostToDevice), "memcopy root to dev_childData");
	cudasafe( cudaMemcpy(octree.dev_brickData, host_brickData, sizeof(int), cudaMemcpyHostToDevice), "memcopy root to dev_brickData");
	cudasafe( cudaMemcpy(octree.dev_localizationCode, host_localizationCode, sizeof(int), cudaMemcpyHostToDevice), "memcopy root to dev_localizationCode");
	cudasafe( cudaMemcpy(octree.dev_localizationDepth, host_localizationDepth, sizeof(char), cudaMemcpyHostToDevice), "memcopy root to dev_localizationDepth");
}

void CudaOctreeBuilder::dumpNodesFromRoot(unsigned int count)
{
	count = mini(count, MAX_NODES);

	int  * host_childData = new int[count];
	int  * host_brickData = new int[count];
	int  * host_localizationCode = new int[count];
	char * host_localizationDepth = new char[count];

	cudasafe( cudaMemcpy(host_childData, octree.dev_childData, count * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_childData to host");
	cudasafe( cudaMemcpy(host_brickData, octree.dev_brickData, count * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_brickData to host");
	cudasafe( cudaMemcpy(host_localizationCode, octree.dev_localizationCode, count * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_localizationCode to host");
	cudasafe( cudaMemcpy(host_localizationDepth, octree.dev_localizationDepth, count * sizeof(char), cudaMemcpyDeviceToHost), "memcopy dev_localizationDepth to host");

	for (unsigned int i = 0; i < count; ++i) {

		NodeChildData childData = unpackNodeChildData(host_childData[i]);
		NodeConstantColor colorData = unpackNodeConstantColor(host_brickData[i]);
		NodeBrickPointer brickPointer = unpackNodeBrickPointer(host_brickData[i]);
		NodeLocalizationCode localizationCode = unpackNodeLocalizationCode(host_localizationCode[i]);
		int localizationDepth = unpackNodeLocalizationDepth(host_localizationDepth[i]);

		NodeLocalization local = getLocalizationFromDepthAndCode(localizationDepth, localizationCode);

		L_DEBUG << "Node " << i << " {";
		L_DEBUG << "   childAddress:   " << childData.childAddress;
		L_DEBUG << "   dataType:       " << (childData.flagDataType == FLAG_DATA_TYPE_CONSTANT ? "CONSTANT" : "BRICK");
		L_DEBUG << "   maxSubDivision: " << (childData.flagMaxSubdivision == FLAG_CAN_SUBDIVIDE ? "NO" : "YES");
		
		if (childData.flagDataType == FLAG_DATA_TYPE_BRICK) {
			L_DEBUG << "   brick:          " << brickPointer.x << ", " << brickPointer.y << ", " << brickPointer.z;
		} else {
			L_DEBUG << "   color:          " << colorData.x << ", " << colorData.y << ", " << colorData.z << ", " << colorData.w;
		}
		
		L_DEBUG << "   localCode.x:    " << get_bits(localizationCode.x);
		L_DEBUG << "   localCode.y:    " << get_bits(localizationCode.y);
		L_DEBUG << "   localCode.z:    " << get_bits(localizationCode.z);
		L_DEBUG << "   localDepth:     " << localizationDepth;
		L_DEBUG << "   localBOXMIN:    " << local.bboxMin.x << ", " << local.bboxMin.y << ", " << local.bboxMin.z;
		L_DEBUG << "   localBOXMAX:    " << local.bboxMax.x << ", " << local.bboxMax.y << ", " << local.bboxMax.z;
		L_DEBUG << "}";

	}

	delete[] host_childData;
	delete[] host_brickData;
	delete[] host_localizationCode;
	delete[] host_localizationDepth;
}

void CudaOctreeBuilder::sanityCheck()
{
	int  * host_childData = new int[MAX_NODES];
	int  * host_brickData = new int[MAX_NODES];
	int  * host_brickTimestamp = new int[MAX_BRICKS];

	std::vector<int3> discoveredBrickpointers = std::vector<int3>();
	std::vector<int> discoveredChildAddress = std::vector<int>();

	cudasafe( cudaMemcpy(host_childData, octree.dev_childData, MAX_NODES * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_childData to host");
	cudasafe( cudaMemcpy(host_brickData, octree.dev_brickData, MAX_NODES * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_brickData to host");
	cudasafe( cudaMemcpy(host_brickTimestamp, octree.dev_brickTimestamp, MAX_BRICKS * sizeof(int), cudaMemcpyDeviceToHost), "memcopy dev_brickTimestamp to host");

	for (unsigned int i = 0; i < MAX_NODES; ++i) {

		NodeChildData childData = unpackNodeChildData(host_childData[i]);

		bool duplicateAddress = false;

		if (childData.childAddress != 0) {

			for (unsigned v = 0; v < discoveredBrickpointers.size(); ++v) {

				int horse = discoveredChildAddress[v];

				if (childData.childAddress == horse) {
					duplicateAddress = true;
					break;
				}

			}
		}

		if (duplicateAddress) {
			L_DEBUG << "alerady seen childaddress: " << childData.childAddress;
		} else {
			discoveredChildAddress.push_back(childData.childAddress);
		}


		if (childData.flagDataType == FLAG_DATA_TYPE_BRICK) {
			NodeBrickPointer brickPointer = unpackNodeBrickPointer(host_brickData[i]);

			bool duplicateBrick = false;



			for (unsigned v = 0; v < discoveredBrickpointers.size(); ++v) {

				int3 horse = discoveredBrickpointers[v];

				if (horse.x == brickPointer.x && horse.y == brickPointer.y && horse.z == brickPointer.z) {
					duplicateBrick = true;
					break;
				}

			}

			if (duplicateBrick) {

				int tsIndex = getBrickTimestampIndexFromBrickPointer(brickPointer);
				int ts = host_brickTimestamp[tsIndex];

				L_DEBUG << "alerady seen brickpointer: " << brickPointer.x  << ", " << brickPointer.y  << ", " << brickPointer.z << " --- ts: " << ts;
				continue;
			} else {
				discoveredBrickpointers.push_back(brickPointer);
			}

		}

	}

	delete[] host_childData;
	delete[] host_brickData;
}