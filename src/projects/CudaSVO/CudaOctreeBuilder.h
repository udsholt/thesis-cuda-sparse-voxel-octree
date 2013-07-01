#ifndef _INC_CUDAOCTREEBUILDER_H
#define _INC_CUDAOCTREEBUILDER_H

#include "cuda/defines.h"
#include "cuda/Octree.h"

class CudaOctreeBuilder
{
public:
	CudaOctreeBuilder();
	~CudaOctreeBuilder();

	void initialize();
	void rebuild();
	void destroy();

	void dumpNodesFromRoot(unsigned int count);
	void sanityCheck();

protected:

	void initializeVolume();
	void initializeOctree();
	void initializeUsageLists();
	void initializeRootNode();

	Octree octree;

	unsigned int requestBufferSize;

	bool done;
};

#endif