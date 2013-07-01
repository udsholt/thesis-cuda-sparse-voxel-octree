#ifndef _INC_CUDA_DEFINES_H
#define _INC_CUDA_DEFINES_H

#define MAX_DEPTH 8

#define MAX_NODES		2097152															// Maximum number of nodes avaiable
#define MAX_NODE_TILES	MAX_NODES/8														// Maximum number of nodes tiles in the pool

#define MAX_SUBDIVIDE_REQUESTS_PER_UPDATE 64											// Handle 64 subdivde requests from the raymarcher each frame (working: 18, 26)
#define MAX_DATA_REQUESTS_PER_UPDATE	  MAX_SUBDIVIDE_REQUESTS_PER_UPDATE*8			// This will yield at most 64*8 data requests (as each node 
																						// will subdivde into 8 nodes that all need data evaulation)

#define VOLUME_DIMENSION 512															// Size of the volume texture
#define BRICK_SIZE		 16																// Size of a brick in the volume texture
#define BRICK_BORDER	 2																// Brick border 2 is needed for normals

#define MAX_BRICKS_PER_SIDE	VOLUME_DIMENSION/BRICK_SIZE									// How many bricks are available on each side of the volume
#define MAX_BRICKS			MAX_BRICKS_PER_SIDE*MAX_BRICKS_PER_SIDE*MAX_BRICKS_PER_SIDE	// Total number of bricks in the volume

#define DENSITY_THRESHOLD_AIR     -0.05f												// Anything lower than this threshold is considered air
#define DENSITY_THRESHOLD_SOLID    0.05f												// Anything higher than this threshold is considered solid

#define ESCAPE_EPSILON 1.192092e-05F													// How far does the raymarcher need to go to escape the current
																						// node and enter the next one. This needs to be somewhat higher
																						// than FLT_EPSILON at grazeing angles

//#define KERNEL_TIMERS_ENABLE															// Enable in kernel profiling

enum 
{
	BIN_ALL = 0,
	BIN_RENDER,
	BIN_FILL_REQUEST_ARRAYS,
	BIN_SUBDIVIDE_ALL,
	BIN_SUBDIVIDE_USAGE_MASK_TILE,
	BIN_SUBDIVIDE_USAGE_MASK_BRICK,
	BIN_SUBDIVIDE_USAGE_MASK_COMPACT,
	BIN_SUBDIVIDE_INVALIDATE_TIMESTAMPS,
	BIN_SUBDIVIDE_INVALIDATE_NODES,
	BIN_SUBDIVIDE_REQUESTS_COMPACT,
	BIN_SUBDIVIDE_REQUESTS_HANDLE,	
	BIN_SUBDIVIDE_REQUESTS_DATA,
	BIN_SUBDIVIDE_REQUESTS_FINALIZE,
	BIN_KERNEL_ALL,
	BIN_KERNEL_INNER,
	BIN_KERNEL_RAYSETUP,
	BIN_KERNEL_GLOBAL_INTERSECT,
	BIN_KERNEL_LOOKUP,
	BIN_KERNEL_LOCALIZE,
	BIN_KERNEL_LOCAL_INTERSECT,
	BIN_KERNEL_MARCH_CONSTANT,
	BIN_KERNEL_MARCH_BRICK,
	BIN_KERNEL_SELECTED_SECTION,
	BIN_COUNT_SUBDIVIDE_REQUEST,
	BIN_LAST						
};



#endif