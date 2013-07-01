#ifndef _INC_CUDA_PREDICATES_H
#define _INC_CUDA_PREDICATES_H

#include "Octree.h"

struct is_zero
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x == 0;
	}
};

struct is_not_negative
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x >= 0;
	}
};

struct is_not_zero
{
	__host__ __device__
	bool operator()(const int x)
	{
		return x != 0;
	}
};

struct tile_usage_mask_is_zero
{
	__host__ __device__
	bool operator()(const int & packed)
	{
		TileUsage usage = unpackTileUsage(packed);
		return usage.flag == 0;
	}
};

struct brick_usage_mask_is_zero
{
	__host__ __device__
	bool operator()(const int & packed)
	{
		BrickUsage usage = unpackBrickUsage(packed);
		return usage.flag == 0;
	}
};

#endif