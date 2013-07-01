#ifndef _RESTLESS_CUDA_CUDA_KERNELTIMER_H
#define _RESTLESS_CUDA_CUDA_KERNELTIMER_H

#include <cuda_runtime.h>

// http://stackoverflow.com/questions/11209228/timing-different-sections-in-cuda-kernel

enum 
{
	timer_raymarcher_all = 0,
	timer_raymarcher_inner,
	timer_raymarcher_raysetup,
	timer_raymarcher_global_intesect,
    timer_raymarcher_lookup,
	timer_raymarcher_localize,
	timer_raymarcher_local_intesect,
	timer_raymarcher_march_constant,
    timer_raymarcher_march_brick,
	timer_raymarcher_selected_section,
    timer_count
};

__device__ float dev_cuda_kernel_timers[timer_count];

#ifdef KERNEL_TIMERS_ENABLE
	#define TIMER_TIC(tid) clock_t tic_##tid; if ( threadIdx.x == 0 ) tic_##tid = clock();
	#define TIMER_TOC(tid) clock_t toc_##tid = clock(); if ( threadIdx.x == 0 ) atomicAdd( &dev_cuda_kernel_timers[tid] , ( toc_##tid > tic_##tid ) ? (toc_##tid - tic_##tid) : ( toc_##tid + (0xffffffff - tic_##tid) ) );
#else
	#define TIMER_TIC(tid)
	#define TIMER_TOC(tid)
#endif

inline cudaError_t cudaResetKernelTimers()
{
	float timers[timer_count];
	for (unsigned int t = 0; t < timer_count; ++t) {
		timers[t] = 0.0f;
	}
	return cudaMemcpyToSymbol(dev_cuda_kernel_timers, timers, timer_count * sizeof(float));
}

inline cudaError_t cudaReadKernelTimers(float * timers)
{
	return cudaMemcpyFromSymbol(timers, dev_cuda_kernel_timers, timer_count * sizeof(float));
}

#endif