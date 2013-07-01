#ifndef _RESTLESS_CUDA_TIMERCOLLECTIONRECORDER_H
#define _RESTLESS_CUDA_TIMERCOLLECTIONRECORDER_H

#include "CudaTimerCollection.h"

#include <string>
#include <iostream>
#include <fstream>

namespace restless
{

	class CudaTimerCollectionRecorder
	{
	public:
		CudaTimerCollectionRecorder(CudaTimerCollection & collection);
		~CudaTimerCollectionRecorder();

		void setDirectory(const char * rootDir);
		void setFilename(const char * toFilename);

		bool start();
		bool stop();

		void update(const float fps);

	protected:

		void writeHeader();

		CudaTimerCollection & timerCollection;

		std::string filename;
		std::string directory;

		std::ofstream file;

		bool recording;

		int counter;
	};

}

#endif