#include "CudaTimerCollectionRecorder.h"

#include <Framework/Util/Log.h>

using namespace std;

namespace restless
{

	CudaTimerCollectionRecorder::CudaTimerCollectionRecorder(CudaTimerCollection & collection) :
		timerCollection(collection),
		filename("default_cuda_timer_recording.csv"),
		directory(""),
		recording(false),
		counter(0)
	{
	}


	CudaTimerCollectionRecorder::~CudaTimerCollectionRecorder()
	{
	}

	void CudaTimerCollectionRecorder::setFilename(const char * toFilename)
	{
		if (recording) {
			return;
		}

		filename = toFilename;

	}

	void CudaTimerCollectionRecorder::setDirectory(const char * rootDir)
	{
		if (recording) {
			return;
		}

		directory = rootDir;
	}


	bool CudaTimerCollectionRecorder::start()
	{
		if (recording) {
			return false;
		}

		file.open(directory + "/" + filename);

		L_DEBUG << "recording to file: " << directory + "/" + filename;

		if (!file.is_open()) {
			L_DEBUG << "file could not be opened";
			return false;
		}

		recording = true;
		counter = 0;

		writeHeader();

		return true;
	}

	void CudaTimerCollectionRecorder::update(const float fps)
	{
		if (!recording) {
			return;
		}

		if (!file.is_open()) {
			return;
		}

		counter++;

		unsigned int binCount = timerCollection.getBinCount();

		file << "\t[";
		file << counter << ", ";
		file << fps << ", ";

		for (unsigned int b = 0; b < binCount; ++b) {

			CudaTimerBin & bin = timerCollection.getBin(b);

			if (bin.enabled) {
				
				file << timerCollection.getBin(b).elapsed << ", ";
			}
			
		}

		file << "0],";

		file << "\n";
	}

	void CudaTimerCollectionRecorder::writeHeader()
	{
		if (!file.is_open()) {
			return;
		}

		unsigned int binCount = timerCollection.getBinCount();

		file << "var data = [\n";

		file << "\t[ ";
		file << "\"Frame\", ";
		file << "\"FPS\", ";

		for (unsigned int b = 0; b < binCount; ++b) {

			CudaTimerBin & bin = timerCollection.getBin(b);

			if (bin.enabled) {
				
				file << "\"" << timerCollection.getBin(b).name << "\", ";
				
			}

			
		}

		file << "\"Empty\"";
		file << "],";
		file << "\n";

		L_DEBUG << "header written";
	}

	bool CudaTimerCollectionRecorder::stop()
	{
		if (!recording) {
			return false;
		}

		file << "\t[]\n";
		file << "]";

		if (file.is_open()) {
			file.close();
		}

		L_DEBUG << "recording stoppedd";

		

		recording = false;
		return true;
	}

	
}