#include "Screenshot.h"

#include <FreeImage.h>

#include <iomanip>
#include <string>
#include <sstream>

#include "../GL.h"
#include "../Util/Log.h"
#include "FileSystem.h"

using namespace std;

namespace restless
{
	bool Screenshot::saveScreenshot(const unsigned int width, const unsigned int height)
	{
		// Rumors have it that put_time requries c++11
		stringstream filenameBuffer;
		time_t t = time(NULL);
		tm tm = * localtime(&t);
		filenameBuffer << "screenshot/" << std::put_time(&tm, "%Y%m%d-%H%M%S") << "_" <<  width << "x" <<  height << ".png";
		string filename = filenameBuffer.str();

		string fullpath = FileSystem::getInstance().path(filename);

		LOG(LOG_RESOURCE, LOG_INFO) << "saving screenshot to: " << fullpath;

		BYTE * pixels = new BYTE[3 * width * height];
		glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);
		FIBITMAP* Image = FreeImage_ConvertFromRawBits(pixels, width, height, width*3, 24, 0xFF0000, 0x00FF00, 0x0000FF, false); 
		delete[] pixels;

		if (!FreeImage_Save(FIF_PNG, Image, fullpath.c_str(), 0)) {
			LOG(LOG_RESOURCE, LOG_INFO) << "somthing bad happened";
			return false;
		}

		return true;
	}

}