#include "FileResource.h"

#include <fstream>
#include "Log.h"

namespace restless
{

	FileResource::FileResource(const char * filename) :
		_filename(filename),
		_contents("")
	{

	}


	FileResource::~FileResource(void)
	{

	}

	const char * FileResource::getFilename()
	{
		return _filename.c_str();
	}

	const char * FileResource::getContents()
	{
		if (_contents.empty() && read() == false) {
			LOG(LOG_RESOURCE, LOG_ERROR) << "unable to read: " << _filename;
		}

		return _contents.c_str();
	}

	bool FileResource::read()
	{
		std::ifstream in(_filename, std::ios::in | std::ios::binary);
		if (!in) {
			return false;
		}

		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();

		_contents = contents;
		return true;
	}
}

