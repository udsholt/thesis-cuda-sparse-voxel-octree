#ifndef _RESTLESS_UTIL_FILERESOURCE_H
#define _RESTLESS_UTIL_FILERESOURCE_H

#include <string>

namespace restless
{

	class FileResource
	{
	public:
		FileResource(const char * filename);
		~FileResource();

		const char * getFilename();
		const char * getContents();

	protected:

		bool read();

		std::string _filename;
		std::string _contents;
	};

}

#endif

