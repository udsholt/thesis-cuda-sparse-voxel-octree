#ifndef _RESTLESS_UTIL_FILESYSTEM_H
#define _RESTLESS_UTIL_FILESYSTEM_H

#include <string>

#include "Singleton.h"

namespace restless
{

	class FileSystem : public Singleton<FileSystem>
	{
	protected:
		friend Singleton<FileSystem>;
		FileSystem();
		virtual ~FileSystem();

	public:
		std::string path(const char * path);
		std::string path(std::string path);

		void setRoot(const char * root);
		const char * getRoot();

	protected:

		std::string resolve(std::string path);
		std::string _root;
	};

}

#endif

