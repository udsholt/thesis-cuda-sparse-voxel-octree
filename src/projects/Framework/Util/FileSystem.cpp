#include "FileSystem.h"

#include <stdlib.h>
#include <direct.h>

#include "Log.h"

namespace restless
{

	FileSystem::FileSystem() :
		_root("")
	{
		char * buffer = _getcwd( NULL, 0 );

		if (buffer != nullptr) {
			setRoot(buffer);
			free(buffer);
		}
	}


	FileSystem::~FileSystem()
	{

	}

	std::string FileSystem::path(const char * filePath)
	{
		return path(std::string(filePath));
	}

	std::string FileSystem::path(std::string filePath)
	{
		return resolve(_root + "/" + filePath);
	}

	void FileSystem::setRoot(const char * root)
	{
		_root = resolve(root);
	}

	const char * FileSystem::getRoot()
	{
		return _root.c_str();
	}

	std::string FileSystem::resolve(std::string path)
	{
		char full[_MAX_PATH];

		if (_fullpath(full, path.c_str(), _MAX_PATH) == NULL) {
			L_ERROR << "invalid path: " << path;
			return "";
		}

		return std::string(full);
	}

}

