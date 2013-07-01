#ifndef _RESTLESS_UTIL_VECTORUTIL_H
#define _RESTLESS_UTIL_VECTORUTIL_H

#include <string>
#include <vector>

namespace restless
{
	std::string findLongestCommonPrefix(const std::string & prefix, const std::vector<std::string> strings);
}

#endif