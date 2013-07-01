#ifndef RESTLESS_UTIL_STRINGTOKENIZER_H
#define RESTLESS_UTIL_STRINGTOKENIZER_H

/*
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
*/

#include <string>

namespace restless 
{

	class StringTokenizer
	{

	   public:

		StringTokenizer(const std::string & str, const std::string & delim);
	   ~StringTokenizer(){};

		int         countTokens();
		bool        hasMoreTokens();
		std::string nextToken(bool remove = true);
		int         nextIntToken();
		float       nextFloatToken();
		std::string nextToken(const std::string & delim, bool remove = true);
		std::string remainingString();
		std::string filterNextToken(const std::string & filterStr);

	   private:

		std::string  _tokenStr;
		std::string  _delim;

	};
}

#endif
