#include "StringTokenizer.h"

using namespace std;

namespace restless {

	StringTokenizer::StringTokenizer(const string & str, const string & delim)
	{
		
		if ((str.length() == 0) || (str.length() == 0)) {
			return;
		}

		_tokenStr = str;
		_delim = delim;

		unsigned int currPos = 0;

		// Remove sequential delimiter
		while(true) {
			if ((currPos = _tokenStr.find(_delim, currPos)) != string::npos) {
				currPos += _delim.length();

				while(_tokenStr.find(_delim, currPos) == currPos) {
					_tokenStr.erase(currPos,_delim.length());
				}

				continue;
			}
			break;
		}

		// Trim leading delimiter
		if (_tokenStr.find(_delim, 0) == 0) {
			_tokenStr.erase(0, _delim.length());
		}

		// Trim ending delimiter
		currPos = 0;
		if ((currPos = _tokenStr.rfind(_delim)) != string::npos) {
			if (currPos != (_tokenStr.length() - _delim.length())) {
				return;
			}

			_tokenStr.erase(_tokenStr.length() - _delim.length(), _delim.length());
		}

	}


	int StringTokenizer::countTokens()
	{
		unsigned int prevPos = 0;
		int numTokens        = 0;

		if (_tokenStr.length() > 0) {
			numTokens = 0;

			unsigned int currPos = 0;
			while(true) {
				if ((currPos = _tokenStr.find(_delim, currPos)) != string::npos) {
					numTokens++;
					prevPos  = currPos;
					currPos += _delim.length();
					continue;
				}
				break;
			}

			return ++numTokens;
		}

		return 0;
	}


	bool StringTokenizer::hasMoreTokens()
	{
		return (_tokenStr.length() > 0);
	}


	string StringTokenizer::nextToken(bool remove)
	{

		return nextToken(_delim, remove);

		if (_tokenStr.length() == 0) {
			return "";
		}

		string  tmp_str = "";
		unsigned int pos     = _tokenStr.find(_delim,0);

		if (pos != string::npos)  {
			tmp_str   = _tokenStr.substr(0,pos);
			_tokenStr = _tokenStr.substr(pos + _delim.length(), _tokenStr.length() - pos);
		} else {
			tmp_str   = _tokenStr.substr(0, _tokenStr.length());
			_tokenStr = "";
		}

		return tmp_str;
	}


	int StringTokenizer::nextIntToken()
	{
		return atoi(nextToken().c_str());
	}


	float StringTokenizer::nextFloatToken()
	{
		return (float) atof(nextToken().c_str());
	}


	std::string StringTokenizer::nextToken(const std::string & delimiter, bool remove)
	{
		if (_tokenStr.length() == 0) {
			return "";
		}

		string  tmpStr = "";
		unsigned int pos = _tokenStr.find(delimiter, 0);

		if (pos != string::npos) {
			tmpStr = _tokenStr.substr(0,pos);
			if (remove) {
				_tokenStr = _tokenStr.substr(pos + delimiter.length(),_tokenStr.length() - pos);
			}
		} else {
			tmpStr = _tokenStr.substr(0,_tokenStr.length());
			if (remove) {
				_tokenStr = "";
			}
		}

		return tmpStr;
	}


	std::string StringTokenizer::remainingString()
	{
		return _tokenStr;
	}


	std::string StringTokenizer::filterNextToken(const std::string & filterStr)
	{
		string  tmpStr = nextToken();
		unsigned int currentPos = 0;

		while((currentPos = tmpStr.find(filterStr,currentPos)) != string::npos) {
			tmpStr.erase(currentPos,filterStr.length());
		}

		return tmpStr;
	}
}