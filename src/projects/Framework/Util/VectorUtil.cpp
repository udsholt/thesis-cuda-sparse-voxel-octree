#include "VectorUtil.h"

using namespace std;

namespace restless
{
	string findLongestCommonPrefix(const string & prefix, const vector<string> strings)
	{
		// No matches, return empty string
		if (strings.size() == 0) {
			return "";
		}

		// One match, return the prefix
		if (strings.size() == 1) {
			return strings[0];
		}

		// Start with the first string and determine the 
		// longest common substring, which can be no longer
		// then the first element
		string longest = strings[0];

		vector<string>::const_iterator it;
		for(it = strings.begin(); it != strings.end(); it++) {
			string current = *it;

			// Check each character from 0 to min(longest.lenght(), current.length())
			// to determine where they are no longer equal
			unsigned int commonPos = 0;
			while (commonPos < longest.length() && commonPos < current.length()) {
				if (current.substr(commonPos, 1).compare(longest.substr(commonPos, 1)) != 0) {
					break;
				}
				commonPos++;
			}

			// Reduce the longest string to the common size
			longest = longest.substr(0, commonPos);
		}

		return longest;
	}
}


