#include "ConsoleLog.h"

using namespace std;

namespace restless
{
	ConsoleLog::ConsoleLog(int s) : size(s),
		added(0),
		log(list<string>())
	{}

	ConsoleLog::~ConsoleLog()
	{}

	const vector<string> ConsoleLog::getLog(const unsigned int length) const
	{
		vector<string> snippet = vector<string>();

		unsigned int i = 0;
		list<string>::const_iterator iter;
		for (iter=log.begin(); iter != log.end() && i++ < length; iter++) snippet.push_back((*iter));

		return snippet;
	}

	void ConsoleLog::add(string line) 
	{
		added++;
		log.push_front(line);
		if (added > size) {
			log.pop_back();
		}
	}

	void ConsoleLog::append(string line) 
	{
		if (log.size() > 0) {
			string org = log.front();
			org.append(line);
			log.pop_front();
			log.push_front(org);
		}
		else {
			add(line);
		}
	}
}
