#ifndef _RESTLESS_CONSOLE_CONSOLELOG_H
#define _RESTLESS_CONSOLE_CONSOLELOG_H

#include <list>
#include <vector>
#include <string>

namespace restless
{

	class ConsoleLog
	{
	public:
		ConsoleLog(int size);
		~ConsoleLog();

		const int getAdded() const { return added; }

		const std::vector<std::string> getLog(unsigned int lenght) const;
		void add(std::string text);
		void append(std::string text);

	protected:
		int size, added;
		std::list<std::string> log;
	};

}

#endif