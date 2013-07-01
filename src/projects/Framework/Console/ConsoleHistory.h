#ifndef _RESTLESS_CONSOLE_CONSOLEHISTORY_H
#define _RESTLESS_CONSOLE_CONSOLEHISTORY_H

#include <list>
#include <string>

namespace restless
{
	class ConsoleHistory
	{
	public:
		ConsoleHistory(int size);
		~ConsoleHistory();

		void add(std::string command);
		std::string getNext();
		std::string getPrevious();

	protected:
		int size;
		int added;
		int pointer;

		std::list<std::string> commands;

		const std::string getCurrentCommand() const;
		const std::string getCommand(int n) const;
	};
}

#endif
