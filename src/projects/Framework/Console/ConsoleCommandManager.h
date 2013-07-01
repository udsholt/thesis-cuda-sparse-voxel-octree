#ifndef _RESTLESS_CONSOLE_CONSOLECOMMANDMANAGER_H
#define _RESTLESS_CONSOLE_CONSOLECOMMANDMANAGER_H

#include <map>
#include <vector>
#include <string>

namespace restless
{
	class ConsoleCommand;
	class ConsoleCommandRequest;
}

namespace restless
{

	class ConsoleCommandManager
	{
	public:
		ConsoleCommandManager();
		virtual ~ConsoleCommandManager();

		virtual bool registerCommand(std::string name, ConsoleCommand & command);

		virtual bool onExecute(ConsoleCommandRequest & request);
		virtual std::string onComplete(ConsoleCommandRequest & request);

	protected:

		std::vector<std::string> findCommandsByPrefix(const std::string & prefix) const;

		std::map<std::string, ConsoleCommand *> commands;
	};

}

#endif
