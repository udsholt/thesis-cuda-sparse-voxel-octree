#ifndef _RESTLESS_CONSOLE_CONSOLECOMMAND_H
#define _RESTLESS_CONSOLE_CONSOLECOMMAND_H

#include <string>

#include "ConsoleCommandRequest.h"

namespace restless
{
	class ConsoleCommand
	{
	public:
		ConsoleCommand();
		virtual ~ConsoleCommand();

		virtual void onExecute(ConsoleCommandRequest & request);
		virtual std::string onComplete(ConsoleCommandRequest & request);
	};

}

#endif
