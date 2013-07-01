#ifndef _RESTLESS_VARIABLE_COMMAND_VARIABLELISTCOMMAND_H
#define _RESTLESS_VARIABLE_COMMAND_VARIABLELISTCOMMAND_H

#include "../../Console/ConsoleCommand.h"

namespace restless
{
	class VariableManager;
}

namespace restless
{
	class VariableListCommand : public ConsoleCommand
	{
	public:
		VariableListCommand(VariableManager & variables);
		virtual ~VariableListCommand();

		virtual void onExecute(ConsoleCommandRequest & request);
		virtual std::string onComplete(ConsoleCommandRequest & request);

	protected:

		VariableManager & vars;

	};
}

#endif