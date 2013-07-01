#ifndef _RESTLESS_VARIABLE_COMMAND_VARIABLESETCOMMAND_H
#define _RESTLESS_VARIABLE_COMMAND_VARIABLESETCOMMAND_H

#include "../../Console/ConsoleCommand.h"

namespace restless
{
	class VariableManager;
}

namespace restless
{
	class VariableSetCommand : public ConsoleCommand
	{
	public:
		VariableSetCommand(VariableManager & variables);
		virtual ~VariableSetCommand();

		virtual void onExecute(ConsoleCommandRequest & request);
		virtual std::string onComplete(ConsoleCommandRequest & request);

	protected:

		VariableManager & vars;

	};
}

#endif