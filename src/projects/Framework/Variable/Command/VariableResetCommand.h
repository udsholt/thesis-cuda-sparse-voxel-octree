#ifndef _RESTLESS_VARIABLE_COMMAND_VARIABLERESETCOMMAND_H
#define _RESTLESS_VARIABLE_COMMAND_VARIABLERESETCOMMAND_H

#include "../../Console/ConsoleCommand.h"

namespace restless
{
	class VariableManager;
}

namespace restless
{
	class VariableResetCommand : public ConsoleCommand
	{
	public:
		VariableResetCommand(VariableManager & variables);
		virtual ~VariableResetCommand();

		virtual void onExecute(ConsoleCommandRequest & request);
		virtual std::string onComplete(ConsoleCommandRequest & request);

	protected:

		VariableManager & vars;

	};
}

#endif