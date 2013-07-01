#include "ConsoleCommand.h"

#include "../Util/Log.h"

using namespace std;

namespace restless
{

	ConsoleCommand::ConsoleCommand()
	{
	}


	ConsoleCommand::~ConsoleCommand()
	{
	}

	void ConsoleCommand::onExecute(ConsoleCommandRequest & request)
	{
		L_ERROR << "command has no onExecute";
	}

	string ConsoleCommand::onComplete(ConsoleCommandRequest & request)
	{
		L_ERROR << "command has no onComplete";
		return request.tokens().remainingString();
	}
}