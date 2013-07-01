#include "VariableListCommand.h"

#include "../VariableManager.h"

using namespace std;

namespace restless
{

	VariableListCommand::VariableListCommand(VariableManager & variables) :
		vars(variables)
	{
	}


	VariableListCommand::~VariableListCommand()
	{
	}

	void VariableListCommand::onExecute(ConsoleCommandRequest & request)
	{
		string prefix = request.tokens().nextToken();
		vector<string> matches = vars.findByPrefix(prefix);

		request.out()  <<"\ncvars \"" << prefix << "*\" { \n";
		for (unsigned int i = 0; i < matches.size(); ++i) {
			string nameType = matches[i] + "[" + vars.get(matches[i]).getTypeName() + "]";
			string value = vars.get(matches[i]).getValue();

			request.out()  << "  ";
			request.out()  << nameType << string(max(abs((int) nameType.length() - 30), 0), '.') << ": ";
			request.out()  << value;
			request.out()  << "\n";
		}
		request.out()  <<"}\n\n";
	}

	string VariableListCommand::onComplete(ConsoleCommandRequest & request)
	{
		return request.tokens().remainingString();
	}

}