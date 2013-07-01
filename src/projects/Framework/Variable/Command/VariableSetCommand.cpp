#include "VariableSetCommand.h"

#include "../VariableManager.h"
#include "../../Util/VectorUtil.h"

using namespace std;

namespace restless
{

	VariableSetCommand::VariableSetCommand(VariableManager & variables) :
		vars(variables)
	{
	}


	VariableSetCommand::~VariableSetCommand()
	{
	}

	void VariableSetCommand::onExecute(ConsoleCommandRequest & request)
	{
		if (request.tokens().countTokens() < 2) {
			request.out() << "usage: [varname] [value]\n";
			return;
		}

		string varname = request.tokens().nextToken();

		if (!vars.exists(varname.c_str())) {
			request.out() << "unknown varname: " << varname << "\n";
			return;
		}

		if (!vars.get(varname.c_str()).setValue(request.tokens().remainingString())) {
			request.out() << "could not set value for: " << varname << "!\n";
			return;
		}

		request.out() << varname << " set to " << vars.get(varname.c_str()).getValue();
	}

	string VariableSetCommand::onComplete(ConsoleCommandRequest & request)
	{
		string variableName = request.tokens().nextToken();

		if (vars.exists(variableName)) {
			request.out() << vars.get(variableName).getValue();
			return variableName + " " + request.tokens().remainingString();
		}

		vector<string> matches = vars.findByPrefix(variableName);

		if (matches.size() == 0) {
			return variableName;
		}

		if (matches.size() == 1) {
			return matches[0] + " " + request.tokens().remainingString();
		}

		if (matches.size() > 1) {
			for (unsigned int i = 0; i < matches.size(); ++i) {
				request.out() << matches[i] << "\n";
			}
		}
		
		return findLongestCommonPrefix(variableName, matches);
	}

}