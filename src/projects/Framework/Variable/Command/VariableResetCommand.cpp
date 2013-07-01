#include "VariableResetCommand.h"

#include "../VariableManager.h"
#include "../../Util/VectorUtil.h"

using namespace std;

namespace restless
{

	VariableResetCommand::VariableResetCommand(VariableManager & variables) :
		vars(variables)
	{
	}


	VariableResetCommand::~VariableResetCommand()
	{
	}

	void VariableResetCommand::onExecute(ConsoleCommandRequest & request)
	{
		string prefix = request.tokens().nextToken();
		vector<string> matches = vars.findByPrefix(prefix);

		for (unsigned int i = 0; i < matches.size(); ++i) {

			string name = matches[i];
			VariableInterface & variable = vars.get(name);

			string currentValue = variable.getValue();
			variable.resetValue();
			string resetValue = variable.getValue();

			if (currentValue.compare(resetValue) != 0 ) {
				request.out() << "reset " << name << ": from " << currentValue << " to " << resetValue << "\n";
			}

		}
	}

	string VariableResetCommand::onComplete(ConsoleCommandRequest & request)
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