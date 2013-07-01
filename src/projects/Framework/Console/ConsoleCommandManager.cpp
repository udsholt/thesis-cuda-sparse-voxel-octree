#include "ConsoleCommandManager.h"

#include "../Util/Log.h"
#include "../Util/VectorUtil.h"

#include "ConsoleCommand.h"

#include "ConsoleCommandRequest.h"

using namespace std;


namespace restless
{

	ConsoleCommandManager::ConsoleCommandManager(void)
	{
	}


	ConsoleCommandManager::~ConsoleCommandManager(void)
	{
	}

	bool ConsoleCommandManager::registerCommand(string name, ConsoleCommand & command)
	{
		if (commands.find(name) != commands.end()) {
			L_ERROR << "command with name: " << name << " already registered";
			return false;
		}

		commands[name] = & command;
		return true;
	}

	bool ConsoleCommandManager::onExecute(ConsoleCommandRequest & request)
	{
		if (request.command().empty()) {
			return false;
		}

		string name = request.command();

		if (commands.find(name) != commands.end()) {
			commands[name]->onExecute(request);
			return true;
		}

		request.out() << "unknown command: " << name;

		return false;
	}

	std::string ConsoleCommandManager::onComplete(ConsoleCommandRequest & request)
	{
		string name = request.command();

		// If exactly one command fits the name, ask the command to
		// fill in the rest
		if (commands.find(name) != commands.end()) {
			return name + " " + commands[name]->onComplete(request);
		}

		// If there are additional tokens, autocomplete would destroy them
		// so return
		if (request.tokens().hasMoreTokens()) {
			return name + " " + request.tokens().remainingString();
		}

		// Find all commands that has name as a prefix
		vector<string> matches = findCommandsByPrefix(name);

		// No commands return the string as we received it
		if (matches.size() == 0) {
			return name;
		}

		// Print the commands that matches the prefix
		request.out() << "\n";
		for (unsigned int i = 0; i < matches.size(); ++i) {
			request.out() << matches[i] << "\n";
		}
		
		// Return the longest common command prefix
		return findLongestCommonPrefix(name, matches);
	}

	vector<string> ConsoleCommandManager::findCommandsByPrefix(const string & prefix) const
	{
		vector<string> matches = vector<string>();

		map<string, ConsoleCommand *>::const_iterator it;
		for(it = commands.begin(); it != commands.end(); it++) {

			const string name = it->first;

			if (name.length() < prefix.length()) {
				continue;
			}

			if (name.substr(0, prefix.length()).compare(prefix) == 0) {
				matches.push_back(name);
			}
		}

		return matches;
	}

}