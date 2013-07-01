#include "ConsoleHistory.h"

#include <iostream>
#include "../Util/Log.h"

using namespace std;

namespace restless
{
	ConsoleHistory::ConsoleHistory(int s) : size(s), 
		added(0), 
		pointer(-1), 
		commands(list<string>())
	{}

	ConsoleHistory::~ConsoleHistory()
	{}

	void ConsoleHistory::add(string cmd) 
	{
		if (!commands.empty() && commands.front().compare(cmd) == 0) {
			return;
		}

		added++;
		pointer = -1;

		commands.push_front(cmd);
		if (added > size) {
			commands.pop_back();
		}
	}

	string ConsoleHistory::getNext() 
	{
		// Make sure that we dont point beyond the list
		// ... and make sure we dont point beyond the number added (for length smaller than size)
		if (pointer < size && pointer < added-1) {
			pointer++;
		}

		return getCurrentCommand();
	}

	string ConsoleHistory::getPrevious() 
	{
		if (pointer >= 0) {
			pointer--;    
		}

		return getCurrentCommand();
	}

	const string ConsoleHistory::getCurrentCommand() const
	{
		// In case we dont have a pointer yet ..happens when key down is pressed as the first key
		if (pointer == -1)
			return "";

		// If the pointer is within range of the list we get that command
		else if (int(commands.size()) > pointer)
			return getCommand(pointer);
		else if (commands.size() > 0)
			return commands.back();
		else 
			return "";
	}

	const string ConsoleHistory::getCommand(int n) const
	{
		int i = 0;
		string value = ""; 

		list<string>::const_iterator iter;
		for (iter=commands.begin(); iter != commands.end(); iter++) {
			if (i == n) value.append(iter->c_str());
			i++;
		}

		return value;
	}

}
