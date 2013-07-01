#ifndef _RESTLESS_CONSOLE_CONSOLECOMMANDREQUEST_H
#define _RESTLESS_CONSOLE_CONSOLECOMMANDREQUEST_H

#include "../Util/StringTokenizer.h"

#include <sstream>

namespace restless
{
	class ConsoleCommandRequest
	{
	public:
		ConsoleCommandRequest(std::string & line, std::ostream & outputStream) :
			_tokens(line, " "),
			_out(outputStream)
		{
			_command = _tokens.nextToken();
		}

		const std::string & command() const
		{
			return _command;
		}

		StringTokenizer & tokens()
		{
			return _tokens;
		}

		std::ostream & out()
		{
			return _out;
		}

	protected:
		std::string _command;
		StringTokenizer _tokens;
		std::ostream & _out;

		
	};
}

#endif